from collections import defaultdict
import pprint
from pathlib import Path

import torch
import numpy as np
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from lib.comm import gather, all_gather
from lib.profiler import PassThroughProfiler, build_profiler
from utils import *
import argparse
import importlib
from distutils.util import strtobool
from pytorch_lightning.utilities import rank_zero_only
import math
from pytorch_lightning.loggers import TensorBoardLogger
from datasets.data import MultiSceneDataModule
from loguru import logger as loguru_logger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin
from collections import defaultdict
from contextlib import redirect_stdout

loguru_logger = get_rank_zero_only_logger(loguru_logger)
import sys

sys.path.insert(0, '../')
from src.config.default import get_cfg_defaults
from terminaltables import AsciiTable


def parse_args():
    # init a costum parser which will be added into pl.Trainer parser
    # check documentation: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('main_cfg_path', type=str, help='main config path')
    parser.add_argument('--exp_name', type=str, default='default_exp_name')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--train_scale', type=int, default=None)
    parser.add_argument('--limit_test_number', type=int, default=None)
    parser.add_argument('--multi_gpu_check', action='store_true')
    parser.add_argument('--center_crop', action='store_true')
    parser.add_argument('--save_pose_init', action='store_true')
    parser.add_argument('--batch_size',
                        type=int,
                        default=4,
                        help='batch_size per gpu')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory',
                        type=lambda x: bool(strtobool(x)),
                        nargs='?',
                        default=True,
                        help='whether loading data to pinned memory or not')
    parser.add_argument('--ckpt_path',
                        type=str,
                        default=None,
                        help='pretrained checkpoint path')
    parser.add_argument(
        '--disable_ckpt',
        action='store_true',
        help='disable checkpoint saving (useful for debugging).')
    parser.add_argument(
        '--profiler_name',
        type=str,
        default=None,
        help='options: [inference, pytorch], or leave it unset')
    parser.add_argument('--parallel_load_data',
                        action='store_true',
                        help='load datasets in with multiple processes.')
    parser.add_argument('--trainer', type=str, default='scene_coord')
    parser.add_argument(
        '--dump_dir',
        type=str,
        default='./test_results/',
        help="if set, the matching results will be dump to dump_dir")
    parser.add_argument('--mode', type=str, default='train')

    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()


class Model(pl.LightningModule):
    def __init__(self,
                 config,
                 pretrained_ckpt=None,
                 profiler=None,
                 dump_dir=None):
        """
        TODO:
            - use the new version of PL logging API.
        """
        super().__init__()
        # Misc
        self.config = config  # full config
        self.profiler = profiler or PassThroughProfiler()
        self.n_vals_plot = max(
            config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1)

        module = importlib.import_module(config.TRAINER.MODEL)
        self.model = module.model(config=self.config)

        # Pretrained weights
        if pretrained_ckpt:
            tp = torch.load(pretrained_ckpt, map_location='cpu')['state_dict']
            tp = {k[6:]: v for k, v in tp.items()}  # TODO fix the model name
            try:
                self.model.load_state_dict(tp)
            except Exception as e:
                print(e)
            loguru_logger.info(
                f"Load \'{pretrained_ckpt}\' as pretrained checkpoint")

        #torch.save({'state_dict': {f"model.{k}":v for k, v in self.model.state_dict().items()} }, 'test.ckpt')
        self.dump_dir = dump_dir
        os.makedirs(dump_dir, exist_ok=True)
        with open(f"{dump_dir}/option.yml", 'w') as f:
            with redirect_stdout(f):
                print(config.dump())

    def configure_optimizers(self):
        # FIXME: The scheduler did not work properly when `--resume_from_checkpoint`
        optimizer = build_optimizer(self, self.config)
        scheduler = build_scheduler(self.config, optimizer)
        return [optimizer], [scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                       optimizer_closure, on_tpu, using_native_amp,
                       using_lbfgs):
        # learning rate warm up
        warmup_step = self.config.TRAINER.WARMUP_STEP
        if self.trainer.global_step < warmup_step:
            if self.config.TRAINER.WARMUP_TYPE == 'linear':
                base_lr = self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
                lr = base_lr + \
                    (self.trainer.global_step / self.config.TRAINER.WARMUP_STEP) * \
                    abs(self.config.TRAINER.TRUE_LR - base_lr)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
            elif self.config.TRAINER.WARMUP_TYPE == 'constant':
                pass
            else:
                raise ValueError(
                    f'Unknown lr warm-up strategy: {self.config.TRAINER.WARMUP_TYPE}'
                )

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def _compute_metrics(self, data, mode='train'):
        metrics = defaultdict(list)
        metrics['identifiers'] = data['bag_names']
        metrics['group'] = data['group']
        metrics.update(self.model.compute_metrics(data, mode))
        data['metrics'] = metrics
        return {'metrics': data['metrics']}, data['bag_names']

    def training_step(self, data, batch_idx):
        with self.profiler.profile("Train Forward"):
            self.model(data, 'train')
        with self.profiler.profile("Compute losses"):
            self.model.loss(data, 'train')
        cur_epoch = self.trainer.current_epoch

        # logging
        if self.trainer.global_rank == 0 and self.global_step % self.trainer.log_every_n_steps == 0:
            # scalars
            for k, v in data['loss_scalars'].items():
                self.logger.experiment.add_scalar(f'train/{k}', v,
                                                  self.global_step)

            figures = make_scene_coord_figures(
                data, self.config, self.config.TRAINER.PLOT_MODE)
            for k, v in figures.items():
                self.logger.experiment.add_figure(f'train_scene_coord/{k}',
                                                  v, self.global_step)

        if data['loss'] is None:
            return None
        else:
            return {'loss': data['loss']}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        if self.trainer.global_rank == 0:
            print('avg loss on epoch ', avg_loss)
            self.logger.experiment.add_scalar('train/avg_loss_on_epoch',
                                              avg_loss,
                                              global_step=self.current_epoch)

    def validation_step(self, data, batch_idx):
        with self.profiler.profile("Val Forward"):
            self.model(data, 'val')
        with self.profiler.profile("Compute losses"):
            self.model.loss(data, 'val')

        cur_epoch = self.trainer.current_epoch
        val_plot_interval = max(
            self.trainer.num_val_batches[0] // self.n_vals_plot, 1)

        ret_dict, _ = self._compute_metrics(data, 'test')

        if batch_idx % val_plot_interval == 0:
            figures = self.model.dump_vis(data,
                                          self.dump_dir,
                                          prefix=f"val_epoch_{cur_epoch}")

        figures = {self.config.TRAINER.PLOT_MODE: []}

        return {
            **ret_dict,
            'loss_scalars': data['loss_scalars'],
            'figures': figures,
        }

    def validation_epoch_end(self, outputs):
        # since pl performs sanity_check at the very begining of the training
        cur_epoch = self.trainer.current_epoch
        if not self.trainer.resume_from_checkpoint and self.trainer.running_sanity_check:
            cur_epoch = -1

        # 1. loss_scalars: dict of list, on cpu
        _loss_scalars = [o['loss_scalars'] for o in outputs]
        loss_scalars = {
            k: flattenList(all_gather([_ls[k] for _ls in _loss_scalars]))
            for k in _loss_scalars[0]
        }

        # 2. val metrics: dict of list, numpy
        _metrics = [o['metrics'] for o in outputs]
        metrics = {
            k:
            flattenList(all_gather(flattenList([_me[k] for _me in _metrics])))
            for k in _metrics[0]
        }
        # np.save('metrics.npy', metrics)
        # NOTE: all ranks need to `aggregate_merics`, but only log at rank-0
        val_metrics = self.aggregate_metrics(metrics, {})

        # 3. figures
        _figures = [o['figures'] for o in outputs]
        figures = {
            k: flattenList(gather(flattenList([_me[k] for _me in _figures])))
            for k in _figures[0]
        }

        # tensorboard records only on rank 0
        if self.trainer.global_rank == 0:
            for k, v in loss_scalars.items():
                mean_v = torch.stack(v).mean().item()
                print('Val avg loss ', k, mean_v)
                self.logger.experiment.add_scalar(f'val/avg_{k}',
                                                  mean_v,
                                                  global_step=cur_epoch)

            for k, v in val_metrics.items():
                self.logger.experiment.add_scalar(f"metrics/{k}",
                                                  v,
                                                  global_step=cur_epoch)

            for k, v in figures.items():
                if self.trainer.global_rank == 0:
                    for plot_idx, fig in enumerate(v):
                        self.logger.experiment.add_figure(
                            f'val_scene_coord/{k}/pair-{plot_idx}',
                            fig,
                            cur_epoch,
                            close=True)
            plt.close('all')

        # log on all ranks for ModelCheckpoint callback to work properly
        key = self.config.TRAINER.MONITOR_KEY
        self.log(key, torch.tensor(val_metrics[key]))  # ckpt monitors on this

    def test_step(self, data, batch_idx):
        with self.profiler.profile("Test/Forward"):
            self.model(data, 'test')
        with self.profiler.profile("Test/Compute losses"):
            self.model.loss(data, 'test')

        ret_dict, _ = self._compute_metrics(data, 'test')
        with self.profiler.profile("dump_results"):
            if args.save_pose_init:
                scale_gt = data['bbox_scale_gt']
                if self.config.TRAINER.TRAIN_SCALE:
                    scale_pred = data['scale_pred'].data.cpu().numpy()[0]
                else:
                    scale_pred = 1.0
                np.save(
                    f"./results_pose_init/{data['bag_names'][0]}.npy",
                    {
                        'pred': data['T_pred'][0].data.cpu().numpy(),
                        'gt': data['poses'][0].data.cpu().numpy(),
                        #'err_pixel':
                        #data['pixel_err'][0].data.cpu().numpy(),
                        'err_R': data['metrics']['pose_err_R'][0],
                        'err_t': data['metrics']['pose_err_t'][0],
                        'scale_pred': scale_pred,
                    })

            figures = self.model.dump_vis(data,
                                          self.dump_dir,
                                          prefix=f"test",
                                          metrics=0)

        return {**ret_dict, 'loss_scalars': data['loss_scalars']}

    def test_epoch_end(self, outputs):
        _metrics = [o['metrics'] for o in outputs]
        metrics = {
            k: flattenList(gather(flattenList([_me[k] for _me in _metrics])))
            for k in _metrics[0]
        }
        _loss_scalars = [o['loss_scalars'] for o in outputs]
        print('collect loss scalars')
        loss_scalars = {
            k: flattenList(all_gather([_ls[k] for _ls in _loss_scalars]))
            for k in _loss_scalars[0]
        }
        for k, v in loss_scalars.items():
            loss_scalars[k] = torch.stack(v).mean()

        if self.trainer.global_rank == 0:
            test_metrics = self.aggregate_metrics(metrics, {})
            print(self.profiler.summary())
            loguru_logger.info('Test Losses\n' + pprint.pformat(loss_scalars))
            self.write_results(metrics,
                               test_metrics,
                               kw=[
                                   'pixel_err', 'pose_err_R', 'pose_err_t',
                                   'scene_coord_err_masked'
                               ])
            np.save(self.dump_dir + '/test_metrics.npy', {
                'metrics': metrics,
                'metrics_agg': test_metrics
            })

    def aggregate_metrics(self, metrics, auc_config):
        ###  filter duplicates
        unq_ids = OrderedDict(
            (iden, id) for id, iden in enumerate(metrics['identifiers']))
        unq_ids = list(unq_ids.values())
        loguru_logger.info(
            f'Aggregating metrics over {len(unq_ids)} unique items...')
        metrics_agg = {}

        groups = list(set(metrics['group']))
        for group in groups:
            mask = np.array(metrics['group'])[unq_ids] == group
            for k, v in metrics.items():
                if type(v[0]) in [np.float, np.float32, np.float64]:
                    vals = np.array(v)[unq_ids][mask]
                    metrics_agg[f"{group}:{k}_mean"] = np.mean(vals)
                    metrics_agg[f"{group}:{k}_median"] = np.median(vals)
                    metrics_agg[f"{group}:Count"] = mask.sum()

        for k, v in metrics.items():
            if not len(v):
                continue
            if type(v[0]) in [np.float, np.float32, np.float64]:
                metrics_agg[f"{k}_mean"] = np.mean(np.array(v)[unq_ids])
                metrics_agg[f"{k}_median"] = np.median(np.array(v)[unq_ids])

        return metrics_agg

    def write_results(self, metrics, aggregated_metrics, kw):
        if self.config.DATASET.name == 'shapenet':
            label2cat = {
                '02691156': 'airplane',
                '02933112': 'cabinet',
                '03001627': 'chair',
                '03636649': 'lamp',
                '04090263': 'rifle',
                '04379243': 'table',
                '04530566': 'watercraft',
                '02828884': 'bench',
                '02958343': 'car',
                '03211117': 'display',
                '03691459': 'loudspeaker',
                '04256520': 'sofa',
                '04401088': 'telephone'
            }
            classes = list(set(metrics['group']))
            table_columns = [[label2cat[label]
                              for label in classes] + ['Overall']]
        else:
            classes = list(set(metrics['group']))
            table_columns = [[label for label in classes] + ['Overall']]

        header = ['classes']
        shorthand = {
            'pixel_err': 'Error_Pix',
            'pose_err_R': 'Error_Rot',
            'pose_err_t': 'Error_Trans',
            'scene_coord_err_masked': 'Error_SC'
        }
        for k in kw:
            if k == 'scene_coord_err_masked': continue
            header.append(shorthand[k])
            vals, vals_mean, vals_median = [], [], []
            vals = []
            for label in classes:
                mean = aggregated_metrics[f'{label}:{k}_mean']
                median = aggregated_metrics[f'{label}:{k}_median']
                vals.append(f'{mean:.3f}/{median:.3f}')
                vals_mean.append(mean)
                vals_median.append(median)
            table_columns.append(vals)
            overall_mean = float(np.mean(vals_mean))
            overall_median = float(np.mean(vals_median))
            table_columns[-1] += [f'{overall_mean:.3f}/{overall_median:.3f}']

        table_data = [header]
        table_rows = list(zip(*table_columns))
        table_data += table_rows
        table = AsciiTable(table_data)
        with open('table.txt', 'w') as f:
            f.write(table.table)
        table.inner_footing_row_border = True
        print('\n' + table.table)
        keys = metrics.keys()
        for k in kw:
            filtered_keys = list(filter(lambda x: k in x, keys))
            filtered_keys = sorted(filtered_keys)
            with open(f"{self.dump_dir}/metric_{k}.txt", 'w') as f:
                for x in filtered_keys:
                    f.write(f"{x} {metrics[x]}\n")


def train(args):
    # init default-cfg and merge it with the main- and data-cfg
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    update_cfg_with_args(config, args)
    pl.seed_everything(config.TRAINER.SEED)  # reproducibility

    # TODO: Use different seeds for each dataloader workers
    # This is needed for data augmentation

    if args.debug:
        if not args.multi_gpu_check:
            args.gpus = 1
        args.num_workers = 0
        args.parallel_load_data = False
        config.TRAINER.N_SAMPLES_PER_SUBSET = 500
    # scale lr and warmup-step automatically
    args.gpus = _n_gpus = setup_gpus(args.gpus)
    config.TRAINER.WORLD_SIZE = _n_gpus * args.num_nodes
    config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size
    _scaling = config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
    config.TRAINER.SCALING = _scaling
    config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR * _scaling
    config.TRAINER.WARMUP_STEP = math.floor(config.TRAINER.WARMUP_STEP /
                                            _scaling)

    # TensorBoard Logger
    base_dir = '.'
    logger = TensorBoardLogger(save_dir=os.path.join(base_dir, 'logs/tb_logs'),
                               name=args.exp_name,
                               default_hp_metric=False)
    ckpt_dir = Path(logger.log_dir) / 'checkpoints'
    loguru_logger.info(f"Results will be saved to {logger.log_dir}")
    dump_dir = Path(logger.log_dir) / 'dump'
    os.makedirs(dump_dir, exist_ok=True)

    # lightning module
    profiler = build_profiler(args.profiler_name)
    model = Model(config,
                  pretrained_ckpt=args.ckpt_path,
                  profiler=profiler,
                  dump_dir=dump_dir)
    loguru_logger.info(f"Model initialized!")

    # lightning data
    data_module = MultiSceneDataModule(args, config)
    loguru_logger.info(f"DataModule initialized!")

    # Callbacks
    # TODO: update ModelCheckpoint to monitor multiple metrics
    monitor_key = config.TRAINER.MONITOR_KEY
    loguru_logger.info(f"Monitor key: {monitor_key}")
    ckpt_callback = ModelCheckpoint(monitor=monitor_key,
                                    verbose=True,
                                    save_top_k=5,
                                    mode='min',
                                    save_last=True,
                                    dirpath=str(ckpt_dir),
                                    filename='{epoch}-{%s:.3f}' % monitor_key)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [lr_monitor]
    if not args.disable_ckpt:
        callbacks.append(ckpt_callback)

    # Lightning Trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        plugins=DDPPlugin(find_unused_parameters=True,
                          num_nodes=args.num_nodes,
                          sync_batchnorm=False),
        gradient_clip_val=config.TRAINER.GRADIENT_CLIPPING,
        callbacks=callbacks,
        logger=logger,
        sync_batchnorm=False,
        replace_sampler_ddp=False,  # use custom sampler
        reload_dataloaders_every_epoch=False,  # avoid repeated samples!
        weights_summary='full',
        profiler=profiler)
    loguru_logger.info(f"Trainer initialized!")
    loguru_logger.info(f"Start training!")
    trainer.fit(model, datamodule=data_module)


def test(args):
    # init default-cfg and merge it with the main- and data-cfg
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)

    update_cfg_with_args(config, args)

    pl.seed_everything(config.TRAINER.SEED)  # reproducibility
    args.num_workers = 4
    args.parallel_load_data = False
    args.batch_size = 1
    if args.debug:
        if not args.multi_gpu_check:
            args.gpus = 1
        args.num_workers = 0

    args.gpus = _n_gpus = setup_gpus(args.gpus)
    if args.save_pose_init:
        os.makedirs('./results_pose_init', exist_ok=True)
    args.profiler_name = "inference"

    # lightning module
    base_dir = '.'
    dump_dir = os.path.join(base_dir, args.dump_dir)
    os.makedirs(dump_dir, exist_ok=True)

    profiler = build_profiler(args.profiler_name)
    model = Model(config,
                  pretrained_ckpt=args.ckpt_path,
                  profiler=profiler,
                  dump_dir=dump_dir)

    # lightning data
    data_module = MultiSceneDataModule(args, config)

    # lightning trainer
    trainer = pl.Trainer.from_argparse_args(args,
                                            replace_sampler_ddp=False,
                                            logger=False)

    loguru_logger.info(f"Start testing!")
    trainer.test(model, datamodule=data_module, verbose=False)


if __name__ == '__main__':
    args = parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
