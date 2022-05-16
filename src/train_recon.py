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
from models.rend_util import get_camera_params
from collections import OrderedDict
import sys

sys.path.insert(0, '../')
from src.config.default import get_cfg_defaults
from terminaltables import AsciiTable

loguru_logger = get_rank_zero_only_logger(loguru_logger)

def parse_args():
    # init a costum parser which will be added into pl.Trainer parser
    # check documentation: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('main_cfg_path', type=str, help='main config path')
    parser.add_argument('--exp_name', type=str, default='default_exp_name')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--multi_gpu_check', action='store_true')
    parser.add_argument('--center_crop', action='store_true')
    parser.add_argument('--batch_size',
                        type=int,
                        default=4,
                        help='batch_size per gpu')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--limit_test_number', type=int, default=None)
    parser.add_argument('--lambda_reg', type=float, default=None)
    parser.add_argument('--lambda_damping', type=float, default=None)
    parser.add_argument('--noise_std', type=float, default=None)
    parser.add_argument('--eval_per_joint_step', action='store_true')
    parser.add_argument('--rigid_align_to_gt', action='store_true')
    parser.add_argument('--use_gt_pose', action='store_true')
    parser.add_argument('--use_noisy_pose', action='store_true')
    parser.add_argument('--use_predicted_pose', action='store_true')
    parser.add_argument('--flownet_normalize_feature_map',
                        type=int,
                        default=None)
    parser.add_argument('--joint_step', type=int, default=None)
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

        self.dump_dir = dump_dir
        os.makedirs(dump_dir, exist_ok=True)
        with open(f"{dump_dir}/option.yml", 'w') as f:
            with redirect_stdout(f):
                print(config.dump())

    def on_after_backward(self) -> None:
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any()
                                       or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break
        if not valid_gradients:
            loguru_logger.warning(
                f'detected inf or nan values in gradients. not updating model parameters'
            )
            self.zero_grad()

    def configure_optimizers(self):
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
                    finite_mask = np.isfinite(np.array(v)[unq_ids][mask])
                    metrics_agg[f"{group}:{k}_mean"] = np.mean(
                        np.array(v)[unq_ids][mask][finite_mask])
                    metrics_agg[f"{group}:{k}_median"] = np.median(
                        np.array(v)[unq_ids][mask][finite_mask])
                    metrics_agg[f"{group}:Count"] = mask.sum()

        for k in auc_config.keys():
            if k in metrics:
                err = np.array(metrics[k])[unq_ids]
                aucs = error_auc(err, auc_config[k],
                                 prefix=k + '_')  # (auc@5, auc@10, auc@20)
                metrics_agg.update(aucs)
        for k, v in metrics.items():
            if not len(v):
                continue
            if type(v[0]) in [np.float, np.float32, np.float64]:
                finite_mask = np.isfinite(np.array(v)[unq_ids])
                metrics_agg[f"{k}_mean"] = np.mean(
                    np.array(v)[unq_ids][finite_mask])
                metrics_agg[f"{k}_median"] = np.median(
                    np.array(v)[unq_ids][finite_mask])

        return metrics_agg

    def write_results(self, metrics, aggregated_metrics, kw):
        unq_ids = OrderedDict(
            (iden, id) for id, iden in enumerate(metrics['identifiers']))
        unq_ids = list(unq_ids.values())
        identifiers = np.array(metrics['identifiers'])[unq_ids]
        idx = np.argsort(identifiers)
        for k in kw:
            ids = [identifiers[x] for x in idx]
            tp = np.array(metrics[k])[unq_ids]
            data = [tp[x] for x in idx]
            with open(f"{self.dump_dir}/detailed_metric_{k}.txt", 'w') as f:
                for i in range(len(ids)):
                    f.write(f"{ids[i]} {data[i]}\n")

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
            table_column_header = [label2cat[label]
                                   for label in classes] + ['Overall']
        else:
            classes = list(set(metrics['group']))
            table_column_header = [label for label in classes] + ['Overall']

        shorthand = {
            'IoU': 'IoU',
            'chamfer-L1': 'ChamferL1',
            'Normal-Consistency': 'Normal',
        }
        header = ['classes']
        table_columns = [table_column_header]
        for k in kw:
            header.append(shorthand[k])
            vals, overall_mean, overall_median = [], [], []
            for label in classes:
                mean = aggregated_metrics[f'{label}:{k}_mean']
                median = aggregated_metrics[f'{label}:{k}_median']
                vals.append(f"{mean:.5f}/{median:.5f}")
                overall_mean.append(mean)
                overall_median.append(median)
            table_columns.append(vals)
            overall_mean = float(np.mean(overall_mean))
            overall_median = float(np.mean(overall_median))
            table_columns[-1] += [f'{overall_mean:.5f}/{overall_median:.5f}']
        table_data = [header]
        table_rows = list(zip(*table_columns))
        table_data += table_rows
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        with open('table.txt', 'w') as f:
            f.write(table.table)
        print('\n' + table.table)

        if self.config.TRAINER.EVAL_PER_JOINT_STEP:
            keys = ['IoU', 'chamfer-L1', 'Normal-Consistency']
            for k in keys:
                header = ['classes']
                table_columns = [table_column_header]
                for i in range(self.config.POSE_REFINE.JOINT_STEP + 1):
                    header.append(f'step{i}')
                    vals, overall_mean, overall_median = [], [], []
                    for label in classes:
                        mean = aggregated_metrics[f'{label}:{k}_step{i}_mean']
                        median = aggregated_metrics[
                            f'{label}:{k}_step{i}_median']
                        vals.append(f"{mean:.5f}/{median:.5f}")
                        overall_mean.append(mean)
                        overall_median.append(median)
                    table_columns.append(vals)
                    overall_mean = float(np.mean(overall_mean))
                    overall_median = float(np.mean(overall_median))
                    table_columns[-1] += [
                        f'{overall_mean:.5f}/{overall_median:.5f}'
                    ]
                table_data = [header]
                table_rows = list(zip(*table_columns))
                table_data += table_rows
                table = AsciiTable(table_data)
                table.inner_footing_row_border = True
                print(f'{k}\n' + table.table + '\n')

        keys = aggregated_metrics.keys()
        for k in kw:
            filtered_keys = list(filter(lambda x: k in x, keys))
            filtered_keys = sorted(filtered_keys)
            with open(f"{self.dump_dir}/metric_{k}.txt", 'w') as f:
                for x in filtered_keys:
                    f.write(f"{x} {aggregated_metrics[x]}\n")

    @torch.no_grad()
    def _compute_metrics(self,
                         data,
                         mode='train',
                         mesh_name='mesh_pred',
                         pose_name='poses_cur'):
        with self.profiler.profile("Compute metrics"):
            metrics = defaultdict(list)
            metrics['identifiers'] = data['bag_names']
            metrics['group'] = data['group']
            metrics.update(
                self.model.compute_metrics(data,
                                           mesh_name,
                                           mode=mode,
                                           pose_name=pose_name))
            data['metrics'] = metrics
        return {'metrics': data['metrics']}, data['bag_names']

    def training_step(self, data, batch_idx):
        with self.profiler.profile("Train Forward"):
            self.model(data, 'train')
        with self.profiler.profile("Compute losses"):
            self.model.loss(data, 'train')
        cur_epoch = self.trainer.current_epoch

        if self.trainer.global_rank == 0 and self.global_step % self.trainer.log_every_n_steps == 0:
            for k, v in data['loss_scalars'].items():
                self.logger.experiment.add_scalar(f'train/{k}', v,
                                                  self.global_step)
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

        self.model.extract_mesh(data)
        ret_dict, _ = self._compute_metrics(data, 'test')

        if batch_idx % val_plot_interval == 0:
            figures = self.model.dump_vis(data,
                                          self.dump_dir,
                                          prefix=f"val_epoch_{cur_epoch}",
                                          mesh_name='mesh_pred')

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
            ## Log losses
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

            ## Log figures
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
        st = cuda_time()
        with self.profiler.profile("Test/Forward"):
            self.model(data, 'test')

        with self.profiler.profile("Test/Compute losses"):
            self.model.loss(data, 'test')

        if self.config.TRAINER.TRAIN_POSE:
            dump_dir = f"{self.dump_dir}/{data['bag_names'][0]}"
            os.makedirs(dump_dir, exist_ok=True)
            self.model.joint_update_pose_and_shape(data,
                                                   write=True,
                                                   dump_dir=dump_dir,
                                                   verbose=args.verbose)
        else:
            self.model.extract_mesh(
                data,
                mesh_fn='mesh_final',
                verbose=args.verbose,
                align=self.config.TRAINER.RIGID_ALIGN_TO_GT)
        metrics = dict()
        if self.config.TRAINER.RIGID_ALIGN_TO_GT:
            if self.config.TRAINER.EVAL_PER_JOINT_STEP:
                for i in range(self.config.POSE_REFINE.JOINT_STEP + 1):
                    mesh_fn = 'mesh_step_%d' % i
                    ret_dict, _ = self._compute_metrics(
                        data,
                        'test',
                        mesh_name=mesh_fn,
                        pose_name='poses_cur_step_%d' % i)
                    for k, v in ret_dict['metrics'].items():
                        metrics[f'{k}_step{i}'] = v
        ret_dict, _ = self._compute_metrics(data,
                                            'test',
                                            mesh_name='mesh_final',
                                            pose_name='poses_cur')
        metrics.update(ret_dict['metrics'])
        if batch_idx % 1 == 0:
            with self.profiler.profile("dump_results"):
                figures = self.model.dump_vis(
                    data,
                    self.dump_dir,
                    prefix=f"test",
                    metrics=ret_dict['metrics']['IoU'],
                    mesh_name='mesh_final')

        return {'metrics': metrics, 'loss_scalars': data['loss_scalars']}

    def test_epoch_end(self, outputs):
        _metrics = [o['metrics'] for o in outputs]
        metrics = {
            k: flattenList(gather(flattenList([_me[k] for _me in _metrics])))
            for k in _metrics[0]
        }

        _loss_scalars = [o['loss_scalars'] for o in outputs]
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
                               kw=['IoU', 'chamfer-L1', 'Normal-Consistency'])
            np.save(self.dump_dir + '/test_metrics.npy', test_metrics)


def train(args):
    # init default-cfg and merge it with the main- and data-cfg
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    update_cfg_with_args(config, args)
    if len(config.DATASET.SHAPENET_CATEGORY_TRAIN):
        val = config.DATASET.SHAPENET_CATEGORY_TRAIN.split('-')
        setattr(config.DATASET, 'SHAPENET_CATEGORY_TRAIN', val)
    if len(config.DATASET.SHAPENET_CATEGORY_TEST):
        val = config.DATASET.SHAPENET_CATEGORY_TEST.split('-')
        config.DATASET.SHAPENET_CATEGORY_TEST = val

    pl.seed_everything(config.TRAINER.SEED)  # reproducibility

    if args.debug:
        if not args.multi_gpu_check:
            args.gpus = 1
        args.num_workers = 0
        args.parallel_load_data = False
        config.TRAINER.N_SAMPLES_PER_SUBSET = 500
    # scale lr and warmup-step automatically
    args.gpus = _n_gpus = setup_gpus(args.gpus)
    if config.TRAINER.BATCH_SIZE > 0:
        args.batch_size = config.TRAINER.BATCH_SIZE
    config.TRAINER.WORLD_SIZE = _n_gpus * args.num_nodes
    config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size
    _scaling = config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
    config.TRAINER.SCALING = _scaling
    config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR * _scaling
    config.TRAINER.WARMUP_STEP = math.floor(config.TRAINER.WARMUP_STEP /
                                            _scaling)

    # TensorBoard Logger
    base_dir = './'
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

    # lightning data
    data_module = MultiSceneDataModule(args, config)

    monitor_key = config.TRAINER.MONITOR_KEY
    ckpt_callback = ModelCheckpoint(monitor=monitor_key,
                                    verbose=True,
                                    save_top_k=-1,
                                    mode='max',
                                    save_last=True,
                                    dirpath=str(ckpt_dir),
                                    filename='{epoch}-{%s:.3f}' % monitor_key)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [lr_monitor]
    if not args.disable_ckpt:
        callbacks.append(ckpt_callback)

    # Lightning Trainer
    sync_batchnorm = False
    trainer = pl.Trainer.from_argparse_args(
        args,
        plugins=DDPPlugin(find_unused_parameters=True,
                          num_nodes=args.num_nodes,
                          sync_batchnorm=sync_batchnorm),
        gradient_clip_val=config.TRAINER.GRADIENT_CLIPPING,
        callbacks=callbacks,
        logger=logger,
        sync_batchnorm=sync_batchnorm,
        replace_sampler_ddp=False,  # use custom sampler
        reload_dataloaders_every_epoch=False,  # avoid repeated samples!
        weights_summary='full',
        profiler=profiler)
    loguru_logger.info(f"Start training!")
    trainer.fit(model, datamodule=data_module)


def test(args):
    # init default-cfg and merge it with the main- and data-cfg
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)

    update_cfg_with_args(config, args)
    if len(config.DATASET.SHAPENET_CATEGORY_TRAIN):
        val = config.DATASET.SHAPENET_CATEGORY_TRAIN.split('-')
        setattr(config.DATASET, 'SHAPENET_CATEGORY_TRAIN', val)
    if len(config.DATASET.SHAPENET_CATEGORY_TEST):
        val = config.DATASET.SHAPENET_CATEGORY_TEST.split('-')
        config.DATASET.SHAPENET_CATEGORY_TEST = val

    pl.seed_everything(config.TRAINER.SEED)  # reproducibility
    args.num_workers = 4
    args.parallel_load_data = False
    args.batch_size = 1
    if args.debug:
        if not args.multi_gpu_check:
            args.gpus = 1
        args.num_workers = 0

    args.profiler_name = "inference"

    # lightning module
    base_dir = './'
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
