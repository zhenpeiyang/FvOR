# # Code obtain from https://zju3dv.github.io/loftr/
import os
import math
import numpy as np
from collections import abc
from loguru import logger
import torch
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from os import path as osp
from pathlib import Path
from joblib import Parallel, delayed

import pytorch_lightning as pl
from torch import distributed as dist
from torch.utils.data import (Dataset, DataLoader, ConcatDataset,
                              DistributedSampler, RandomSampler, dataloader)
from utils import tqdm_joblib, get_local_split
from lib import comm
from torch.utils.data import Sampler, ConcatDataset


class RandomConcatSampler(Sampler):
    """ Random sampler for ConcatDataset. At each epoch, `n_samples_per_subset` samples will be draw from each subset
    in the ConcatDataset. If `subset_replacement` is ``True``, sampling within each subset will be done with replacement.
    However, it is impossible to sample data without replacement between epochs, unless bulding a stateful sampler lived along the entire training phase.
    
    For current implementation, the randomness of sampling is ensured no matter the sampler is recreated across epochs or not and call `torch.manual_seed()` or not.
    Args:
        shuffle (bool): shuffle the random sampled indices across all sub-datsets.
        repeat (int): repeatedly use the sampled indices multiple times for training.
            [arXiv:1902.05509, arXiv:1901.09335]
    NOTE: Don't re-initialize the sampler between epochs (will lead to repeated samples)
    NOTE: This sampler behaves differently with DistributedSampler.
          It assume the dataset is splitted across ranks instead of replicated.
    TODO: Add a `set_epoch()` method to fullfill sampling without replacement across epochs.
          ref: https://github.com/PyTorchLightning/pytorch-lightning/blob/e9846dd758cfb1500eb9dba2d86f6912eb487587/pytorch_lightning/trainer/training_loop.py#L373
    """
    def __init__(self,
                 data_source: ConcatDataset,
                 n_samples_per_subset: int,
                 subset_replacement: bool = True,
                 shuffle: bool = True,
                 repeat: int = 1,
                 seed: int = None):
        if not isinstance(data_source, ConcatDataset):
            raise TypeError(
                "data_source should be torch.utils.data.ConcatDataset")

        self.data_source = data_source
        self.n_subset = len(self.data_source.datasets)
        self.n_samples_per_subset = n_samples_per_subset
        self.n_samples = self.n_subset * self.n_samples_per_subset * repeat
        self.subset_replacement = subset_replacement
        self.repeat = repeat
        self.shuffle = shuffle
        self.generator = torch.manual_seed(seed)
        assert self.repeat >= 1

    def __len__(self):
        return self.n_samples

    def __iter__(self):
        indices = []
        # sample from each sub-dataset
        for d_idx in range(self.n_subset):
            low = 0 if d_idx == 0 else self.data_source.cumulative_sizes[d_idx
                                                                         - 1]
            high = self.data_source.cumulative_sizes[d_idx]
            if self.subset_replacement:
                rand_tensor = torch.randint(low,
                                            high,
                                            (self.n_samples_per_subset, ),
                                            generator=self.generator,
                                            dtype=torch.int64)
            else:  # sample without replacement
                len_subset = len(self.data_source.datasets[d_idx])
                rand_tensor = torch.randperm(len_subset,
                                             generator=self.generator) + low
                if len_subset >= self.n_samples_per_subset:
                    rand_tensor = rand_tensor[:self.n_samples_per_subset]
                else:  # padding with replacement
                    rand_tensor_replacement = torch.randint(
                        low,
                        high, (self.n_samples_per_subset - len_subset, ),
                        generator=self.generator,
                        dtype=torch.int64)
                    rand_tensor = torch.cat(
                        [rand_tensor, rand_tensor_replacement])
            indices.append(rand_tensor)
        indices = torch.cat(indices)
        if self.shuffle:  # shuffle the sampled dataset (from multiple subsets)
            rand_tensor = torch.randperm(len(indices),
                                         generator=self.generator)
            indices = indices[rand_tensor]

        # repeat the sampled indices (can be used for RepeatAugmentation or pure RepeatSampling)
        if self.repeat > 1:
            repeat_indices = [indices.clone() for _ in range(self.repeat - 1)]
            if self.shuffle:
                _choice = lambda x: x[torch.randperm(len(x),
                                                     generator=self.generator)]
                repeat_indices = map(_choice, repeat_indices)
            indices = torch.cat([indices, *repeat_indices], 0)

        assert indices.shape[0] == self.n_samples
        return iter(indices.tolist())


class MultiSceneDataModule(pl.LightningDataModule):
    """ 
    For distributed training, each training process is assgined
    only a part of the training scenes to reduce memory overhead.
    """
    def __init__(self, args, config):
        super().__init__()

        self.debug = args.debug
        self.args = args
        self.config = config
        self.data_sampler = config.TRAINER.DATA_SAMPLER
        self.n_samples_per_subset = config.TRAINER.N_SAMPLES_PER_SUBSET
        self.subset_replacement = config.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT
        self.shuffle = config.TRAINER.SB_SUBSET_SHUFFLE
        self.repeat = config.TRAINER.SB_REPEAT

        # 3.loader parameters
        self.train_loader_params = {
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'pin_memory': getattr(args, 'pin_memory', True)
        }
        self.val_loader_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': args.num_workers,
            'pin_memory': getattr(args, 'pin_memory', True)
        }
        self.test_loader_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': args.num_workers,
            'pin_memory': True
        }

        # misc configurations
        self.parallel_load_data = getattr(args, 'parallel_load_data', False)
        self.seed = config.TRAINER.SEED  # 66

    def setup(self, stage=None):
        """
        Setup train / val / test dataset. This method will be called by PL automatically.
        Args:
            stage (str): 'fit' in training phase, and 'test' in testing phase.
        """
        assert stage in ['fit', 'test'], "stage must be either fit or test"
        try:
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        except AssertionError as ae:
            self.world_size = 1
            self.rank = 0
            logger.warning(str(ae) + " (set wolrd_size=1 and rank=0)")

        if stage == 'fit':
            self.train_dataset = self._setup_dataset(mode='train')
            self.val_dataset = self._setup_dataset(mode='val')
            #self.val_dataset = self._setup_dataset(mode='test')
            if self.debug:
                self.val_dataset = self.train_dataset
        else:  # stage == 'test
            self.test_dataset = self._setup_dataset(mode='test')

    def _setup_dataset(self, mode):
        """ Setup train / val / test set"""
        scene_list_path = getattr(self.config.DATASET,
                                  f"{mode.upper()}_LIST_PATH")
        with open(scene_list_path, 'r') as f:
            npz_names = [name.strip() for name in f.readlines()]

        if self.config.DATASET.name == 'shapenet':
            ## Filter certain category for ShapeNet dataset
            if mode == 'train' and len(
                    self.config.DATASET.SHAPENET_CATEGORY_TRAIN):
                npz_names = list(
                    filter(
                        lambda x: x.split('_')[0] in self.config.DATASET.
                        SHAPENET_CATEGORY_TRAIN, npz_names))

        if self.debug:
            idx = 30 if len(npz_names) > 30 else 0
            npz_names = [npz_names[idx]]

        if mode == 'train':
            len0 = len(npz_names)
            local_npz_names = get_local_split(npz_names, self.world_size,
                                              self.rank, self.seed)
            len1 = len(local_npz_names)
            logger.info(
                f"[rank:{self.rank}]: Get local split of TRAIN dataset [{self.rank}]/[{self.world_size}] [{len0}]/[{len1}]!"
            )
        else:
            local_npz_names = npz_names
        return self._build_concat_dataset(local_npz_names, mode)

    def _build_concat_dataset(
        self,
        npz_names,
        mode,
    ):
        datasets = []
        data_source = self.config.DATASET.DATA_SOURCE
        if self.config.DATASET.DATA_SOURCE.lower() == 'shapenet':
            from src.datasets.shapenet import ShapeNetDataset as Dataset
        elif self.config.DATASET.DATA_SOURCE.lower() == 'hm3d_abo':
            from src.datasets.hm3d_abo import HM3DABODataset as Dataset
        else:
            raise Exception("Data Source Not Implemented")
        for npz_name in tqdm(
                npz_names,
                desc=f'[rank:{self.rank}] loading {mode} datasets',
                disable=int(self.rank) != 0):
            datasets.append(
                Dataset(
                    npz_name,
                    self.args,
                    self.config,
                    mode=mode,
                ))
        datasets = list(filter(lambda x: len(x), datasets))
        return ConcatDataset(datasets)

    def train_dataloader(self):
        """ Build training dataloader for ScanNet / MegaDepth. """
        assert self.data_sampler in ['scene_balance']
        logger.info(
            f'[rank:{self.rank}/{self.world_size}]: Train Sampler and DataLoader re-init (should not re-init between epochs!).'
        )
        if self.data_sampler == 'scene_balance':
            sampler = RandomConcatSampler(self.train_dataset,
                                          self.n_samples_per_subset,
                                          self.subset_replacement,
                                          self.shuffle, self.repeat, self.seed)
        else:
            sampler = None
        dataloader = DataLoader(self.train_dataset,
                                sampler=sampler,
                                **self.train_loader_params)
        print('train dataloader size', len(dataloader))
        return dataloader

    def val_dataloader(self):
        """ Build validation dataloader for ScanNet / MegaDepth. """
        logger.info(
            f'[rank:{self.rank}/{self.world_size}]: Val Sampler and DataLoader re-init.'
        )

        if not isinstance(self.val_dataset, abc.Sequence):
            sampler = DistributedSampler(self.val_dataset, shuffle=False)
            dataloader = DataLoader(self.val_dataset,
                                    sampler=sampler,
                                    **self.val_loader_params)
            print('val dataloader size', len(dataloader))
            return dataloader

    def test_dataloader(self, *args, **kwargs):
        sampler = DistributedSampler(self.test_dataset, shuffle=False)
        return DataLoader(self.test_dataset,
                          sampler=sampler,
                          **self.test_loader_params)


def _build_dataset(dataset: Dataset, *args, **kwargs):
    return dataset(*args, **kwargs)
