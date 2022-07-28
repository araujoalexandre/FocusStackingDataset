import logging
from os.path import join, exists

import numpy as np
import torch
import torchvision.transforms as transforms
from torch import multiprocessing
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import Compose

from core.datasets.datasets import DdDpCanonDataset, FocusDataset
from core.datasets.synthetic_burst_generation import SyntheticBurst


def get_data_dir(data_dir, dataset_name):
  paths = data_dir.split(':')
  data_dir = None
  for path in paths:
    if exists(join(path, dataset_name)):
      data_dir = path
      break
  if data_dir is None:
    raise ValueError("Data directory not found.")
  return join(data_dir, dataset_name)



class BaseReader:

  def __init__(self, config, batch_size, is_distributed, is_training):
    self.config = config
    self.batch_size = batch_size
    self.is_distributed = is_distributed
    self.is_training = is_training
    self.num_workers = 10
    self.prefetch_factor = self.batch_size * 2
    self.crop_sz = self.config.data.crop_sz
    self.height, self.width = self.crop_sz, self.crop_sz
    self.img_size = (None, 3, self.height, self.height)

  def load_dataset(self):
    """Load or download dataset."""
    sampler = None
    if self.is_distributed:
      sampler = DistributedSampler(self.dataset, shuffle=self.is_training)
    loader = DataLoader(self.dataset,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        # shuffle=self.is_training and not sampler,
                        pin_memory=False,
                        prefetch_factor=self.prefetch_factor,
                        sampler=sampler)
    return loader, sampler



class SimulatedBlurReader(BaseReader):

  def __init__(self, config, batch_size, is_distributed, is_training):
    super(SimulatedBlurReader, self).__init__(config, batch_size, is_distributed, is_training)
    self.dataset_name = 'simulated_blur'
    if self.is_training:
      n_patches_by_images = (1092 // config.data.crop_sz) * (728 // config.data.crop_sz)
    else:
      n_patches_by_images = 1
    self.n_train_files = 350 * n_patches_by_images
    self.n_test_files = 75 * n_patches_by_images
    self.path = get_data_dir(config.project.data_dir, 'dd_dp_dataset_canon')
    split = 'train' if self.is_training else 'val'
    self.dataset = SyntheticBurst(
      DdDpCanonDataset(self.path, split=split), config)


class FocusStackReader(BaseReader):

  def __init__(self, config, batch_size, is_distributed, is_training):
    super(FocusStackReader, self).__init__(config, batch_size, is_distributed, is_training)
    self.dataset_name = 'focus_stack'
    self.path = get_data_dir(config.project.data_dir, 'focus_stack_dataset')
    split = 'train' if self.is_training else 'test'
    self.dataset = FocusDataset(self.path, split=split)
    n_patches_by_images = (5184 // 128) * (3888 // 128)
    self.n_train_files = self.dataset.n_train_files * n_patches_by_images
    self.n_test_files = self.dataset.n_test_files * n_patches_by_images


readers_config  = {
  'simulated_blur': SimulatedBlurReader,
  'focus': FocusStackReader
}

