
import os
import torch
import glob
import cv2
import natsort
import logging
import numpy as np
import pandas as pd
import PIL.Image as Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from os.path import join
from torch.utils.data import Dataset
from cv2 import imread
from core.data_utils import unpack
import core.datasets.camera_pipeline as camera


class FocusDataset:

  def __init__(self, config, root, split='train', transform=None):
    self.dataset_id = config.project.dataset_id
    self.crop_sz = config.data.crop_sz
    self.split = split
    self.add_noise = config.data.add_noise
    root = join(root, 'dataset')
    df = pd.read_csv(join(root, 'dataset.csv'), sep=";")
    self.n_train_files = (df['set'] == 'train').sum()
    self.n_test_files = (df['set'] == 'test').sum()
    df = df[df['set'] == split][['lens', 'photo']]
    if config.data.overfit is not None and split == 'train':
      df = df[df['photo'] == config.data.overfit]
      self.n_train_files = 1
    self.bursts_list = df.apply(lambda x: join(root, split, x[0], x[1]), axis=1).values
    self.bursts_list = list(self.bursts_list)
    self.crop_list = []
    ncrop = len(glob.glob(join(self.bursts_list[0], f'crops{self.dataset_id}', 'crop*')))
    for burst_path in self.bursts_list:
      for i in range(ncrop):
        crop_path = f'{burst_path}/crops{self.dataset_id}/crop{i}.pkl'
        self.crop_list.append(crop_path)
    self.raw = False
    if 'raw' in self.dataset_id:
      self.raw = True

  def __len__(self):
    return len(self.crop_list)

  def __getitem__(self, idx):
    burst_path = self.crop_list[idx]
    data = torch.load(burst_path)
    burst, target = data['burst'], data['target']
    if not self.raw:
      burst = burst.float() / 255
    target = target.float() / 255
    if self.split == 'train' and not self.raw:
      # random crop for jpg pipeline
      crop = transforms.RandomCrop(size=self.crop_sz)
      params = crop.get_params(burst, (self.crop_sz, self.crop_sz))
      burst, target = F.crop(burst, *params), F.crop(target, *params)
    elif self.split == 'train' and self.raw:
      # center crop for raw pipeline
      burst = unpack(burst)
      crop = (burst.shape[-1] - self.crop_sz) // 2
      burst = burst[:, :, crop:-crop or None, crop:-crop or None]
      target = target[:, :, crop:-crop or None, crop:-crop or None]

      if self.add_noise:
        shot_noise_level, read_noise_level = camera.random_noise_levels()
        burst = camera.add_noise(burst, shot_noise_level, read_noise_level)

    return burst, target
    

