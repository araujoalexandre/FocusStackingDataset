
import os
import torch
import glob
import cv2
import natsort
import logging
import numpy as np
import random
import pandas as pd
import PIL.Image as Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from os.path import join
from torch.utils.data import Dataset
from cv2 import imread
from core.data_utils import unpack, pack
import core.datasets.camera_pipeline as camera


def get_tmat(image_shape, translation, rotation, scale_factors):
  """ Generates a transformation matrix corresponding to the input transformation parameters """
  im_h, im_w = image_shape
  t_mat = np.identity(3)
  t_mat[0, 2] = translation[0]
  t_mat[1, 2] = translation[1]
  t_rot = cv2.getRotationMatrix2D((im_w * 0.5, im_h * 0.5), rotation, 1.0)
  t_rot = np.concatenate((t_rot, np.array([0.0, 0.0, 1.0]).reshape(1, 3)))
  t_scale = np.array([[scale_factors[0], 0.0, 0.0],
                      [0.0, scale_factors[1], 0.0],
                      [0.0, 0.0, 1.0]])
  t_mat = t_scale @ t_rot @ t_mat
  t_mat = t_mat[:2, :]
  return t_mat


def burst_data_augmentation(burst, config, interpolation_type='bilinear'):
  """ Generates a burst of size burst_size from the input image by applying random transformations defined by
  transformation_params, and downsampling the resulting burst by downsample_factor.
  """
  if interpolation_type == 'bilinear':
    interpolation = cv2.INTER_LINEAR
  elif interpolation_type == 'lanczos':
    interpolation = cv2.INTER_LANCZOS4
  else:
    raise ValueError

  burst = pack(burst)

  h, w = burst.shape[-2], burst.shape[-1]
  burst = burst.permute(0, 2, 3, 1)
  burst = burst.numpy()

  max_translation = config.data.max_translation
  max_rotation = config.data.max_rotation
  max_scale = config.data.max_scale
  downsample_factor = config.data.downsample_factor
  add_noise = config.data.add_noise
    
  burst_t = []
  for i in range(len(burst)):

    translation = (random.uniform(-max_translation, max_translation),
                   random.uniform(-max_translation, max_translation))

    rotation = random.uniform(-max_rotation, max_rotation)
    
    scale_factor = np.exp(random.uniform(-max_scale, max_scale))
    scale_factor = (scale_factor, scale_factor)
    
    # Generate a affine transformation matrix corresponding to the sampled parameters
    t_mat = get_tmat((h, w), translation, rotation, scale_factor)
    
    # Apply the sampled affine transformation
    image_t = cv2.warpAffine(burst[i], t_mat,  (w, h), None, flags=interpolation, borderMode=cv2.BORDER_REFLECT)
      
    # Downsample the image
    factor = 1/downsample_factor
    image_t = cv2.resize(image_t, None, fx=factor, fy=factor, interpolation=interpolation)
    
    image_t = torch.FloatTensor(image_t).float()
    burst_t.append(image_t)
  
  burst_t = torch.stack(burst_t).permute(0, 3, 1, 2)
  
  if add_noise:
    shot_noise_level, read_noise_level = camera.random_noise_levels()
    burst_t = camera.add_noise(burst_t, shot_noise_level, read_noise_level)

  burst_t = unpack(burst_t)
  return burst_t



class FocusDataset:

  def __init__(self, config, root, split='train', transform=None):
    self.config = config
    self.dataset_id = config.project.dataset_id
    self.crop_sz = config.data.crop_sz
    self.split = split
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

      # data augmentation on raw burst
      burst = burst_data_augmentation(burst, self.config)

    return burst, target
    

