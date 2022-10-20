import logging
import random
import numpy as np
import torch
import cv2
import torch.nn.functional as F
from torch.utils.data import Dataset

import core.datasets.camera_pipeline as camera
from core.utils import torch_to_npimage, npimage_to_torch, flatten_raw_image

""" File based on https://github.com/Algolzw/EBSR/blob/main/datasets/synthetic_burst_train_set.py and 
https://github.com/Algolzw/EBSR/blob/main/data_processing/synthetic_burst_generation.py """


class SyntheticRaw(Dataset):
  """ The burst is converted to linear sensor space using the inverse camera pipeline employed in [1].
  The generated raw burst is then mosaicked, and corrupted by random noise to obtain the RAW burst.

  [1] Unprocessing Images for Learned Raw Denoising, Brooks, Tim and Mildenhall, Ben and Xue, Tianfan and Chen,
  Jiawen and Sharlet, Dillon and Barron, Jonathan T, CVPR 2019
  """
  def __init__(self, base_dataset, config):

    self.base_dataset = base_dataset
    self.n_train_files = self.base_dataset.n_train_files
    self.n_test_files = self.base_dataset.n_test_files

    self.burst_size = config.data.burst_size
    self.downsample_factor = config.data.downsample_factor

    self.burst_transformation_params = {'max_translation': config.data.max_translation,
                                        'max_rotation': config.data.max_rotation,
                                        'max_shear': config.data.max_shear,
                                        'max_scale': config.data.max_scale
                                       }
    self.image_processing_params = {'random_ccm': True, 
                                    'random_gains': True, 
                                    'smoothstep': True, 
                                    'gamma': True, 
                                    'add_noise': True}
    self.interpolation_type = 'bilinear'

    # Sample camera pipeline params
    if self.image_processing_params['random_ccm']:
      self.rgb2cam = camera.random_ccm()
    else:
      self.rgb2cam = torch.eye(3).float()
    self.cam2rgb = self.rgb2cam.inverse()

    # Sample gains
    if self.image_processing_params['random_gains']:
      self.rgb_gain, self.red_gain, self.blue_gain = camera.random_gains()
    else:
      self.rgb_gain, self.red_gain, self.blue_gain = (1.0, 1.0, 1.0)

  def __len__(self):
    return len(self.base_dataset)

  def rgbburst2rawburst(self, burst): 

    # Approximately inverts global tone mapping.
    smoothstep = self.image_processing_params['smoothstep']
    if smoothstep:
      burst = camera.invert_smoothstep(burst)

    # Inverts gamma compression.
    gamma = self.image_processing_params['gamma']
    if gamma:
      burst = camera.gamma_expansion(burst)

    # Inverts color correction.
    burst = camera.apply_ccm(burst, self.rgb2cam)

    # Approximately inverts white balance and brightening.
    burst = camera.safe_invert_gains(
      burst, self.rgb_gain, self.red_gain, self.blue_gain)

    # Clip saturated pixels.
    burst = burst.clamp(0.0, 1.0)

    # mosaic
    raw_burst = camera.mosaic(burst.clone())

    # Add noise
    if self.image_processing_params['add_noise']:
      shot_noise_level, read_noise_level = camera.random_noise_levels()
      raw_burst = camera.add_noise(raw_burst, shot_noise_level, read_noise_level)
    else:
      shot_noise_level = 0
      read_noise_level = 0

    # Clip saturated pixels.
    raw_burst = raw_burst.clamp(0.0, 1.0)

    meta_info = {
      'rgb2cam': self.rgb2cam, 'cam2rgb': self.cam2rgb, 
      'rgb_gain': self.rgb_gain, 'red_gain': self.red_gain, 'blue_gain': self.blue_gain, 
      'smoothstep': smoothstep, 'gamma': gamma,
      'shot_noise_level': shot_noise_level, 'read_noise_level': read_noise_level
    }

    return raw_burst, meta_info

  def __getitem__(self, index):
    burst, image_gt = self.base_dataset[index]
    raw_burst, meta_info = self.rgbburst2rawburst(burst)
    raw_burst = flatten_raw_image(raw_burst)
    return raw_burst, image_gt



