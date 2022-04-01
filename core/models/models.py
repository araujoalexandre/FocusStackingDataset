
import logging
import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

class FusionLinearModel(nn.Module):

  def __init__(self, config):
    super(FusionLinearModel, self).__init__()
    burst_size = config.data.burst_size
    channels = 3
    self.conv = nn.Conv3d(burst_size, 1, kernel_size=channels, padding=channels//2)

  def forward(self, x):
    return self.conv(x)

class FusionNonLinearModel(nn.Module):

  def __init__(self, config):
    super(FusionNonLinearModel, self).__init__()
    burst_size = config.data.burst_size
    channels = 3
    self.conv1 = nn.Conv3d(burst_size, burst_size, kernel_size=channels, padding=channels//2)
    self.conv2 = nn.Conv3d(burst_size, burst_size, kernel_size=channels, padding=channels//2)
    self.conv3 = nn.Conv3d(burst_size, 1, kernel_size=channels, padding=channels//2)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.relu(self.conv1(x))
    x = self.relu(self.conv2(x))
    x = self.conv3(x)
    return x


models_config = {
  'FusionLinearModel': FusionLinearModel,
  'FusionNonLinearModel': FusionNonLinearModel
}
