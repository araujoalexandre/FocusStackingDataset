import logging
import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast


class ChannelPool(nn.MaxPool1d):

  def forward(self, x):
    b, n, c, w, h = x.size()
    x = x.view(b, n, c * w * h).permute(0, 2, 1)
    pooled = F.max_pool1d(
        x,
        self.kernel_size,
        self.stride,
        self.padding,
        self.dilation,
        self.ceil_mode,
        self.return_indices,
    )
    pooled = pooled.permute(0, 2, 1)
    return pooled.view(b, c, w, h)


class MultiResolutionLayerDown(nn.Module):

  def __init__(self, channels, wn):
    super(MultiResolutionLayerDown, self).__init__()
    ksize = 3
    padding = ksize // 2
    self.conv1 = wn(nn.Conv2d(channels, channels, kernel_size=ksize, padding=padding, stride=2, bias=True))
    self.conv2 = wn(nn.Conv2d(channels, channels, kernel_size=ksize, padding=padding, stride=2, bias=True))
    self.conv3 = wn(nn.Conv2d(channels, channels, kernel_size=ksize, padding=padding, stride=2, bias=True))
    self.conv4 = wn(nn.Conv2d(channels, channels, kernel_size=ksize, padding=padding, stride=2, bias=True))
    self.pool = ChannelPool(30)

  def forward(self, x):
    b, n, c, w, h = x.shape
    x = x.reshape(b*n, c, h, w)
    x1 = self.conv1(x).reshape(b, n, c, h//2, w//2)
    x2 = self.pool(self.conv2(x).reshape(b, n, c, h//2, w//2))
    x3 = self.pool(self.conv3(x).reshape(b, n, c, h//2, w//2))
    x4 = self.pool(self.conv4(x).reshape(b, n, c, h//2, w//2))
    return x1, x2, x3, x4

class MultiResolutionLayerUp(nn.Module):

  def __init__(self, features, wn):
    super(MultiResolutionLayerUp, self).__init__()
    self.conv = wn(nn.ConvTranspose2d(features, features, kernel_size=2, stride=2, padding=0, bias=True))

  def forward(self, x):
    x = self.conv(x)[:, :, ::2, ::2]
    return x

class MultiResolutionFusion(nn.Module):
  
  def __init__(self, config):
    super(MultiResolutionFusion, self).__init__()

    wn = torch.nn.utils.weight_norm
    channels = 3
    features = config.archi.n_feats
    burst_size = config.data.burst_size

    self.conv1 = wn(nn.Conv2d(3, features, kernel_size=3, padding=1, stride=1, bias=True))
    self.conv2 = wn(nn.Conv2d(features, features, kernel_size=3, padding=1, stride=1, bias=True))
    self.conv3 = wn(nn.Conv2d(features, features, kernel_size=3, padding=1, stride=1, bias=True))

    self.conv4 = wn(nn.Conv2d(features, features, kernel_size=3, padding=1, bias=True))
    self.conv5 = wn(nn.Conv2d(features, 3, kernel_size=3, padding=1, bias=True))

    self.layer1 = MultiResolutionLayerDown(features, wn)
    self.layer2 = MultiResolutionLayerDown(features, wn)
    self.layer3 = MultiResolutionLayerDown(features, wn)
    self.layer4 = MultiResolutionLayerDown(features, wn)
    self.pool = ChannelPool(burst_size)
    self.layer5 = MultiResolutionLayerUp(features, wn)
    self.layer6 = MultiResolutionLayerUp(features, wn)
    self.layer7 = MultiResolutionLayerUp(features, wn)
    self.layer8 = MultiResolutionLayerUp(features, wn)
    self.act = nn.ReLU()

  def concat(self, x1, x2, x3, x4):
    return torch.cat([
        torch.cat([x1, x2], axis=2),
        torch.cat([x3, x4], axis=2),
      ], axis=3)
  
  def forward(self, x):
    batch_size, burst_size, channels, h, w = x.shape

    x = x.reshape(batch_size*burst_size, channels, h, w)
    x = self.act(self.conv1(x))
    x = self.act(self.conv2(x))
    x = self.act(self.conv3(x))
    _, channels, h, w = x.shape
    x = x.reshape(batch_size, burst_size, channels, h, w)

    x1_1, x2_1, x3_1, x4_1 = self.layer1(x)
    x1_2, x2_2, x3_2, x4_2 = self.layer2(x1_1)
    x1_3, x2_3, x3_3, x4_3 = self.layer3(x1_2)
    x1_4, x2_4, x3_4, x4_4 = self.layer4(x1_3)
    x1_1, x1_2, x1_3, x1_4 = self.pool(x1_1), self.pool(x1_2), self.pool(x1_3), self.pool(x1_4)
    x1_3 = self.layer5(self.concat(x1_4, x2_4, x3_4, x4_4))
    x1_2 = self.layer6(self.concat(x1_3, x2_3, x3_3, x4_3))
    x1_1 = self.layer7(self.concat(x1_2, x2_2, x3_2, x4_2))
    x = self.layer8(self.concat(x1_1, x2_1, x3_1, x4_1))

    x = self.act(self.conv4(x))
    x = self.conv5(x)
    return x


class FusionNonLinearWeightedAverageModel(nn.Module):

  def __init__(self, config):
    super(FusionNonLinearWeightedAverageModel, self).__init__()
    channels = 3
    burst_size = config.data.burst_size
    crop_size = config.data.crop_sz
    self.conv1 = nn.Conv3d(burst_size, burst_size, kernel_size=channels, padding=channels//2, bias=True)
    self.conv2 = nn.Conv3d(burst_size, burst_size, kernel_size=channels, padding=channels//2, bias=True)
    self.conv3 = nn.Conv3d(burst_size, burst_size, kernel_size=channels, padding=channels//2, bias=True)
    self.avg_weights = nn.Parameter(torch.randn(burst_size, 1, crop_size, crop_size), requires_grad=True)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.relu(self.conv1(x))
    x = self.relu(self.conv2(x))
    x = self.relu(self.conv3(x))
    x = x * torch.softmax(self.avg_weights, dim=0)
    x = x.sum(1)
    return x

