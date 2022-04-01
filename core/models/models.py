
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


class FusionLinearWeightedAverageModel(nn.Module):

  def __init__(self, config):
    super(FusionWeightedAverageModel, self).__init__()
    burst_size = config.data.burst_size
    channels = 3
    burst_size = config.data.burst_size
    crop_size = config.data.crop_sz
    self.conv1 = nn.Conv3d(burst_size, burst_size, kernel_size=channels, padding=channels//2)
    self.avg_weights = nn.Parameter(torch.randn(burst_size, 1, crop_size, crop_size), requires_grad=True)

  def forward(self, x):
    x = self.conv1(x)
    x = x * torch.softmax(self.avg_weights, dim=0)
    x = x.sum(1)
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


class BasicBlock(nn.Module):
  
  def __init__(self, cin, cout, ksize=3, stride=1, padding=1, bias=False):
    super(BasicBlock, self).__init__()
    
    self.conv1 = nn.Conv2d(cin, cout, kernel_size=ksize, stride=stride, padding=padding, bias=bias)
    self.conv2 = nn.Conv2d(cout, cout, kernel_size=ksize, stride=stride, padding=padding, bias=bias)
    self.conv3 = nn.Conv2d(cout, cout, kernel_size=ksize, stride=stride, padding=padding, bias=bias)
    self.relu = nn.ReLU()
    
  def forward(self, x):
    x = self.relu(self.conv1(x))
    x = self.relu(self.conv2(x))
    x = self.relu(self.conv3(x))
    return x
    

class UNet(nn.Module):

  def __init__(self, config):
    super(UNet, self).__init__()
    features = config.archi.n_channels
    in_channels = 3 * config.data.burst_size
    self.encoder1 = BasicBlock(in_channels, features)
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.encoder2 = BasicBlock(features, features * 2)
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.encoder3 = BasicBlock(features * 2, features * 4)
    self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.encoder4 = BasicBlock(features * 4, features * 8)
    self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.bottleneck = BasicBlock(features * 8, features * 16)
    self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
    self.decoder4 = BasicBlock((features * 8) * 2, features * 8)
    self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
    self.decoder3 = BasicBlock((features * 4) * 2, features * 4)
    self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
    self.decoder2 = BasicBlock((features * 2) * 2, features * 2)
    self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
    self.decoder1 = BasicBlock(features * 2, features)
    self.conv = nn.Conv2d(features, 3, kernel_size=1)

  def forward(self, x):
    batch_size, burst_size, channels, h, w = x.shape
    x = x.reshape(batch_size, burst_size*channels, h, w)
    enc1 = self.encoder1(x)
    enc2 = self.encoder2(self.pool1(enc1))
    enc3 = self.encoder3(self.pool2(enc2))
    enc4 = self.encoder4(self.pool3(enc3))
    bottleneck = self.bottleneck(self.pool4(enc4))
    dec4 = self.upconv4(bottleneck)
    dec4 = torch.cat((dec4, enc4), dim=1)
    dec4 = self.decoder4(dec4)
    dec3 = self.upconv3(dec4)
    dec3 = torch.cat((dec3, enc3), dim=1)
    dec3 = self.decoder3(dec3)
    dec2 = self.upconv2(dec3)
    dec2 = torch.cat((dec2, enc2), dim=1)
    dec2 = self.decoder2(dec2)
    dec1 = self.upconv1(dec2)
    dec1 = torch.cat((dec1, enc1), dim=1)
    dec1 = self.decoder1(dec1)
    out = self.conv(dec1)
    return out



class ConvBlock(nn.Module):

  def __init__(self, in_channels, out_channels):
    super(ConvBlock, self).__init__()
    # number of input channels is a number of filters in the previous layer
    # number of output channels is a number of filters in the current layer
    # "same" convolutions
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

  def forward(self, x):
    return self.conv(x)


class UpConv(nn.Module):

  def __init__(self, in_channels, out_channels):
    super(UpConv, self).__init__()
    self.up = nn.Sequential(
      nn.Upsample(scale_factor=2),
      nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True)
    )

  def forward(self, x):
    x = self.up(x)
    return x


class AttentionBlock(nn.Module):
  """Attention block with learnable parameters"""

  def __init__(self, F_g, F_l, n_coefficients):
    """
    :param F_g: number of feature maps (channels) in previous layer
    :param F_l: number of feature maps in corresponding encoder layer, transferred via skip connection
    :param n_coefficients: number of learnable multi-dimensional attention coefficients
    """
    super(AttentionBlock, self).__init__()

    self.W_gate = nn.Sequential(
        nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
        nn.BatchNorm2d(n_coefficients)
    )

    self.W_x = nn.Sequential(
        nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
        nn.BatchNorm2d(n_coefficients)
    )

    self.psi = nn.Sequential(
        nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
        nn.BatchNorm2d(1),
        nn.Sigmoid()
    )

    self.relu = nn.ReLU(inplace=True)

  def forward(self, gate, skip_connection):
    """
    :param gate: gating signal from previous layer
    :param skip_connection: activation from corresponding encoder layer
    :return: output activations
    """
    g1 = self.W_gate(gate)
    x1 = self.W_x(skip_connection)
    psi = self.relu(g1 + x1)
    psi = self.psi(psi)
    out = skip_connection * psi
    return out


class AttentionUNet(nn.Module):

  def __init__(self, config):
    super(AttentionUNet, self).__init__()

    in_channels = 3 * config.data.burst_size

    self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

    self.Conv1 = ConvBlock(in_channels, 64)
    self.Conv2 = ConvBlock(64, 128)
    self.Conv3 = ConvBlock(128, 256)
    self.Conv4 = ConvBlock(256, 512)
    self.Conv5 = ConvBlock(512, 1024)

    self.Up5 = UpConv(1024, 512)
    self.Att5 = AttentionBlock(F_g=512, F_l=512, n_coefficients=256)
    self.UpConv5 = ConvBlock(1024, 512)

    self.Up4 = UpConv(512, 256)
    self.Att4 = AttentionBlock(F_g=256, F_l=256, n_coefficients=128)
    self.UpConv4 = ConvBlock(512, 256)

    self.Up3 = UpConv(256, 128)
    self.Att3 = AttentionBlock(F_g=128, F_l=128, n_coefficients=64)
    self.UpConv3 = ConvBlock(256, 128)

    self.Up2 = UpConv(128, 64)
    self.Att2 = AttentionBlock(F_g=64, F_l=64, n_coefficients=32)
    self.UpConv2 = ConvBlock(128, 64)

    self.Conv = nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0)

  def forward(self, x):
    batch, burst, channels, h, w = x.shape
    x = x.reshape(batch, burst*channels, h, w)
    e1 = self.Conv1(x)
    e2 = self.MaxPool(e1)
    e2 = self.Conv2(e2)
    e3 = self.MaxPool(e2)
    e3 = self.Conv3(e3)
    e4 = self.MaxPool(e3)
    e4 = self.Conv4(e4)
    e5 = self.MaxPool(e4)
    e5 = self.Conv5(e5)
    d5 = self.Up5(e5)
    s4 = self.Att5(gate=d5, skip_connection=e4)
    # concatenate attention-weighted skip connection with previous layer output
    d5 = torch.cat((s4, d5), dim=1)
    d5 = self.UpConv5(d5)
    d4 = self.Up4(d5)
    s3 = self.Att4(gate=d4, skip_connection=e3)
    d4 = torch.cat((s3, d4), dim=1)
    d4 = self.UpConv4(d4)
    d3 = self.Up3(d4)
    s2 = self.Att3(gate=d3, skip_connection=e2)
    d3 = torch.cat((s2, d3), dim=1)
    d3 = self.UpConv3(d3)
    d2 = self.Up2(d3)
    s1 = self.Att2(gate=d2, skip_connection=e1)
    d2 = torch.cat((s1, d2), dim=1)
    d2 = self.UpConv2(d2)
    out = self.Conv(d2)
    return out









models_config = {
  'FusionLinearModel': FusionLinearModel,
  'FusionNonLinearModel': FusionNonLinearModel,
  'FusionNonLinearWeightedAverageModel': FusionNonLinearWeightedAverageModel,
  'UNet': UNet,
  'AttentionUNet': AttentionUNet
}
