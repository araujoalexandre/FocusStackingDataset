import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexDaubechiesWavelets(nn.Module):

  def __init__(self, levels, learned):
    super(ComplexDaubechiesWavelets, self).__init__()

    self.levels = levels
    self.learned_wavelets = bool(learned)
    logging.info(f"Learning Wavelets: {self.learned_wavelets}")

    self.lo_pass = torch.Tensor([
      [-0.0662912607,  0.1104854346,  0.6629126074, 0.6629126074,  0.1104854346, -0.0662912607],
      [-0.0855816496, -0.0855816496,  0.1711632992, 0.1711632992, -0.0855816496, -0.0855816496]
    ]).reshape(2, 1, 1, 6)

    self.hi_pass = torch.Tensor([
       [-0.0662912607, -0.1104854346, 0.6629126074, -0.6629126074,  0.1104854346, 0.0662912607],
       [ 0.0855816496, -0.0855816496, -0.1711632992, 0.1711632992, 0.0855816496, -0.0855816496]
    ]).reshape(2, 1, 1, 6)

    self.lo_pass = nn.Parameter(self.lo_pass, requires_grad=self.learned_wavelets)
    self.hi_pass = nn.Parameter(self.hi_pass, requires_grad=self.learned_wavelets)

    self.filter_len = len(self.lo_pass)

  def _pad(self, src, vertical, transpose=False):
    pad = 3 if not transpose else 1
    if vertical:
      return F.pad(src.transpose(3, 2), (pad, pad, 0, 0), mode='circular')
    return F.pad(src, (pad, pad, 0, 0), mode='circular')


  def _complex_conv(self, images, kernel, vertical):
    n_batch, n_channels, _, h, w = images.shape
    h_out, w_out = h, w // 2
    kernel = kernel.tile((n_channels, 1, 1, 1))
    images_flipped = torch.flip(images, dims=[2])
    images = images.reshape(n_batch, n_channels*2, h, w)
    images_flipped = images_flipped.reshape(n_batch, n_channels*2, h, w)
    images = self._pad(images, vertical)
    images_flipped = self._pad(images_flipped, vertical)
    M1 = F.conv2d(images, kernel, padding=0, stride=1, groups=n_channels*2)
    M2 = F.conv2d(images_flipped, kernel, padding=0, stride=1, groups=n_channels*2)
    _, _, h, w = M1.shape
    M1 = M1.reshape(n_batch, n_channels, 2, h, w)
    M2 = M2.reshape(n_batch, n_channels, 2, h, w)
    Mr_kr = M1[:, :, 0].reshape(n_batch, n_channels, 1, h, w)
    Mi_ki = M1[:, :, 1].reshape(n_batch, n_channels, 1, h, w)
    Mi_kr = M2[:, :, 0].reshape(n_batch, n_channels, 1, h, w)
    Mr_ki = M2[:, :, 1].reshape(n_batch, n_channels, 1, h, w)
    result = torch.cat((Mr_kr - Mi_ki, Mr_ki + Mi_kr), dim=2)[:, :, :, :, 0:-1:2]
    return result

  def _complex_conv_transpose(self, wavelets, kernel, vertical):
    n_batch, n_channels, _, h, w = wavelets.shape
    kernel = kernel.tile((n_channels, 1, 1, 1))
    wavelets_flipped = torch.flip(wavelets, dims=[2])
    wavelets = wavelets.reshape(n_batch, n_channels*2, h, w)
    wavelets_flipped = wavelets_flipped.reshape(n_batch, n_channels*2, h, w)
    wavelets = self._pad(wavelets, vertical, True)
    wavelets_flipped = self._pad(wavelets_flipped, vertical, True)
    M1 = F.conv_transpose2d(wavelets, kernel, padding=0, stride=2, groups=n_channels*2)
    M2 = F.conv_transpose2d(wavelets_flipped, kernel, padding=0, stride=2, groups=n_channels*2)
    _, _, h, w = M1.shape
    M1 = M1.reshape(n_batch, n_channels, 2, h, w)
    M2 = M2.reshape(n_batch, n_channels, 2, h, w)
    M1 = torch.roll(M1[:, :, :, ::2, 4:-4], -1, dims=4)
    M2 = torch.roll(M2[:, :, :, ::2, 4:-4], -1, dims=4)
    _, _, _, h, w = M1.shape
    Mr_kr = M1[:, :, 0].reshape(n_batch, n_channels, 1, h, w)
    Mi_ki = M1[:, :, 1].reshape(n_batch, n_channels, 1, h, w)
    Mi_kr = M2[:, :, 0].reshape(n_batch, n_channels, 1, h, w)
    Mr_ki = M2[:, :, 1].reshape(n_batch, n_channels, 1, h, w)
    return Mr_kr, Mr_ki, Mi_kr, Mi_ki


  def _decompose(self, image):
    result = torch.cat((
      self._complex_conv(image, self.lo_pass, True),
      self._complex_conv(image, self.hi_pass, True)
    ), dim=-1).transpose(4, 3)
    result = torch.cat((
      self._complex_conv(result, self.lo_pass, False),
      self._complex_conv(result, self.hi_pass, False),
    ), dim=-1)
    return result
 
  def _compose(self, wavelets):
    batch_size, n_channels, _, h, w = wavelets.shape
    conv = self._complex_conv_transpose
    result = wavelets.clone()
    for vertical in [True, False]:
      if vertical:
        lo_band, hi_band = result[:, :, :, :h//2, :], result[:, :, :, h//2:, :]
      else:
        lo_band, hi_band = result[:, :, :, :, :w//2], result[:, :, :, :, w//2:]
      Mr_lo_kr_lo, Mr_lo_ki_lo, Mi_lo_kr_lo, Mi_lo_ki_lo = conv(lo_band, self.lo_pass, vertical)
      Mr_hi_kr_hi, Mr_hi_ki_hi, Mi_hi_kr_hi, Mi_hi_ki_hi = conv(hi_band, self.hi_pass, vertical)
      Mr_lo_kr_hi, Mr_lo_ki_hi, Mi_lo_kr_hi, Mi_lo_ki_hi = conv(lo_band, self.hi_pass, vertical)
      Mr_hi_kr_lo, Mr_hi_ki_lo, Mi_hi_kr_lo, Mi_hi_ki_lo = conv(hi_band, self.lo_pass, vertical)
      real = (Mr_lo_kr_lo + Mr_hi_kr_hi + Mi_lo_ki_lo + Mi_hi_ki_hi)
      imag = (Mi_lo_kr_lo + Mi_hi_kr_hi - Mr_lo_ki_lo - Mr_hi_ki_hi)
      if vertical:
        real, imag = real.transpose(4, 3), imag.transpose(4, 3)
      result = torch.cat((real, imag), dim=2)
    return result
  
  def forward(self, images):
    """ Apply the Daubechies Wavelet Transform on the image. """
    batch_size, n_channels, _, h, w = images.shape
    result = torch.zeros((batch_size, n_channels, 2, h, w), dtype=torch.float32)
    result = result.to(images.device)
    self.lo_pass = self.lo_pass.to(images.device)
    self.hi_pass = self.hi_pass.to(images.device)
    for i in range(self.levels):
      w = images.shape[4] >> i
      h = images.shape[3] >> i
      dstarea = result[:, :, :, 0:h, 0:w]
      if i == 0:
        srcarea = images[:, :, :, 0:h, 0:w]
      else:
        srcarea = dstarea
      result[:, :, :, 0:h, 0:w] = self._decompose(srcarea)
    return result

  def inverse(self, wavelets):
    """ Perform the inverse Daubechies Wavelet Transform on the wavelet. """
    batch_size, n_channels, _, h, w = wavelets.shape
    result = torch.zeros((batch_size, n_channels, 2, h, w), dtype=torch.float32)
    result = result.to(wavelets.device)
    self.lo_pass = self.lo_pass.to(wavelets.device)
    self.hi_pass = self.hi_pass.to(wavelets.device)
    for i in range(self.levels)[::-1]:
      w = wavelets.shape[4] >> i
      h = wavelets.shape[3] >> i
      srcarea = wavelets[:, :, :, 0:h, 0:w]
      result[:, :, :, 0:h, 0:w] = self._compose(srcarea)
      wavelets[:, :, :, 0:h, 0:w] = result[:, :, :, 0:h, 0:w]
    result = result[:, :, 0, :, :]
    max_tensor, min_tensor = torch.Tensor([255]), torch.Tensor([0])
    max_tensor, min_tensor = max_tensor.to(result.device), min_tensor.to(result.device)
    result = torch.minimum(max_tensor, torch.maximum(min_tensor, result))
    return result


class FusionWavelets(nn.Module):

  def __init__(self, config):
    super(FusionWavelets, self).__init__()
    self.wavelets = ComplexDaubechiesWavelets(6, False)

  def merge(self, wavelets):
    batch_size, burst_size, channels, _, h, w = wavelets.shape
    abs_wavelets = (wavelets**2).sum(axis=3)
    flatten_abs_wavelets = abs_wavelets.reshape(batch_size, burst_size, channels*h*w)
    _, indices = F.max_pool1d(flatten_abs_wavelets.permute(0, 2, 1), burst_size, return_indices=True)
    # burst, batch_size, channels, complex, h, w 
    wavelets = wavelets.permute(1, 0, 2, 3, 4, 5)
    # burst, complex, batch_size, channels, h, w 
    wavelets = wavelets.permute(0, 3, 1, 2, 4, 5)
    flatten_wavelets = wavelets.reshape(burst_size, 2, batch_size*channels*h*w)
    real = flatten_wavelets[indices.reshape(-1), 0, range(batch_size*channels*h*w)]
    imag = flatten_wavelets[indices.reshape(-1), 1, range(batch_size*channels*h*w)]
    real = real.reshape(batch_size, channels, 1, h, w)
    imag = imag.reshape(batch_size, channels, 1, h, w)
    merged = torch.cat([real, imag], dim=2)
    return merged

  def forward(self, x):
    # add complex channels
    batch_size, burst_size, channels, h, w = x.shape
    x= x.reshape(batch_size, burst_size, channels, -1)
    x = F.pad(x, (0, h*w))
    x = x.reshape(batch_size, burst_size, channels, 2, h, w)

    x = x.reshape(batch_size*burst_size, channels, 2, h, w)
    wavelets = self.wavelets.forward(x)
    wavelets = wavelets.reshape(batch_size, burst_size, channels, 2, h, w)
    # merge over the burst dimension
    merged = self.merge(wavelets)
    final = self.wavelets.inverse(merged)

    return final

