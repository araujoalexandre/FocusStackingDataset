import logging
import numpy as np
import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F


class ComplexDaubechiesWavelets(nn.Module):

  def __init__(self):
    super(ComplexDaubechiesWavelets, self).__init__()

    self.levels = 6

    self.lo_pass = torch.Tensor([
      [-0.0662912607,  0.1104854346,  0.6629126074, 0.6629126074,  0.1104854346, -0.0662912607],
      [-0.0855816496, -0.0855816496,  0.1711632992, 0.1711632992, -0.0855816496, -0.0855816496]
    ]).reshape(2, 1, 1, 6)

    self.hi_pass = torch.Tensor([
       [-0.0662912607, -0.1104854346, 0.6629126074, -0.6629126074,  0.1104854346, 0.0662912607],
       [ 0.0855816496, -0.0855816496, -0.1711632992, 0.1711632992, 0.0855816496, -0.0855816496]
    ]).reshape(2, 1, 1, 6)

    self.lo_pass = nn.Parameter(self.lo_pass, requires_grad=False)
    self.hi_pass = nn.Parameter(self.hi_pass, requires_grad=False)

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


  def _decompose(self, image, lo_pass, hi_pass):
    result = torch.cat((
      self._complex_conv(image, lo_pass, True),
      self._complex_conv(image, hi_pass, True)
    ), dim=-1).transpose(4, 3)
    result = torch.cat((
      self._complex_conv(result, lo_pass, False),
      self._complex_conv(result, hi_pass, False),
    ), dim=-1)
    return result
 
  def _compose(self, wavelets, lo_pass, hi_pass):
    batch_size, n_channels, _, h, w = wavelets.shape
    conv = self._complex_conv_transpose
    result = wavelets.clone()
    for vertical in [True, False]:
      if vertical:
        lo_band, hi_band = result[:, :, :, :h//2, :], result[:, :, :, h//2:, :]
      else:
        lo_band, hi_band = result[:, :, :, :, :w//2], result[:, :, :, :, w//2:]
      Mr_lo_kr_lo, Mr_lo_ki_lo, Mi_lo_kr_lo, Mi_lo_ki_lo = conv(lo_band, lo_pass, vertical)
      Mr_hi_kr_hi, Mr_hi_ki_hi, Mi_hi_kr_hi, Mi_hi_ki_hi = conv(hi_band, hi_pass, vertical)
      Mr_lo_kr_hi, Mr_lo_ki_hi, Mi_lo_kr_hi, Mi_lo_ki_hi = conv(lo_band, hi_pass, vertical)
      Mr_hi_kr_lo, Mr_hi_ki_lo, Mi_hi_kr_lo, Mi_hi_ki_lo = conv(hi_band, lo_pass, vertical)
      real = (Mr_lo_kr_lo + Mr_hi_kr_hi + Mi_lo_ki_lo + Mi_hi_ki_hi)
      imag = (Mi_lo_kr_lo + Mi_hi_kr_hi - Mr_lo_ki_lo - Mr_hi_ki_hi)
      if vertical:
        real, imag = real.transpose(4, 3), imag.transpose(4, 3)
      result = torch.cat((real, imag), dim=2)
    return result
  
  def forward(self, images):
    """ Apply the Daubechies Wavelet Transform on the image. """
    batch_size, n_channels, h, w = images.shape
    images = images.reshape(batch_size, n_channels, 1, h, w)
    zeros = torch.zeros_like(images)
    images = torch.cat([images, zeros], dim=2)
    result = torch.zeros((batch_size, n_channels, 2, h, w), dtype=torch.float32)
    result = result.to(images.device)
    lo_pass = self.lo_pass.to(images.device) / np.sqrt(2)
    hi_pass = self.hi_pass.to(images.device) / np.sqrt(2)
    for i in range(self.levels):
      w = images.shape[4] >> i
      h = images.shape[3] >> i
      dstarea = result[:, :, :, 0:h, 0:w]
      if i == 0:
        srcarea = images[:, :, :, 0:h, 0:w]
      else:
        srcarea = dstarea
      result[:, :, :, 0:h, 0:w] = self._decompose(srcarea, lo_pass, hi_pass)
    return result

  def inverse(self, wavelets):
    """ Perform the inverse Daubechies Wavelet Transform on the wavelet. """
    batch_size, n_channels, _, h, w = wavelets.shape
    result = torch.zeros((batch_size, n_channels, 2, h, w), dtype=torch.float32)
    result = result.to(wavelets.device)
    lo_pass = self.lo_pass.to(wavelets.device) * np.sqrt(2)
    hi_pass = self.hi_pass.to(wavelets.device) * np.sqrt(2)
    for i in range(self.levels)[::-1]:
      w = wavelets.shape[4] >> i
      h = wavelets.shape[3] >> i
      srcarea = wavelets[:, :, :, 0:h, 0:w]
      result[:, :, :, 0:h, 0:w] = self._compose(srcarea, lo_pass, hi_pass)
      wavelets[:, :, :, 0:h, 0:w] = result[:, :, :, 0:h, 0:w]
    result = result[:, :, 0, :, :]
    result = result.clamp(0, 255)
    return result


class FusionWavelets(nn.Module):

  def __init__(self, config):
    super(FusionWavelets, self).__init__()
    self.wavelets = ComplexDaubechiesWavelets()

  def reassign(self, aligned_grayscales, aligned_images):
    aligned_grayscales = aligned_grayscales[0, :, 0].long().cpu().numpy()
    aligned_images = aligned_images.permute(0, 1, 3, 4, 2)
    aligned_images = aligned_images[0].long().cpu().numpy()
    width = aligned_grayscales[0].shape[1]
    height = aligned_grayscales[0].shape[0]
    m_colors = [None] * (width * height * (len(aligned_grayscales) + 1))
    m_counts = [None] * (width * height)
    colors_wrpos = 0 # pointer on m_colors 
    counts_wrpos = 0 # pointer on m_counts
    gray_seen = [0] * 256
    pixel_idx = 1
    for y in range(height):
      for x in range(width):
        pixel_idx += 1
        color_count = 0
        for i in range(len(aligned_images)):
          gray = aligned_grayscales[i][y, x]
          if gray_seen[gray] != pixel_idx:
            gray_seen[gray] = pixel_idx
            m_colors[colors_wrpos] = [gray, aligned_images[i][y, x]]
            colors_wrpos += 1
            color_count += 1
        m_counts[counts_wrpos] = color_count - 1
        counts_wrpos += 1
    return m_colors, m_counts

  def reassign_colors(self, m_colors, m_counts, merged_gray):
    merged_gray = merged_gray[0, 0].cpu().numpy()
    height, width = merged_gray.shape
    m_result = np.zeros((height, width, 3), dtype=np.uint8)
    colors = 0 # pointer on m_colors
    counts = 0 # pointer on m_counts
    for y in range(height):
      for x in range(width):
        color_count = m_counts[counts]
        counts += 1
        pos = colors
        colors += color_count
        gray = np.int32(merged_gray[y, x])
        closest = m_colors[pos]
        colors += 1
        error = np.abs(np.int32(closest[0]) - gray)
        while (error > 0 and pos != colors):
          candidate = m_colors[pos]
          pos += 1
          distance = abs(np.int32(candidate[0]) - gray)
          if (distance < error):
            error = distance
            closest = candidate
        m_result[y, x] = closest[1] 
    return torch.FloatTensor(m_result)

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

  def compute_pca(self, image):  
    """ compute PCA for grayscale conversion """
    assert image.dim() == 3 and image.shape[0] == 3
    channels, h, w = image.shape
    image = image.permute(1, 2, 0)
    flat_img = image.reshape(-1, channels).cpu().numpy()
    mean, eigenvectors = cv2.PCACompute(flat_img, np.array(()), cv2.PCA_DATA_AS_ROW, 2)
    T = np.zeros((1, 2), dtype=np.float32)
    T[0, 0] = 1
    m_weights = cv2.PCABackProject(T, mean, eigenvectors)
    m_weights -= cv2.PCABackProject(np.zeros((1, 2), dtype=np.float32), mean, eigenvectors)
    m_weights /= np.sum(m_weights)
    return m_weights[0]

  def convert_to_grayscale(self, x):
    batch_size, burst_size, channels, h, w = x.shape
    ref = burst_size // 2
    x_grayscale = []
    for burst in x:
      weights = self.compute_pca(burst[ref])
      w0, w1, w2 = weights
      gray = burst[:, 0] * w0 + burst[:, 1] * w1 + burst[:, 2] * w2
      x_grayscale.append(gray[None])
    x_grayscale = torch.cat(x_grayscale, 0)
    x_grayscale = x_grayscale.clamp(0, 255)
    return x_grayscale

  def forward(self, x):
    normalize = False
    if x.max() < 2:
      normalize = True
      x = x * 255
    batch_size, burst_size, channels, h, w = x.shape
    x_grayscale = x.mean(2, keepdim=True).reshape(batch_size*burst_size, 1, h, w)
    # x_grayscale = self.convert_to_grayscale(x)
    # x_grayscale = x_grayscale.reshape(batch_size*burst_size, 1, h, w)
    wavelets = self.wavelets.forward(x_grayscale)
    wavelets = wavelets.reshape(batch_size, burst_size, 1, 2, h, w)
    merged = self.merge(wavelets)
    final_gray = self.wavelets.inverse(merged)
    x_grayscale = x_grayscale.reshape(batch_size, burst_size, 1, h, w)
    final_ = []
    for ix in range(batch_size):
      m_colors, m_counts = self.reassign(x_grayscale[ix][None], x[ix][None])
      final_.append(self.reassign_colors(m_colors, m_counts, final_gray[ix][None])[None])
    final = torch.cat(final_, 0) 
    if normalize:
      final = final / 255
    final = torch.FloatTensor(final).permute(0, 3, 1, 2).cuda()
    return final





def test1():
  size = 64
  levels = 4
  batch_size = 3
  channels = 64
  images = torch.randint(0, 256, size=(batch_size, channels, size, size)).to(torch.float32)
  images_real = images.clone()
  images = images.reshape(batch_size, channels, -1)
  images = F.pad(images, (0, size*size))
  images = images.reshape(batch_size, channels, 2, size, size)
  images = images.cuda()
  wavelet = ComplexDaubechiesWavelets(levels, False)
  transform = wavelet.forward(images)
  images_out = wavelet.inverse(transform)
  torch.testing.assert_allclose(images_real.cpu(), images_out.cpu(), rtol=0.01, atol=0.01)

def test2():

  batch_size = 3
  burst_size = 14
  channels = 32
  size = 64
  images = torch.randint(0, 256, size=(batch_size, burst_size, channels, size, size)).to(torch.float32)
  fusion = FusionWavelets(4, False)
  merged = fusion(images)
  print('merged', merged.shape)


if __name__ == "__main__":
  test1()
  test2()





