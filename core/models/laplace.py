import logging
import numpy as np
import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F


class Laplacian(nn.Module):

  def __init__(self, config):
    super(Laplacian, self).__init__()

  def compute_laplacian(self, x):
    batch_size, burst_size, channels, h, w = x.shape
    x = x.reshape(batch_size*burst_size, channels, h, w)
    kernel_size, blur_size = 5, 5
    laplacians = []
    for image in x.cpu().numpy():
      image = image.mean(0)
      blurred = cv2.GaussianBlur(image, (blur_size, blur_size), 0)
      laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=kernel_size)
      laplacian = np.stack([laplacian, laplacian, laplacian])
      laplacians.append(laplacian)
    laplacians = np.array(laplacians)
    laplacians = laplacians.reshape(batch_size, burst_size, channels, h, w)
    laplacians = np.argmax(laplacians, 1)
    laplacians = laplacians.reshape(-1)
    return torch.Tensor(laplacians).long()

  def merge(self, x, laplacians):
    batch_size, burst_size, channels, h, w = x.shape
    x = x.permute(1, 0, 2, 3, 4)
    x = x.reshape(burst_size, -1)
    indexes = list(range(batch_size * channels * h * w))
    merged = x[laplacians, indexes]
    merged = merged.reshape(batch_size, channels, h, w)
    return merged

  def forward(self, x):
    device = x.device
    batch_size, burst_size, channels, h, w = x.shape
    laplacians = self.compute_laplacian(x)
    x = self.merge(x, laplacians)
    return x




