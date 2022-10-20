
import logging
import numpy as np
import torch
import cv2
from os.path import join


def clamp(image):
  assert image.max() > 2
  image = np.minimum(255, np.maximum(0, image))
  return image

def compute_pca(image):  
  flat_img = image.reshape(-1, 3)
  mean, eigenvectors = cv2.PCACompute(flat_img, np.array(()), cv2.PCA_DATA_AS_ROW, 2)
  T = np.zeros((1, 2), dtype=np.float32)
  T[0, 0] = 1
  weights = cv2.PCABackProject(T, mean, eigenvectors)
  weights -= cv2.PCABackProject(np.zeros((1, 2), dtype=np.float32), mean, eigenvectors)
  weights /= np.sum(weights)
  return weights[0]

def convert_to_grayscale(image, weights):
  assert image.shape[-1] == 3
  w0, w1, w2 = weights[0], weights[1], weights[2]
  result = image[..., 0] * w0 + image[..., 1] * w1 + image[..., 2] * w2
  result = clamp(result)
  return result

def align_images(refcolor, srccolor, transformation, pca_weights):
  refcolor, srccolor = np.float32(refcolor), np.float32(srccolor)
  if np.allclose(refcolor, srccolor):
    return srccolor, np.float32(np.eye(2, 3))
  refgray = convert_to_grayscale(refcolor, pca_weights)
  srcgray = convert_to_grayscale(srccolor, pca_weights)
  transform = find_transform(srcgray, refgray, transformation)
  image = apply_transform(srccolor, transform)
  image = clamp(image)
  return image, transform


def find_transform(src, ref, transformation, max_res=2048):
  resolution = np.max(ref.shape)
  scale_ratio = 1.
  if resolution > max_res:
    scale_ratio = max_res / resolution
    ref = cv2.resize(ref, None, None, scale_ratio, scale_ratio, cv2.INTER_AREA)
    src = cv2.resize(src, None, None, scale_ratio, scale_ratio, cv2.INTER_AREA)
  transformation[:, 2] *= scale_ratio
  number_of_iterations = 50
  termination_eps = 0.001
  warp_mode = cv2.MOTION_AFFINE
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
  cc, transformation = cv2.findTransformECC(src, ref, transformation, warp_mode, criteria)
  transformation[:, 2] /= scale_ratio
  return transformation


def apply_transform(image, transformation):
  h, w = image.shape[:2]
  image = cv2.warpAffine(image, transformation, (w, h), None, cv2.INTER_CUBIC, cv2.BORDER_REFLECT)
  return image


def align_burst_raw(burst, burst_path):
  burst = burst.permute(0, 2, 3, 1).numpy()
  transforms = torch.load(join(burst_path, 'transforms.pth'))
  aligned_images = [None] * len(burst)
  aligned_images[0] = burst[0]
  for i in range(len(burst)-1):
    transform = np.array(transforms[i])
    transform[:, 2] /= 2
    aligned_images[i+1] = apply_transform(burst[i+1]*255, transform) / 255
  aligned_images = np.array(aligned_images)
  aligned_images = torch.FloatTensor(aligned_images).permute(0, 3, 1, 2)
  return aligned_images


def align_burst_jpg(burst):
  burst = burst * 255
  burst = burst.permute(0, 2, 3, 1).numpy()
  pca_weights = compute_pca(burst[0])
  aligned_images = [None] * len(burst)
  aligned_images[0] = burst[0]
  transform = np.float32(np.eye(2, 3))
  for i in range(len(burst)-1):
    aligned_images[i+1], transform = align_images(
      aligned_images[i], burst[i+1], transform, pca_weights)
  aligned_images = np.array(aligned_images)
  aligned_images = torch.FloatTensor(aligned_images).permute(0, 3, 1, 2)
  return aligned_images / 255


def align_burst(burst, config):
  if config.eval.image_type == 'raw':
    data_dir = config.eval.eval_data_dir
    burst_name = config.eval.burst_name
    path = join(data_dir, burst_name)
    return align_burst_raw(burst, path)
  return align_burst_jpg(burst)





