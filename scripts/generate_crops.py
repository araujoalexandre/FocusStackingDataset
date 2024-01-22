import submitit
import os
import glob
import numpy as np
import torch
import pandas as pd
import PIL.Image as Image
import torchvision.transforms as T
import natsort
import pickle
from os.path import join
import cv2

def pickle_dump(file, path):
  """ Function to dump pickle object """
  with open(path, 'wb') as f:
    pickle.dump(file, f, -1)

def clamp(image):
  image = np.minimum(255, np.maximum(0, image))
  assert image.max() > 0
  return np.uint8(image)

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

def align_images(refcolor, srccolor, initial_guess, pca_weights):
  refcolor, srccolor = np.float32(refcolor), np.float32(srccolor)
  if np.allclose(refcolor, srccolor):
    return srccolor, np.float32(np.eye(2, 3))
  refgray = convert_to_grayscale(refcolor, pca_weights)
  srcgray = convert_to_grayscale(srccolor, pca_weights)
  transform = find_transform(srcgray, refgray, initial_guess)
  image = apply_transform(srccolor, transform)
  image = clamp(image)
  return image 

def find_transform(src, ref, transformation, max_res=2048):
  resolution = np.max(ref.shape)
  scale_ratio = 1.
  if resolution > max_res:
    scale_ratio = max_res / resolution
    ref = cv2.resize(ref, None, None, scale_ratio, scale_ratio, cv2.INTER_AREA)
    src = cv2.resize(src, None, None, scale_ratio, scale_ratio, cv2.INTER_AREA)
  transformation[0, 2] *= scale_ratio
  transformation[1, 2] *= scale_ratio
  number_of_iterations = 50
  termination_eps = 0.001
  warp_mode = cv2.MOTION_AFFINE
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
  cc, transformation = cv2.findTransformECC(src, ref, transformation, warp_mode, criteria)
  transformation[0, 2] /= scale_ratio
  transformation[1, 2] /= scale_ratio
  return transformation

def apply_transform(image, transformation):
  h, w = image.shape[:2]
  image = cv2.warpAffine(image, transformation, (w, h), None, cv2.INTER_CUBIC, cv2.BORDER_REFLECT)
  return image


class GenerateCrops:

  def __init__(self, crop_size):
    self.crop_size = crop_size

  def split_image(self, image):
    crop_size = self.crop_size
    burst_size, channels, imsize1, imsize2 = image.shape
    npatch1 = imsize1 // crop_size
    npatch2 = imsize2 // crop_size
    image = image[:, :, :npatch1*crop_size, :npatch2*crop_size]
    image = image.reshape(burst_size, channels, npatch1, crop_size, npatch2, crop_size)
    image = image.permute(0, 2, 4, 1, 3, 5)
    image = image.reshape(burst_size, npatch1*npatch2, channels, crop_size, crop_size)
    image = image.permute(1, 0, 2, 3, 4)
    return image

  def __call__(self, burst_path):

    print(burst_path)
    imgs_path = glob.glob(join(burst_path, 'aligned', '*.jpg'))
    imgs_path = natsort.natsorted(imgs_path)

    # load target
    target = cv2.imread(join(burst_path, 'helicon_focus.jpg'), cv2.IMREAD_COLOR)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    dim = (int(target.shape[1]), int(target.shape[0]))
    
    images = []
    for i, path in enumerate(imgs_path):
      image = cv2.imread(path, cv2.IMREAD_COLOR)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
      images.append(image)
    images = np.array(images)
      
    assert all([images[0].shape == x.shape for x in images])
    assert target.shape == images[0].shape

    images = torch.Tensor(images)
    images = images.to(torch.uint8)
    images = images.permute(0, 3, 1, 2)

    target = torch.Tensor(target)[None]
    target = target.to(torch.uint8)
    target = target.permute(0, 3, 1, 2)

    images = self.split_image(images)
    target = self.split_image(target)

    assert images.shape[0] == target.shape[0], f'{images.shape}'
    assert images.shape[0] > 0

    outpath = join(burst_path, 'crops3')
    os.makedirs(outpath, exist_ok=True)
    for i, (burst_crop, target_crop) in enumerate(zip(images, target)):
      burst_crop = burst_crop.clone()
      target_crop = target_crop.clone()
      data = {'burst': burst_crop, 'target': target_crop}
      torch.save(data, join(outpath, f'crop{i}.pkl'))


if __name__ == '__main__':

  crop_size = 128
  for split in ['train', 'test']:
    df = pd.read_csv('dataset.csv', sep=";")
    df = df[df['set'] == split][['lens', 'photo']]
    bursts_list = df.apply(lambda x: join(split, x[0], x[1]), axis=1).values
    bursts_list = list(bursts_list)

    executor = submitit.AutoExecutor(
      folder='./', cluster='slurm')
    executor.update_parameters(
      stderr_to_stdout=True,
      slurm_gpus_per_node=4,
      slurm_nodes=1,
      slurm_job_name=f'crops',
      slurm_signal_delay_s=0,
      slurm_timeout_min=15,
      slurm_exclusive=True
    )

    generate = GenerateCrops(crop_size)
    with executor.batch():
      for burst_path in bursts_list:
        job = executor.submit(generate, burst_path)

