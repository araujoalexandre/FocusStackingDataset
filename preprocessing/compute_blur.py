
import os
import sys
import time
import argparse
import pickle
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import submitit
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

from collections import OrderedDict
from os.path import join, exists
from scipy.ndimage import gaussian_filter
from torchvision.transforms import Compose

from dpt.models import DPTDepthModel
from dpt.midas_net import MidasNet_large
from dpt.transforms import Resize, NormalizeImage, PrepareForNet

from filter_disk import FilterDisk


def pickle_dump(file, path):
  """function to dump picke object."""
  with open(path, 'wb') as f:
    pickle.dump(file, f, -1)

def pickle_load(path):
  with open(path, 'rb') as f:
    return pickle.load(f)

def adjust_gamma(image, gamma=1.0):
  # build a lookup table mapping the pixel values [0, 255] to
  # their adjusted gamma values
  invGamma = 1.0 / gamma
  table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")
  # apply gamma correction using the lookup table
  if type(image) == np.asarray:
    image = image.astype(np.uint8)
    return cv2.LUT(image, table)
  image = image.detach().numpy().astype(np.uint8)
  return torch.Tensor(cv2.LUT(image, table)).to(torch.uint8)


class ProgressiveBlur:

  def __init__(self, name, image, output_dir='./imsave', depth_dir='./saved_depthmap',
               scale=1, batch_size=1000, coc_max=40):

    self.name = name
    self.image = torch.Tensor(self.resize_image(image, scale))
    self.h, self.w, _ = self.image.shape
    print(f"{name}: Loaded image: {self.image.shape}")

    self.filter_disk = pickle_load('filter_disk.pkl')

    self.batch_size = batch_size
    self.coc_max = coc_max
    self.ksize = coc_max + 1
    self.padding = self.ksize // 2

    print(f"{name}: Loading or Computing depthmap")
    self.depthmap = self.load_depthmap(self.image)
    # depth_dir = 'saved_depthmap'
    # if depth_dir is not None:
    #   if not exists(depth_dir): os.mkdir(depth_dir)
    #   name = f'./{depth_dir}/{name}_depthmap_{self.h}x{self.w}.pkl'
    #   if exists(name):
    #     self.depthmap = pickle_load(name)
    #   else:
    #     self.depthmap = self.load_depthmap(self.image)
    #     pickle_dump(self.depthmap, name)
    # else:
    #   self.depthmap = self.load_depthmap(self.image)
    print(f"{name}: depth stats: min {self.depthmap.min()} / max {self.depthmap.max()}")

    self.output_dir = output_dir
    if not exists(self.output_dir):
      os.mkdir(self.output_dir)

    image_gamma_corrected = adjust_gamma(self.image, 0.45)
    print(image_gamma_corrected.shape)
    self.image1_split = self.unfold_and_split(image_gamma_corrected, self.ksize, self.padding)
    self.image2_split = self.unfold_and_split(self.image, self.ksize, self.padding)
    self.depth_split = self.unfold_and_split(self.depthmap, self.ksize, self.padding)

  def resize_image(self, image, scale):
    image = cv2.resize(image, None, None, scale, scale, cv2.INTER_AREA)
    return image

  def load_depthmap(self, image):
    model_path = "./dpt_large-midas-2f21e586.pt"
    model = DPTDepthModel(
      path=None,
      backbone="vitl16_384",
      non_negative=True,
      enable_attention_hooks=False,
    )
    model.load(model_path)
    depthmap = self._compute_depth_map(image.numpy(), model)
    depthmap = depthmap - depthmap.min()
    depthmap = torch.abs(depthmap - depthmap.max())
    return depthmap

  def _compute_depth_map(self, image, model):
    """Run MonoDepthNN to compute depthmap
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load network
    net_w = net_h = 384
    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transform = Compose([
      Resize( net_w,
          net_h,
          resize_target=None,
          keep_aspect_ratio=True,
          ensure_multiple_of=32,
          resize_method="minimal",
          image_interpolation_method=cv2.INTER_CUBIC,
        ),
        normalization,
        PrepareForNet(),
    ])
    model.eval()
  
    if device == torch.device("cuda"):
      model = model.to(memory_format=torch.channels_last)
      model = model.half()
    model.to(device)
  
    img_input = transform({"image": image})["image"]
  
    with torch.no_grad():
      sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
      if device == torch.device("cuda"):
        sample = sample.to(memory_format=torch.channels_last)
        sample = sample.half()
      prediction = model.forward(sample)
      prediction = torch.nn.functional.interpolate(
          prediction.unsqueeze(1),
          size=image.shape[:2],
          mode="bicubic",
          align_corners=False,
        ).squeeze().cpu()
    return prediction.float()

  def unfold_and_split(self, src, ksize, padding):
    src = src.float()
    if src.dim() == 2:
      h, w = src.shape
      c = 1
      src = src[None, None]
    else:
      h, w, c = src.shape
      src = src.permute(2, 0, 1)
      src = src[None]
    src_unfold = F.unfold(src, (self.ksize, self.ksize), padding=self.padding)
    _, h, w = src_unfold.shape
    src_unfold = src_unfold.reshape(c, h // c, w)
    src_unfold = src_unfold.transpose(1, 2)
    return torch.split(src_unfold, self.batch_size, dim=1)

  def cuda_parallel_compute(self, ksize, image_split, coc_split, depth_split):
    result = []
    l = len(image_split) // 4
    for i, (image, coc, depth) in enumerate(zip(image_split, coc_split, depth_split)):
      ksize = coc.shape[-1]
      mid = ksize // 2
      coc = coc[:, :, mid:-mid]
      coc = coc.flatten().numpy()
      kernel = torch.cat([self.filter_disk['{:.2f}'.format(c)].reshape(1, -1) for c in coc], dim=0)
      # mid_depth = depth[:, :, mid:-mid]
      # mask = (depth >= mid_depth) * 1.
      # kernel = kernel * mask
      # kernel = kernel.div(kernel.sum(-1, keepdim=True))
      out = image.mul(kernel).sum(axis=-1)[..., None]
      result.append(out)
    return torch.cat(result, dim=1)

  def compute_blur(self, coc, gamma_corrected=False):
    if gamma_corrected:
      image_split = self.image1_split
    else:
      image_split = self.image2_split
    depth_split = self.depth_split
    coc_split = self.unfold_and_split(coc, self.ksize, self.padding)
    result = self.cuda_parallel_compute(self.ksize, image_split, coc_split, depth_split)
    result = result.transpose(1, 2)
    result = F.fold(result, (self.h, self.w), (1, 1)).reshape(3, self.h, self.w)
    return result.permute(1, 2, 0).cpu()
  
  def get_circle_confusion(self, depth, focus, focal, aperture):
    N = focal / aperture
    radii = (torch.abs(depth - focus) / depth) * (focal**2 / (N * (focus - focal)))
    # radii = radii / 0.0033
    radii = torch.abs(radii)
    radii = torch.nan_to_num(radii, posinf=self.coc_max)
    radii = radii.clamp(max=self.coc_max)
    return radii.div(2.)
  
  def compute(self, depth, focus, focal, aperture):
    """ Compute a blur image with respect to camera parameters """
    circle_conf_map = self.get_circle_confusion(depth, focus, focal, aperture)
    bokeh = self.compute_blur(circle_conf_map, gamma_corrected=True)
    bokeh = adjust_gamma(bokeh, 2.2)
    blur_disk = self.compute_blur(circle_conf_map, gamma_corrected=False)
    result = torch.maximum(bokeh, blur_disk)
    return result.to(torch.uint8).detach().numpy()



def main(name, photo_path, output_dir, scale, aperture, focal):

  image = cv2.imread(photo_path, cv2.IMREAD_COLOR)

  progressive_blur = ProgressiveBlur(
    name,
    image,
    scale=scale,
    batch_size=1000000,
    depth_dir=None
  )

  n = 15
  step = 10
  f1_values = np.linspace(80, 80+n*step - step, n)

  for f1 in f1_values:

    depthmap1 = progressive_blur.depthmap * 10
    S1 = ( (f1 * focal) / (f1 - focal) )
    S2 = depthmap1 - S1
  
    print(f'{name} - f1: {f1}')
    # focus_dir = join(output_dir, f'ap_{aperture}_focal_{focal}')
    # os.makedirs(focus_dir, exist_ok=True)

    result = progressive_blur.compute(S2, S1, focal, aperture)
    cv2.imwrite(join(output_dir, f'step_{f1}.jpg'), result)



if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument("--name", type=str)
  parser.add_argument("--input_dir", type=str)
  parser.add_argument("--output_dir", type=str)

  parser.add_argument("--scale", type=float, default=0.65)
  parser.add_argument("--aperture", type=float, default=4)
  parser.add_argument("--focal", type=float, default=60)

  parser.add_argument("--local", action='store_true')
  args = parser.parse_args()

  if args.name:

    args.photo_path = join('./photos', args.name + '.jpg')
    folder_params_name = f'scale_{args.scale}_ap_{args.aperture}_focal_{args.focal}'
    args.output_dir = join('./imsave', folder_params_name, args.name)
    os.makedirs(args.output_dir, exist_ok=True)

    folder = f'slurm_{args.name}'
    cluster = 'slurm' if not args.local else 'local'
    executor = submitit.AutoExecutor(folder=folder, cluster=cluster)
    executor.update_parameters(
      gpus_per_node=4,
      nodes=1,
      slurm_account='dci@gpu',
      tasks_per_node=1,
      cpus_per_task=40,
      stderr_to_stdout=True,
      slurm_job_name='process',
      slurm_partition='gpu_p13',
      timeout_min=300,
    )
    job = executor.submit(main,
                          args.name, args.photo_path, args.output_dir,
                          args.scale, args.aperture, args.focal)
    job_id = job.job_id
    print(f"Submitted batch job {job_id} in folder {args.name}")

  else:

    # dataset 
    input_folder = args.input_dir
    output_folder = args.output_dir

    cluster = 'slurm' if not args.local else 'local'
    executor = submitit.AutoExecutor(folder='./slurm_dataset', cluster=cluster)
    executor.update_parameters(
      gpus_per_node=4,
      nodes=1,
      slurm_account='dci@gpu',
      tasks_per_node=1,
      cpus_per_task=40,
      stderr_to_stdout=True,
      slurm_job_name='process',
      slurm_partition='gpu_p13',
      timeout_min=300,
    )

    with executor.batch():
      for folder in ['train_c', 'val_c', 'test_c']:
        paths = glob.glob(join(input_folder, folder, 'target', '*png'))
        folder = folder.split('_')[0]
        for photo_path in paths:
          args.name = photo_path.split('/')[-1].split('.')[0]
          args.photo_path = photo_path
          folder_params_name = f'scale_{args.scale}_ap_{args.aperture}_focal_{args.focal}'
          args.output_dir = join(output_folder, folder_params_name, folder, args.name)
          os.makedirs(args.output_dir, exist_ok=True)
          job = executor.submit(main,
                                args.name, args.photo_path, args.output_dir,
                                args.scale, args.aperture, args.focal)

    job_id = job.job_id
    print(f"Submitted batch job {job_id} in folder slurm_dataset")


