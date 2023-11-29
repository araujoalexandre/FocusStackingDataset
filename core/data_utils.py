import os
import glob
import logging
import torch
import numpy as np
import rawpy
import cv2
from natsort import natsorted
from os.path import join

import core.datasets.camera_pipeline as camera
from core.utils import torch_to_npimage, npimage_to_torch, flatten_raw_image


def demosaic_bilinear(image):
  assert isinstance(image, torch.Tensor)
  image = image.clamp(0.0, 1.0) * 255

  if image.dim() == 4:
    num_images = image.shape[0]
    batch_input = True
  else:
    num_images = 1
    batch_input = False
    image = image.unsqueeze(0)

  # Generate single channel input for opencv
  im_sc = torch.zeros((num_images, image.shape[-2] * 2, image.shape[-1] * 2, 1))
  im_sc[:, ::2, ::2, 0] = image[:, 0, :, :]
  im_sc[:, ::2, 1::2, 0] = image[:, 1, :, :]
  im_sc[:, 1::2, ::2, 0] = image[:, 2, :, :]
  im_sc[:, 1::2, 1::2, 0] = image[:, 3, :, :]

  im_sc = im_sc.numpy().astype(np.uint8)
  out = []
  for im in im_sc:
    im_dem_np = cv2.cvtColor(im, cv2.COLOR_BAYER_BG2RGB)
    # Convert to torch image
    im_t = npimage_to_torch(im_dem_np, input_bgr=False)
    out.append(im_t)

  if batch_input:
    return torch.stack(out, dim=0)
  return out[0]

def unpack(rggb):
  b, c, h, w = rggb.shape
  raw = torch.zeros((b, 1, h * 2, w * 2)).to(rggb.device)
  raw[:, 0, 0::2, 0::2] = rggb[:, 0, :, :]
  raw[:, 0, 0::2, 1::2] = rggb[:, 1, :, :]
  raw[:, 0, 1::2, 0::2] = rggb[:, 2, :, :]
  raw[:, 0, 1::2, 1::2] = rggb[:, 3, :, :]
  return raw

def pack(raw):
  b, c, h, w = raw.shape
  rggb = torch.zeros((b, 4, h // 2, w // 2)).to(raw.device)
  rggb[:, 0, :, :] = raw[:, 0, 0::2, 0::2]
  rggb[:, 1, :, :] = raw[:, 0, 0::2, 1::2]
  rggb[:, 2, :, :] = raw[:, 0, 1::2, 0::2]
  rggb[:, 3, :, :] = raw[:, 0, 1::2, 1::2]
  return rggb

def get_cam2rgb(xyz2cam):
  """Generates random RGB -> Camera color correction matrices."""
  # Takes a random convex combination of XYZ -> Camera CCMs.
  xyz2cam = torch.tensor(xyz2cam)[0:3]

  # Multiplies with RGB -> XYZ to get RGB -> Camera CCM.
  rgb2xyz = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                          [0.2126729, 0.7151522, 0.0721750],
                          [0.0193339, 0.1191920, 0.9503041]])
  rgb2cam = torch.mm(xyz2cam, rgb2xyz)

  # Normalizes each row.
  rgb2cam = rgb2cam / rgb2cam.sum(dim=-1, keepdims=True)
  cam2rgb = torch.inverse(rgb2cam)
  return cam2rgb

def rollPattern(raw, visible=True):
  pattern = raw.raw_pattern
  rx,ry = np.where(pattern==0) # check Red channel location on pattern
  rx = int(rx); ry = int(ry)
  if not visible:
    rolledImraw = raw.raw_image[rx:-rx, ry:-ry]
  else:
    rolledImraw = raw.raw_image_visible[rx:-rx or None, ry:-ry or None] # does not include margin
  return rolledImraw[8:3896, 8:5192]

def pack_raw_image(im_raw):
  if isinstance(im_raw, np.ndarray):
    im_out = np.zeros_like(im_raw, shape=(4, im_raw.shape[0] // 2, im_raw.shape[1] // 2))
  elif isinstance(im_raw, torch.Tensor):
    im_out = torch.zeros((4, im_raw.shape[0] // 2, im_raw.shape[1] // 2), dtype=im_raw.dtype)
  else:
    raise Exception
  im_out[0, :, :] = im_raw[0::2, 0::2]
  im_out[1, :, :] = im_raw[0::2, 1::2]
  im_out[2, :, :] = im_raw[1::2, 0::2]
  im_out[3, :, :] = im_raw[1::2, 1::2]
  return im_out

def getColorDestIdx(raw):
  if raw.color_desc == b'RGBG':
    idx = [0, 1, 3, 2]
  elif raw.color_desc == b'RGGB':
    idx = [0, 1, 2, 3]
  else:
    raise NotImplementedError('unknown color desc pattern raw file')
  return idx


def load_raw_image(path, scaling, visible=True):
  # raw_value_visible
  raw = rawpy.imread(path)
  im_raw = rollPattern(raw, visible=visible)
  im_raw = pack_raw_image(im_raw).astype(np.int16) # convert to rggb 4 channels
  im_raw = im_raw.transpose(1, 2, 0)
  if scaling < 1:
    im_raw = cv2.resize(im_raw, None, None, scaling, scaling, cv2.INTER_AREA)
  im_raw = im_raw.transpose(2, 0, 1)
  im_raw = torch.from_numpy(im_raw)
  idx_channels = getColorDestIdx(raw)
  bl = raw.black_level_per_channel
  if raw.camera_white_level_per_channel is not None:
    white_level = raw.camera_white_level_per_channel
  else:
    white_level = (raw.white_level, ) * 4
  cam_wb = raw.camera_whitebalance
  daylight_wb = raw.daylight_whitebalance
  black_level = [bl[i] for i in idx_channels]
  cam_wb = [cam_wb[i] for i in idx_channels]
  daylight_wb = [daylight_wb[i] for i in idx_channels]
  white_level = [white_level[i] for i in idx_channels]
  meta_info = {'black_level': black_level, 'white_level': white_level,'cam_wb': cam_wb,
               'daylight_wb': daylight_wb,'color_mat': raw.rgb_xyz_matrix.tolist()}
  return im_raw.float()[None], meta_info 


def load_jpg_image(path, scaling):
  image = cv2.imread(path)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  if scaling < 1:
    image = cv2.resize(image, None, None, scaling, scaling, cv2.INTER_AREA)
  image = torch.FloatTensor(image)
  image = image.permute(2, 0, 1)
  image = image[None]
  image = image / 255
  return image, None

class Burst:

  def __init__(self, data_dir, burst_name, burst_size, image_type, scaling):
    path = join(data_dir, burst_name, image_type)
    self.burst_size = burst_size
    self.image_type = image_type
    self.scaling = scaling
    self.visible = True
    self.files_list = self.get_files_list(path, burst_size)
    if len(self.files_list) < self.burst_size:
      self.files_list = self.expand_list(self.files_list)

  def expand_list(self, files_list):
    expand_factor = self.burst_size // len(files_list)
    files_new = [[files_list[i]] * expand_factor for i in range(len(files_list))]
    files_new = [x for sublist in files_new for x in sublist]
    return files_new

  def get_files_list(self, path, burst_size):
    burst_list = natsorted(glob.glob(f'{path}/*'))
    if len(burst_list) > burst_size:
      burst_list = burst_list[::int(len(burst_list)//burst_size)][:burst_size]
    return burst_list

  def get_burst(self):
    burst_image_data = []
    for path in self.files_list:
      if self.image_type == 'raw':
        image, meta_info = load_raw_image(path, self.scaling, self.visible)
      elif self.image_type in ['jpg', 'aligned', '']:
        image, meta_info = load_jpg_image(path, self.scaling)
      burst_image_data.append(image)
    burst = torch.cat(burst_image_data, 0).float()
    if self.image_type == 'raw':
      burst = postprocess_raw(burst, meta_info,
                              linearize=True, demosaic=False, wb=False, dwb=False,
                              ccm=False, brightness=False, gamma=False, tonemap=False)
    return burst, meta_info




def postprocess_raw(burst, meta_info, linearize=False, demosaic=False, wb=False, dwb=False,
                    ccm=False, brightness=False, gamma=False, tonemap=False):

  if linearize:  # linearize in [0,1]
    black = torch.tensor(meta_info['black_level']).view(4, 1, 1)
    saturation = torch.tensor(meta_info['white_level']).view(4, 1, 1)
    burst = (burst - black) / (saturation - black)
    burst = burst.clip(0, 1)

  if demosaic:
    burst = demosaic_bilinear(burst)
  
  # white balance
  if wb:
    cam_wb = torch.tensor(meta_info['cam_wb'])
    burst = burst * cam_wb[[0, 1, -1]].view(3, 1, 1) / cam_wb[1]
  elif dwb:
    daylight_wb = torch.tensor(meta_info['daylight_wb'])
    burst = burst * daylight_wb[[0, 1, -1]].view(3, 1, 1) / daylight_wb[1]

  # color matrix
  if ccm:
    xyz2cam = meta_info['color_mat']
    cam2rgb = get_cam2rgb(xyz2cam)
    if burst.dim() == 4:
      bursts = []
      for i in range(burst.shape[0]):
        bursts.append(camera.apply_ccm(burst[i], cam2rgb)[None])
      burst = torch.cat(bursts, 0)
    else:
      burst = camera.apply_ccm(burst, cam2rgb)
    burst = burst.clamp(0.0, 1.0)

  if brightness:
    grayscale = 0.25 / burst.mean()
    burst = burst * grayscale

  burst = burst.clamp(0.0, 1.0)

  if gamma:
    burst = (burst+1e-8) ** (1.0 / 2.2)

  if tonemap:
    burst = 3 * burst ** 2 - 2 * burst ** 3

  return burst




