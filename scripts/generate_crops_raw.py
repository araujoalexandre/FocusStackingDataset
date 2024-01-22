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
import rawpy
import torch.nn.functional as F


def clamp(image):
  image = np.minimum(255, np.maximum(0, image))
  return image


def apply_transform(image, transformation):
  assert image.max() > 2
  shape = list(image.shape[:2])[::-1]
  image = cv2.warpAffine(image, transformation, shape, None, cv2.INTER_CUBIC, cv2.BORDER_REFLECT)
  image = clamp(image)
  return image


def rollPattern(raw, visible=True):
  pattern = raw.raw_pattern
  rx,ry = np.where(pattern==0) # check Red channel location on pattern
  rx = int(rx); ry = int(ry)
  if not visible:
    rolledImraw = raw.raw_image[rx:-rx, ry:-ry]
  else:
    rolledImraw = raw.raw_image_visible[rx:-rx or None, ry:-ry or None] # does not include margin
  rolledImraw = rolledImraw[8:3896, 8:5192]
  return rolledImraw


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


def load_raw_image(path, visible=True):
  # raw_value_visible
  raw = rawpy.imread(path)
  im_raw = rollPattern(raw, visible=True)
  im_raw = pack_raw_image(im_raw).astype(np.int16) # convert to rggb 4 channels
  im_raw = im_raw.transpose(1, 2, 0)
  im_raw = np.float32(im_raw)
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
  return im_raw, meta_info 


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
    im_t = numpy_to_torch(im_dem_np) / 255
    out.append(im_t)

  if batch_input:
    return torch.stack(out, dim=0)
  return out[0]


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


def apply_ccm(image, ccm):
  """Applies a color correction matrix."""
  if ccm.dim() == 3: ccm = ccm[0]
  shape = image.shape
  if image.dim() == 3:
    assert image.shape[0] == 3
    image = image.view(3, -1)
    ccm = ccm.to(image.device).type_as(image)
    image = torch.mm(ccm, image)
    return image.view(shape)
  elif image.dim() == 4:
    assert image.shape[1] == 3
    burst_size = shape[0]
    image = image.view(burst_size, 3, -1)
    image = image.permute(0, 2, 1)
    ccm = ccm.to(image.device).type_as(image)
    image = torch.matmul(image, ccm.T)
    image = image.permute(0, 2, 1)
    return image.view(shape)


def numpy_to_torch(a: np.ndarray):
  return torch.from_numpy(a).float().permute(2, 0, 1)


def postprocess_raw(burst, meta_info, linearize=False, demosaic=False, wb=False, dwb=False,
                    ccm=False, brightness=False, gamma=False, tonemap=False):

  if linearize:  # linearize in [0,1]
    black = torch.tensor(meta_info['black_level']).view(4, 1, 1)
    saturation = torch.tensor(meta_info['white_level']).view(4, 1, 1)
    burst = (burst - black) / (saturation - black)
    burst = burst.clip(0, 1)

  assert burst.max() < 2

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
        bursts.append(apply_ccm(burst[i], cam2rgb)[None])
      burst = torch.cat(bursts, 0)
    else:
      burst = apply_ccm(burst, cam2rgb)
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




class GenerateCrops:

  def __init__(self, crop_size):
    self.crop_size = crop_size

  def split_image(self, image, raw=False):
    if raw:
      crop_size = self.crop_size // 2 
    else:
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
    # load target
    target = cv2.imread(join(burst_path, 'helicon_focus_aligned_tiff.jpg'), cv2.IMREAD_COLOR)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    dim = (int(target.shape[1]), int(target.shape[0]))

    imgs_path = glob.glob(join(burst_path, 'raw', '*.rw2'))
    imgs_path = natsort.natsorted(imgs_path)
    
    images_raw = []
    for i, path in enumerate(imgs_path):
      image_raw, meta_info = load_raw_image(path, 1)
      image_raw = torch.FloatTensor(image_raw)[None].permute(0, 3, 1, 2)
      image_raw = postprocess_raw(image_raw, meta_info, linearize=True)
      image_raw = image_raw[0].permute(1, 2, 0).numpy()
      images_raw.append(image_raw)
        
    aligned_raw = [None] * len(images_raw)
    aligned_raw[0] = images_raw[0]

    transforms = torch.load(join(burst_path, 'transforms.pth'))
    
    for i in range(len(images_raw)-1):
      transform = np.array(transforms[i])
      transform[:, 2] /= 2
      aligned_raw[i+1] = apply_transform(images_raw[i+1]*255, transform) / 255
      
    aligned_raw = np.array(aligned_raw)

    aligned_raw = torch.Tensor(aligned_raw)
    aligned_raw = aligned_raw.permute(0, 3, 1, 2)

    target = torch.Tensor(target)[None]
    target = target.to(torch.uint8)
    target = target.permute(0, 3, 1, 2)

    raw_crops = self.split_image(aligned_raw, raw=True)
    target_crops = self.split_image(target)

    assert raw_crops.shape[0] == target_crops.shape[0], f'{aligned_raw.shape}'

    outpath = join(burst_path, 'crops_raw')
    os.makedirs(outpath, exist_ok=True)
    for i, (burst_crop, target_crop) in enumerate(zip(raw_crops, target_crops)):
      burst_crop = burst_crop.clone()
      target_crop = target_crop.clone()
      data = {'burst': burst_crop, 'target': target_crop, 'meta_info': meta_info}
      torch.save(data, join(outpath, f'crop{i}.pkl'))

    outpath = join(burst_path, 'aligned_raw')
    os.makedirs(outpath, exist_ok=True)
    params = dict(demosaic=True, wb=True, dwb=False, ccm=True, brightness=True, gamma=True, tonemap=True)
    for i, image in enumerate(aligned_raw):
      image = torch.FloatTensor(image)[None]
      image = postprocess_raw(image, meta_info, linearize=False, **params)
      image = image[0].permute(1, 2, 0).numpy()
      image = (image*255).astype(np.uint8)
      Image.fromarray(image).save(join(outpath, f'burst_{i}.jpg'))



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


