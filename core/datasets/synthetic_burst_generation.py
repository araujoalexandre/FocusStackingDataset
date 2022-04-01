import logging
import random
import numpy as np
import torch
import cv2
import torchvision.transforms as tfm
import torch.nn.functional as F
from torch.utils.data import Dataset

import core.datasets.camera_pipeline as camera
from core.utils import torch_to_npimage, npimage_to_torch

""" File based on https://github.com/Algolzw/EBSR/blob/main/datasets/synthetic_burst_train_set.py and 
https://github.com/Algolzw/EBSR/blob/main/data_processing/synthetic_burst_generation.py """

class RandomCrop:
  """ Extract a random crop of size crop_sz from the input frames. If the crop_sz is larger than the input image size,
  then the largest possible crop of same aspect ratio as crop_sz will be extracted from frames, and upsampled to
  crop_sz.
  """
  def __init__(self, shape, crop_sz):

    if not isinstance(crop_sz, (tuple, list)):
      crop_sz = (crop_sz, crop_sz)
    crop_sz = torch.tensor(crop_sz).float()
    self.crop_sz = crop_sz

    # Select scale_factor. Ensure the crop fits inside the image
    max_scale_factor = torch.tensor(shape[-2:]).float() / crop_sz
    max_scale_factor = max_scale_factor.min().item()

    if max_scale_factor < 1.0:
      self.scale_factor = max_scale_factor
    else:
      self.scale_factor = 1.0

    # Extract the crop
    orig_crop_sz = (crop_sz * self.scale_factor).floor()

    assert orig_crop_sz[-2] <= shape[-2] and orig_crop_sz[-1] <= shape[-1], 'Bug in crop size estimation!'

    self.r1 = random.randint(0, shape[-2] - orig_crop_sz[-2])
    self.c1 = random.randint(0, shape[-1] - orig_crop_sz[-1])

    self.r2 = self.r1 + orig_crop_sz[0].int().item()
    self.c2 = self.c1 + orig_crop_sz[1].int().item()

  def get_crop_coord(self):
    return (self.r1, self.c1), (self.r2, self.c2)

  def crop(self, frames, frames_gt):
    (r1, c1), (r2, c2) = self.get_crop_coord()
    frames_crop = frames[:, :, r1:r2, c1:c2]
    frames_crop_gt = frames_gt[:, :, r1:r2, c1:c2]
    # Resize to crop_sz
    if self.scale_factor < 1.0:
      frames_crop = F.interpolate(frames_crop.unsqueeze(0),
                                  size=self.crop_sz.int().tolist(), mode='bilinear').squeeze(0)
    return frames_crop, frames_crop_gt



def get_tmat(image_shape, translation, theta, shear_values, scale_factors):
  """ Generates a transformation matrix corresponding to the input transformation parameters """
  im_h, im_w = image_shape
  t_mat = np.identity(3)
  t_mat[0, 2] = translation[0]
  t_mat[1, 2] = translation[1]
  t_rot = cv2.getRotationMatrix2D((im_w * 0.5, im_h * 0.5), theta, 1.0)
  t_rot = np.concatenate((t_rot, np.array([0.0, 0.0, 1.0]).reshape(1, 3)))
  t_shear = np.array([[1.0, shear_values[0], -shear_values[0] * 0.5 * im_w],
                      [shear_values[1], 1.0, -shear_values[1] * 0.5 * im_h],
                      [0.0, 0.0, 1.0]])
  t_scale = np.array([[scale_factors[0], 0.0, 0.0],
                      [0.0, scale_factors[1], 0.0],
                      [0.0, 0.0, 1.0]])
  t_mat = t_scale @ t_rot @ t_shear @ t_mat
  t_mat = t_mat[:2, :]
  return t_mat


def single2lrburst(image, depthmap, burst_size, downsample_factor=1, transformation_params=None,
                   interpolation_type='bilinear'):
  """ Generates a burst of size burst_size from the input image by applying random transformations defined by
  transformation_params, and downsampling the resulting burst by downsample_factor.
  """
  if interpolation_type == 'bilinear':
    interpolation = cv2.INTER_LINEAR
  elif interpolation_type == 'lanczos':
    interpolation = cv2.INTER_LANCZOS4
  else:
    raise ValueError

  image = torch_to_npimage(image).astype(np.uint8)

  burst = []
  sample_pos_inv_all = []

  rvs, cvs = torch.meshgrid([torch.arange(0, image.shape[0]),
                             torch.arange(0, image.shape[1])], indexing='ij')

  sample_grid = torch.stack((cvs, rvs, torch.ones_like(cvs)), dim=-1).float()

  for i in range(burst_size):
    if i == 0:
      # For base image, do not apply any random transformations.
      # We only translate the image to center the sampling grid
      shift = (downsample_factor / 2.0) - 0.5
      translation = (shift, shift)
      theta = 0.0
      shear_factor = (0.0, 0.0)
      scale_factor = (1.0, 1.0)
    else:
      # Sample random image transformation parameters
      max_translation = transformation_params.get('max_translation', 0.0)

      if max_translation <= 0.01:
        shift = (downsample_factor / 2.0) - 0.5
        translation = (shift, shift)
      else:
        translation = (random.uniform(-max_translation, max_translation),
                       random.uniform(-max_translation, max_translation))

      max_rotation = transformation_params.get('max_rotation', 0.0)
      theta = random.uniform(-max_rotation, max_rotation)

      max_shear = transformation_params.get('max_shear', 0.0)
      shear_x = random.uniform(-max_shear, max_shear)
      shear_y = random.uniform(-max_shear, max_shear)
      shear_factor = (shear_x, shear_y)

      max_ar_factor = transformation_params.get('max_ar_factor', 0.0)
      ar_factor = np.exp(random.uniform(-max_ar_factor, max_ar_factor))

      max_scale = transformation_params.get('max_scale', 0.0)
      scale_factor = np.exp(random.uniform(-max_scale, max_scale))

      scale_factor = (scale_factor, scale_factor * ar_factor)

    output_sz = (image.shape[1], image.shape[0])

    # Generate a affine transformation matrix corresponding to the sampled parameters
    t_mat = get_tmat((image.shape[0], image.shape[1]), translation, theta, shear_factor, scale_factor)
    t_mat_tensor = torch.from_numpy(t_mat)

    # Apply the sampled affine transformation
    image_t = cv2.warpAffine(image, t_mat, output_sz, flags=interpolation, borderMode=cv2.BORDER_CONSTANT)

    t_mat_tensor_3x3 = torch.cat((t_mat_tensor.float(), torch.tensor([0.0, 0.0, 1.0]).view(1, 3)), dim=0)
    t_mat_tensor_inverse = t_mat_tensor_3x3.inverse()[:2, :].contiguous()

    sample_pos_inv = torch.mm(sample_grid.view(-1, 3), t_mat_tensor_inverse.t().float()).view(
        *sample_grid.shape[:2], -1)

    if transformation_params.get('border_crop') is not None:
      border_crop = transformation_params.get('border_crop')
      image_t = image_t[border_crop:-border_crop, border_crop:-border_crop, :]
      sample_pos_inv = sample_pos_inv[border_crop:-border_crop, border_crop:-border_crop, :]

    # Downsample the image
    image_t = cv2.resize(image_t, None, fx=1.0 / downsample_factor, fy=1.0 / downsample_factor,
                         interpolation=interpolation)
    sample_pos_inv = cv2.resize(sample_pos_inv.numpy(), None, fx=1.0 / downsample_factor,
                                fy=1.0 / downsample_factor,
                                interpolation=interpolation)

    sample_pos_inv = torch.from_numpy(sample_pos_inv).permute(2, 0, 1).contiguous()

    image_t = npimage_to_torch(image_t).float()
    burst.append(image_t)
    sample_pos_inv_all.append(sample_pos_inv / downsample_factor)


  burst_images = torch.stack(burst)
  sample_pos_inv_all = torch.stack(sample_pos_inv_all)

  # Compute the flow vectors to go from the i'th burst image to the base image
  flow_vectors = sample_pos_inv_all - sample_pos_inv_all[:, :1, ...]

  return burst_images, depthmap, flow_vectors



def flatten_raw_image_batch(x):
  out = torch.zeros((x.shape[0], 1, x.shape[2] * 2, x.shape[3] * 2), dtype=x.dtype)
  out = out.to(x.device)
  out[:, 0, 0::2, 0::2] = x[:, 0, :, :]
  out[:, 0, 0::2, 1::2] = x[:, 1, :, :]
  out[:, 0, 1::2, 0::2] = x[:, 2, :, :]
  out[:, 0, 1::2, 1::2] = x[:, 3, :, :]
  return out




class SyntheticBurst(Dataset):
  """ Synthetic burst dataset for joint denoising, demosaicking, and super-resolution. RAW Burst sequences are
  synthetically generated on the fly as follows. First, a single image is loaded from the base_dataset. The sampled
  image is converted to linear sensor space using the inverse camera pipeline employed in [1]. A burst
  sequence is then generated by adding random translations and rotations to the converted image. The generated burst
  is then converted is then mosaicked, and corrupted by random noise to obtain the RAW burst.

  [1] Unprocessing Images for Learned Raw Denoising, Brooks, Tim and Mildenhall, Ben and Xue, Tianfan and Chen,
  Jiawen and Sharlet, Dillon and Barron, Jonathan T, CVPR 2019
  """
  def __init__(self, base_dataset, config, transform=None):

    self.base_dataset = base_dataset
    self.burst_size = config.data.burst_size
    self.crop_sz = config.data.crop_sz
    self.downsample_factor = config.data.downsample_factor

    self.burst_transformation_params = {'max_translation': config.data.max_translation,
                                        'max_rotation': config.data.max_rotation,
                                        'max_shear': config.data.max_shear,
                                        'max_scale': config.data.max_scale,
                                        'border_crop': config.data.border_crop}
    self.image_processing_params = {'random_ccm': True, 
                                    'random_gains': True, 
                                    'smoothstep': True, 
                                    'gamma': True, 
                                    'add_noise': True}
    self.interpolation_type = 'bilinear'

    # Sample camera pipeline params
    if self.image_processing_params['random_ccm']:
      self.rgb2cam = camera.random_ccm()
    else:
      self.rgb2cam = torch.eye(3).float()
    self.cam2rgb = self.rgb2cam.inverse()

    # Sample gains
    if self.image_processing_params['random_gains']:
      self.rgb_gain, self.red_gain, self.blue_gain = camera.random_gains()
    else:
      self.rgb_gain, self.red_gain, self.blue_gain = (1.0, 1.0, 1.0)

  def __len__(self):
    return len(self.base_dataset)

  def rgb2rawburst(self, image, depthmap): 
    """ Generates a synthetic LR RAW burst from the input image.
    The input sRGB image is first converted to linear sensor space using an inverse camera pipeline.
    A LR burst is then generated by applying random transformations defined by 
    burst_transformation_params to the input image, and downsampling it by the downsample_factor.
    The generated burst is then mosaicekd and corrputed by random noise.
    """

    # Approximately inverts global tone mapping.
    use_smoothstep = self.image_processing_params['smoothstep']
    if use_smoothstep:
      image = camera.invert_smoothstep(image)

    # Inverts gamma compression.
    use_gamma = self.image_processing_params['gamma']
    if use_gamma:
      image = camera.gamma_expansion(image)

    # Inverts color correction.
    image = camera.apply_ccm(image, self.rgb2cam)

    # Approximately inverts white balance and brightening.
    image = camera.safe_invert_gains(
      image, self.rgb_gain, self.red_gain, self.blue_gain)

    # Clip saturated pixels.
    image = image.clamp(0.0, 1.0)

    border_crop = self.burst_transformation_params.get('border_crop', 0)
    image_pad = F.pad(image, (border_crop, border_crop, border_crop, border_crop), mode='reflect')

    # Generate LR burst
    image_burst_rgb, depthmap, flow_vectors = \
        single2lrburst(image_pad, depthmap, burst_size=self.burst_size,
                       downsample_factor=self.downsample_factor,
                       transformation_params=self.burst_transformation_params,
                       interpolation_type=self.interpolation_type)

    random_crop = RandomCrop(image_burst_rgb.shape, self.crop_sz)
    random_crop_coord = random_crop.get_crop_coord()

    if not self.no_crop_synthetic_burst:
      # Extract a random crop from the image
      crop_sz = self.crop_sz
      image_burst_rgb, image = random_crop.crop(image_burst_rgb, image)

    # mosaic
    image_burst = camera.mosaic(image_burst_rgb.clone())

    # Add noise
    if self.image_processing_params['add_noise']:
      shot_noise_level, read_noise_level = camera.random_noise_levels()
      image_burst = camera.add_noise(image_burst, shot_noise_level, read_noise_level)
    else:
      shot_noise_level = 0
      read_noise_level = 0

    # Clip saturated pixels.
    image_burst = image_burst.clamp(0.0, 1.0)

    meta_info = {
      'rgb2cam': self.rgb2cam, 'cam2rgb': self.cam2rgb, 
      'rgb_gain': self.rgb_gain, 'red_gain': self.red_gain, 'blue_gain': self.blue_gain, 
      'smoothstep': use_smoothstep, 'gamma': use_gamma,
      'shot_noise_level': shot_noise_level, 'read_noise_level': read_noise_level}

    new_image_burst_rgb = []
    if not self.no_crop_synthetic_burst:
      new_image_burst_rgb.append(
         camera.process_linear_image_rgb(burst_, meta_info, return_np=False)
      )
    image_burst_rgb = torch.cat(new_image_burst_rgb, dim=0)

    return image_burst, image, image_burst_rgb, flow_vectors, meta_info


  def process_jpg_burst(self, burst_rgb, image_gt):

    border_crop = self.burst_transformation_params.get('border_crop', 0)
    burst_pad = F.pad(burst_rgb, (border_crop, border_crop, border_crop, border_crop), mode='reflect')
    image_gt_pad = F.pad(image_gt, (border_crop, border_crop, border_crop, border_crop), mode='reflect')

    random_crop = RandomCrop(burst_pad.shape, self.crop_sz)
    random_crop_coord = random_crop.get_crop_coord()

    # Extract a random crop from the image
    burst_rgb_crop, image_gt_crop = random_crop.crop(burst_pad, image_gt_pad)

    return burst_rgb_crop, image_gt_crop

  def __getitem__(self, index):
    """ Generates a synthetic burst
    args:
      index: Index of the image in the base_dataset used to generate the burst

    returns:
      burst: 
        Generated LR RAW burst, a torch tensor of shape
        [burst_size, 4, crop_sz / (2*downsample_factor), crop_sz / (2*downsample_factor)]
        The 4 channels correspond to 'R', 'G', 'G', and 'B' values in the RGGB bayer mosaick.
        The extra factor 2 in the denominator (2*downsample_factor) corresponds to the mosaicking
        operation.

      frame_gt:
        The HR RGB ground truth in the linear sensor space, a torch tensor of shape [3, crop_sz, crop_sz]

      flow_vectors:
        The ground truth flow vectors between a burst image and the base image (i.e. the first image in the burst).
        The flow_vectors can be used to warp the burst images to the base frame, using the 'warp' function in utils.warp package.
        flow_vectors is torch tensor of shape [burst_size, 2, crop_sz / downsample_factor, crop_sz / downsample_factor].
        Note that the flow_vectors are in the LR RGB space, before mosaicking. Hence it has twice
        the number of rows and columns, compared to the output burst.

        NOTE: The flow_vectors are only available during training for the purpose of using any
        auxiliary losses if needed. The flow_vectors will NOT be provided for the bursts in the
        test set

      meta_info: A dictionary containing the parameters used to generate the synthetic burst.
    """
    burst, image_gt = self.base_dataset[index]
    # Generate JPG burst
    burst, image_gt = self.process_jpg_burst(burst, image_gt)
    return burst, image_gt.squeeze(0)



