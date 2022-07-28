
import logging
import numpy as np
import torch
import torch.nn.functional as F
from core.inference.im2col import Im2Col, Col2Im


class IterPatchesImage:

  def __init__(self, burst, flows, config, center_pad=False, avg=True, pad_mode='constant'):

    self.burst = burst
    self.flows = flows

    self.n_channels = self.burst.shape[1]

    self.batch_size = config.eval.eval_batch_size
    self.crop_size = config.eval.crop // 4 * 4 # force even crop factor to keep the correct rggb pattern
    self.block_size = config.eval.patch_size
    stride = config.eval.stride
    if config.eval.stride is None:
      stride = config.eval.patch_size
    self.upscale = config.data.downsample_factor
    self.block_stride = stride - 2 * self.crop_size // self.upscale
    self.center_pad = center_pad
    self.avg = avg
    self.pad_mode = pad_mode

    self.of_scale = config.eval.of_scale

    self.blocks_coord = self.get_coord_blocks(self.burst)

    # batch, n_blocks, channels, hb, wb = blocks.shape
    self.n_blocks = self.blocks_coord[0].shape[1]
    patch_size = self.block_size * self.upscale - 2 * self.crop_size
    self.batch_out_blocks = torch.zeros(self.n_blocks, self.n_channels, patch_size, patch_size)

    self.all_indexes = list(range(0, self.n_blocks, self.batch_size))

  def shape_pad_even(self, tensor_shape, patch, stride):
    assert len(tensor_shape) == 4
    b, c, h, w = tensor_shape
    required_pad_h = (patch - (h - patch) % stride) % patch
    required_pad_w = (patch - (w - patch) % stride) % patch
    return required_pad_h, required_pad_w

  def make_blocks(self, image):
    """
    :param image: (1,C,H,W)
    :return: raw block (batch,C,block_size,block_size), tulple shape augmented image
    """
    pad_even = self.shape_pad_even(image.shape, self.block_size, self.block_stride)
    pad_h, pad_w =  pad_even

    if self.center_pad:
      pad_ = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
    else:
      pad_ = (0, pad_w, 0, pad_h)
    self.pad = pad_

    # add half kernel cause block edges are dumped
    image_padded_even = F.pad(image, self.pad, mode=self.pad_mode)
    self.padded_shape = image_padded_even.shape
    batch_blocks = Im2Col(image_padded_even,
                          kernel_size=self.block_size, stride=self.block_stride, padding=0)
    batch_blocks = batch_blocks.permute(0, 2, 1)
    batch_blocks = batch_blocks.view(1, -1, self.n_channels, self.block_size, self.block_size)
    return batch_blocks

  def get_coord_blocks(self,image):
    b,c,h,w = image.shape
    x = torch.arange(0, h)
    y = torch.arange(0, w)
    xx, yy = torch.meshgrid(x, y)
    img_coord = torch.stack((xx, yy, yy)).unsqueeze(0).float()
    tiles = self.make_blocks(img_coord)
    coord_upleft = tiles[..., :-1, 0, 0]
    coord_downright = coord_upleft + self.block_size
    return coord_upleft, coord_downright

  def extract_patch(self, i):

    # coord patch (up,down) to (x, y, h, w)
    up, down = self.blocks_coord[0][0, i, :].int(), self.blocks_coord[1][0, i, :].int()
    x, y = up
    h, w = down - up

    # get crop on each frame corrected with optical flow
    scale = self.of_scale
    xf, yf, hf, wf = x // scale, y // scale, h // scale, w // scale
    self.flow_crop = self.flows[:, :, xf:xf+hf, yf:yf+wf]

    # take mean optical on patch and round by 2 for exact interpolation
    # and for keeping the correct RGGB pattern
    mean_flow = self.flow_crop.mean((2, 3))
    offset = mean_flow.mul(0.5).round().mul(2).int()

    # patch idx corrected w offset for each frame
    x_offset, y_offset = x + offset[:, 1], y + offset[:, 0]  

    # extract coarsly aligned patch on each frame
    burst = torch.zeros(self.burst.shape[0], 3, self.block_size, self.block_size)
    for j, (xc, yc) in enumerate(zip(x_offset, y_offset)):
      crops = self.burst[j, :, xc:xc + h, yc:yc + w]
      burst[j, :, :crops.shape[-2], :crops.shape[-1]] = crops

    return burst

  def __iter__(self):
    self.ix = 0
    return self

  def __next__(self):
    if self.ix < len(self.all_indexes):
      i = self.all_indexes[self.ix]
      self.ix += 1
      self.indexes = np.arange(self.n_blocks)[i:i+self.batch_size]
      crops = []
      for j in self.indexes:
        crops.append(self.extract_patch(j)[None])
      return torch.cat(crops, dim=0)
    raise StopIteration

  def add_processed_patches(self, batch_patches):
    crop = self.crop_size
    batch_patches = batch_patches[:, :, crop:-crop or None, crop:-crop or None]
    start = self.indexes[0]
    self.batch_out_blocks[start:start+self.batch_size, ...] = batch_patches

  def agregate_blocks(self):
    """
    :param blocks: processed blocks
    :return: image of averaged estimates
    """
    h_pad, w_pad = self.padded_shape[2:]

    pad = self.pad
    pad = tuple(i * self.upscale for i in pad)

    batch_out_blocks_flatten = self.batch_out_blocks.contiguous().view(
      1, -1, self.n_channels * (self.block_size*self.upscale-2*self.crop_size)**2)
    batch_out_blocks_flatten = batch_out_blocks_flatten.permute(0, 2, 1)

    output_size = (h_pad*self.upscale-2*self.crop_size, w_pad*self.upscale-2*self.crop_size)
    output_padded = Col2Im(batch_out_blocks_flatten,
                           output_size=output_size,
                           kernel_size=self.block_size*self.upscale-2*self.crop_size,
                           stride=self.block_stride * self.upscale,
                           padding=0,
                           avg=self.avg)

    output = output_padded[:, :, pad[2]:-pad[3] or None, pad[0]:-pad[1] or None]
    return output



