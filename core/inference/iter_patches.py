
import logging
import numpy as np
import torch
import torch.nn.functional as F
from core.inference.flow import calcFlow
from core.utils import flatten_raw_image
from core.data_utils import postprocess_raw


class IterPatchesImage:

  def __init__(self, burst, meta_info, config, optical_flows=True):

    if optical_flows:
      logging.info(f'Computing optical flow ...')
      if config.eval.image_type == 'raw':
        burst_rgb = postprocess_raw(burst, meta_info, linearize=False, demosaic=True,
                                    wb=True, ccm=False, brightness=True,
                                    gamma=True, tonemap=True)
      else:
        burst_rgb = burst
      self.flows, _ = calcFlow(burst_rgb, D=config.eval.of_scale, poly_n=7,
                               poly_sigma=1.5, interpolate=False, winsize=config.eval.of_win)
    else:
      self.flows = torch.zeros_like(burst)

    if config.eval.image_type == 'raw':
      burst = flatten_raw_image(burst)

    logging.info(f"burst size: {burst.shape}")

    # self.burst, self.original_padding = self.padding(burst, config.eval.patch_size, 'replicate')
    # self.flows, _ = self.padding(self.flows, config.eval.patch_size, 'constant')

    self.burst = burst
    self.original_padding = (0, 0, 0, 0)

    self.burst_size = self.burst.shape
    self.in_channels = config.archi.n_colors
    self.out_channels = 3 # we always output rgb images
    self.output_size = burst.shape[2:]
    self.batch_size = config.eval.eval_batch_size
    self.of_scale = config.eval.of_scale
    self.upscale = config.data.downsample_factor
    self.block_stride = 32
    self.last_padding = 16

    self.block_size_in = config.eval.patch_size + self.last_padding * 2
    self.block_size_out = config.eval.patch_size * self.upscale

    self.blocks_coord = self.get_coord_blocks(self.burst)

    self.n_blocks = self.blocks_coord[0].shape[1]
    self.batch_out_blocks = torch.empty(
      self.n_blocks, self.out_channels, self.block_size_out, self.block_size_out)

    self.all_indexes = list(range(0, self.n_blocks, self.batch_size))
    logging.info(f'Number of batches: {len(self.all_indexes)}')

  def padding(self, x, factor, pad_mode):
    expand_x = np.int32(2**np.ceil(np.log2(x.shape[-2])) - x.shape[-2])
    expand_y = np.int32(2**np.ceil(np.log2(x.shape[-1])) - x.shape[-1])
    padding = (0, expand_y, 0, expand_x)
    x = F.pad(x, padding, mode=pad_mode)
    return x, padding
  
  def remove_padding(self, x):
    padding = self.original_padding
    return x[:, :, :-padding[3] or None, :-padding[1] or None]
    
  def im2col(self, x):
    batch = x.shape[0]
    out = F.unfold(x, kernel_size=self.block_size_in, stride=self.block_stride, padding=0, dilation=1)
    return out

  def col2im(self, x):
    output_size = (np.array(self.burst_size[2:]) - 2 * self.last_padding) * self.upscale
    kernel_size = self.block_size_out # * self.upscale
    stride = self.block_stride * self.upscale
    fold_params = dict(output_size=output_size, kernel_size=kernel_size,
                 stride=stride, padding=0, dilation=1)
    out = F.fold(x, **fold_params)
    mean = F.fold(torch.ones_like(x), **fold_params)    
    out = out / mean
    return out

  def make_blocks(self, image):
    """
    :param image: (1, C, H, W)
    :return: raw block (batch, C, block_size, block_size), tulple shape augmented image
    """
    batch_blocks = self.im2col(image)
    batch_blocks = batch_blocks.permute(0, 2, 1)
    batch_blocks = batch_blocks.reshape(1, -1, 3, self.block_size_in, self.block_size_in)
    return batch_blocks

  def get_coord_blocks(self, image):
    batch_size, n_channels, h, w = image.shape
    x = torch.arange(0, h)
    y = torch.arange(0, w)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    img_coord = torch.stack((xx, yy, yy)).unsqueeze(0).float()
    tiles = self.make_blocks(img_coord)
    coord_upleft = tiles[..., :-1, 0, 0]
    coord_downright = coord_upleft + self.block_size_in
    return coord_upleft, coord_downright

  def extract_patch(self, i):

    # coord patch (up,down) to (x, y, h, w)
    up, down = self.blocks_coord[0][0, i, :].int(), self.blocks_coord[1][0, i, :].int()
    x, y = up
    h, w = down - up

    # get crop on each frame corrected with optical flow
    xf = torch.div(x, self.of_scale, rounding_mode='floor')
    yf = torch.div(y, self.of_scale, rounding_mode='floor')
    hf = torch.div(h, self.of_scale, rounding_mode='floor')
    wf = torch.div(w, self.of_scale, rounding_mode='floor')
    self.flow_crop = self.flows[:, :, xf:xf+hf, yf:yf+wf]

    # take mean optical on patch and round by 2 for exact interpolation
    # and for keeping the correct RGGB pattern
    mean_flow = self.flow_crop.mean((2, 3))
    offset = mean_flow.mul(0.5).round().mul(2).int()

    # patch idx corrected w offset for each frame
    x_offset, y_offset = x + offset[:, 1], y + offset[:, 0]  
    x_offset, y_offset = torch.relu(x_offset), torch.relu(y_offset)

    # extract coarsly aligned patch on each frame
    burst = torch.zeros(self.burst.shape[0], self.in_channels, self.block_size_in, self.block_size_in)
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
    start = self.indexes[0]
    p = self.last_padding * self.upscale
    batch_patches_cropped = batch_patches[..., p:-p, p:-p]
    self.batch_out_blocks[start:start+self.batch_size, ...] = batch_patches_cropped

  def agregate_blocks(self):
    """
    :param blocks: processed blocks
    :return: image of averaged estimates
    """
    del self.burst, self.flows, self.blocks_coord
    n_blocks, out_channels, block_size_out, block_size_out = self.batch_out_blocks.shape
    batch_out_blocks_flatten = self.batch_out_blocks.reshape(
      1, n_blocks, out_channels * block_size_out**2)
    batch_out_blocks_flatten = batch_out_blocks_flatten.permute(0, 2, 1)
    output = self.col2im(batch_out_blocks_flatten)
    return self.remove_padding(output)



