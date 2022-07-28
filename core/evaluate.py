import os
import time
import random
import datetime
import pprint
import socket
import logging
import shutil
import glob
import subprocess
import cv2
from os.path import join, basename

from core import utils
from core.models import models_config
from core.datasets.readers import readers_config
from core.inference.iter_patches import IterPatchesImage
from core.inference.flow import calcFlow
from core.inference.align_key_points import AlignImages
from core.data_utils import Burst
from core.datasets.readers import get_data_dir

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import nn
from torch.distributed.elastic.multiprocessing.errors import record


class Evaluator:

  def __init__(self, config, submitit=True):
    self.config = config
    self.submitit = submitit

  def load_state(self, ckpt_path=None):
    if ckpt_path is None:
      ckpt_path = join(
        self.config.train_dir, "checkpoints", "best_model.ckpt.pth")
    checkpoint = torch.load(ckpt_path)
    model_state = checkpoint['model_state_dict']
    self.model.load_state_dict(model_state)

  def get_patches(self):
    # compute optical flow
    self.of_scale = self.config.eval.of_scale
    self.of_win = self.config.eval.of_win
    logging.info(f'Computing optical flow ...')
    self.flows, _ = calcFlow(
      self.burst, D=self.of_scale, poly_n=7, poly_sigma=1.5, interpolate=False, winsize=self.of_win)
    # define the patch generator
    generator_patches = IterPatchesImage(self.burst, self.flows, self.config,
      pad_mode='constant', center_pad=False)
    return generator_patches

  def preprocess(self, x):
    if self.config.archi.model not in ['wavelets', 'laplacian']:
      return x - 0.5
    return x

  def postprocess(self, output):
    if self.config.archi.model not in ['wavelets', 'laplacian']:
      return output + 0.5
    return output

  def eval_loop(self, epoch, ckpt, best_psnr, best_epoch, best_ckpt):
    # eval model on test: compute metrics 
    criterion = utils.L1Loss(boundary_ignore=4)
    psnr = utils.PSNR(boundary_ignore=4)
    # load test dataset
    Reader = readers_config[self.config.project.dataset]
    self.reader = Reader(self.config, self.batch_size, False, is_training=False)
    data_loader, _ = self.reader.load_dataset()

    running_loss, running_psnr = 0., 0.
    for n_batch, data in enumerate(data_loader):
      burst, frame_gt = data
      burst = burst.cuda(non_blocking=True)
      frame_gt = frame_gt.cuda(non_blocking=True)

      if self.config.archi.n_colors == 1:
        frame_gt = frame_gt.mean(2)

      burst, frame_gt = self.preprocess(burst), self.preprocess(frame_gt)
      with torch.no_grad():
        outputs = self.model(burst)
      outputs = outputs.squeeze(1)
      frame_gt = frame_gt.squeeze(1)
      running_loss += criterion(outputs, frame_gt)
      running_psnr += psnr(outputs, frame_gt)

    final_loss = running_loss / (n_batch + 1)
    final_psnr = running_psnr / (n_batch + 1)
    if best_psnr < final_psnr:
      best_psnr = final_psnr
      best_epoch = epoch
      best_ckpt = ckpt

    self.message.add('epoch', epoch, format='.0f')
    self.message.add("L1", final_loss, format=".4f")
    self.message.add("PSNR", final_psnr, format=".4f")
    self.message.add("Best PSNR", best_psnr, format=".4f")
    logging.info(self.message.get_message())
    return best_psnr, best_epoch, best_ckpt

  def inference_full(self):
    """ inference on all image """
    with torch.no_grad():
      burst = self.preprocess(self.burst[None])
      output = self.model(burst)
      output = self.postprocess(output)
    return final_image

  def inference_by_patch(self):
    """ inference by patch """
    generator_patches = self.get_patches()
    logging.info('Block inference ...')
    for i, x in enumerate(generator_patches):
      if i == 0:
        logging.info(f'x: {x.shape}')
      x = x.cuda()
      x = self.preprocess(x)
      with torch.no_grad():
        output = self.model(x)
      output = output.float().cpu()
      output = self.postprocess(output)
      generator_patches.add_processed_patches(output) 
    logging.info('Agregating blocks ...')
    final_image = generator_patches.agregate_blocks()
    return final_image

  def save_output(self, final_image):
    logging.info('Saving Image ...')
    final_image = final_image.clamp(0, 1)
    final_image = final_image[0]
    final_image = final_image.permute(1, 2, 0)
    final_image = final_image.mul(255.0)
    final_image = final_image.cpu().numpy().astype(np.uint8)
    if final_image.shape[-1] == 1:
      final_image = final_image.squeeze(-1)
    logging.info(f"Final shape: {final_image.shape}")
    # save result
    patch = 'patch' if self.eval_by_patch else 'full'
    image_type = self.config.eval.image_type
    name = f'prediction_{self.burst_name}_{patch}_{image_type}.jpg'
    prediction_path = join(self.prediction_folder, name)
    Image.fromarray(final_image).save(prediction_path, quality=95)

  def eval_model(self):
    # load model
    Model = models_config[self.config.archi.model]
    self.model = Model(self.config)
    self.model = nn.DataParallel(self.model)
    self.model = self.model.cuda()
    self.model.eval()

    if self.burst_name == 'test':

      best_ckpt, best_epoch, best_psnr = None, None, 0.
      ckpts = utils.get_list_checkpoints(self.train_dir)
      for ckpt_id, ckpt in enumerate(ckpts):
        self.load_state(ckpt)
        epoch = utils.get_epochs_from_ckpt(ckpt)
        best_psnr, best_epoch, best_ckpt = self.eval_loop(
          epoch, ckpt, best_psnr, best_epoch, best_ckpt)

      new_name_best_ckpt = join(self.train_dir, "checkpoints", "best_model.ckpt.pth")
      shutil.copyfile(best_ckpt, new_name_best_ckpt)
      self.message.add('Best epoch', best_epoch, format='.0f')
      self.message.add('Best PSNR', best_psnr, format='.5f')
      logging.info(self.message.get_message())
      logging.info("Done with batched inference.")

    else:

      # load best checkpoint
      self.load_state()

      if not self.eval_by_patch:
        final_image = self.inference_full()
      else:
        final_image = self.inference_by_patch()
      self.save_output(final_image)

  def eval_wavelets(self):

    # load model
    Model = models_config[self.config.archi.model]
    self.model = Model(self.config)
    self.model = nn.DataParallel(self.model)
    self.model = self.model.cuda()
    self.model.eval()

    if self.burst_name == 'test':

      best_ckpt, best_epoch, best_psnr, epoch, ckpt = None, 0., 0., 0., None
      best_psnr, best_epoch, best_ckpt = self.eval_loop(
        epoch, ckpt, best_psnr, best_epoch, best_ckpt)

      self.message.add('Best epoch', best_epoch, format='.0f')
      self.message.add('PSNR score', best_psnr, format='.5f')
      logging.info(self.message.get_message())
      logging.info("Done with batched inference.")
 
    else:

      original_burst_size = self.burst.shape
      factor = 1 << 6
      if not self.eval_by_patch:
        # resize image to fit wavelets transform
        if (self.burst.shape[-2] % factor != 0 or self.burst.shape[-1] % factor != 0): 
          expand_x = factor * int(np.ceil(self.burst.shape[-2] / factor)) - self.burst.shape[-2]
          expand_y = factor * int(np.ceil(self.burst.shape[-1] / factor)) - self.burst.shape[-1]
          padding = (expand_y // 2, expand_y - expand_y // 2, expand_x // 2, expand_x -  expand_x // 2)
          self.burst = F.pad(self.burst, padding, mode='replicate')
        assert self.burst.shape[-2] % factor == 0 and self.burst.shape[-1] % factor == 0
        final_image = self.inference_full()
      else:
        assert self.patch_size % factor == 0, f'patch_size needs to be a factor of {factor}'
        # add half block size padding because block edge are dumped
        padding = (self.patch_size // 2, self.patch_size // 2, self.patch_size // 2, self.patch_size // 2)
        self.burst = F.pad(self.burst, padding, mode='replicate')
        final_image = self.inference_by_patch()

      r_x = (final_image.shape[-2] - original_burst_size[-2]) // 2
      r_y = (final_image.shape[-1] - original_burst_size[-1]) // 2
      if r_x > 0 and r_y > 0:
        final_image = final_image[..., r_x:-r_x, r_y:-r_y]
      self.save_output(final_image)

  @record
  def __call__(self):
    """Performs training and eval
    """
    # Setup logging & print infos
    utils.setup_logging(self.config.project, 0)
    self.message = utils.MessageBuilder()
    pp = pprint.PrettyPrinter(indent=2, compact=True)
    logging.info(pp.pformat(self.config.to_dict()))
    logging.info(f"PyTorch version: {torch.__version__}.")
    logging.info(f"NCCL Version {torch.cuda.nccl.version()}")
    logging.info(f"Hostname: {socket.gethostname()}.")

    self.train_dir = self.config.project.train_dir
    self.ngpus = self.config.cluster.ngpus
    cudnn.benchmark = True

    self.data_dir = self.config.eval.eval_data_dir
    self.prediction_folder = self.config.eval.prediction_folder
    self.burst_name = self.config.eval.burst_name
    self.batch_size = self.config.eval.eval_batch_size
    self.align = self.config.eval.align
    self.patch_size = self.config.eval.patch_size
    self.eval_by_patch = self.config.eval.eval_by_patch

    if self.burst_name != 'test':
      burst_size = self.config.data.burst_size
      image_type = self.config.eval.image_type
      scaling = self.config.eval.image_scaling
      path = join(self.data_dir, self.burst_name)
      logging.info(f'Reading images: {self.burst_name}')
      burst_obj = Burst(path, burst_size, image_type, scaling)
      self.burst, meta_info_burst = burst_obj.get_burst()
      logging.info(f'Burst shape: {self.burst.shape}')

      if self.align:
        logging.info(f'Aligning Images ...')
        align_images = AlignImages()
        self.burst = align_images(self.burst)

    if self.config.archi.model in ['wavelets', 'laplacian']:
      final_image = self.eval_wavelets()
    else:
      final_image = self.eval_model()

