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
from os.path import join, basename, exists

from core import utils
from core.models import models_config
from core.datasets.readers import readers_config
from core.inference.iter_patches import IterPatchesImage
from core.inference.align_key_points import align_burst
from core.data_utils import Burst
from core.datasets.readers import get_data_dir
import core.datasets.camera_pipeline as camera

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import nn
from torch.distributed.elastic.multiprocessing.errors import record

classic_models = ['wavelets', 'laplacian']


class Evaluator:

  def __init__(self, config, submitit=True):
    self.config = config
    self.submitit = submitit

  def load_state(self, ckpt_path=None):
    if ckpt_path is None:
      ckpt_path = join(
        self.train_dir, "checkpoints", "best_model.ckpt.pth")
      if not exists(ckpt_path):
        checkpoints = glob.glob(join(self.train_dir, "checkpoints", "model.ckpt-*.pth"))
        get_model_id = lambda x: int(x.strip('.pth').strip('model.ckpt-'))
        ckpt_name = sorted([ckpt.split('/')[-1] for ckpt in checkpoints], key=get_model_id)[-1]
        ckpt_path = join(self.train_dir, "checkpoints", ckpt_name)
    checkpoint = torch.load(ckpt_path)
    model_state = checkpoint['model_state_dict']
    self.model.load_state_dict(model_state)

  def preprocess(self, x):
    if self.model_name not in classic_models:
      return x - 0.5
    return x

  def postprocess(self, output):
    if self.model_name not in classic_models:
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

      if n_batch == 0:
        logging.info(f'burst: {burst.shape}')
        logging.info(f'frame_gt: {frame_gt.shape}')

      with torch.no_grad():
        burst = self.preprocess(burst)
        outputs = self.model(burst)
        outputs = self.postprocess(outputs)
        if n_batch == 0:
          logging.info(f'outputs: {outputs.shape}')

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
    generator_patches = IterPatchesImage(
      self.burst, self.meta_info_burst, self.config, optical_flows=True)
    logging.info('Block inference ...')
    for i, x in enumerate(generator_patches):
      if i == 0:
        logging.info(f'x: {x.shape}')
      x = x.cuda()
      x = self.preprocess(x)
      if self.config.project.autocast:
        x = x.half()
      with torch.no_grad():
        output = self.model(x)
      if i == 0:
        logging.info(f'output: {output.shape}')
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
    patch_size = f'{self.config.eval.patch_size}'
    name = f'prediction_{self.burst_name}_{patch}_{image_type}_{patch_size}.jpg'
    prediction_path = join(self.prediction_folder, name)
    logging.info(f'prediction_path: {prediction_path}')
    Image.fromarray(final_image).save(prediction_path, quality=100)

  def eval_model(self):
    # load best checkpoint
    original_burst_size = np.array(self.burst.shape)
    original_burst_size = original_burst_size * self.config.data.downsample_factor
    self.load_state()
    if not self.eval_by_patch:
      final_image = self.inference_full()
    else:
      if self.config.eval.image_type == 'raw':
        p = self.patch_size // 4
      else:
        p = self.patch_size // 2
      padding = (p, p, p, p)
      self.burst = F.pad(self.burst, padding, mode='replicate')
      logging.info(f'self.burst before inference: {self.burst.shape}')
      final_image = self.inference_by_patch()

    logging.info(f'final_image: {final_image.shape}')
    if self.config.eval.image_type == 'raw':
      r_x = (final_image.shape[-2] - original_burst_size[-2]*2) // 2
      r_y = (final_image.shape[-1] - original_burst_size[-1]*2) // 2
    else:
      r_x = (final_image.shape[-2] - original_burst_size[-2]) // 2
      r_y = (final_image.shape[-1] - original_burst_size[-1]) // 2
    if r_x > 0 and r_y > 0:
      final_image = final_image[..., r_x:-r_x, r_y:-r_y]
    logging.info(f'final_image: {final_image.shape}')
    self.save_output(final_image)

  def eval_classic_models(self):

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
      # assert self.patch_size % factor == 0, f'patch_size needs to be a factor of {factor}'
      # add half block size padding because block edge are dumped
      logging.info(f'self.burst: {self.burst.shape}')
      padding = (self.patch_size // 2, self.patch_size // 2, self.patch_size // 2, self.patch_size // 2)
      self.burst = F.pad(self.burst, padding, mode='replicate')
      logging.info(f'self.burst: {self.burst.shape}')
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

    self.model_name = self.config.archi.model
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
      logging.info(f'Reading images: {self.burst_name}')
      burst_obj = Burst(self.data_dir, self.burst_name, burst_size, image_type, scaling)
      self.burst, self.meta_info_burst = burst_obj.get_burst()
      logging.info(f'Burst shape: {self.burst.shape}')

      # if config.eval.noise:
      #   logging.info('ADDING NOISE')
      #   shot_noise_level, read_noise_level = camera.random_noise_levels(config.eval.noise)
      #   self.burst = camera.add_noise(self.burst, shot_noise_level, read_noise_level)

      if self.align:
        logging.info(f'Aligning Images ...')
        # align_images = AlignImages(self.config)
        # self.burst = align_images(self.burst)
        self.burst = align_burst(self.burst, self.config)
      logging.info(f"Aligning Images ok... {self.burst.shape}")

    # load model
    Model = models_config[self.model_name]
    self.model = Model(self.config)
    self.model = nn.DataParallel(self.model)
    self.model = self.model.cuda()
    self.model.eval()

    if self.model_name in classic_models and basename(self.train_dir) in classic_models:
      final_image = self.eval_classic_models()
    else:
      final_image = self.eval_model()

