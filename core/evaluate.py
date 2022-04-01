import os
import time
import random
import datetime
import pprint
import socket
import logging
import glob
import subprocess
import cv2
from os.path import join, basename

from core import utils
from core.models.models import FusionLinearModel, FusionNonLinearModel
from core.models.wavelets import FusionWavelets
from core.inference.iter_patches import IterPatchesImage
from core.inference.flow import calcFlow
from core.inference.align_key_points import AlignImages
from core.data_utils import Burst, postprocess_raw

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

  def load_state(self):
    # load last checkpoint
    checkpoints = glob.glob(join(self.train_dir, "checkpoints", "model.ckpt-*.pth"))
    get_model_id = lambda x: int(x.strip('.pth').strip('model.ckpt-'))
    checkpoints = sorted(
      [ckpt.split('/')[-1] for ckpt in checkpoints], key=get_model_id)
    checkpoint = checkpoints[-1]
    path_ckpt = join(self.train_dir, "checkpoints", checkpoint)
    self.checkpoint = torch.load(path_ckpt)
    self.model.load_state_dict(self.checkpoint['model_state_dict'])
    logging.info('Loading checkpoint {}'.format(checkpoint))

  @record
  def __call__(self):
    """Performs training and eval
    """
    self.train_dir = self.config.project.train_dir
    self.ngpus = self.config.cluster.ngpus
    cudnn.benchmark = True

    self.batch_size = self.config.eval.eval_batch_size
    self.data_dir = self.config.eval.eval_data_dir
    self.prediction_folder = self.config.eval.prediction_folder
    self.burst_name = self.config.eval.burst_name
    self.burst_size = self.config.data.burst_size
    self.crop_sz = self.config.eval.crop // 4 * 4 # force even crop factor to keep the correct rggb pattern
    self.window = self.config.eval.window
    self.superres_factor = self.config.data.downsample_factor

    # params optical flow
    self.of_scale = self.config.eval.of_scale
    self.of_win = self.config.eval.of_win

    stride = self.config.eval.stride
    if self.config.eval.stride is None:
      stride = self.window
    assert stride <= self.window

    # Setup logging & print infos
    utils.setup_logging(self.config.project, 0)
    self.message = utils.MessageBuilder()
    pp = pprint.PrettyPrinter(indent=2, compact=True)
    logging.info(pp.pformat(self.config.to_dict()))
    logging.info(f"PyTorch version: {torch.__version__}.")
    logging.info(f"NCCL Version {torch.cuda.nccl.version()}")
    logging.info(f"Hostname: {socket.gethostname()}.")

    # load model
    Model = {
      'FusionLinearModel': FusionLinearModel,
      'FusionNonLinearModel': FusionNonLinearModel,
      'FusionWavelets': FusionWavelets,
    }[self.config.archi.model]
    self.model = Model(self.config)
    self.model = nn.DataParallel(self.model)
    if self.config.archi.model in ['FusionLinearModel', 'FusionNonLinearModel']:
      self.load_state()
    self.model = self.model.cuda()
    self.model.eval()

    path = join(self.data_dir, self.burst_name, 'raw')
    burst_obj = Burst(
      root=path, burst_size=self.burst_size, visible=True)
    logging.info(f'Reading images: {self.burst_name}')
    self.burst, meta_info_burst = burst_obj.get_burst()

    # convert raw burst to RGB burst
    self.burst = postprocess_raw(self.burst, meta_info_burst,
                          linearize=True, demosaic=True, wb=True, dwb=False,
                          ccm=True, gamma=True, tonemap=True, brightness=True)
    logging.info(f'Burst shape: {self.burst.shape}')

    # logging.info(f'Aligning Images ...')
    # align_images = AlignImages()
    # self.burst = align_images(self.burst)

    logging.info(f'Computing optical flow ...')
    self.flows, _ = calcFlow(
      self.burst, D=self.of_scale, poly_n=7, poly_sigma=1.5, interpolate=False, winsize=self.of_win)

    # define the patch generator
    generator_patches = IterPatchesImage(self.burst, self.flows, self.config,
      pad_mode='constant', center_pad=False)

    # block inference
    logging.info('Block inference ...')
    for batch_patches in generator_patches:

      with torch.no_grad():
        batch_patches = batch_patches.cuda()
        output = self.model(batch_patches)
        output = output.float()
        output = output.cpu()

      generator_patches.add_processed_patches(output) 

    logging.info('Agregating blocks ...')
    final_image = generator_patches.agregate_blocks()

    logging.info('Saving Image ...')
    final_image = final_image.clamp(0, 1)
    final_image = final_image[0]
    final_image = final_image.permute(1, 2, 0)
    final_image = final_image.mul(255.0)
    final_image = final_image.numpy().astype(np.uint8)
    logging.info(f"Final shape: {final_image.shape}")

    # save result
    name = f'prediction_{self.burst_name}.jpg'
    prediction_path = join(self.prediction_folder, name)
    Image.fromarray(final_image).save(prediction_path, quality=95)

