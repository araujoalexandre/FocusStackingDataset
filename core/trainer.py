import os
import sys
import time
import random
import datetime
import pprint
import socket
import logging
import glob
import submitit
from os.path import join, exists

from core import utils
from core.models import models_config
from core.datasets.readers import readers_config

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import nn
from torch.nn import SyncBatchNorm
from torch.optim import lr_scheduler
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.elastic.multiprocessing.errors import record
from torch.cuda.amp import autocast, GradScaler



class Trainer:
  """A Trainer to train a PyTorch."""

  def __init__(self, config):
    self.config = config

  def _load_state(self):
    # load last checkpoint
    checkpoints = glob.glob(join(self.train_dir, "checkpoints", "model.ckpt-*.pth"))
    get_model_id = lambda x: int(x.strip('.pth').strip('model.ckpt-'))
    checkpoints = sorted(
      [ckpt.split('/')[-1] for ckpt in checkpoints], key=get_model_id)
    path_last_ckpt = join(self.train_dir, "checkpoints", checkpoints[-1])
    self.checkpoint = torch.load(path_last_ckpt)
    self.model.load_state_dict(self.checkpoint['model_state_dict'])
    self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
    self.scheduler.load_state_dict(self.checkpoint['scheduler'])
    self.saved_ckpts.add(self.checkpoint['epoch'])
    epoch = self.checkpoint['epoch']
    if self.local_rank == 0:
      logging.info('Loading checkpoint {}'.format(checkpoints[-1]))

  def _save_ckpt(self, step, epoch, final=False, best=False):
    """Save ckpt in train directory."""
    freq_ckpt_epochs = self.config.training.save_checkpoint_epochs
    if (epoch % freq_ckpt_epochs == 0 and self.is_master \
        and epoch not in self.saved_ckpts) \
         or (final and self.is_master) or best:
      prefix = "model" if not best else "best_model"
      ckpt_name = f"{prefix}.ckpt-{step}.pth"
      ckpt_path = join(self.train_dir, 'checkpoints', ckpt_name)
      if exists(ckpt_path) and not best: return 
      self.saved_ckpts.add(epoch)
      state = {
        'epoch': epoch,
        'global_step': step,
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'scheduler': self.scheduler.state_dict()
      }
      logging.debug("Saving checkpoint '{}'.".format(ckpt_name))
      torch.save(state, ckpt_path)


  @record
  def __call__(self):
    """Performs training and evaluation
    """
    cudnn.benchmark = True

    self.start_new_model = self.config.training.start_new_model
    self.train_dir = self.config.project.train_dir
    self.ngpus = self.config.cluster.ngpus

    job_env = submitit.JobEnvironment()
    self.rank = int(job_env.global_rank)
    self.local_rank = int(job_env.local_rank)
    self.num_nodes = int(job_env.num_nodes) 
    self.num_tasks = int(job_env.num_tasks)
    self.is_master = bool(self.rank == 0)

    # Setup logging
    utils.setup_logging(self.config.project, self.rank)

    self.message = utils.MessageBuilder()
    # print self.config parameters
    if self.start_new_model and self.local_rank == 0:
      logging.info(self.config.cmd)
      pp = pprint.PrettyPrinter(indent=2, compact=True)
      logging.info(pp.pformat(self.config.to_dict()))
    # print infos
    if self.local_rank == 0:
      logging.info(f"PyTorch version: {torch.__version__}.")
      logging.info(f"NCCL Version {torch.cuda.nccl.version()}")
      logging.info(f"Hostname: {socket.gethostname()}.")

    # ditributed settings
    self.world_size = 1
    self.is_distributed = False
    if self.num_nodes > 1 or self.num_tasks > 1:
      self.is_distributed = True
      self.world_size = self.num_nodes * self.ngpus
    if self.num_nodes > 1:
      logging.info(
        f"Distributed Training on {self.num_nodes} nodes")
    elif self.num_nodes == 1 and self.num_tasks > 1:
      logging.info(f"Single node Distributed Training with {self.num_tasks} tasks")
    else:
      assert self.num_nodes == 1 and self.num_tasks == 1
      logging.info("Single node training.")

    if not self.is_distributed:
      self.batch_size = self.config.training.batch_size * self.ngpus
    else:
      self.batch_size = self.config.training.batch_size

    self.global_batch_size = self.batch_size * self.world_size
    logging.info('World Size={} => Total batch size {}'.format(
      self.world_size, self.global_batch_size))

    torch.cuda.set_device(self.local_rank)

    # load dataset
    Reader = readers_config[self.config.project.dataset]
    self.reader = Reader(self.config, self.batch_size, self.is_distributed, is_training=True)
    if self.local_rank == 0:
      logging.info(f"Using dataset: {self.config.project.dataset}")

    # load model
    Model = models_config[self.config.archi.model]
    self.model = Model(self.config)
    self.model = SyncBatchNorm.convert_sync_batchnorm(self.model)
    self.model = self.model.cuda()
    nb_parameters = np.sum([p.numel() for p in self.model.parameters() if p.requires_grad])
    logging.info(f'Number of parameters to train: {nb_parameters}')

    # setup distributed process if training is distributed 
    # and use DistributedDataParallel for distributed training
    if self.is_distributed:
      utils.setup_distributed_training(self.world_size, self.rank)
      self.model = DistributedDataParallel(
        self.model, device_ids=[self.local_rank], output_device=self.local_rank)
      if self.local_rank == 0:
        logging.info('Model defined with DistributedDataParallel')
    else:
      self.model = nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))

    # define set for saved ckpt
    self.saved_ckpts = set([0])

    data_loader, sampler = self.reader.load_dataset()
    if sampler is not None:
      assert sampler.num_replicas == self.world_size

    if self.is_distributed:
      n_files = sampler.num_samples
    else:
      n_files = self.reader.n_train_files

    # define optimizer
    self.optimizer = utils.get_optimizer(
      self.config.training, self.model.parameters())

    # Gradient Scaler for Mix Precision Training
    self.scaler = GradScaler()

    # define the loss
    self.criterion = utils.L1Loss(boundary_ignore=4)
    self.psnr = utils.PSNR(boundary_ignore=4)

    # define learning rate scheduler
    milestones = list(map(int, self.config.training.decay.split('-')))
    self.scheduler = lr_scheduler.MultiStepLR(
      self.optimizer, milestones=milestones, gamma=self.config.training.gamma)

    # if start_new_model is False, we restart training
    if not self.start_new_model:
      if self.local_rank == 0:
        logging.info('Restarting training...')
      self._load_state()

    # if start_new_model is True, global_step = 0
    # else we get global step from checkpoint
    if self.start_new_model:
      start_epoch, global_step = 0, 0
    else:
      start_epoch = self.checkpoint['epoch']
      global_step = self.checkpoint['global_step']

    if self.local_rank == 0:
      logging.info("Number of files on worker: {}".format(n_files))
      logging.info("Start training")

    # training loop
    self.best_checkpoint = None
    self.best_accuracy = None
    for epoch_id in range(start_epoch, self.config.training.epochs):
      if self.is_distributed:
        sampler.set_epoch(epoch_id)
      for n_batch, data in enumerate(data_loader):
        if global_step == 2 and self.is_master:
          start_time = time.time()
        epoch = (int(global_step) * self.global_batch_size) / self.reader.n_train_files
        self.one_step_training(data, epoch_id, epoch, global_step)
        self._save_ckpt(global_step, epoch_id)
        if global_step == 20 and self.is_master:
          self._print_approximated_train_time(start_time)
        global_step += 1
      self.scheduler.step()

    self._save_ckpt(global_step, epoch_id, final=True)
    logging.info("Done training -- epoch limit reached.")

  def _print_approximated_train_time(self, start_time):
    total_steps = self.reader.n_train_files * self.config.training.epochs / self.global_batch_size
    total_seconds = total_steps * ((time.time() - start_time) / 18)
    n_days = total_seconds // 86400
    n_hours = (total_seconds % 86400) / 3600
    logging.info(
      'Approximated training time: {:.0f} days and {:.1f} hours'.format(
        n_days, n_hours))

  def _to_print(self, step):
    frequency = self.config.training.frequency_log_steps
    if frequency is None:
      return False
    return (step % frequency == 0 and self.local_rank == 0) or \
        (step == 1 and self.local_rank == 0)

  def one_step_training(self, data, epoch_id, epoch, step):

    self.optimizer.zero_grad()

    batch_start_time = time.time()
    burst, frame_gt = data
    burst, frame_gt = burst.cuda(), frame_gt.cuda()

    if step == 0 and self.local_rank == 0:
      logging.info(f'burst {burst.shape}')
      logging.info(f'ground truth {frame_gt.shape}')

    burst = burst - 0.5
    frame_gt = frame_gt - 0.5

    if self.config.project.autocast:
      burst = burst.half()
      frame_gt = frame_gt.half()

    with autocast(enabled=self.config.project.autocast):
      outputs = self.model(burst)
      if step == 0 and self.local_rank == 0:
        logging.info(f'outputs {outputs.shape}')

      outputs = outputs.squeeze(1)
      frame_gt = frame_gt.squeeze(1)

      loss = self.criterion(outputs, frame_gt)
      psnr = self.psnr(outputs, frame_gt)

    if self.config.project.autocast:
      self.scaler.scale(loss).backward()
      self.scaler.step(self.optimizer)
      self.scaler.update()
    else:
      loss.backward()
      self.optimizer.step()

    seconds_per_batch = time.time() - batch_start_time
    examples_per_second = self.batch_size / seconds_per_batch
    examples_per_second *= self.world_size

    if self._to_print(step):
      lr = self.optimizer.param_groups[0]['lr']
      self.message.add("epoch", epoch, format="4.2f")
      self.message.add("step", step, width=5, format=".0f")
      self.message.add("lr", lr, format=".6f")
      self.message.add("L1", loss, format=".4f")
      self.message.add("PSNR", psnr, format=".4f")
      self.message.add("imgs/sec", examples_per_second, width=5, format=".0f")
      logging.info(self.message.get_message())

