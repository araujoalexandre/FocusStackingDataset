
import os
import sys
import re
import shutil
import json
import logging
import glob
import copy
import subprocess
import pickle
import natsort
import cv2
from os.path import join, exists

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist



def numpy_to_torch(a: np.ndarray):
  return torch.from_numpy(a).float().permute(2, 0, 1)

def torch_to_numpy(a: torch.Tensor):
  return a.permute(1, 2, 0).cpu().numpy()

def torch_to_npimage(a: torch.Tensor, unnormalize=True):
  a_np = torch_to_numpy(a)
  if unnormalize:
    a_np = a_np * 255
  a_np = a_np.astype(np.uint8)
  return cv2.cvtColor(a_np, cv.COLOR_RGB2BGR)

def npimage_to_torch(a, normalize=True, input_bgr=True):
  if input_bgr:
    a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
  a_t = numpy_to_torch(a)
  if normalize:
    a_t = a_t / 255.0
  return a_t


def flatten_raw_image(x):

  def get_empty_array(x, shape):
    if isinstance(x, np.ndarray):
      x_out = np.zeros(shape, dtype=x.dtype)
    elif isinstance(x, torch.Tensor):
      x_out = torch.zeros(shape, dtype=x.dtype)
      x_out = x_out.to(x.device)
    else:
      raise ValueError(f'x should be numpy of torch array')
    return x_out

  if len(x.shape) == 3 and x.shape[0] == 4:
    x_out = get_empty_array(x, (x.shape[1] * 2, x.shape[2] * 2))
    x_out[0::2, 0::2] = x[0, :, :]
    x_out[0::2, 1::2] = x[1, :, :]
    x_out[1::2, 0::2] = x[2, :, :]
    x_out[1::2, 1::2] = x[3, :, :]
    return im_out
  elif len(x.shape) == 4 and x.shape[1] == 4: # batch
    x_out = get_empty_array(x, (x.shape[0], 1, x.shape[2] * 2, x.shape[3] * 2))
    x_out[:, 0, 0::2, 0::2] = x[:, 0, :, :]
    x_out[:, 0, 0::2, 1::2] = x[:, 1, :, :]
    x_out[:, 0, 1::2, 0::2] = x[:, 2, :, :]
    x_out[:, 0, 1::2, 1::2] = x[:, 3, :, :]
  else:
    raise ValueError(f'x.shape not recognized: {x.shape}')
  return x_out



def pickle_load(path):
  """ Function to load pickle object """
  with open(path, 'rb') as f:
    return pickle.load(f, encoding='latin1')

def pickle_dump(file, path):
  """ Function to dump pickle object """
  with open(path, 'wb') as f:
    pickle.dump(file, f, -1)


class Config:

  def __init__(self, cluster, project, data, training,
               eval, archi, *args, **kwargs):
    self.cluster = cluster
    self.project = project
    self.data = data
    self.training = training
    self.eval = eval
    self.archi = archi

  def load(self):
    path = f'{self.project.train_dir}/config.pkl'
    if exists(path):
      mode = self.project.mode
      config = pickle_load(path)
      self.archi = config.archi
      self.project = config.project
      # get the mode from the command line
      self.project.mode = mode
  
  def save(self):
    path = f'{self.project.train_dir}/config.pkl'
    if not exists(path):
      pickle_dump(self, path) 

  def to_dict(self):
    return {
      'cluster': vars(self.cluster),
      'project': vars(self.project),
      'data': vars(self.data),
      'training': vars(self.training),
      'eval': vars(self.eval),
      'archi': vars(self.archi)
    }


def setup_distributed_training(world_size, rank):
  """ find a common host name on all nodes and setup distributed training """
  # make sure http proxy are unset, in order for the nodes to communicate
  for var in ['http_proxy', 'https_proxy']:
    if var in os.environ:
      del os.environ[var]
    if var.upper() in os.environ:
      del os.environ[var.upper()]
  # get distributed url 
  cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
  stdout = subprocess.check_output(cmd.split())
  host_name = stdout.decode().splitlines()[0]
  dist_url = f'tcp://{host_name}:9000'
  # setup dist.init_process_group
  dist.init_process_group(backend='nccl', init_method=dist_url,
    world_size=world_size, rank=rank)

def get_epochs_from_ckpt(filename):
  regex = "(?<=ckpt-)[0-9]+"
  return int(re.findall(regex, filename)[-1])

def get_list_checkpoints(train_dir):
  files = glob.glob(join(train_dir, 'checkpoints', 'model.ckpt-*.pth'))
  files = natsort.natsorted(files, key=get_epochs_from_ckpt)
  return [filename for filename in files]

def get_checkpoint(train_dir, last_global_step):
  files = get_list_checkpoints(train_dir)
  if not files:
    return None, None
  for filename in files:
    global_step = get_global_step_from_ckpt(filename)
    if last_global_step < global_step:
      return filename, global_step
  return None, None


class MessageBuilder:

  def __init__(self):
    self.msg = []

  def add(self, name, values, align=">", width=0, format=None):
    if name:
      metric_str = "{}: ".format(name)
    else:
      metric_str = ""
    values_str = []
    if type(values) != list:
      values = [values]
    for value in values:
      if format:
        values_str.append("{value:{align}{width}{format}}".format(
          value=value, align=align, width=width, format=format))
      else:
        values_str.append("{value:{align}{width}}".format(
          value=value, align=align, width=width))
    metric_str += '/'.join(values_str)
    self.msg.append(metric_str)

  def get_message(self):
    message = " | ".join(self.msg)
    self.clear()
    return message

  def clear(self):
    self.msg = []


def setup_logging(params, rank):
  level = {'DEBUG': 10, 'ERROR': 40, 'FATAL': 50,
    'INFO': 20, 'WARN': 30
  }[params.logging_verbosity]
  format_ = "[%(asctime)s %(filename)s:%(lineno)s] %(message)s"
  filename = '{}/log_{}_{}.logs'.format(params.train_dir, params.mode, rank)
  logging.basicConfig(filename=filename, level=level, format=format_, datefmt='%H:%M:%S')



class L2Loss(nn.Module):

  def __init__(self, boundary_ignore=None):
    super(L2Loss, self).__init__()
    self.boundary_ignore = boundary_ignore

  def forward(self, pred, gt, valid=None,**kwargs):
    if self.boundary_ignore is not None:
      pred = pred[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]
      gt = gt[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]
      if valid is not None:
        valid = valid[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]
    pred_m = pred
    gt_m = gt
    if valid is None:
      mse = F.mse_loss(pred_m, gt_m)
    else:
      mse = F.mse_loss(pred_m, gt_m, reduction='none')
      eps = 1e-12
      elem_ratio = mse.numel() / valid.numel()
      mse = (mse * valid.float()).sum() / (valid.float().sum()*elem_ratio + eps)
    return mse


class PSNR(nn.Module):

  def __init__(self, boundary_ignore=None, max_value=1.0):
    super(PSNR, self).__init__()
    self.l2 = L2Loss(boundary_ignore=boundary_ignore)
    self.max_value = max_value

  def psnr(self, pred, gt, valid=None):
    mse = self.l2(pred, gt, valid=valid)
    psnr = 20 * np.log10(self.max_value) - 10.0 * mse.log10()
    return psnr

  def processed(self, image):
    if image.dim() == 3:
      image = image[None]
    if image.shape[-1] == 3:
      image = image.permute(0, 3, 1, 2)
    if image.max() > 2 and self.max_value == 1:
      image = image / 255.
    return image

  def forward(self, pred, gt, valid=None, **kwargs):
    pred, gt = self.processed(pred), self.processed(gt)
    assert pred.dim() == 4 and pred.shape == gt.shape, f'{pred.shape}, {gt.shape}'
    if valid is None:
      psnr_all = [self.psnr(p.unsqueeze(0), g.unsqueeze(0)) for p, g in zip(pred, gt)]
    else:
      psnr_all = [self.psnr(p.unsqueeze(0), g.unsqueeze(0), v.unsqueeze(0)) for p, g, v in zip(pred, gt, valid)]
    psnr = sum(psnr_all) / len(psnr_all)
    return psnr


class L1Loss(nn.Module):

  def __init__(self, boundary_ignore=None):
    super(L1Loss, self).__init__()
    self.boundary_ignore = boundary_ignore

  def forward(self, pred, gt, **kwargs):
    if self.boundary_ignore is not None:
      pred = pred[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]
      gt = gt[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]
    pred_m = pred
    gt_m = gt
    l1_loss = F.l1_loss(pred_m, gt_m)
    return l1_loss



def get_scheduler(optimizer, lr_scheduler, num_steps):
  """Return a learning rate scheduler schedulers."""
  if lr_scheduler == 'CosineAnnealingLR':
    return torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer, T_max=num_steps)
  else:
    raise ValueError("scheduler was not recognized")
  return scheduler

def get_optimizer(config, model_parameters):
  """Returns the optimizer that should be used based on params."""
  optimizer = config.optimizer
  init_lr = config.lr
  weight_decay = config.wd
  if optimizer == 'sgd':
    opt = torch.optim.SGD(
      model_parameters, lr=init_lr, weight_decay=weight_decay,
      momentum=0.9, nesterov=True)
  elif optimizer == 'rmsprop':
    opt = torch.optim.RMSprop(
      model_parameters, lr=init_lr, weight_decay=weight_decay)
  elif optimizer == 'adam':
    opt = torch.optim.Adam(
      model_parameters, lr=init_lr, weight_decay=weight_decay)
  else:
    raise ValueError("Optimizer was not recognized")
  return opt

