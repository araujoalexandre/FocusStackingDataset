import os
import sys
import glob
import natsort
import cv2
from PIL import Image
from os.path import join, exists, basename

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import lpips
from pytorch_msssim import ssim

lpips_metric = lpips.LPIPS(net='alex', model_path='/gpfswork/rech/yxj/uuc79vj/lpips_ckpts/alexnet-owt-7be5be79.pth')

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


def compute_psnr(preds, target):
  psnr_metric = PSNR()
  preds, target = torch.FloatTensor(preds), torch.FloatTensor(target)
  preds = preds[None].permute(0, 3, 1, 2)
  target = target[None].permute(0, 3, 1, 2)
  preds = preds / 255
  target = target / 255
  return psnr_metric(preds, target).item()

def compute_ssim(preds, target):
  preds, target = torch.FloatTensor(preds), torch.FloatTensor(target)
  preds = preds[None].permute(0, 3, 1, 2)
  target = target[None].permute(0, 3, 1, 2)
  return ssim(preds, target).item()

def compute_lpips(preds, target):
  preds, target = torch.FloatTensor(preds), torch.FloatTensor(target)
  preds = preds[None].permute(0, 3, 1, 2)
  target = target[None].permute(0, 3, 1, 2)
  preds = preds / 255
  target = target / 255
  preds = (preds - 0.5) * 2
  target = (target - 0.5) * 2
  return lpips_metric(preds, target).item()

def compute_metrics(folder):

  preds_path = '/gpfswork/rech/yxj/uuc79vj/focus_stack/imsave'
  target_path = '/gpfsscratch/rech/yxj/uuc79vj/data/focus_stack_dataset/dataset'
  for split in ['train', 'test']:
    for lens in ['lumix_lens', 'olympus_macro_lens']:
      image_paths = glob.glob(join(preds_path, folder, 'focus_stack_dataset', split, lens, '**.jpg'))

      print(split, lens, len(image_paths))

      running_psnr = 0
      running_ssim = 0
      running_lpips = 0
      running_images = 0
      for img_path in image_paths:
        image_name = basename(img_path).split('prediction_')[1].split('_patch')[0]
        helicon_focus_path = join(target_path, split, lens, image_name, 'helicon_focus_aligned_tiff.jpg')

        crop_target = 9
        preds = np.array(Image.open(img_path))
        target = np.array(Image.open(helicon_focus_path))
        # target = target[crop_target:-crop_target, crop_target:-crop_target]

        psrn_score = compute_psnr(preds, target)
        ssim_score = compute_ssim(preds, target)
        lpips_score = compute_lpips(preds, target)

        running_psnr += psrn_score
        running_ssim += ssim_score
        running_lpips += lpips_score
        running_images += 1

        # print(lpips_score)

      mean_psnr = running_psnr / running_images
      mean_ssim = running_ssim / running_images
      mean_lpips = running_lpips / running_images
      print(f'{split}, {lens}, {running_images}: psnr:{mean_psnr:.4f}, ssim:{mean_ssim:.4f}, lpips:{mean_lpips:.4f}')


if __name__ == '__main__':
  folder = sys.argv[1]
  print(f'folder: {folder}')
  compute_metrics(folder)


