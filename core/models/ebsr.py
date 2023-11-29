import logging
import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast


class EBSR(nn.Module):

  def __init__(self, config):
    super(EBSR, self).__init__()
    self.config = config

    # hack: we need to load the layers here in order to be able to use submitit
    # without gpu on the front machine
    from core.models.layers import make_layer, WideActResBlock, LRSCWideActResGroup, \
        PCD_Align, CrossNonLocal_Fusion

    nf = self.config.archi.n_feats
    n_resblocks = self.config.archi.n_resblocks
    nframes = self.config.data.burst_size
    front_RBs = 5
    back_RBs = self.config.archi.n_resgroups # 20
    groups = 8
    n_colors = self.config.archi.n_colors

    wn = torch.nn.utils.weight_norm
    # wn = torch.nn.Identity
    self.center = 0
    self.lrcn = self.config.archi.lrcn
    self.fusion_type = self.config.archi.fusion

    WARB = functools.partial(WideActResBlock, nf=nf)
    if self.lrcn:
      LRCN = functools.partial(LRSCWideActResGroup, n_resblocks=n_resblocks, nf=nf)

    #### extract features (for each frame)
    self.conv_first = wn(nn.Conv2d(n_colors, nf, 3, 1, 1, bias=True))
    self.feature_extraction = make_layer(WARB, front_RBs)
    self.fea_L2_conv1 = wn(nn.Conv2d(nf, nf*2, 3, 2, 1, bias=True))
    self.fea_L3_conv1 = wn(nn.Conv2d(nf*2, nf*4, 3, 2, 1, bias=True))

    ############### Feature Enhanced PCD Align #####################
    # Top layers
    self.toplayer = wn(nn.Conv2d(nf*4, nf, kernel_size=1, stride=1, padding=0))
    # Smooth layers
    self.smooth1 = wn(nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1))
    self.smooth2 = wn(nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1))
    # Lateral layers
    self.latlayer1 = wn(nn.Conv2d(nf*2, nf, kernel_size=1, stride=1, padding=0))
    self.latlayer2 = wn(nn.Conv2d(nf*1, nf, kernel_size=1, stride=1, padding=0))

    self.pcd_align = PCD_Align(nf=nf, groups=groups, wn=wn)
    #################################################################

    if self.fusion_type == 'conv':
      self.fusion = wn(nn.Conv2d(nframes * nf, nf, 1, 1, bias=True))
    elif self.fusion_type == 'non_local':
      self.fusion = CrossNonLocal_Fusion(nf=nf, nframes=nframes, center=self.center, wn=wn)

    #### reconstruction
    if not self.lrcn:
      self.recon_trunk = make_layer(WARB, back_RBs)
    else:
      self.recon_trunk = nn.Sequential(
          make_layer(LRCN, back_RBs, idx=True),
          wn(nn.Conv2d(nf*(back_RBs+1), nf, 1)))

    #### upsampling
    downsample_factor = self.config.data.downsample_factor ** 2
    self.upconv1 = wn(nn.Conv2d(nf, nf * downsample_factor, 3, 1, 1, bias=True))
    self.upconv2 = wn(nn.Conv2d(nf, 64, 3, 1, 1, bias=True))
    self.pixel_shuffle = nn.PixelShuffle(int(np.sqrt(downsample_factor)))
    self.HRconv = wn(nn.Conv2d(64, 64, 3, 1, 1, bias=True))
    self.conv_last = wn(nn.Conv2d(64, 3, 3, 1, 1, bias=True))

    #### skip #############
    self.skip_pixel_shuffle = nn.PixelShuffle(int(np.sqrt(downsample_factor)))
    self.skipup1 = wn(nn.Conv2d(n_colors, nf * downsample_factor, 3, 1, 1, bias=True))
    self.skipup2 = wn(nn.Conv2d(nf, 3, 3, 1, 1, bias=True))

    #### activation function
    self.relu = nn.ReLU(inplace=False)

  def _upsample_add(self, x, y):
      return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) + y

  def forward(self, x):

    with autocast(enabled=self.config.project.autocast):

      B, N, C, H, W = x.size()  # N video frames
      x_center = x[:, self.center, :, :, :].contiguous()

      #### skip module ########
      skip1 = self.relu(self.skip_pixel_shuffle(self.skipup1(x_center)))
      # skip2 = self.skip_pixel_shuffle(self.skipup2(skip1))
      skip2 = self.skipup2(skip1)

      #### extract LR features
      L1_fea = self.relu(self.conv_first(x.view(-1, C, H, W)))
      L1_fea = self.feature_extraction(L1_fea)
      L2_fea = self.relu(self.fea_L2_conv1(L1_fea))
      L3_fea = self.relu(self.fea_L3_conv1(L2_fea))

      # FPN enhance features
      L3_fea = self.relu(self.toplayer(L3_fea))
      L2_fea = self.smooth1(self._upsample_add(L3_fea, self.latlayer1(L2_fea)))
      L1_fea = self.smooth2(self._upsample_add(L2_fea, self.latlayer2(L1_fea)))

      L1_fea = L1_fea.view(B, N, -1, H, W)
      L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2)
      L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4)

      #### PCD align
      # ref feature list
      aligned_fea = []
      for i in range(N):
        i1 = np.maximum(i-1, 0)
        ref_fea_l = [
          L1_fea[:, i1, :, :, :].clone(),
          L2_fea[:, i1, :, :, :].clone(),
          L3_fea[:, i1, :, :, :].clone(),
        ]
        nbr_fea_l = [
          L1_fea[:, i, :, :, :].clone(),
          L2_fea[:, i, :, :, :].clone(),
          L3_fea[:, i, :, :, :].clone(),
        ]
        aligned_fea.append(self.pcd_align(nbr_fea_l, ref_fea_l))
      aligned_fea = torch.stack(aligned_fea, dim=1)  # [B, N, C, H, W] --> [B, T, C, H, W]

      if self.fusion_type in ['conv', 'multi_resolution']:
        aligned_fea = aligned_fea.view(B, -1, H, W)

      fea = self.relu(self.fusion(aligned_fea))

      out = self.recon_trunk(fea)
      out = self.relu(self.pixel_shuffle(self.upconv1(out)))
      out = skip1 + out
      # out = self.relu(self.pixel_shuffle(self.upconv2(out)))
      out = self.relu(self.upconv2(out))
      out = self.relu(self.HRconv(out))
      out = self.conv_last(out)

      out = skip2 + out
      return out
