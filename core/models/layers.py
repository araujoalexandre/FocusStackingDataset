import logging
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from core.models.non_local.non_local_cross_dot_product import NONLocalBlock2D as NonLocalCross
from core.models.non_local.non_local_dot_product import NONLocalBlock2D as NonLocal

# from dcn_v2 import DCN_sep as DCN
from torchvision.ops import DeformConv2d



def make_layer(block, n_layers, idx=False):
  layers = []
  for i in range(n_layers):
    b = block() if not idx else block(idx=i)
    layers.append(b)
  return nn.Sequential(*layers)


class PyConv2d(nn.Module):
  """PyConv2d with padding (general case). Applies a 2D PyConv over an input signal composed of several input planes.
  Args:
    in_channels (int): Number of channels in the input image
    out_channels (list): Number of channels for each pyramid level produced by the convolution
    pyconv_kernels (list): Spatial size of the kernel for each pyramid level
    pyconv_groups (list): Number of blocked connections from input channels to output channels for each pyramid level
    stride (int or tuple, optional): Stride of the convolution. Default: 1
    dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
    bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``False``
  """
  def __init__(self, in_channels, out_channels, pyconv_kernels, pyconv_groups, stride=1, dilation=1, bias=False):
    super(PyConv2d, self).__init__()
    assert len(out_channels) == len(pyconv_kernels) == len(pyconv_groups)
    self.pyconv_levels = [None] * len(pyconv_kernels)
    for i in range(len(pyconv_kernels)):
      self.pyconv_levels[i] = nn.Conv2d(in_channels, out_channels[i], kernel_size=pyconv_kernels[i],
                                        stride=stride, padding=pyconv_kernels[i] // 2, groups=pyconv_groups[i],
                                        dilation=dilation, bias=bias)
    self.pyconv_levels = nn.ModuleList(self.pyconv_levels)

  def forward(self, x):
    out = []
    for level in self.pyconv_levels:
      out.append(level(x))
    return torch.cat(out, 1)



class WideActResBlock(nn.Module):

  def __init__(self, nf=64):
    super(WideActResBlock, self).__init__()
    self.res_scale = 1
    body = []
    expand = 6
    linear = 0.8
    kernel_size = 3
    wn = lambda x: torch.nn.utils.weight_norm(x)
    act = nn.ReLU(True)
    body.append(wn(nn.Conv2d(nf, nf*expand, 1, padding=1//2)))
    body.append(act)
    body.append(wn(nn.Conv2d(nf*expand, int(nf*linear), 1, padding=1//2)))
    body.append(wn(nn.Conv2d(int(nf*linear), nf, kernel_size, padding=kernel_size//2)))
    self.body = nn.Sequential(*body)

  def forward(self, x):
    res = self.body(x) * self.res_scale
    res += x
    return res

class LRSCWideActResBlock(nn.Module):

  def __init__(self, nf=64, idx=0):
    super(LRSCWideActResBlock, self).__init__()
    self.res_scale = 1

    expand = 6
    linear = 0.8
    kernel_size = 3
    wn = lambda x: torch.nn.utils.weight_norm(x)
    act = nn.ReLU(True)
    head = [wn(nn.Conv2d(nf*(idx+1), nf, 1, bias=True))] if idx > 0 else []

    body = []
    body.append(wn(nn.Conv2d(nf, nf*expand, 1, padding=1//2)))
    body.append(act)
    body.append(wn(nn.Conv2d(nf*expand, int(nf*linear), 1, padding=1//2)))
    body.append(wn(nn.Conv2d(int(nf*linear), nf, kernel_size, padding=kernel_size//2)))

    self.head = nn.Sequential(*head)
    self.body = nn.Sequential(*body)

  def forward(self, x):
    res = self.head(x)
    res = self.body(res)
    res  = torch.cat([res, x], dim=1)
    return res




class LRSCWideActResGroup(nn.Module):
  """ Long-Range Skip-connect Residual Group (RG) """

  def __init__(self, nf, n_resblocks, idx=0):
    super(LRSCWideActResGroup, self).__init__()
    kernel_size = 3

    conv = PyConv2d
    wn = lambda x: torch.nn.utils.weight_norm(x)

    modules_head = [wn(nn.Conv2d(nf*(idx+1), nf, 1, 1, 0, bias=True))] if idx > 0 else []
    modules_body = [
        LRSCWideActResBlock(nf=nf, idx=i) for i in range(n_resblocks)]
    modules_tail = [wn(nn.Conv2d(nf*(n_resblocks+1), nf, 1))]
    self.head = nn.Sequential(*modules_head)
    self.body = nn.Sequential(*modules_body)
    self.tail = nn.Sequential(*modules_tail)

  def forward(self, x):
    res = self.head(x)
    res = self.body(res)
    res = self.tail(res)
    res  = torch.cat([res, x], dim=1)
    return res




class PCD_Align(nn.Module):
  """ Alignment module using Pyramid, Cascading and Deformable convolution
  with 3 pyramid levels. [From EDVR]
  """

  def __init__(self, nf=64, groups=8, wn=None):
    super(PCD_Align, self).__init__()
    # L3: level 3, 1/4 spatial size
    self.L3_offset_conv1 = wn(nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True))  # concat for diff
    # self.L3_offset_conv2 = wn(nn.Conv2d(nf, nf, 3, 1, 1, bias=True))
    self.L3_offset_conv2 = wn(nn.Conv2d(nf, 2*groups*3*3, 3, 1, 1, bias=True))
    # self.L3_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
    self.L3_dcnpack = DeformConv2d(nf, nf, 3, stride=1, padding=1, dilation=1, groups=groups)

    # L2: level 2, 1/2 spatial size
    self.L2_offset_conv1 = wn(nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True))  # concat for diff
    self.L2_offset_conv2 = wn(nn.Conv2d(nf + 2*groups*3*3, nf, 3, 1, 1, bias=True))  # concat for offset
    # self.L2_offset_conv3 = wn(nn.Conv2d(nf, nf, 3, 1, 1, bias=True))
    self.L2_offset_conv3 = wn(nn.Conv2d(nf, 2*groups*3*3, 3, 1, 1, bias=True))
    # self.L2_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
    self.L2_dcnpack = DeformConv2d(nf, nf, 3, stride=1, padding=1, dilation=1, groups=groups)
    self.L2_fea_conv = wn(nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True))  # concat for fea

    # L1: level 1, original spatial size
    self.L1_offset_conv1 = wn(nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True))  # concat for diff
    self.L1_offset_conv2 = wn(nn.Conv2d(nf + 2*groups*3*3, nf, 3, 1, 1, bias=True))  # concat for offset
    # self.L1_offset_conv3 = wn(nn.Conv2d(nf, nf, 3, 1, 1, bias=True))
    self.L1_offset_conv3 = wn(nn.Conv2d(nf, 2*groups*3*3, 3, 1, 1, bias=True))
    # self.L1_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
    self.L1_dcnpack = DeformConv2d(nf, nf, 3, stride=1, padding=1, dilation=1, groups=groups)
    self.L1_fea_conv = wn(nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True))  # concat for fea

    # Cascading DCN
    self.cas_offset_conv1 = wn(nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True))  # concat for diff
    # self.cas_offset_conv2 = wn(nn.Conv2d(nf, nf, 3, 1, 1, bias=True))
    self.cas_offset_conv2 = wn(nn.Conv2d(nf, 2*groups*3*3, 3, 1, 1, bias=True))
    # self.cas_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
    self.cas_dcnpack = DeformConv2d(nf, nf, 3, stride=1, padding=1, dilation=1, groups=groups)

    self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

  def forward(self, nbr_fea_l, ref_fea_l):
    """ Align other neighboring frames to the reference frame in the feature level
    nbr_fea_l, ref_fea_l: [L1, L2, L3], each with [B,C,H,W] features
    """
    # L3
    L3_offset = torch.cat([nbr_fea_l[2], ref_fea_l[2]], dim=1)
    L3_offset = self.lrelu(self.L3_offset_conv1(L3_offset))
    L3_offset = self.lrelu(self.L3_offset_conv2(L3_offset))
    L3_fea = self.lrelu(self.L3_dcnpack(nbr_fea_l[2], L3_offset))
    # L2
    L3_offset = F.interpolate(L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
    L2_offset = torch.cat([nbr_fea_l[1], ref_fea_l[1]], dim=1)
    L2_offset = self.lrelu(self.L2_offset_conv1(L2_offset))
    L2_offset = self.lrelu(self.L2_offset_conv2(torch.cat([L2_offset, L3_offset*2], dim=1)))
    L2_offset = self.lrelu(self.L2_offset_conv3(L2_offset))
    L2_fea = self.L2_dcnpack(nbr_fea_l[1], L2_offset)
    L3_fea = F.interpolate(L3_fea, scale_factor=2, mode='bilinear', align_corners=False)
    L2_fea = self.lrelu(self.L2_fea_conv(torch.cat([L2_fea, L3_fea], dim=1)))
    # L1
    L2_offset = F.interpolate(L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
    L1_offset = torch.cat([nbr_fea_l[0], ref_fea_l[0]], dim=1)
    L1_offset = self.lrelu(self.L1_offset_conv1(L1_offset))
    L1_offset = self.lrelu(self.L1_offset_conv2(torch.cat([L1_offset, L2_offset * 2], dim=1)))
    L1_offset = self.lrelu(self.L1_offset_conv3(L1_offset))
    L1_fea = self.L1_dcnpack(nbr_fea_l[0], L1_offset)
    L2_fea = F.interpolate(L2_fea, scale_factor=2, mode='bilinear', align_corners=False)
    L1_fea = self.L1_fea_conv(torch.cat([L1_fea, L2_fea], dim=1))
    # Cascading
    offset = torch.cat([L1_fea, ref_fea_l[0]], dim=1)
    offset = self.lrelu(self.cas_offset_conv1(offset))
    offset = self.lrelu(self.cas_offset_conv2(offset))
    L1_fea = self.lrelu(self.cas_dcnpack(L1_fea, offset))
    return L1_fea


class CrossNonLocal_Fusion(nn.Module):
  """ Cross Non Local fusion module. """

  def __init__(self, nf=64, nframes=5, center=2, wn=None):
    super(CrossNonLocal_Fusion, self).__init__()
    self.center = center

    self.non_local_T = nn.ModuleList()
    self.non_local_F = nn.ModuleList()

    for i in range(nframes):
      self.non_local_T.append(NonLocalCross(nf, inter_channels=nf//2, sub_sample=True, bn_layer=False))
      self.non_local_F.append(NonLocal(nf, inter_channels=nf//2, sub_sample=True, bn_layer=False))

    # fusion conv: using 1x1 to save parameters and computation
    self.fea_fusion = wn(nn.Conv2d(nframes * nf * 2, nf, 1, 1, bias=True))

    self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

  def forward(self, aligned_fea):
    B, N, C, H, W = aligned_fea.size()  # N video frames
    ref = aligned_fea[:, self.center, :, :, :].clone()

    cor_l = []
    non_l = []
    for i in range(N):
      nbr = aligned_fea[:, i, :, :, :]
      non_l.append(self.non_local_F[i](nbr))
      cor_l.append(self.non_local_T[i](nbr, ref))

    aligned_fea_T = torch.cat(cor_l, dim=1)
    aligned_fea_F = torch.cat(non_l, dim=1)
    aligned_fea = torch.cat([aligned_fea_T, aligned_fea_F], dim=1)

    #### fusion
    fea = self.fea_fusion(aligned_fea)

    return fea
