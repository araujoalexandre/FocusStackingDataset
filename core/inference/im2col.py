import torch
from torch.nn import functional as F

def Im2Col(input_tensor, kernel_size, stride, padding,dilation=1,tensorized=False,):
    batch = input_tensor.shape[0]
    out = F.unfold(input_tensor, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation)
    if tensorized:
        lh, lw = im2col_shape(input_tensor.shape[1:], kernel_size=kernel_size,
                              stride=stride, padding=padding,dilation=dilation)[-2:]
        out = out.view(batch,-1,lh,lw)
    return out

def Col2Im(input_tensor, output_size, kernel_size, stride, padding, dilation=1, avg=False, 
         input_tensorized=False, return_w=False):
  if input_tensorized:
    input_tensor = input_tensor.flatten(2, 3)
  out = F.fold(input_tensor, output_size=output_size, kernel_size=kernel_size,
               padding=padding, stride=stride,dilation=dilation)
  if avg:
    me = F.fold(torch.ones_like(input_tensor), output_size=output_size,
                kernel_size=kernel_size, padding=padding, stride=stride,dilation=dilation)
    out = out / me
  if return_w:
    me = F.fold(torch.ones_like(input_tensor), output_size=output_size,
                kernel_size=kernel_size, padding=padding, stride=stride,dilation=dilation)
    return out, me
  return out


