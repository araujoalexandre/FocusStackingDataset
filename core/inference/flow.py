import torch
import cv2 as cv
import numpy as np
import torch.nn.functional as F



def calcFlow(frames, D=4, pyr_scale=0.5, levels=1, winsize=300,
           iterations=1, poly_n=10, poly_sigma=1.7, interpolate=False, return_np=False):
  """
  compute optical flow wrt. first frame
  :param frames: b,c,h,w torch tensor of frames
  :param D: scale factor to compute the flow
  :param return_np:  return numpy flow and rgbs
  :param kwargs:
  :return:
  """

  assert frames.shape[0] > 1
  assert len(frames.shape) == 4

  frames_downscaled = F.interpolate(frames, scale_factor=1 / D, mode="bilinear")
  frames_downscaled_np = (frames_downscaled.permute(0,2,3,1).numpy() * 255.0).astype(np.uint8)

  first_frame = frames_downscaled_np[0]
  first_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
  mask = np.zeros_like(first_frame)
  mask[..., 1] = 255

  flows = []
  rgbs = []
  for frame_idx in range(frames.shape[0]-1):

    frame = frames_downscaled_np[frame_idx+1]
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(
      first_gray, gray, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, 0)
    flows.append(flow)

    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
    mask[..., 0] = angle * 180 / np.pi / 2
    mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
    rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
    rgbs.append(rgb)

  flows.insert(0, np.zeros_like(flow))
  rgbs.insert(0, np.zeros_like(rgb))

  flows = np.stack(flows) * D # rescale optical flow
  rgbs = np.stack(rgbs)

  if not return_np:
    flows = torch.from_numpy(flows).permute(0, 3, 1,2)
    rgbs = torch.from_numpy(rgbs).permute(0, 3, 1, 2).float() / 255

    if interpolate:
      flows = F.interpolate(flows, scale_factor=4, mode='bilinear')
      rgbs = F.interpolate(rgbs, scale_factor=4, mode='bilinear')

  return flows, rgbs


def calcFlowGray(frames, D=4, pyr_scale=0.5, levels=1, winsize=300,
                 iterations=3, poly_n=10, poly_sigma=1.7, interpolate=False, return_np=False):
  """
  compute optical flow wrt. first frame
  :param frames: b,1,h,w torch tensor of GRAY frames
  :param D: scale factor to compute the flow
  :param return_np:  return numpy flow and rgbs
  :param kwargs:
  :return:
  """

  assert frames.shape[0] > 1
  assert len(frames.shape) == 4
  inp_shape = frames.shape[-2:]
  frames_downscaled = F.interpolate(frames, scale_factor=1/D, mode="bilinear")
  frames_downscaled_np = (frames_downscaled.permute(0, 2, 3, 1).numpy() * 255.0).astype(np.uint8)

  first_frame = frames_downscaled_np[0]
  h, w, _ = first_frame.shape
  mask = np.zeros((h, w, 3)).astype('uint8')
  mask[..., 1] = 255

  flows, rgbs = [], []
  for frame_idx in range(frames.shape[0]-1):

    frame = frames_downscaled_np[frame_idx+1]
    flow = cv.calcOpticalFlowFarneback(
      first_frame, frame, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, 0)
    flows.append(flow)

    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
    mask[..., 0] = angle * 180 / np.pi / 2
    mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
    rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
    rgbs.append(rgb)

  flows.insert(0, np.zeros_like(flow))
  rgbs.insert(0, np.zeros_like(rgb))

  flows = np.stack(flows) * D # rescale optical flow
  rgbs = np.stack(rgbs)

  if not return_np:
    flows = torch.from_numpy(flows).permute(0, 3, 1, 2)
    rgbs = torch.from_numpy(rgbs).permute(0, 3, 1, 2).float() / 255

    if interpolate:
      flows = F.interpolate(flows, size=inp_shape, mode='bilinear')
      rgbs = F.interpolate(rgbs, size=inp_shape, mode='bilinear')

  return flows, rgbs

# def calcFlow(frame1, frame2, pyr_scale=0.5, levels=1, winsize=300,
#              iterations=1, poly_n=10, poly_sigma=1.7, interpolate=False):
#   """
#   compute optical flow wrt. first frame
#   :param frames: b, 1, h, w torch tensor of GRAY frames
#   :param D: scale factor to compute the flow
#   :param return_np: return numpy flow and rgbs
#   :param kwargs:
#   :return:
#   """
#
#   first_frame = frame1
#   h,w,_ = frame1.shape
#   mask = np.zeros((h,w,3)).astype('uint8')
#   mask[..., 1] = 255
#
#   frame = frame2
#
#   flow = cv.calcOpticalFlowFarneback(
#     first_frame, frame, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, 0)
#
#   magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
#   mask[..., 0] = angle * 180 / np.pi / 2
#   mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
#   rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
#
#   return flow, rgb
