
import logging
import torch
import numpy as np
import cv2
  

class AlignImages:

  def __init__(self):

    self.m_transformation = np.float32(np.eye(2, 3))

    self.contrast = np.zeros((5, ))
    self.contrast[0] = 1.

    self.whitebalance = np.zeros((6, 1))
    self.whitebalance[1, 0] = 1.
    self.whitebalance[3, 0] = 1.
    self.whitebalance[5, 0] = 1.

  def compute_pca(self, image):  
    """ compute PCA for grayscale conversion """
    flat_img = image.reshape(-1, self.n_channels)
    mean, eigenvectors = cv2.PCACompute(flat_img, np.array(()), cv2.PCA_DATA_AS_ROW, 2)
    T = np.zeros((1, 2), dtype=np.float32)
    T[0, 0] = 1
    m_weights = cv2.PCABackProject(T, mean, eigenvectors)
    m_weights -= cv2.PCABackProject(np.zeros((1, 2), dtype=np.float32), mean, eigenvectors)
    m_weights /= np.sum(m_weights)
    self.weights = m_weights[0]

  def convert_to_grayscale(self, img):
    if self.n_channels == 4:
      w0, w1, w2, w3 = self.weights
      result = img[..., 0] * w0 + img[..., 1] * w1 + img[..., 2] * w2 + img[..., 3] * w3
    else:
      w0, w1, w2 = self.weights
      result = img[..., 0] * w0 + img[..., 1] * w1 + img[..., 2] * w2
    result = np.minimum(255, np.maximum(0, result))
    result = np.uint8(result)
    return result

  def find_constrast(self, src, ref):
    src = self.apply_transform(src)
    xsamples, ysamples = 64, 64
    total = xsamples * ysamples
    ref = cv2.resize(ref, (xsamples, ysamples), 0, 0, cv2.INTER_AREA)
    src = cv2.resize(src, (xsamples, ysamples), 0, 0, cv2.INTER_AREA)
    contrast = np.zeros((total, 1), dtype=np.float32)
    positions = np.zeros((total, 5), dtype=np.float32) 
    for y in range(ysamples):
      for x in range(xsamples):
        idx = y * xsamples + x
        yd = (y - ref.shape[0] / 2.0) / ref.shape[0]
        xd = (x - ref.shape[1] / 2.0) / ref.shape[1]
        refpix = ref[y, x]
        srcpix = src[y, x]
        c = 1.0
        if (refpix > 4 and srcpix > 4):
          c = refpix / srcpix
        contrast[idx] = c
        positions[idx, 0] = 1.0
        positions[idx, 1] = xd
        positions[idx, 2] = xd**2
        positions[idx, 3] = yd
        positions[idx, 4] = yd**2
    _, m_contrast = cv2.solve(positions, contrast, None, cv2.DECOMP_SVD)
    return np.float32(m_contrast.flatten())

  def find_white_balance(self, src, ref):
    src = self.apply_transform(src)
    src = self.apply_contrast_whitebalance(src)
    xsamples, ysamples = 64, 64
    total = xsamples * ysamples
    ref = cv2.resize(ref, (xsamples, ysamples), 0, 0, cv2.INTER_AREA)
    src = cv2.resize(src, (xsamples, ysamples), 0, 0, cv2.INTER_AREA)
    targets = np.zeros((total * 3, 1), dtype=np.float32)
    factors = np.zeros((total * 3, 6), dtype=np.float32)
    for y in range(ysamples):
      for x in range(xsamples):
        idx = y * xsamples + x
        srcpixel = src[y, x]
        refpixel = ref[y, x]
        targets[idx * 3 + 0, 0] = refpixel[0]
        targets[idx * 3 + 1, 0] = refpixel[1]
        targets[idx * 3 + 2, 0] = refpixel[2]
        factors[idx * 3 + 0, 0] = 1.0
        factors[idx * 3 + 0, 1] = srcpixel[0]
        factors[idx * 3 + 1, 2] = 1.0
        factors[idx * 3 + 1, 3] = srcpixel[1]
        factors[idx * 3 + 2, 4] = 1.0
        factors[idx * 3 + 2, 5] = srcpixel[2]
    _, m_whitebalance = cv2.solve(factors, targets, None, cv2.DECOMP_SVD)
    return np.float32(m_whitebalance.flatten())

  def _apply_contrast_whitebalance_gray(self, image):
    h, w = image.shape
    y_values = (np.arange(h) - h/2.0) / h
    x_values = (np.arange(w) - w/2.0) / w
    xd, yd = np.meshgrid(x_values, y_values, indexing='xy')
    c = self.contrast[0]
    c += xd * (self.contrast[1] + self.contrast[2] * xd)
    c += yd * (self.contrast[3] + self.contrast[4] * yd)
    image = np.uint8(np.minimum(255, np.maximum(0, image * c)))
    return image

  def _apply_contrast_whitebalance_color(self, image):
    h, w, _ = image.shape
    y_values = (np.arange(h) - h/2.0) / h
    x_values = (np.arange(w) - w/2.0) / w
    xd, yd = np.meshgrid(x_values, y_values, indexing='xy')
    c = self.contrast[0]
    c += xd * (self.contrast[1] + self.contrast[2] * xd)
    c += yd * (self.contrast[3] + self.contrast[4] * yd)
    whitebalance = self.whitebalance
    image[..., 0] = image[..., 0] * c # * whitebalance[1] + whitebalance[0]
    image[..., 1] = image[..., 1] * c # * whitebalance[3] + whitebalance[2]
    image[..., 2] = image[..., 2] * c # * whitebalance[5] + whitebalance[4]
    image[..., 0] = np.uint8(np.minimum(255, np.maximum(0, image[..., 0])))
    image[..., 1] = np.uint8(np.minimum(255, np.maximum(0, image[..., 1])))
    image[..., 2] = np.uint8(np.minimum(255, np.maximum(0, image[..., 2])))
    return image

  def apply_contrast_whitebalance(self, image):
    if len(image.shape) == 2 or image.shape[2] == 1:
      return self._apply_contrast_whitebalance_gray(image)
    return self._apply_contrast_whitebalance_color(image)

  def find_transform(self, src, ref, rough=False, max_res=2048):

    resolution = np.max(ref.shape)
    scale_ratio = 1.
    if resolution > max_res:
      scale_ratio = max_res / resolution
      ref = cv2.resize(ref, None, None, scale_ratio, scale_ratio, cv2.INTER_AREA)
      src = cv2.resize(src, None, None, scale_ratio, scale_ratio, cv2.INTER_AREA)

    src = self.apply_contrast_whitebalance(src)

    self.m_transformation[0, 2] *= scale_ratio
    self.m_transformation[1, 2] *= scale_ratio

    if rough:
      number_of_iterations = 25
      termination_eps = 0.01
      warp_mode = cv2.MOTION_AFFINE
      criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
    else:
      number_of_iterations = 50
      termination_eps = 0.001
      warp_mode = cv2.MOTION_AFFINE
      criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    cc, self.m_transformation = cv2.findTransformECC(
      src, ref, self.m_transformation, warp_mode, criteria)

    self.m_transformation[0, 2] /= scale_ratio
    self.m_transformation[1, 2] /= scale_ratio

    return self.m_transformation

  def apply_transform(self, image):
    h, w = image.shape[:2]
    image = cv2.warpAffine(
      image, self.m_transformation, (w, h), None, cv2.INTER_CUBIC, cv2.BORDER_REFLECT)
    return image


  def align_images(self, m_refgray, m_refcolor, m_srcgray, m_srccolor, m_initial_guess): 

    if np.allclose(m_refcolor, m_srccolor):
      return m_srccolor

    self.m_transform = m_initial_guess
    self.m_transform = self.find_transform(m_srcgray, m_refgray, True, 256)

    self.contrast = self.find_constrast(m_srcgray, m_refgray)
    self.whitebalance = self.find_white_balance(m_srccolor, m_refcolor)

    m_srcgray = self.apply_contrast_whitebalance(m_srcgray)
    
    self.m_transform = self.find_transform(m_srcgray, m_refgray, False)

    newimage = self.apply_transform(m_srccolor)
    newimage = self.apply_contrast_whitebalance(newimage)
    return newimage, self.m_transform


  def __call__(self, images):

    if images.max() < 2:
      images = images.permute(0, 2, 3, 1)
      images = (images * 255).to(torch.uint8)
    images = images.numpy()

    self.n_channels = images.shape[-1]

    refidx = 0
    ref = images[refidx]

    self.compute_pca(ref)
    grayscale_images = []
    for i, image in enumerate(images):
      img_gray = self.convert_to_grayscale(image)
      grayscale_images.append(img_gray)

    transforms = [None] * len(images)
    aligned_images = [None] * len(images)
    aligned_grayscales = [None] * len(images)

    aligned_images[refidx] = images[refidx]
    aligned_grayscales[refidx] = grayscale_images[refidx]

    indexes = list(range(refidx+1, len(images))) + list(range(refidx))[::-1]
    for i in indexes:
      neighbour = refidx;
      if (i < refidx): neighbour = i + 1
      if (i > refidx): neighbour = i - 1

      if i != refidx:
        aligned, transform = self.align_images(aligned_grayscales[neighbour],
              aligned_images[neighbour], grayscale_images[i], images[i], transforms[neighbour])
        transforms[i] = transform
      else:
        aligned = aligned_images[refidx]
      aligned_images[i] = aligned
      aligned_grayscales[i] = self.convert_to_grayscale(aligned)

    return torch.Tensor(aligned_images).permute(0, 3, 1, 2) / 255




