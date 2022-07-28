
import os
import torch
import glob
import cv2
import natsort
import logging
import numpy as np
import pandas as pd
import PIL.Image as Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from os.path import join
from torch.utils.data import Dataset
from cv2 import imread


class DdDpCanonDataset:

  def __init__(self, root, split='train', transform=None):
    self.root = root
    self.scale = 0.65
    self.ap = 4
    self.focal = 60
    params_folder = f'scale_{self.scale}_ap_{self.ap}_focal_{self.focal}'
    folder = join(self.root, 'bursts', params_folder, split)
    self.bursts_list = glob.glob(join(folder, '**'))
    self.target_path = join(self.root, 'png', f'{split}_c', 'target')

  def __len__(self):
    return len(self.bursts_list)

  def cv2torch(self, image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.FloatTensor(image)
    image = image.permute(2, 0, 1)
    image = image[None]
    image = image / 255
    return image

  def __getitem__(self, idx):
    burst_path = self.bursts_list[idx]
    image_name = burst_path.split('/')[-1]
    imgs_path = glob.glob(join(burst_path, '*.jpg'))
    burst = []
    for img_path in imgs_path:
      image = imread(img_path)
      image = self.cv2torch(image) 
      burst.append(image)
    burst = torch.cat(burst, 0)
    target = imread(join(self.target_path, f'{image_name}.png'))
    target = cv2.resize(target, None, None, self.scale, self.scale, cv2.INTER_AREA)
    target = self.cv2torch(target)
    return burst, target




class FocusDataset:

  def __init__(self, root, split='train', transform=None):
    root = join(root, 'dataset')
    df = pd.read_csv(join(root, 'dataset.csv'), sep=";")
    self.n_train_files = (df['set'] == 'train').sum()
    self.n_test_files = (df['set'] == 'test').sum()
    df = df[df['set'] == split][['lens', 'photo']]
    self.bursts_list = df.apply(lambda x: join(root, split, x[0], x[1]), axis=1).values
    self.bursts_list = list(self.bursts_list)
    self.crop_list = []
    ncrop = len(glob.glob(join(self.bursts_list[0], 'crops', 'crop*')))
    for burst_path in self.bursts_list:
      for i in range(ncrop):
        crop_path = f'{burst_path}/crops/crop{i}.pkl'
        self.crop_list.append(crop_path)

  def __len__(self):
    return len(self.crop_list)

  def __getitem__(self, idx):
    crop_sz = 64
    crop = transforms.RandomCrop(size=crop_sz)
    burst_path = self.crop_list[idx]
    data = torch.load(burst_path)
    burst, target = data['burst'], data['target']
    burst, target = burst.float() / 255, target.float() / 255
    params = crop.get_params(burst, (crop_sz, crop_sz))
    burst, target = F.crop(burst, *params), F.crop(target, *params)
    return burst, target
    



