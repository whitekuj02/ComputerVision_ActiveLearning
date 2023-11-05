import os
import time

import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

class CUB2011(Dataset):
    def __init__(self, transform, mode='train'):
        self.transform = transform
        self.mode = mode

        if self.mode == "train":
            self.image_folder = os.listdir('/home/gw6/CV_Active/dataset/CUB_200_2011_repackage_class50/datasets/train')
        elif self.mode == "valid":
            self.image_folder = os.listdir('/home/gw6/CV_Active/dataset/CUB_200_2011_repackage_class50/datasets/valid')
        elif self.mode == "test":
            self.image_folder = os.listdir('/home/gw6/CV_Active/dataset/CUB_200_2011_repackage_class50/datasets/test')

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, idx):
        # 따로 전처리 해야할게 있나..
        img_path = self.image_folder[idx]
        img = Image.open(os.path.join("/home/gw6/CV_Active/dataset/CUB_200_2011_repackage_class50/datasets", self.mode, img_path)).convert("RGB")
        img = self.transform(img)

        label = img_path.split('_')[-1].split('.')[0]
        label = int(label)
        return (img, label)

   
        