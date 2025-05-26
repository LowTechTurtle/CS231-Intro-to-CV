import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import numpy as np
import glob
import torch

class RainDataset(Dataset):
    def __init__(self, opt, is_eval=False, is_test=False, transform=None, horizontal_flip=False):
        super(RainDataset, self).__init__()

        if is_test:
            self.dataset = opt.test_dataset
        elif is_eval:
            self.dataset = opt.eval_dataset
        else:
            self.dataset = opt.train_dataset
        # dataset = open(self.dataset, 'r').read().split()
        self.img_list = sorted(glob.glob(self.dataset+'/data/*'))
        self.gt_list = sorted(glob.glob(self.dataset+'/gt/*'))
        if not transform:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform
        self.horizontal_flip = horizontal_flip
        
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        gt_name = self.gt_list[idx]

        img = Image.open(img_name).convert("RGB")
        gt = Image.open(gt_name).convert("RGB")
        
        # Apply transforms
        if self.horizontal_flip:
            p = np.random.rand()
            if p > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
        img = self.transform(img)
        gt = transforms.Resize((224, 224))(gt)
        gt = transforms.ToTensor()(gt)
        
        return [img,gt]
