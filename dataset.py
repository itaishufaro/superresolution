import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
from torch.nn.utils.rnn import pad_sequence
from PIL import Image, ImageFilter
import os
from torch.utils.data import Dataset
import torchvision.transforms as T
from kornia.augmentation import AugmentationSequential


class StuffDataset(Dataset):
    def __init__(self, root_dir, inputH=512, inputW=512, scale_factor=4, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.H = inputH
        self.W = inputW
        self.scale_factor = scale_factor
        self.df = sorted(os.listdir(root_dir))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df[idx]
        image = Image.open(os.path.join(self.root_dir, img_name)).convert('L')
        if self.transforms:
            image = self.transforms(image)
        low_res = image
        # low_res = T.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))(image)
        low_res = T.Resize((self.W//self.scale_factor, self.H//self.scale_factor))(low_res)
        return low_res, image
