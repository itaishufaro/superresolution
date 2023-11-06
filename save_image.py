import pandas as pd
from torch.utils.data import DataLoader
import torchvision.transforms as T
from dataset import StuffDataset
import torch
from torch import nn
import matplotlib.pyplot as plt
import wandb
import os
import numpy as np
from PIL import Image
import models
import argparse
import train
import dataset


if __name__ == '__main__':
    path_to_model = 'models/train_3500.pth'
    model = models.SarSubPixel()
    model.load_state_dict(torch.load(path_to_model))
    model.eval()
    transform = T.Compose([T.ToTensor(), T.Resize((512, 512))])
    trainloader = DataLoader(dataset.StuffDataset('train2017', transform), batch_size=1, shuffle=True)
    validloader = DataLoader(dataset.StuffDataset('val2017', transform), batch_size=1, shuffle=True)
    for lr, hr in iter(validloader):
        out = model(lr)
        img_out = Image.fromarray(np.uint8(out[0].cpu().detach().numpy().transpose(1, 2, 0)*255), 'RGB')
        img_out.save('out.png')
        img_hr = Image.fromarray(np.uint8(hr[0].numpy().transpose(1, 2, 0)*255), 'RGB')
        img_hr.save('hr.png')
        img_lr = Image.fromarray(np.uint8(lr[0].numpy().transpose(1, 2, 0)*255), 'RGB')
        img_lr.save('lr.png')
        break

