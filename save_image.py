import pandas as pd
from torch.utils.data import DataLoader
import torchvision.transforms as T

import gan
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


def show_image(lr, model, hr):
    model.eval()
    out = model(lr)
    img = Image.fromarray(np.squeeze(np.uint8(out[0].cpu().detach().numpy()*255)), 'L')
    plt.imshow(np.squeeze(img), cmap='gray')
    plt.show()
    plt.imshow(np.squeeze(hr[0].cpu().detach().numpy()), cmap='gray')
    plt.show()
    plt.imshow(np.squeeze(lr[0].cpu().detach().numpy()), cmap='gray')


if __name__ == '__main__':
    path_to_model = 'models/model_gan2_100.pth'
    model = gan.Generator2(scale_factor=2)
    model.load_state_dict(torch.load(path_to_model))
    model.eval()
    transform = T.Compose([T.ToTensor()])
    trainloader = DataLoader(dataset.StuffDataset('trainSAR', transforms=transform, inputH=256, inputW=256, scale_factor=2), batch_size=1, shuffle=True)
    validloader = DataLoader(dataset.StuffDataset('trainSAR', transforms=transform, inputH=256, inputW=256, scale_factor=2), batch_size=1, shuffle=True)
    for lr, hr in iter(validloader):
        out = model(lr)
        img_out = Image.fromarray(np.squeeze(np.uint8(out[0].cpu().detach().numpy()*255)), 'L')
        img_out.save('out.png')
        img_hr = Image.fromarray(np.squeeze(np.uint8(hr[0].cpu().detach().numpy()*255)), 'L')
        img_hr.save('hr.png')
        img_lr = Image.fromarray(np.squeeze(np.uint8(lr[0].cpu().detach().numpy()*255)), 'L')
        img_lr.save('lr.png')
        break

