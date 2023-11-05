import pandas as pd
from torch.utils.data import DataLoader
import torchvision.transforms as T
from dataset import StuffDataset
import torch
from torch import nn
from transformers import BertTokenizer
import matplotlib.pyplot as plt
import wandb
import os
import numpy as np
from PIL import Image
import models
import argparse
import train
import dataset

WANDB_KEY = "01717b5e711e2653d9cc50175f88588ce40619df"
WANDB_ENTITY = "itai-shufaro"

def parse_args():
    parser = argparse.ArgumentParser(description="Train a superres model")
    parser.add_argument()

if __name__ == '__main__':
    # wandb.login(key=WANDB_KEY)
    # wandb.init(entity=WANDB_ENTITY,
    #            project="superres",
    #            name="superres_run1")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model = models.SarSubPixel()
   #  model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
    criterion = nn.MSELoss()
    model.load_state_dict(torch.load('models/train_3500.pth'))
    model.eval()
    transform = T.Compose([T.ToTensor(), T.Resize((512, 512))])
    trainloader = DataLoader(dataset.StuffDataset('train2017', transform), batch_size=1, shuffle=True)
    validloader = DataLoader(dataset.StuffDataset('val2017', transform), batch_size=1, shuffle=True)
    for lr, hr in iter(validloader):
        # x = lr.to(device)
        out = model(lr)
        img_out = Image.fromarray(np.uint8(out[0].cpu().detach().numpy().transpose(1, 2, 0)*255), 'RGB')
        img_out.save('out.png')
        img_hr = Image.fromarray(np.uint8(hr[0].numpy().transpose(1, 2, 0)*255), 'RGB')
        img_hr.save('hr.png')
        img_lr = Image.fromarray(np.uint8(lr[0].numpy().transpose(1, 2, 0)*255), 'RGB')
        img_lr.save('lr.png')
        break

