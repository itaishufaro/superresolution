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

WANDB_KEY = "01717b5e711e2653d9cc50175f88588ce40619df"
WANDB_ENTITY = "itai-shufaro"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a super-resolution model")
    parser.add_argument()


if __name__ == '__main__':
    use_logger = False
    train_dir = 'trainSAR'
    valid_dir = 'trainSAR'
    if use_logger:
        wandb.login(key=WANDB_KEY)
        wandb.init(entity=WANDB_ENTITY,
                   project="superres",
                   name="superres_run1")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model = models.SarSubPixel(colors=1, drop_prob=0.1)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=0.0001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5, verbose=True)
    criterion = nn.MSELoss().to(device)
    transform = T.Compose([T.ToTensor()])
    trainLoader = DataLoader(dataset.StuffDataset(train_dir, transforms=transform), batch_size=10, shuffle=True)
    validLoader = DataLoader(dataset.StuffDataset(valid_dir, transforms=transform), batch_size=10, shuffle=True)
    train.train_epochs(num_epochs=500,
                                   model=model,
                                   trainloader=trainLoader,
                                   validloader=validLoader,
                                   optimizer=optimizer,
                                   scheduler=scheduler,
                                   criterion_train=criterion,
                                   criterion_valid=train.PSNR,
                                   device=device,
                                   save_every=100,
                                   perceptual_loss=False)

