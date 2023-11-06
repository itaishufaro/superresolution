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
    train_dir = 'train2017'
    valid_dir = 'val2017'
    if use_logger:
        wandb.login(key=WANDB_KEY)
        wandb.init(entity=WANDB_ENTITY,
                   project="superres",
                   name="superres_run1")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model = models.SarSubPixel()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    gamma = 0.95
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer)
    criterion = nn.MSELoss().to(device)
    transform = T.Compose([T.ToTensor(), T.Resize((256, 256))])
    trainLoader = DataLoader(dataset.StuffDataset(train_dir, transform), batch_size=32, shuffle=True)
    validLoader = DataLoader(dataset.StuffDataset(valid_dir, transform), batch_size=32, shuffle=True)
    loss, val = train.train_epochs(num_epochs=1,
                                   model=model,
                                   trainloader=trainLoader,
                                   validloader=validLoader,
                                   optimizer=optimizer,
                                   scheduler=scheduler,
                                   criterion_train=criterion,
                                   criterion_valid=criterion,
                                   device=device,
                                   save_every=1,
                                   perceptual_loss=True)

