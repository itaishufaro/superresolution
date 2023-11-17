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
from sklearn.model_selection import train_test_split

WANDB_KEY = "01717b5e711e2653d9cc50175f88588ce40619df"
WANDB_ENTITY = "itai-shufaro"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a super-resolution model")
    parser.add_argument()


if __name__ == '__main__':
    use_logger = True
    train_dir = 'trainSAR'
    valid_dir = 'trainSAR'
    wandb.login(key=WANDB_KEY)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(.99, 0)
    print(f'Using device: {device}')
    # model = models.SarSubPixel(colors=1, drop_prob=0)
    # model = model.to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, weight_decay=0.0001, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5, verbose=True)
    # criterion = nn.MSELoss().to(device)
    # transform = T.Compose([T.ToTensor()])
    # # trainLoader = DataLoader(dataset.StuffDataset(train_dir, transforms=transform, inputH=256, inputW=256),
    # #                          batch_size=512, shuffle=True)
    # # validLoader = DataLoader(dataset.StuffDataset(valid_dir, transforms=transform, inputH=256, inputW=256),
    # #                          batch_size=512, shuffle=True)
    # SARDataset = dataset.StuffDataset(train_dir, transforms=transform, inputH=256, inputW=256)
    # # split the dataset into train, validation and test data loaders
    # trainFull_set, test_set = train_test_split(SARDataset, test_size=0.2, random_state=42)
    # train_set, val_set = train_test_split(trainFull_set, test_size=0.2, random_state=42)
    # train_loader = DataLoader(train_set, batch_size=512, shuffle=True)
    # valid_loader = DataLoader(val_set, batch_size=512, shuffle=True)
    # test_loader = DataLoader(test_set, batch_size=512, shuffle=True)
    # perform hyperparameter search using wandb
    # wandb_logger = WandbLogger()
    # wandb_logger.watch(model)
    # wandb_logger.log_hyperparams({'lr': 1e-4, 'batch_size': 512, 'epochs': 100, 'weight_decay': 0.0001})
    sweep_config = {
        'method': 'bayes',
        'name': 'superres',
        'metric': {
            'name': 'val_loss',
            'goal': 'maximize'
        },
        'parameters': {
            'lr': {'max':1e-3, 'min': 1e-5},
            'step_size': {'max': 20, 'min': 5},
            'gamma': {'max': 0.9, 'min': 0.1},
            'weight_decay': {'max': 0.0001, 'min': 0.00001},
            'momentum': {'max': 0.9, 'min': 0.1},
            'drop_prob': {'max': 0.2, 'min': 0.0},
            'train_dir': {'value': 'trainSAR'},
        }}
    sweep_id = wandb.sweep(sweep_config, project="superres", entity=WANDB_ENTITY)
    wandb.agent(sweep_id, train.hyperparameter_search, count=100)



