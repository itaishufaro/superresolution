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
import gan

WANDB_KEY = "01717b5e711e2653d9cc50175f88588ce40619df"
WANDB_ENTITY = "itai-shufaro"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a super-resolution model")
    parser.add_argument()


if __name__ == '__main__':
    use_logger = True
    train_dir = 'trainSAR'
    valid_dir = 'trainSAR'
    # wandb.login(key=WANDB_KEY)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(.99, 0)
    print(f'Using device: {device}')
    generator = gan.Generator()
    discriminator = gan.Discriminator()
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, weight_decay=0.0001, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5, verbose=True)
    # criterion = nn.MSELoss().to(device)
    gen_optimizer = torch.optim.SGD(generator.parameters(), lr=1e-3, weight_decay=0.0001, momentum=0.9)
    disc_optimizer = torch.optim.SGD(discriminator.parameters(), lr=1e-3, weight_decay=0.0001, momentum=0.9)
    gen_scheduler = torch.optim.lr_scheduler.StepLR(gen_optimizer, step_size=20, gamma=0.5, verbose=True)
    disc_scheduler = torch.optim.lr_scheduler.StepLR(disc_optimizer, step_size=20, gamma=0.5, verbose=True)
    transform = T.Compose([T.ToTensor()])
    # trainLoader = DataLoader(dataset.StuffDataset(train_dir, transforms=transform, inputH=256, inputW=256),
    #                          batch_size=512, shuffle=True)
    # validLoader = DataLoader(dataset.StuffDataset(valid_dir, transforms=transform, inputH=256, inputW=256),
    #                          batch_size=512, shuffle=True)
    SARDataset = dataset.StuffDataset(train_dir, transforms=transform, inputH=512, inputW=512, scale_factor=4)
    # split the dataset into train, validation and test data loaders
    # trainFull_set, test_set = train_test_split(SARDataset, test_size=0.2, random_state=42)
    # train_set, val_set = train_test_split(trainFull_set, test_size=0.2, random_state=42)
    train_loader = DataLoader(SARDataset, batch_size=1, shuffle=True)
    # valid_loader = DataLoader(val_set, batch_size=1, shuffle=True)
    # test_loader = DataLoader(test_set, batch_size=1, shuffle=True)
    train.train_gan(num_epochs=100, generator=generator,
            discriminator=discriminator, trainloader=train_loader,
            testloader=train_loader, gen_optimizer=gen_optimizer,
            disc_optimizer=disc_optimizer, device=device,
            aug=None, start_epoch=0, save_every=10, save_name='model', wandb_logger=None,
            alpha=1e-5, gen_scheduler=gen_scheduler, disc_scheduler=disc_scheduler)




