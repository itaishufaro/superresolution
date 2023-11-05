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
import logging
import datetime
from torchvision.models import resnet50, ResNet50_Weights

def train_one_epoch(model, dataloader, optimizer, criterion, device, aug=None,
                    wandb_logger=None, epoch=0, perceptual_loss=False):
    '''

    :param model:
    :param dataloader:
    :param optimizer:
    :param criterion:
    :param device:
    :param aug:
    :return:
    '''
    model.train()
    i = 1
    tot_loss = 0
    len_dataloader = len(dataloader)
    log_section = 'train'
    for low_res, high_res in iter(dataloader):
        low_res = low_res.to(device)
        high_res = high_res.to(device)
        if not aug is None:
            low_res = aug(low_res)
            high_res = aug(high_res)
        low_res = low_res.view(low_res.shape[0], 3, 128, 128)
        high_res = high_res.view(high_res.shape[0], 3, 512, 512)
        optimizer.zero_grad()
        # out_train, out_real = model(low_res, high_res)
        out_real = high_res
        out_train = model(low_res)
        loss = criterion(out_train, out_real)
        loss.backward()
        optimizer.step()
        if wandb_logger is not None:
            wandb_logger.log({f"{log_section}/episode": len_dataloader * epoch + i},
                {f"{log_section}/lr": optimizer.param_groups[0]['lr']},
                {f"{log_section}/avg_loss": loss.item()})
        tot_loss += loss.item()
        if i % 1 == 0:
            print(f'Batch {i}/{len_dataloader} Loss: {loss.item()}')
        i += 1
        if i % 500 == 0:
            torch.save(model.state_dict(), f'models/train_{i}.pth')
        torch.cuda.empty_cache()
    return tot_loss / len_dataloader


def validate(model, dataloader, criterion, device):
    '''

    :param model:
    :param dataloader:
    :param criterion:
    :param device:
    :return:
    '''
    model.eval()
    eval_loss = 0
    for lr, hr in iter(dataloader):
        hr_model = model(lr.to(device))
        eval_loss += criterion(hr_model, hr.to(device)).item()
    return eval_loss / len(dataloader)


def train_epochs(num_epochs, model, trainloader, validloader, optimizer, criterion_train, criterion_valid, device,
                 aug=None, start_epoch=0, save_every=10, save_name='model', wandb_logger=None,
                 perceptual_loss=False):
    '''

    :param num_epochs:
    :param model:
    :param trainloader:
    :param validloader:
    :param optimizer:
    :param criterion_train:
    :param criterion_valid:
    :param device:
    :param aug:
    :param start_epoch:
    :param save_every:
    :param save_name:
    :return:
    '''
    loss_points = []
    valid_points = []
    for epoch in range(num_epochs):
        loss = train_one_epoch(model, trainloader, optimizer, criterion_train, device, aug,
                               wandb_logger=wandb_logger, epoch=epoch + start_epoch)
        print(f'Epoch {epoch + 1 + start_epoch} Loss: {loss}')
        loss_points.append(loss)
        val = validate(model.superResolution, validloader, criterion_valid, device)
        log_section = 'val'
        print(f'Epoch {epoch + 1 + start_epoch} Validation Loss: {val}')
        valid_points.append(val)
        if wandb_logger is not None:
            wandb_logger.log({f"{log_section}/epoch": epoch + 1 + start_epoch},
                {f"{log_section}/avg_error": val},
                {f"{log_section}/avg_loss": loss})
        if (epoch + 1) % save_every == 0:
            torch.save(model.superResolution.state_dict(), f'models/{save_name}_{epoch + 1 + start_epoch}.pth')
