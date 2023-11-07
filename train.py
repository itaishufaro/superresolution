import pandas as pd
from torch.utils.data import DataLoader
import torchvision.transforms as T

import gan
import models
from dataset import StuffDataset
import torch
from torch import nn
import matplotlib.pyplot as plt
import wandb
import os
import numpy as np
import logging
import datetime
from torchvision.models import resnet50, ResNet50_Weights
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from sklearn.model_selection import train_test_split
import dataset
WANDB_KEY = "01717b5e711e2653d9cc50175f88588ce40619df"
WANDB_ENTITY = "itai-shufaro"

# SSIM = StructuralSimilarityIndexMeasure().to(device='cuda')
SEED = 1234
def total_variation(img):
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).sum()
    tv_w = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).sum()
    return (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)


def PSNR(img1, img2):
    x = torch.pow(img1 - img2, 2).mean()
    return 10 * torch.log10(1 / x)


def GAN_loss(fake_pred, real_pred):
    fake_loss = torch.log(1 - fake_pred).mean()
    real_loss = torch.log(real_pred).mean()
    return fake_loss + real_loss

def train_one_epoch(model, dataloader, optimizer, criterion, device, aug=None,
                    wandb_logger=None, epoch=0, perceptual_loss=False,
                    PerceptualLoss=None, save_every=2000, lam=0.01):
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
        # low_res = low_res.view(low_res.shape[0], 3, 64, 64)
        # high_res = high_res.view(high_res.shape[0], 3, 256, 256)
        optimizer.zero_grad()
        out_real = high_res
        out_train = model(low_res)
        tv = total_variation(out_train)
        if perceptual_loss:
            out_real = out_real.repeat(1, 3, 1, 1)
            out_train = out_train.repeat(1, 3, 1, 1)
            loss = PerceptualLoss(out_train, out_real)
        else:
            loss = criterion(out_train, out_real)
        loss -= lam * tv
        loss.retain_grad()
        loss.backward()
        if i % 100 == 0:
            # print(loss.is_leaf)
            print(f'Batch {i}/{len_dataloader} Loss: {loss.item()} Grad: {loss.grad}')
        optimizer.step()
        if wandb_logger is not None:
            wandb_logger.log({f"{log_section}/episode": len_dataloader * epoch + i},
                {f"{log_section}/lr": optimizer.param_groups[0]['lr']},
                {f"{log_section}/avg_loss": loss.item()})
        tot_loss += loss.item()
        i += 1
        if i % save_every == 0:
            torch.save(model.state_dict(), f'models/train_epoch_{epoch}_iter_{i}.pth')
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
    x = 0
    for lr, hr in iter(dataloader):
        hr_model = model(lr.to(device))
        j = hr.to(device)
        eval_loss += criterion(hr_model, j).item()
        # x += SSIM(hr_model, j)
        torch.cuda.empty_cache()
    # print(f"SSIM: {x/len(dataloader)}")
    return eval_loss / len(dataloader)


def train_epochs(num_epochs, model, trainloader, validloader, optimizer, scheduler,
                 criterion_train, criterion_valid, device,
                 aug=None, start_epoch=0, save_every=10, save_name='model', wandb_logger=None,
                 perceptual_loss=False):
    '''

    :param num_epochs:
    :param model:
    :param trainloader:
    :param validloader:
    :param optimizer:
    :param scheduler:
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
    Per = None
    if perceptual_loss:
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
        resnet = nn.Sequential(*list(resnet.children())[:-1])
        resnet.eval()
        Per = models.PerceptualLoss(feature_extractor=resnet)
    for epoch in range(num_epochs):
        loss = train_one_epoch(model, trainloader, optimizer, criterion_train, device, aug,
                               wandb_logger=wandb_logger, epoch=epoch + start_epoch, perceptual_loss=perceptual_loss,
                               PerceptualLoss=Per)
        loss_points.append(loss)
        val = validate(model, validloader, criterion_valid, device)
        scheduler.step()
        log_section = 'val'
        print(f'Epoch {epoch + 1 + start_epoch} Validation Loss: {val}')
        valid_points.append(val)
        if wandb_logger is not None:
            wandb_logger.log({f"{log_section}/epoch": epoch + 1 + start_epoch},
                {f"{log_section}/avg_error": val},
                {f"{log_section}/avg_loss": loss})
        if (epoch + 1) % save_every == 0:
            torch.save(model.state_dict(), f'models/{save_name}_{epoch + 1 + start_epoch}.pth')
    return loss_points, valid_points


# A function for hyperparameter search
def hyperparameter_search():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wandb.init()
    params = wandb.config
    model = models.SarSubPixel(colors=1, drop_prob=0.1).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'],
                                momentum=params['momentum'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params['step_size'], gamma=params['gamma'],
                                                verbose=True)
    criterion = nn.MSELoss().to(device)
    valid_criterion = PeakSignalNoiseRatio().to(device)
    transform = T.Compose([T.ToTensor()])
    SARDataset = dataset.StuffDataset(params['train_dir'], transforms=transform, inputH=256, inputW=256)
    # split the dataset into train, validation and test data loaders
    trainFull_set, test_set = train_test_split(SARDataset, test_size=0.2, random_state=SEED)
    train_set, val_set = train_test_split(trainFull_set, test_size=0.2, random_state=SEED)
    train_loader = DataLoader(train_set, batch_size=512, shuffle=True)
    valid_loader = DataLoader(val_set, batch_size=512, shuffle=True)
    # test_loader = DataLoader(test_set, batch_size=512, shuffle=True)
    loss_points, valid_points = train_epochs(num_epochs=10, model=model,
                                             trainloader=train_loader,
                                             validloader=valid_loader,
                                             optimizer=optimizer, scheduler=scheduler,
                                             criterion_train=criterion,
                                             criterion_valid=valid_criterion, device=device,
                                             aug=None, start_epoch=0,
                                             save_every=10,
                                             save_name=f"model_{params['lr']}_{params['weight_decay']}_{params['momentum']}_{params['step_size']}_{params['gamma']}",
                                             wandb_logger=None,
                                             perceptual_loss=False)
    wandb.log({'val_loss': valid_points[-1]})


# A function for training GAN with alternative training
def train_gan(num_epochs, generator, discriminator, trainloader, testloader, gen_optimizer, disc_optimizer, device,
              aug=None, start_epoch=0, save_every=10, save_name='model', wandb_logger=None,
              alpha=1e-5, gen_scheduler=None, disc_scheduler=None):
    '''

    :param model:
    :param discriminator:
    :param trainloader:
    :param optimizer:
    :param criterion:
    :param device:
    :param aug:
    :param start_epoch:
    :param save_every:
    :return:
    '''
    loss_points = []
    valid_points = []
    criterion_train = nn.MSELoss().to(device)
    criterion_valid = PeakSignalNoiseRatio().to(device)
    for epoch in range(start_epoch, start_epoch + num_epochs):
        for lr, hr in iter(trainloader):
            lr = lr.to(device)
            hr = hr.to(device)
            # Train the discriminator
            disc_optimizer.zero_grad()
            fake_hr = generator(lr)
            disc_fake = discriminator(fake_hr.detach())
            disc_real = discriminator(hr)
            disc_loss = GAN_loss(disc_fake, disc_real)
            disc_loss.backward()
            disc_optimizer.step()
            # Train the generator
            gen_optimizer.zero_grad()
            fake_hr = generator(lr)
            disc_fake = discriminator(fake_hr)
            disc_real = discriminator(hr)
            gen_loss = criterion_train(fake_hr, hr) - alpha * GAN_loss(disc_fake, disc_real)
            gen_loss.backward()
            gen_optimizer.step()
            torch.cuda.empty_cache() # empty cache to avoid memory leak
            loss_points.append(gen_loss.item())
        val = validate(generator, testloader, criterion_valid, device)
        valid_points.append(val)
        if gen_scheduler is not None:
            gen_scheduler.step()
        if disc_scheduler is not None:
            disc_scheduler.step()
        print(f'Epoch {epoch + 1} Validation Loss: {val}')
        if wandb_logger is not None:
            wandb_logger.log({"val/epoch": epoch + 1},
                {"val/avg_error": val})
        if (epoch + 1) % save_every == 0:
            torch.save(generator.state_dict(), f'models/{save_name}_{epoch + 1}.pth')
    return loss_points, valid_points

def gan_hyperparameter():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wandb.init()
    params = wandb.config
    generator = gan.Generator().to(device)
    discriminator = gan.Discriminator().to(device)
    gen_optimizer = torch.optim.SGD(generator.parameters(), lr=params['lr'], weight_decay=params['weight_decay'],
                                momentum=params['momentum'])
    disc_optimizer = torch.optim.SGD(discriminator.parameters(), lr=params['lr'], weight_decay=params['weight_decay'],
                                momentum=params['momentum'])
    gen_scheduler = torch.optim.lr_scheduler.StepLR(gen_optimizer, step_size=params['step_size'], gamma=params['gamma'])
    disc_scheduler = torch.optim.lr_scheduler.StepLR(disc_optimizer, step_size=params['step_size'], gamma=params['gamma'])
    transform = T.Compose([T.ToTensor()])
    SARDataset = dataset.StuffDataset(params['train_dir'], transforms=transform, inputH=256, inputW=256)
    # split the dataset into train, validation and test data loaders
    trainFull_set, test_set = train_test_split(SARDataset, test_size=0.2, random_state=SEED)
    train_set, val_set = train_test_split(trainFull_set, test_size=0.2, random_state=SEED)
    train_loader = DataLoader(train_set, batch_size=512, shuffle=True)
    valid_loader = DataLoader(val_set, batch_size=512, shuffle=True)
    # test_loader = DataLoader(test_set, batch_size=512, shuffle=True)
    loss_points, valid_points = train_gan(num_epochs=15, generator=generator,
                    discriminator=discriminator, trainloader=train_loader,
                    testloader=train_loader, gen_optimizer=gen_optimizer,
                    disc_optimizer=disc_optimizer, device=device,
                    aug=None, start_epoch=0, save_every=15,
                    save_name=f"model_{params['lr']}_{params['weight_decay']}_{params['momentum']}_{params['step_size']}_{params['gamma']}",
                    wandb_logger=None,
                    alpha=params['alpha'], gen_scheduler=gen_scheduler, disc_scheduler=disc_scheduler)
    wandb.log({'val_loss': valid_points[-1]})


