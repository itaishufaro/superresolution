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
    wandb.login(key=WANDB_KEY)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(.99, 0)
    print(f'Using device: {device}')
    sweep_config = {
        'method': 'bayes',
        'name': 'superres',
        'metric': {
            'name': 'val_loss',
            'goal': 'maximize'
        },
        'parameters': {
            'lr': {'max': 1e-3, 'min': 1e-5},
            'step_size': {'max': 20, 'min': 5},
            'gamma': {'max': 0.9, 'min': 0.1},
            'weight_decay': {'max': 0.0001, 'min': 0.00001},
            'momentum': {'max': 0.9, 'min': 0.1},
            'drop_prob': {'max': 0.2, 'min': 0.0},
            'train_dir': {'value': 'trainSAR'},
            'alpha': {'max': 1e-3, 'min': 1e-6},
        }}
    sweep_id = wandb.sweep(sweep_config, project="gan_superres", entity=WANDB_ENTITY)
    wandb.agent(sweep_id, train.hyperparameter_search, count=1000)




