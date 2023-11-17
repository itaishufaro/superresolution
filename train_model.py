from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch
import wandb
import argparse
import train
import dataset
import gan


def parse_args():
    """
    The parse_args function is a command line argument parser that returns a namespace containing arguments from the
    command line.
    The arguments include:

    1. '--epochs': An integer that defines the number of epochs for training. The default value is 100.
    2. '--batch_size': An integer that sets the batch size for training. Its default value is 16.
    3. '--learning_rate': A float that sets the learning rate for the optimizer. Its default value is 0.001.
    4. '--upscale_factor': An integer that defines the upscaling factor for super-resolution with only supports 2 or 4
    as values. Its default value is 2.
    5. '--train_folder': A string that specifies the path to the training dataset.
    6. '--val_folder': A string that specifies the path to the validation dataset.
    7. '--wandb_key' : Your WANDB key for logging training process.
    8. '--wandb_entity' : Your WANDB entity where the project is located.
    9. '--use_logger' : Whether to use WANDB logger for training. Default is False.
    Returns:
        args: A namespace that contains these command line argument values.
    """
    par = argparse.ArgumentParser(description="Train a super-resolution model")
    par.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    par.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    par.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer.')
    par.add_argument('--upscale_factor', type=int, default=2, help='Upscaling factor for super-resolution \
                        (currently supports only 2 or 4).')
    par.add_argument('--train_folder', type=str, default='train/',
                     help='Path to the training dataset.')
    par.add_argument('--val_folder', type=str, default='val/',
                     help='Path to the validation dataset.')

    par.add_argument('--wandb_key', type=str, help='Your WANDB key for logging training process.')
    par.add_argument('--wandb_entity', type=str, help='Your WANDB entity where the project is located.')
    par.add_argument('--use_logger', type=bool, default=False,
                     help='Whether to use WANDB logger for training. Default is False.')

    args = par.parse_args()
    return args


if __name__ == '__main__':
    # Parse the arguments
    parser = parse_args()

    # Initialize logging, if required
    wandb_logger = None
    if parser.use_logger:
        wandb.login(key=parser.wandb_key)
        wandb_logger = wandb.init(project="gan_superres", entity=parser.wandb_entity)

    # Setup for GPU, if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(.99, 0)

    print(f'Using device: {device}')

    # Create Generator and Discriminator instances
    generator = gan.Generator(scale_factor=parser.upscale_factor)
    discriminator = gan.Discriminator()

    # Move the models to GPU/CPU
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # Creating optimizers for generator and DISCRIMINATOR, Adam optimizer is used
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=parser.learning_rate, weight_decay=0.0001)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=parser.learning_rate, weight_decay=0.0001)

    # Initialize learning rate schedulers for the optimizers
    gen_scheduler = torch.optim.lr_scheduler.StepLR(gen_optimizer, step_size=10, gamma=0.5)
    disc_scheduler = torch.optim.lr_scheduler.StepLR(disc_optimizer, step_size=10, gamma=0.5)

    # Compose image transformations, converting images to tensors
    transform = T.Compose([T.ToTensor()])

    # Setup the train and validation datasets
    trainset = dataset.StuffDataset(parser.train_folder, transforms=transform, inputH=256, inputW=256,
                                    scale_factor=parser.upscale_factor)
    valset = dataset.StuffDataset(parser.val_folder, transforms=transform, inputH=256, inputW=256,
                                  scale_factor=parser.upscale_factor)

    # Create data loaders for train and validation datasets
    train_loader = DataLoader(trainset, batch_size=parser.batch_size, shuffle=True)
    valid_loader = DataLoader(valset, batch_size=parser.batch_size, shuffle=True)

    # Start the training process for GAN model
    train.train_gan(num_epochs=parser.epochs, generator=generator, discriminator=discriminator,
                    trainloader=train_loader, testloader=valid_loader, gen_optimizer=gen_optimizer,
                    disc_optimizer=disc_optimizer, device=device, aug=None, start_epoch=0, save_every=1,
                    save_name='SAR_gan', wandb_logger=wandb_logger, alpha=1e-5, gen_scheduler=gen_scheduler,
                    disc_scheduler=disc_scheduler)
