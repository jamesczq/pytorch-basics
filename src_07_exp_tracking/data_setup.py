"""
Contains functionality for creating PyTorch DataLoaders for 
images classification data.
"""
import os

import torch
import torchvision

NUM_WORKERS = os.cpu_count()
NUM_WORKERS = 0


def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: torchvision.transforms.Compose,
    batch_size: int = 32,
    num_workers: int = NUM_WORKERS,
):
    """
    Creates training and testing DataLoaders.

    Takes in training/testing directory paths and turns them into
    PyTorch Datasets and then DataLoaders.

    Args:

    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names) where
        class_names is a list of target classes.

    Example:
        train_dataloader, test_dataloader, class_names =
         create_dataloaders(
            tr_dir, tst_dir, transform, batch_size, num_workers)
    """
    # Use ImageFolder to create dataset(s)
    train_data = torchvision.datasets.ImageFolder(train_dir, transform=transform)
    test_data = torchvision.datasets.ImageFolder(test_dir, transform=transform)

    class_names = train_data.classes

    # Turn images to dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader, class_names
