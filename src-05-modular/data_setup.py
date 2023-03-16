"""
Contains functionality for creating PyTorch DataLoaders for 
images classification data.
"""
import os

import torch
import torchvision

NUM_WORKERS = os.cpu_count()


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
