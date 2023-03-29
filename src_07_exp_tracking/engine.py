"""
Contains functions for training/testing a PyTorch model.
"""
import os
import torch

from datetime import datetime
from tqdm.auto import tqdm
from typing import Dict, List, Tuple


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Trains a PyTorch model per epoch.

    Sets a target PyTorch model to "train" mode and then steps through the
    forward/backward pass: forward, loss, loss backward, optim i.e. gradient descent.

    Args:
        ***

    Returns:
        A tuple of (training loss, training accuracy), e.g., (0.1111, 0.8765)
    """
    model.train()

    train_loss, train_acc = 0.0, 0.0

    # Loop through the batches in given DataLoader
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Forward
        y_pred = model(X)

        # Compute loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # Loss backward
        optimizer.zero_grad()
        loss.backward()

        # Grad descent
        optimizer.step()

        # Metrics
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y)

    # Modify accumulated (over all batches) metrics to be avg metrics per batch
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Tests a PyTorch model per epoch.

    Sets a target PyTorch model to "eval" mode and then perform the
    forward step.

    Args:
        ***

    Returns:
        A tuple of (test loss, test accuracy), e.g., (0.1111, 0.8765)
    """
    model.eval()

    test_loss, test_acc = 0.0, 0.0

    with torch.inference_mode():
        # Loop through the batches in given DataLoader
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Forward
            test_pred_logits = model(X)

            # Compute loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Compute accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

        # Modify accumulated metrics to be avg metrics per batch
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)

        return test_loss, test_acc


def create_writer(
    experiment_name: str, model_name: str, extra: str
) -> torch.utils.tensorboard.writer.SummaryWriter():
    """
    Create a tensorboard SummaryWriter instance to save to special-named log_dir as follows:
    runs/YYYY-MM-DD/experiment_name/model_name/extra.

    Args:
     ***

    Returns:
     Instance of torch.utils.tensorboard.writer.SummaryWriter()

    Example:
        ***
    """
    time_stamp = datetime.now().strftime("%Y-%m-%d")

    if extra:
        log_dir = os.path.join("runs", time_stamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", time_stamp, experiment_name, model_name)

    print(f"[INFO] Created SummaryWriter: saving to {log_dir}...")

    return torch.utils.tensorboard.writer.SummaryWriter(log_dir=log_dir)


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
    writer: torch.utils.tensorboard.SummaryWriter,
) -> Dict[str, List]:
    """
    Defines the training loop over multiple epochs.

    Args:
        ***

    Returns:
        A dict of {
        train_loss: [...], # List constains values over all epochs
        train_acc: [...],
        test_loss: [...],
        test_acc: [...]}
    """
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        test_loss, test_acc = test_step(
            model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device
        )

        print(
            f"Epoch {epoch + 1}:",
            f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} | ",
            f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}",
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        # New: add experiment tracking device: tensorboard.SummaryWriter()
        writer.add_scalars(
            main_tag="Loss",
            tag_scalar_dict={"train_loss": train_loss, "test_loss": test_loss},
            global_step=epoch,
        )

        writer.add_scalars(
            main_tag="Accuracy",
            tag_scalar_dict={"train_acc": train_acc, "test_acc": test_acc},
            global_step=epoch,
        )

        # Tracking the Pytorch model architecture
        writer.add_graph(
            model=model,
            # Pass in an example input
            input_to_model=torch.randn(32, 3, 224, 224).to(device),
        )

    print(f"\n>>>Tensorboard logs saved in {writer.log_dir}\n")

    writer.close()

    return results
