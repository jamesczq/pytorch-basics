"""
Contains various utility functions for PyTorch model training and saving.
"""

import pathlib
import platform
import torch


def get_device() -> str:
    myplatform = platform.system().lower()
    if myplatform == "windows":
        return "cuda" if torch.cuda.is_available() else "cpu"
    elif myplatform in {"mac", "darwin"}:
        return "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        return "cpu"


def save_model(model: torch.nn.Module, target_dir: str, model_name: str):
    """
    Saves a PyTorch model to disk at target directory.

    Args:
        ***

    Example:
        save_model(model0, "models/", "05_modular_tiny_vgg.pt")
    """
    target_dir_path = pathlib.Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    model_save_path = target_dir_path / model_name

    print(f"[INFO] Saving model to {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)
