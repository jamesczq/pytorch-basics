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
