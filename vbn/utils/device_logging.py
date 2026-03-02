import os
from typing import Optional

import torch


def get_device_string(device: Optional[torch.device] = None) -> str:
    """
    Human-readable torch device description.
    If `device` is None, infers best-available (cuda if available, else cpu).
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda":
        return "cpu"

    idx = device.index if device.index is not None else 0
    name = torch.cuda.get_device_name(idx)
    total_mem_gb = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
    return f"cuda:{idx} ({name}, {total_mem_gb:.1f}GB)"


def log_device(
    logger,
    phase: str,
    device: Optional[torch.device] = None,
    once_key: Optional[str] = None,
) -> None:
    """
    Log the device used for computations, optionally only once per process for a given once_key.
    Uses env var guard to avoid duplicate spam across phases if desired.
    """
    if once_key is not None:
        env_key = f"VBN_LOGGED_DEVICE_{once_key.upper()}"
        if os.environ.get(env_key) == "1":
            return
        os.environ[env_key] = "1"

    dev_str = get_device_string(device)
    msg = (
        f"[vbn] Phase={phase} | Device: {dev_str} | "
        f"torch={torch.__version__} | cuda_available={torch.cuda.is_available()}"
    )
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)
