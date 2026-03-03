from __future__ import annotations

import importlib
import random
import sys
from typing import Iterable, Mapping, Sequence, Tuple

import numpy as np
import torch


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def auto_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def print_env_header(example_name: str, device: str) -> None:
    py_ver = sys.version.split()[0]
    torch_ver = torch.__version__
    print(f"{example_name} | Python {py_ver} | Torch {torch_ver} | Device {device}")


def require_optional(module: str, purpose: str):
    try:
        return importlib.import_module(module)
    except ImportError:
        pkg = module.split(".")[0]
        print(
            f"Optional dependency '{pkg}' is required for {purpose}. "
            f"Install it with `pip install {pkg}`."
        )
        sys.exit(0)


def format_prob(value: float, precision: int = 4) -> str:
    return f"{value:.{precision}f}"


def format_distribution(
    items: Mapping[object, float] | Sequence[Tuple[object, float]],
    precision: int = 4,
) -> str:
    if isinstance(items, Mapping):
        pairs: Iterable[Tuple[object, float]] = [(k, items[k]) for k in sorted(items)]
    else:
        pairs = items
    return ", ".join(f"{k}={v:.{precision}f}" for k, v in pairs)
