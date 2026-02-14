from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Optional, Tuple

import torch


def resolve_device(device: Optional[str | torch.device]) -> torch.device:
    if device is None or (isinstance(device, str) and device.lower() == "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_tensor(
    x, device: torch.device, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


def ensure_2d(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        return x.unsqueeze(-1)
    if x.dim() == 2:
        return x
    raise ValueError(f"Expected 1D or 2D tensor, got shape {tuple(x.shape)}")


def broadcast_samples(x: torch.Tensor, n_samples: int) -> torch.Tensor:
    if x.dim() == 2:
        return x.unsqueeze(1).expand(-1, n_samples, -1)
    if x.dim() == 3:
        return x
    raise ValueError(f"Expected 2D or 3D tensor, got shape {tuple(x.shape)}")


def flatten_samples(x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    if x.dim() != 3:
        raise ValueError(f"Expected 3D tensor, got shape {tuple(x.shape)}")
    b, s, d = x.shape
    return x.reshape(b * s, d), b, s


def unflatten_samples(x: torch.Tensor, b: int, s: int) -> torch.Tensor:
    if x.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got shape {tuple(x.shape)}")
    return x.reshape(b, s, x.shape[-1])


def to_plain_dict(conf) -> dict:
    if isinstance(conf, dict):
        return dict(conf)
    if hasattr(conf, "to_dict") and callable(conf.to_dict):
        return dict(conf.to_dict())
    if hasattr(conf, "as_dict") and callable(conf.as_dict):
        return dict(conf.as_dict())
    if is_dataclass(conf):
        return asdict(conf)
    if hasattr(conf, "model_dump") and callable(conf.model_dump):
        return conf.model_dump()
    raise TypeError(
        f"Unsupported config type '{type(conf).__name__}'. "
        "Expected dict, ConfigItem, dataclass, or pydantic model."
    )
