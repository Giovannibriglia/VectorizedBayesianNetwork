from __future__ import annotations

import torch


def safe_softplus(x: torch.Tensor, min_val: float = 1e-4) -> torch.Tensor:
    return torch.nn.functional.softplus(x) + float(min_val)


def stable_log(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.log(x.clamp_min(float(eps)))


def normalize_probs(p: torch.Tensor) -> torch.Tensor:
    return p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
