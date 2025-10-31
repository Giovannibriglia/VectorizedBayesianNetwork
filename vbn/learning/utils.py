from __future__ import annotations

from typing import Dict, List

import torch

from vbn.utils import Tensor


def _cat_parents(parents: Dict[str, Tensor]) -> Tensor:
    """Concatenate parent tensors along last dim; assumes batch-first.
    Accepts dict {name: (N, d_i)} or (N,) which will be unsqueezed.
    Returns X of shape (N, D). If parents is empty -> returns None.
    """
    if len(parents) == 0:
        return None
    xs: List[Tensor] = []
    for _, t in parents.items():
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        xs.append(t)
    return torch.cat(xs, dim=-1)


def _stable_logsumexp(x: Tensor, dim: int = -1) -> Tensor:
    m = x.max(dim=dim, keepdim=True).values
    return (x - m).exp().sum(dim=dim).log() + m.squeeze(dim)
