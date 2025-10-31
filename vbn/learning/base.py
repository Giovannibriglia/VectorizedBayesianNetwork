from __future__ import annotations

from typing import Dict

import torch
from torch import nn

Tensor = torch.Tensor


class BaseCPD(nn.Module):
    """Torch CPD base: log_prob(), sample(), fit(), update()."""

    def __init__(
        self, name: str, parents: Dict[str, int], device: str | torch.device = "cpu"
    ):
        super().__init__()
        self.name = name
        self.parents = parents  # parent -> feature dim (or 1 for discrete codes)
        self.device = torch.device(device)
        self.to(self.device)

    def forward(self, parents: Dict[str, Tensor]):
        raise NotImplementedError

    def log_prob(self, y: Tensor, parents: Dict[str, Tensor]) -> Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def sample(self, parents: Dict[str, Tensor], n_samples: int) -> Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def fit(self, parents: Dict[str, Tensor], y: Tensor) -> None:
        pass

    @torch.no_grad()
    def update(self, parents: Dict[str, Tensor], y: Tensor, alpha: float = 1.0) -> None:
        self.fit(parents, y)

    def training_loss(self, y: Tensor, parents: Dict[str, Tensor]) -> Tensor:
        # default: NLL
        return -self.log_prob(y, parents).mean()
