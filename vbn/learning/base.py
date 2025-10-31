from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from vbn.utils import Tensor


class BaseCPD(nn.Module):
    """Torch-wrapped CPD base.

    Conventions
    - parents: Dict[parent_name, Tensor] with matching batch size N
    - y: Tensor target values, shape (N, out_dim) or discrete as (N,) long
    - forward(parents) returns distribution *parameters* for convenience
    - log_prob(y, parents) returns per-sample log-likelihood (N,)
    - sample(parents, n_samples) returns (N, n_samples, out_dim)

    Learning interfaces
    - fit(data_parents, y): resets/initializes parameters in batch (offline)
    - update(data_parents, y): incremental update (online / partial_fit)
    """

    def __init__(
        self, name: str, parents: Dict[str, int], device: str | torch.device = "cpu"
    ):
        super().__init__()
        self.name = name
        self.parents = parents  # mapping parent->dim/card
        self.device = torch.device(device)
        self.to(self.device)

    # ——— pure likelihood API ———
    def forward(self, parents: Dict[str, Tensor]):
        raise NotImplementedError

    def log_prob(self, y: Tensor, parents: Dict[str, Tensor]) -> Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def sample(self, parents: Dict[str, Tensor], n_samples: int) -> Tensor:
        raise NotImplementedError

    # ——— learning API ———
    @torch.no_grad()
    def fit(self, parents: Dict[str, Tensor], y: Tensor) -> None:
        """(Re)initialize parameters from scratch using provided batch."""
        pass

    @torch.no_grad()
    def update(self, parents: Dict[str, Tensor], y: Tensor, alpha: float = 1.0) -> None:
        """Online/partial update. Default: call fit if alpha>=1, else EMA where possible."""
        self.fit(parents, y)
