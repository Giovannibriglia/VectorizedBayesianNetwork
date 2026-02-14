from __future__ import annotations

import math
from typing import Optional

import torch

from vbn.core.base import BaseCPD
from vbn.core.registry import register_cpd
from vbn.core.utils import broadcast_samples, ensure_2d


@register_cpd("kde")
class KDECPD(BaseCPD):
    """Gaussian KDE CPD with optional conditional weighting on parent space."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        device: torch.device,
        seed: Optional[int] = None,
        bandwidth: float = 1.0,
        parent_bandwidth: Optional[float] = None,
        max_points: int = 1000,
        min_scale: float = 1e-3,
    ) -> None:
        super().__init__(
            input_dim=input_dim, output_dim=output_dim, device=device, seed=seed
        )
        self.bandwidth = float(bandwidth)
        self.parent_bandwidth = (
            float(parent_bandwidth)
            if parent_bandwidth is not None
            else float(bandwidth)
        )
        self.max_points = int(max_points)
        self.min_scale = float(min_scale)
        self._parents: Optional[torch.Tensor] = None
        self._targets: Optional[torch.Tensor] = None

    def _check_fitted(self) -> None:
        if self._targets is None:
            raise RuntimeError("KDECPD is not fitted yet.")

    def fit(self, parents: Optional[torch.Tensor], x: torch.Tensor, **kwargs) -> None:
        if parents is None:
            parents = torch.zeros(x.shape[0], 0, device=self.device)
        self._parents = ensure_2d(parents)
        self._targets = ensure_2d(x)
        if self._parents.shape[0] != self._targets.shape[0]:
            raise ValueError("parents and x must have the same number of rows")

    def update(
        self, parents: Optional[torch.Tensor], x: torch.Tensor, **kwargs
    ) -> None:
        if parents is None:
            parents = torch.zeros(x.shape[0], 0, device=self.device)
        parents = ensure_2d(parents)
        x = ensure_2d(x)
        if self._targets is None:
            self._parents = parents
            self._targets = x
            return
        self._parents = torch.cat([self._parents, parents], dim=0)
        self._targets = torch.cat([self._targets, x], dim=0)
        if self._targets.shape[0] > self.max_points:
            self._parents = self._parents[-self.max_points :]
            self._targets = self._targets[-self.max_points :]

    def _kernel_log_prob(self, diff: torch.Tensor, bandwidth: float) -> torch.Tensor:
        scale = bandwidth + self.min_scale
        return -0.5 * (
            (diff / scale) ** 2 + math.log(2 * math.pi) + 2 * math.log(scale)
        )

    def log_prob(
        self, x: torch.Tensor, parents: Optional[torch.Tensor]
    ) -> torch.Tensor:
        self._check_fitted()
        if x.dim() <= 2:
            x = ensure_2d(x)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        targets = self._targets
        b, s, dx = x.shape
        n = targets.shape[0]
        diff_y = x.unsqueeze(2) - targets.view(1, 1, n, dx)
        log_ky = self._kernel_log_prob(diff_y, self.bandwidth).sum(dim=-1)  # [B,S,N]

        if self.input_dim == 0:
            log_prob = torch.logsumexp(log_ky, dim=2) - math.log(n)
            return log_prob

        if parents is None:
            raise ValueError("parents cannot be None when input_dim > 0")
        parents = broadcast_samples(parents, s)
        diff_p = parents.unsqueeze(2) - self._parents.view(1, 1, n, self.input_dim)
        log_kp = self._kernel_log_prob(diff_p, self.parent_bandwidth).sum(dim=-1)
        log_weights = log_kp
        log_prob = torch.logsumexp(log_weights + log_ky, dim=2) - torch.logsumexp(
            log_weights, dim=2
        )
        return log_prob

    def sample(self, parents: Optional[torch.Tensor], n_samples: int) -> torch.Tensor:
        self._check_fitted()
        targets = self._targets
        n = targets.shape[0]
        if self.input_dim == 0:
            b = 1 if parents is None else parents.shape[0]
            weights = torch.full((b, n_samples, n), 1.0 / n, device=self.device)
        else:
            if parents is None:
                raise ValueError("parents cannot be None when input_dim > 0")
            parents = broadcast_samples(parents, n_samples)
            diff_p = parents.unsqueeze(2) - self._parents.view(1, 1, n, self.input_dim)
            log_kp = self._kernel_log_prob(diff_p, self.parent_bandwidth).sum(dim=-1)
            weights = torch.softmax(log_kp, dim=-1)
        b, s, _ = weights.shape
        idx = torch.multinomial(weights.reshape(b * s, n), num_samples=1).reshape(b, s)
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, self.output_dim)
        selected = torch.gather(
            targets.unsqueeze(0).unsqueeze(0).expand(b, s, -1, -1),
            dim=2,
            index=idx_exp.unsqueeze(2),
        ).squeeze(2)
        noise = torch.randn_like(selected) * (self.bandwidth + self.min_scale)
        return selected + noise
