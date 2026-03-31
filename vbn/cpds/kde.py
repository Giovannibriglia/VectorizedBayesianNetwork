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
        self._chunk_size = 512
        self._parents: Optional[torch.Tensor] = None
        self._targets: Optional[torch.Tensor] = None

    def get_init_kwargs(self) -> dict:
        return {
            "bandwidth": self.bandwidth,
            "parent_bandwidth": self.parent_bandwidth,
            "max_points": self.max_points,
            "min_scale": self.min_scale,
        }

    def get_extra_state(self) -> Optional[dict]:
        return {"parents": self._parents, "targets": self._targets}

    def set_extra_state(self, state: Optional[dict]) -> None:
        if not state:
            self._parents = None
            self._targets = None
            return
        parents = state.get("parents")
        targets = state.get("targets")
        self._parents = parents.to(self.device) if parents is not None else None
        self._targets = targets.to(self.device) if targets is not None else None

    def _check_fitted(self) -> None:
        if self._targets is None:
            raise RuntimeError("KDECPD is not fitted yet.")

    def _limit_points(
        self, parents: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = targets.shape[0]
        if n <= self.max_points:
            return parents, targets
        idx = torch.randperm(n, device=targets.device)[: self.max_points]
        return parents[idx], targets[idx]

    def fit(self, parents: Optional[torch.Tensor], x: torch.Tensor, **kwargs) -> None:
        if parents is None:
            parents = torch.zeros(x.shape[0], 0, device=self.device)
        parents = ensure_2d(parents)
        targets = ensure_2d(x)
        parents, targets = self._limit_points(parents, targets)
        self._parents = parents
        self._targets = targets
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
        parents_all = torch.cat([self._parents, parents], dim=0)
        targets_all = torch.cat([self._targets, x], dim=0)
        parents_all, targets_all = self._limit_points(parents_all, targets_all)
        self._parents = parents_all
        self._targets = targets_all

    def _kernel_log_prob(self, diff: torch.Tensor, bandwidth: float) -> torch.Tensor:
        scale = max(float(bandwidth), 1e-3) + self.min_scale
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
        flat_x = x.reshape(b * s, dx)
        flat_parents = None
        if self.input_dim != 0:
            if parents is None:
                raise ValueError("parents cannot be None when input_dim > 0")
            parents = broadcast_samples(parents, s)
            flat_parents = parents.reshape(b * s, self.input_dim)

        chunks = []
        for start in range(0, flat_x.shape[0], self._chunk_size):
            end = min(start + self._chunk_size, flat_x.shape[0])
            x_chunk = flat_x[start:end]
            diff_y = x_chunk.unsqueeze(1) - targets.unsqueeze(0)
            log_ky = self._kernel_log_prob(diff_y, self.bandwidth).sum(dim=-1)
            if self.input_dim == 0:
                log_prob_chunk = torch.logsumexp(log_ky, dim=1) - math.log(float(n))
            else:
                parents_chunk = flat_parents[start:end]
                diff_p = parents_chunk.unsqueeze(1) - self._parents.unsqueeze(0)
                log_kp = self._kernel_log_prob(diff_p, self.parent_bandwidth).sum(
                    dim=-1
                )
                log_prob_chunk = torch.logsumexp(
                    log_kp + log_ky, dim=1
                ) - torch.logsumexp(log_kp, dim=1)
            chunks.append(log_prob_chunk)
        log_prob = torch.cat(chunks, dim=0).reshape(b, s)
        return log_prob

    def sample(self, parents: Optional[torch.Tensor], n_samples: int) -> torch.Tensor:
        self._check_fitted()
        targets = self._targets
        n = targets.shape[0]
        b = 1 if parents is None else parents.shape[0]
        flat_out = torch.empty(
            b * n_samples, self.output_dim, device=targets.device, dtype=targets.dtype
        )
        flat_parents = None
        if self.input_dim != 0:
            if parents is None:
                raise ValueError("parents cannot be None when input_dim > 0")
            parents = broadcast_samples(parents, n_samples)
            flat_parents = parents.reshape(b * n_samples, self.input_dim)

        bw = max(float(self.bandwidth), 1e-3)
        for start in range(0, flat_out.shape[0], self._chunk_size):
            end = min(start + self._chunk_size, flat_out.shape[0])
            if self.input_dim == 0:
                idx = torch.randint(0, n, (end - start,), device=targets.device)
            else:
                parents_chunk = flat_parents[start:end]
                diff_p = parents_chunk.unsqueeze(1) - self._parents.unsqueeze(0)
                log_kp = self._kernel_log_prob(diff_p, self.parent_bandwidth).sum(
                    dim=-1
                )
                weights = torch.softmax(log_kp, dim=-1)
                idx = torch.multinomial(weights, num_samples=1).squeeze(-1)
            selected = targets[idx]
            noise = torch.randn_like(selected) * (bw + self.min_scale)
            flat_out[start:end] = selected + noise
        return flat_out.reshape(b, n_samples, self.output_dim)
