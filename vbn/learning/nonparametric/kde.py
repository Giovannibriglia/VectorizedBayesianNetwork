from __future__ import annotations

import math
from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F

from vbn.learning.base import BaseCPD
from vbn.learning.utils import _cat_parents, _stable_logsumexp
from vbn.utils import Tensor


class KDECPD(BaseCPD):
    # ─────────────────────────────────────────────────────────────────────────────
    # KDE CPD: Gaussian kernel density on (x, y), then condition via ratio trick
    # For efficiency we implement conditional log p(y|x) using joint KDE and marginal KDE.
    # Buffer holds past samples; bandwidth is learned (single scalar per-dim via softplus).
    # ─────────────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        name: str,
        parents: Dict[str, int],
        y_dim: int,
        buffer_size: int = 5000,
        device: str | torch.device = "cpu",
    ):
        super().__init__(name, parents, device)
        self.x_dim = int(sum(parents.values())) if len(parents) else 0
        self.y_dim = int(y_dim)
        self.buffer_size = int(buffer_size)
        self.register_buffer(
            "xy_buf", torch.empty(0, self.x_dim + self.y_dim, device=self.device)
        )
        # per-dimension unconstrained bandwidth params
        self.h_un = nn.Parameter(
            torch.zeros(self.x_dim + self.y_dim, device=self.device)
        )

    @property
    def h(self) -> Tensor:
        return F.softplus(self.h_un) + 1e-6  # (d,)

    def _append(self, X: Tensor, Y: Tensor):
        XY = torch.cat([X, Y], dim=-1) if self.x_dim else Y
        if self.xy_buf.numel() == 0:
            self.xy_buf = XY[-self.buffer_size :].detach()
        else:
            self.xy_buf = torch.cat([self.xy_buf, XY], dim=0)[
                -self.buffer_size :
            ].detach()

    @torch.no_grad()
    def fit(self, parents: Dict[str, Tensor], y: Tensor) -> None:
        X = (
            _cat_parents(parents)
            if self.x_dim
            else torch.empty((y.shape[0], 0), device=self.device)
        )
        self._append(X.to(self.device), y.to(self.device))

    @torch.no_grad()
    def update(self, parents: Dict[str, Tensor], y: Tensor, alpha: float = 1.0) -> None:
        self.fit(parents, y)

    def _log_kde(self, Z: Tensor, points: Tensor, h: Tensor) -> Tensor:
        # Gaussian product kernel: log sum_i N(z | points_i, diag(h^2))
        # Z: (N, d), points: (M, d), returns (N,)
        d = Z.shape[-1]
        Z = Z.unsqueeze(1)  # (N,1,d)
        dif = (Z - points.unsqueeze(0)) / h
        q = -0.5 * (dif**2).sum(-1)  # (N, M)
        norm = -(d / 2) * math.log(2 * math.pi) - h.log().sum()
        return _stable_logsumexp(q, dim=1) + norm - math.log(points.shape[0] + 1e-8)

    def log_prob(self, y: Tensor, parents: Dict[str, Tensor]) -> Tensor:
        X = (
            _cat_parents(parents)
            if self.x_dim
            else torch.empty((y.shape[0], 0), device=self.device)
        )
        # log p(x, y) - log p(x)
        if self.xy_buf.numel() == 0:
            return torch.full((y.shape[0],), -math.inf, device=self.device)
        h = self.h
        xy = (
            torch.cat([X, y.to(self.device)], dim=-1)
            if self.x_dim
            else y.to(self.device)
        )
        log_joint = self._log_kde(xy, self.xy_buf, h)
        if self.x_dim:
            log_marg_x = self._log_kde(X, self.xy_buf[:, : self.x_dim], h[: self.x_dim])
        else:
            log_marg_x = torch.zeros_like(log_joint)
        return log_joint - log_marg_x

    @torch.no_grad()
    def sample(self, parents: Dict[str, Tensor], n_samples: int) -> Tensor:
        # conditional sampling via nearest neighbors + noise
        X = (
            _cat_parents(parents)
            if self.x_dim
            else torch.empty(
                (len(next(iter(parents.values()))) if parents else 1, 0),
                device=self.device,
            )
        )
        N = (
            X.shape[0]
            if self.x_dim
            else (next(iter(parents.values())).shape[0] if parents else 1)
        )
        if self.xy_buf.numel() == 0:
            return torch.zeros(N, n_samples, self.y_dim, device=self.device)
        # naive: pick random buffer rows, add Gaussian noise with bandwidth for y part
        idx = torch.randint(0, self.xy_buf.shape[0], (N, n_samples), device=self.device)
        Y_ref = self.xy_buf[idx, self.x_dim :]
        eps = torch.randn_like(Y_ref) * self.h[self.x_dim :]
        return Y_ref + eps
