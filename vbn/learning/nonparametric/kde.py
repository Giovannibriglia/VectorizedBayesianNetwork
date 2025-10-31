from __future__ import annotations

import math
from typing import Dict

import torch

from torch import nn
from torch.nn import functional as F

from ..base import BaseCPD, Tensor


def _cat_parents(parents: Dict[str, Tensor]) -> Tensor | None:
    if not parents:
        return None
    xs = []
    for t in parents.values():
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        xs.append(t)
    return torch.cat(xs, dim=-1)


def _logsumexp(x: Tensor, dim=-1):
    m = x.max(dim=dim, keepdim=True).values
    return (x - m).exp().sum(dim=dim).log() + m.squeeze(dim)


class KDECPD(BaseCPD):
    def __init__(
        self,
        name: str,
        parents: Dict[str, int],
        y_dim: int,
        buffer_size: int = 6000,
        device: str | torch.device = "cpu",
    ):
        super().__init__(name, parents, device)
        self.x_dim = int(sum(parents.values())) if parents else 0
        self.y_dim = int(y_dim)
        self.buffer_size = int(buffer_size)
        self.register_buffer(
            "xy_buf", torch.empty(0, self.x_dim + self.y_dim, device=self.device)
        )
        self.h_un = nn.Parameter(
            torch.zeros(self.x_dim + self.y_dim, device=self.device)
        )

    @property
    def h(self) -> Tensor:
        return F.softplus(self.h_un) + 1e-6

    def _append(self, X: Tensor, Y: Tensor):
        XY = torch.cat([X, Y], dim=-1) if self.x_dim else Y
        if self.xy_buf.numel() == 0:
            self.xy_buf = XY[-self.buffer_size :].detach()
        else:
            self.xy_buf = torch.cat([self.xy_buf, XY], 0)[-self.buffer_size :].detach()

    @torch.no_grad()
    def fit(self, parents: Dict[str, Tensor], y: Tensor) -> None:
        X = (
            _cat_parents(parents)
            if self.x_dim
            else torch.empty((y.shape[0], 0), device=self.device)
        )
        if self.x_dim:
            X = X.to(self.device).float()
        self._append(X, y.to(self.device))

    @torch.no_grad()
    def update(self, parents: Dict[str, Tensor], y: Tensor, alpha: float = 1.0) -> None:
        self.fit(parents, y)

    def _log_kde(self, Z: Tensor, points: Tensor, h: Tensor) -> Tensor:
        d = Z.shape[-1]
        Z = Z.unsqueeze(1)
        dif = (Z - points.unsqueeze(0)) / h
        q = -0.5 * (dif**2).sum(-1)
        norm = -(d / 2) * math.log(2 * math.pi) - h.log().sum()
        return _logsumexp(q, 1) + norm - math.log(points.shape[0] + 1e-8)

    def log_prob(self, y: Tensor, parents: Dict[str, Tensor]) -> Tensor:
        X = (
            _cat_parents(parents)
            if self.x_dim
            else torch.empty((y.shape[0], 0), device=self.device)
        )
        if self.x_dim:
            X = X.to(self.device).float()
        if self.xy_buf.numel() == 0:
            return torch.full((y.shape[0],), -math.inf, device=self.device)
        h = self.h
        xy = (
            torch.cat([X, y.to(self.device)], dim=-1)
            if self.x_dim
            else y.to(self.device)
        )
        log_joint = self._log_kde(xy, self.xy_buf, h)
        log_marg_x = (
            self._log_kde(X, self.xy_buf[:, : self.x_dim], h[: self.x_dim])
            if self.x_dim
            else torch.zeros_like(log_joint)
        )
        return log_joint - log_marg_x

    @torch.no_grad()
    def sample(self, parents: Dict[str, Tensor], n_samples: int) -> Tensor:
        # Conditional sampling: p(y|x) as a KDE mixture with weights from x-kernel
        N = next(iter(parents.values())).shape[0] if parents else 1
        if self.xy_buf.numel() == 0:
            return torch.zeros(N, n_samples, self.y_dim, device=self.device)

        # Split buffer
        X_buf = (
            self.xy_buf[:, : self.x_dim]
            if self.x_dim
            else torch.empty((self.xy_buf.shape[0], 0), device=self.device)
        )
        Y_buf = self.xy_buf[:, self.x_dim :]

        if self.x_dim:
            X = _cat_parents(parents).to(self.device).float()  # (N, x_dim)
            # Compute kernel weights w_i(x) ∝ exp(-0.5 * ||(x - x_i)/h_x||^2)
            hx = self.h[: self.x_dim]  # (x_dim,)
            Xn = (X.unsqueeze(1) - X_buf.unsqueeze(0)) / hx  # (N, M, x_dim)
            logw = -0.5 * (Xn**2).sum(-1)  # (N, M)
            w = torch.softmax(logw, dim=-1)  # (N, M)
            # For each x_n, sample component indices according to w_n
            idx = torch.multinomial(
                w, num_samples=n_samples, replacement=True
            )  # (N, n_samples)
            # Gather reference Y_i for each sampled component
            Y_ref = Y_buf[idx]  # (N, n_samples, y_dim)
        else:
            # No parents → unconditional: pick random buffer rows
            M = self.xy_buf.shape[0]
            idx = torch.randint(0, M, (N, n_samples), device=self.device)
            Y_ref = Y_buf[idx]

        # Add kernel noise in Y-space
        hy = self.h[self.x_dim :]  # (y_dim,)
        eps = torch.randn_like(Y_ref) * hy
        return Y_ref + eps
