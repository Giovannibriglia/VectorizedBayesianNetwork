from __future__ import annotations

import math
from typing import Dict, Tuple

import torch

from torch import nn

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


class LinearGaussianCPD(BaseCPD):
    """y|x ~ N(Wx+b, diag(sigma^2))"""

    def __init__(
        self,
        name: str,
        parents: Dict[str, int],
        out_dim: int,
        device: str | torch.device = "cpu",
    ):
        super().__init__(name, parents, device)
        D = int(sum(parents.values())) if parents else 0
        self.W = nn.Linear(max(D, 1), out_dim, bias=True).to(self.device)
        self.log_var = nn.Parameter(
            torch.full((out_dim,), math.log(0.1), device=self.device)
        )

    def forward(self, parents: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        X = _cat_parents(parents)
        if X is None:
            N = next(iter(parents.values())).shape[0] if parents else 1
            X = torch.ones((N, 1), device=self.device)
        else:
            X = X.to(self.device).float()
        mu = self.W(X)
        var = self.log_var.exp().clamp_min(1e-8)
        return mu, var

    def log_prob(self, y: Tensor, parents: Dict[str, Tensor]) -> Tensor:
        mu, var = self.forward(parents)
        resid = y.to(self.device) - mu
        return -0.5 * (resid.pow(2) / var).sum(-1) - 0.5 * (
            var.log().sum() + y.shape[-1] * math.log(2 * math.pi)
        )

    @torch.no_grad()
    def fit(self, parents: Dict[str, Tensor], y: Tensor) -> None:
        X = _cat_parents(parents)
        if X is None:
            X = torch.ones((y.shape[0], 1), device=self.device)
        else:
            X = X.to(self.device, non_blocking=True).float().contiguous()
        Xb = torch.cat([X, torch.ones_like(X[:, :1])], dim=-1).contiguous()
        y_ = y.to(self.device, non_blocking=True).float().contiguous()
        XtX = Xb.T @ Xb
        XtY = Xb.T @ y_
        try:
            theta = torch.linalg.solve(XtX, XtY)
        except RuntimeError:
            theta = torch.linalg.pinv(XtX) @ XtY
        W, b = theta[:-1].T, theta[-1]
        self.W.weight.data.copy_(W)
        self.W.bias.data.copy_(b)
        mu = X @ self.W.weight.data.T + self.W.bias.data
        resid = y.to(self.device) - mu
        var = resid.pow(2).mean(dim=0).clamp_min(1e-6)
        self.log_var.data.copy_(var.log())

    @torch.no_grad()
    def update(
        self, parents: Dict[str, Tensor], y: Tensor, lr: float = 1e-2, steps: int = 1
    ) -> None:
        """Small ELBO/NLL step(s) to assimilate a new batch."""
        self.train()
        opt = torch.optim.Adam(self.parameters(), lr=lr)

        with torch.enable_grad():
            for _ in range(int(steps)):
                opt.zero_grad(set_to_none=True)
                loss = -self.log_prob(y, parents).mean()
                loss.backward()
                opt.step()

    @torch.no_grad()
    def sample(self, parents: Dict[str, Tensor], n_samples: int) -> Tensor:
        mu, var = self.forward(parents)
        eps = torch.randn((mu.shape[0], n_samples, mu.shape[-1]), device=self.device)
        return mu.unsqueeze(1) + eps * var.sqrt().unsqueeze(0)
