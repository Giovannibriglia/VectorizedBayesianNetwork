from __future__ import annotations

import math

from typing import Dict, Tuple

import torch
from torch import nn

from vbn.learning.base import BaseCPD
from vbn.learning.utils import _cat_parents

from vbn.utils import Tensor


class LinearGaussianCPD(BaseCPD):
    # ─────────────────────────────────────────────────────────────────────────────
    # Linear-Gaussian CPD: y | x ~ N(Wx + b, diag(exp(log_var)))
    # Continuous parents (concatenated); y can be multi-dim
    # ─────────────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        name: str,
        parents: Dict[str, int],
        out_dim: int,
        device: str | torch.device = "cpu",
    ):
        super().__init__(name, parents, device)
        D = int(sum(parents.values())) if len(parents) else 0
        self.W = nn.Linear(D if D > 0 else 1, out_dim, bias=True).to(self.device)
        self.log_var = nn.Parameter(
            torch.full((out_dim,), math.log(0.1), device=self.device)
        )

    def forward(self, parents: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        X = _cat_parents(parents)
        if X is None:
            X = torch.ones(
                (next(iter(parents.values())).shape[0] if parents else 1, 1),
                device=self.device,
            )
        mu = self.W(X)
        var = self.log_var.exp().clamp_min(1e-8)
        return mu, var

    def log_prob(self, y: Tensor, parents: Dict[str, Tensor]) -> Tensor:
        mu, var = self.forward(parents)
        return -0.5 * ((y.to(self.device) - mu) ** 2 / var).sum(-1) - 0.5 * (
            var.log().sum() + y.shape[-1] * math.log(2 * math.pi)
        )

    @torch.no_grad()
    def fit(self, parents: Dict[str, Tensor], y: Tensor) -> None:
        # closed-form OLS for W, then set variance to residual var
        X = _cat_parents(parents)
        if X is None:
            X = torch.ones((y.shape[0], 1), device=self.device)
        X_ = torch.cat([X, torch.ones_like(X[:, :1])], dim=-1)  # add bias
        # theta = (X^T X)^-1 X^T Y
        XtX = X_.T @ X_
        XtY = X_.T @ y.to(self.device)
        theta, _ = (
            torch.solve(XtY, XtX)
            if hasattr(torch, "solve")
            else (XtX.inverse() @ XtY, None)
        )
        W, b = theta[:-1].T, theta[-1]
        self.W.weight.data.copy_(W)
        self.W.bias.data.copy_(b)
        mu = X @ self.W.weight.data.T + self.W.bias.data
        resid = y.to(self.device) - mu
        var = resid.pow(2).mean(dim=0).clamp_min(1e-6)
        self.log_var.data.copy_(var.log())

    @torch.no_grad()
    def update(self, parents: Dict[str, Tensor], y: Tensor, alpha: float = 0.1) -> None:
        # single gradient step on NLL (small online adaptation)
        self.train()
        opt = torch.optim.SGD(self.parameters(), lr=alpha)
        opt.zero_grad(set_to_none=True)
        nll = -self.log_prob(y, parents).mean()
        nll.backward()
        opt.step()

    @torch.no_grad()
    def sample(self, parents: Dict[str, Tensor], n_samples: int) -> Tensor:
        mu, var = self.forward(parents)
        eps = torch.randn((mu.shape[0], n_samples, mu.shape[-1]), device=self.device)
        return mu.unsqueeze(1) + eps * var.sqrt().unsqueeze(0)
