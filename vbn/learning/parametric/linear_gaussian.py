# vbn/learning/parametric/linear_gaussian.py
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

Tensor = torch.Tensor


# helper
def _stack_X(parents: Dict[str, Tensor], order: List[str]) -> Tensor:
    """Concatenate parent tensors in a fixed order as features (N, d).
    Each parent tensor may be (N,) or (N, d_i). We cast to float."""
    if not order:
        N = next(iter(parents.values())).shape[0] if parents else 0
        dev = next(iter(parents.values())).device if parents else None
        return torch.empty((N, 0), device=dev)
    cols = []
    dev = None
    for p in order:
        x = parents[p]
        dev = x.device if dev is None else dev
        if x.dim() == 1:
            x = x.unsqueeze(1)
        # ⬇️ ensure float dtype for linear algebra (handles Long discrete parents)
        x = x.to(dtype=torch.get_default_dtype())
        cols.append(x)
    return torch.cat(cols, dim=1)


class LinearGaussianCPD(nn.Module):
    """
    Scalar linear-Gaussian CPD:
        y = X W + b + eps,   eps ~ N(0, sigma^2)
    - parents: Dict[parent_name, feature_dim]  (dims are summed into in_dim)
    - out_dim must be 1 (scalar); raise otherwise
    Exposes attributes used by GaussianExact: .W, .b, .sigma2
    """

    def __init__(
        self,
        name: str,
        parents: Dict[str, int],
        out_dim: int = 1,
        device=None,
        **kwargs,
    ):
        super().__init__()
        if out_dim != 1:
            raise ValueError(
                f"LinearGaussianCPD supports scalar output only (got out_dim={out_dim})."
            )
        self.name = name
        self.parents_meta = dict(parents)  # {parent_name: dim}
        self.parent_order = list(parents.keys())
        self.in_dim = int(sum(int(d) for d in parents.values()))
        dev = torch.device(device) if device is not None else torch.device("cpu")

        # Parameters
        self.W = (
            nn.Parameter(torch.zeros(self.in_dim, device=dev))
            if self.in_dim > 0
            else nn.Parameter(torch.zeros(0, device=dev), requires_grad=False)
        )
        self.b = nn.Parameter(torch.zeros((), device=dev))
        self.log_var = nn.Parameter(torch.log(torch.tensor(0.1, device=dev)))

    @property
    def sigma2(self) -> torch.Tensor:
        return torch.exp(self.log_var)

    def _design(self, parents: Dict[str, Tensor]) -> Tensor:
        if self.in_dim == 0:
            # Return an empty (N,0) matrix
            if len(parents) == 0:
                raise RuntimeError(f"{self.name}: no parents and no batch provided.")
            N = next(iter(parents.values())).shape[0]
            return torch.empty((N, 0), device=next(iter(parents.values())).device)
        X = _stack_X(parents, self.parent_order)
        assert (
            X.shape[1] == self.in_dim
        ), f"{self.name}: expected in_dim={self.in_dim}, got {X.shape[1]}"
        return X

    def forward(self, parents: Dict[str, Tensor]) -> Tensor:
        """Return mean μ(parents) with shape (N,)."""
        if self.in_dim == 0:
            # infer batch size N from any provided tensor; fallback to 1
            N = next((t.shape[0] for t in parents.values()), 1)
            return self.b.expand(N)
        X = self._design(parents)  # (N, d)
        mu = X @ self.W + self.b  # (N,)
        return mu

    def log_prob(self, y: Tensor, parents: Dict[str, Tensor]) -> Tensor:
        """Elementwise log N(y; μ, σ²) -> (N,)"""
        if y.dim() == 2 and y.shape[1] == 1:
            y = y.squeeze(1)
        var = self.sigma2.clamp_min(1e-12)
        if self.in_dim == 0:
            # No parents: μ is a constant vector of size N = len(y)
            N = y.shape[0]
            mu = self.b.expand(N)
            return -0.5 * (torch.log(2 * torch.pi * var) + (y - mu) ** 2 / var)
        # With parents, use the usual design
        mu = self.forward(parents)
        return -0.5 * (torch.log(2 * torch.pi * var) + (y - mu) ** 2 / var)

    @torch.no_grad()
    def sample(self, parents: Dict[str, Tensor], n_samples: int) -> Tensor:
        """Return samples with shape (N, n_samples)."""
        if self.in_dim == 0:
            # Infer N from any provided tensor; if none, sample a single draw
            N = next((t.shape[0] for t in parents.values()), 1)
            mu = self.b.expand(N)
        else:
            mu = self.forward(parents)  # (N,)
        var = self.sigma2
        eps = torch.randn(mu.shape[0], n_samples, device=mu.device, dtype=mu.dtype)
        return mu.unsqueeze(1) + eps * var.sqrt()

    @torch.no_grad()
    def fit(
        self,
        parents: Dict[str, Tensor],
        y: Tensor,
        ridge: float = 1e-8,
        unbiased_var: bool = False,
    ):
        """Closed-form ridge least squares: fits W,b and a SINGLE scalar variance."""
        if y.dim() == 2 and y.shape[1] == 1:
            y = y.squeeze(1)
        assert (
            y.dim() == 1
        ), f"{self.name}: y must be (N,) or (N,1), got {tuple(y.shape)}"
        dev = y.device
        N = y.shape[0]

        if self.in_dim == 0:
            # b = mean(y); var = mean squared residual
            b_hat = y.mean()
            res = y - b_hat
            var = (
                res.pow(2).mean()
                if not unbiased_var
                else res.pow(2).sum() / max(N - 1, 1)
            )
            # commit
            if self.W.numel() > 0:
                self.W.data.zero_()
            self.b.data.copy_(b_hat)
            self.log_var.data.copy_(var.clamp_min(1e-12).log())
            return

        X = self._design(parents)  # (N, d)
        ones = torch.ones(N, 1, device=dev, dtype=X.dtype)
        A = torch.cat([X, ones], dim=1).double()  # (N, d+1)
        y_d = y.double()

        AtA = A.T @ A
        lamI = torch.eye(AtA.shape[0], device=AtA.device, dtype=AtA.dtype) * ridge
        theta = torch.linalg.solve(AtA + lamI, A.T @ y_d)  # (d+1,)
        W_hat = theta[:-1].to(self.W.dtype)
        b_hat = theta[-1].to(self.b.dtype)

        mu = (X @ W_hat) + b_hat
        res = y - mu
        var = (
            res.pow(2).mean()
            if not unbiased_var
            else res.pow(2).sum() / max(N - self.in_dim - 1, 1)
        )

        # commit
        self.W.data.copy_(W_hat)
        self.b.data.copy_(b_hat)
        self.log_var.data.copy_(var.clamp_min(1e-12).log())

    def update(
        self, parents: Dict[str, Tensor], y: Tensor, lr: float = 1e-2, steps: int = 1
    ) -> None:
        self.train()
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        with torch.enable_grad():
            for _ in range(int(steps)):
                opt.zero_grad(set_to_none=True)
                nll = -self.log_prob(y, parents).mean()
                nll.backward()
                opt.step()

    def training_loss(self, y: Tensor, parents: Dict[str, Tensor]) -> Tensor:
        # negative log-likelihood (mean over batch)
        return (-self.log_prob(y, parents)).mean()
