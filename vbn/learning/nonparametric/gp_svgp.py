from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from vbn.learning.base import BaseCPD
from vbn.learning.utils import _cat_parents
from vbn.utils import Tensor


class SVGPRegCPD(BaseCPD):
    # ─────────────────────────────────────────────────────────────────────────────
    # Sparse Variational GP (SVGP) — minimal Gaussian regression CPD
    # RBF kernel, diagonal q(u) covariance, Gaussian likelihood with learnable noise
    # NOTE: simplified and meant for small/medium D and M (inducing points)
    # ─────────────────────────────────────────────────────────────────────────────

    def __init__(
        self,
        name: str,
        parents: Dict[str, int],
        out_dim: int,
        M: int = 64,
        device: str | torch.device = "cpu",
    ):
        super().__init__(name, parents, device)
        self.x_dim = int(sum(parents.values())) if len(parents) else 0
        self.out_dim = int(out_dim)
        self.M = int(M)
        # kernel params (RBF): lengthscale ℓ per-dim, variance σ_f^2
        self.log_ell = nn.Parameter(
            torch.zeros(self.x_dim if self.x_dim else 1, device=self.device)
        )
        self.log_sf2 = nn.Parameter(torch.zeros(1, device=self.device))
        self.log_sn2 = nn.Parameter(
            torch.log(torch.tensor(0.05, device=self.device)).unsqueeze(0)
        )
        # inducing inputs Z and variational params q(u)=N(m, S) with diag S
        self.Z = nn.Parameter(torch.randn(M, max(self.x_dim, 1), device=self.device))
        self.m = nn.Parameter(torch.zeros(M, self.out_dim, device=self.device))
        self.log_Sdiag = nn.Parameter(torch.zeros(M, self.out_dim, device=self.device))

    def _kernel(self, X: Tensor, Y: Tensor) -> Tensor:
        ell = F.softplus(self.log_ell) + 1e-6
        sf2 = F.softplus(self.log_sf2) + 1e-6
        X_ = X / ell
        Y_ = Y / ell
        # ||x-y||^2 = x^2 + y^2 - 2xy
        x2 = (X_**2).sum(-1, keepdim=True)
        y2 = (Y_**2).sum(-1, keepdim=True).transpose(0, 1)
        dist2 = x2 + y2 - 2 * (X_ @ Y_.T)
        return sf2 * torch.exp(-0.5 * dist2)

    def forward(self, parents: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        X = _cat_parents(parents)
        if X is None:
            X = torch.zeros(
                (next(iter(parents.values())).shape[0] if parents else 1, 1),
                device=self.device,
            )
        # predictive mean/var under SVGP
        Kzz = (
            self._kernel(self.Z, self.Z) + torch.eye(self.M, device=self.device) * 1e-4
        )
        Kxz = self._kernel(X, self.Z)
        L = torch.linalg.cholesky(Kzz)
        # A = Kxz Kzz^{-1}
        A = torch.cholesky_solve(Kxz.T, L).T  # (N, M)
        # mean: A m
        mu = A @ self.m  # (N, out)
        # var: diag(Kxx - A Kzx + A S A^T) + sn2
        Kxx_diag = (F.softplus(self.log_sf2) + 1e-6).expand(
            X.shape[0]
        )  # RBF k(x,x)=sf2
        Sdiag = F.softplus(self.log_Sdiag) + 1e-6  # (M, out)
        # A Kzx diag term
        AKzx = (A @ self._kernel(self.Z, X)).diagonal(dim1=-2, dim2=-1)  # (N,)
        var_shared = (Kxx_diag - AKzx).unsqueeze(-1)
        var_q = (A**2) @ Sdiag  # (N, out)
        sn2 = F.softplus(self.log_sn2)
        var = var_shared + var_q + sn2
        return mu, var.clamp_min(1e-9)

    def log_prob(self, y: Tensor, parents: Dict[str, Tensor]) -> Tensor:
        mu, var = self.forward(parents)
        return -0.5 * ((y.to(self.device) - mu) ** 2 / var).sum(-1) - 0.5 * (
            var.log().sum(-1) + self.out_dim * math.log(2 * math.pi)
        )

    def elbo(self, y: Tensor, parents: Dict[str, Tensor]) -> Tensor:
        # ELBO ≈ E_q[f] log p(y|f) - KL[q(u)||p(u)] with p(u)=N(0, Kzz)
        mu, var = self.forward(parents)
        nll = 0.5 * (((y.to(self.device) - mu) ** 2) / var + var.log()).sum(
            -1
        ) + 0.5 * self.out_dim * math.log(2 * math.pi)
        # KL diagonal q(u)
        Kzz = (
            self._kernel(self.Z, self.Z) + torch.eye(self.M, device=self.device) * 1e-6
        )
        L = torch.linalg.cholesky(Kzz)
        # tr(Kzz^{-1} S) + m^T Kzz^{-1} m - M + log |Kzz| - log |S|
        # Sinv_tr = torch.cholesky_solve((F.softplus(self.log_Sdiag) + 1e-6), L)  # wrong shape; compute per out-dim below
        # compute KL per output dim
        kl = 0.0
        Sdiag = F.softplus(self.log_Sdiag) + 1e-6  # (M,out)
        for j in range(self.out_dim):
            Sj = torch.diag(Sdiag[:, j])
            tr_term = torch.trace(torch.cholesky_solve(Sj, L))
            m_term = (
                self.m[:, j : j + 1].T @ torch.cholesky_solve(self.m[:, j : j + 1], L)
            ).squeeze()
            logdet_K = 2 * torch.log(torch.diagonal(L)).sum()
            logdet_S = torch.log(torch.diagonal(Sj)).sum()
            kl = kl + 0.5 * (tr_term + m_term - self.M + logdet_K - logdet_S)
        return -nll.mean() - kl / y.shape[0]

    @torch.no_grad()
    def fit(self, parents: Dict[str, Tensor], y: Tensor) -> None:
        # initialize Z by subsampling X; m zeros; Sdiag small
        X = _cat_parents(parents)
        if X is None:
            X = torch.zeros((y.shape[0], 1), device=self.device)
        N = X.shape[0]
        if N >= self.M:
            idx = torch.randperm(N, device=self.device)[: self.M]
            self.Z.data.copy_(X[idx])
        else:
            # pad/repeat
            reps = math.ceil(self.M / N)
            self.Z.data.copy_(X.repeat(reps, 1)[: self.M])
        self.m.data.zero_()
        self.log_Sdiag.data.fill_(math.log(1e-2))

    @torch.no_grad()
    def update(
        self, parents: Dict[str, Tensor], y: Tensor, alpha: float = 1e-2
    ) -> None:
        # one or few ELBO steps for online refinement
        self.train()
        opt = torch.optim.Adam(self.parameters(), lr=alpha)
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            loss = -self.elbo(y, parents)
            loss.backward()
            opt.step()

    @torch.no_grad()
    def sample(self, parents: Dict[str, Tensor], n_samples: int) -> Tensor:
        mu, var = self.forward(parents)
        eps = torch.randn((mu.shape[0], n_samples, mu.shape[-1]), device=self.device)
        return mu.unsqueeze(1) + eps * var.sqrt().unsqueeze(0)
