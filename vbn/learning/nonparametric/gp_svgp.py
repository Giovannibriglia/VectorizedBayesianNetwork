from __future__ import annotations

import math
from typing import Dict, Tuple

import torch

from torch import nn
from torch.nn import functional as F

from ..base import BaseCPD, Tensor
from ..utils import _cat_parents


def _symmetrize(K: torch.Tensor) -> torch.Tensor:
    return 0.5 * (K + K.transpose(-1, -2))


def _stable_cholesky(
    K: torch.Tensor, jitter_init: float = 1e-6, max_tries: int = 7
) -> torch.Tensor:
    """Try Cholesky with exponentially increasing jitter; final fallback: eigenvalue repair."""
    Id = torch.eye(K.shape[-1], device=K.device, dtype=K.dtype)
    jitter = jitter_init
    for _ in range(max_tries):
        try:
            return torch.linalg.cholesky(_symmetrize(K) + jitter * Id)
        except RuntimeError:
            jitter *= 10.0
    # Eigenvalue repair (final fallback)
    evals, evecs = torch.linalg.eigh(_symmetrize(K))
    evals = torch.clamp(evals, min=1e-8)
    K_pd = (evecs * evals) @ evecs.transpose(-1, -2)
    return torch.linalg.cholesky(_symmetrize(K_pd) + 1e-8 * Id)


class SVGPRegCPD(BaseCPD):
    """Minimal RBF SVGP for regression (diag q(u), Gaussian noise) with
    input/output standardization and a linear mean function."""

    def __init__(
        self,
        name: str,
        parents: Dict[str, int],
        out_dim: int,
        M: int = 128,
        device: str | torch.device = "cpu",
    ):
        super().__init__(name, parents, device)
        self.x_dim = int(sum(parents.values())) if parents else 0
        self.out_dim = int(out_dim)
        self.M = int(M)

        # mean function m(x) = Ax + b (learn residuals with GP)
        self.mean_lin = nn.Linear(max(self.x_dim, 1), self.out_dim, bias=True).to(
            self.device
        )

        # RBF kernel hyperparams
        self.log_ell = nn.Parameter(
            torch.zeros(max(self.x_dim, 1), device=self.device)
        )  # per-dim ℓ
        self.log_sf2 = nn.Parameter(
            torch.zeros(1, device=self.device)
        )  # signal variance
        self.log_sn2 = nn.Parameter(
            torch.log(torch.tensor(0.05, device=self.device)).unsqueeze(0)
        )  # noise variance

        # Inducing variables
        self.Z = nn.Parameter(
            torch.randn(self.M, max(self.x_dim, 1), device=self.device)
        )
        self.m = nn.Parameter(
            torch.zeros(self.M, self.out_dim, device=self.device)
        )  # q(u) mean
        self.log_Sdiag = nn.Parameter(
            torch.zeros(self.M, self.out_dim, device=self.device)
        )  # q(u) diag cov (softplus)

        # standardization buffers
        self.register_buffer(
            "x_mean", torch.zeros(1, max(self.x_dim, 1), device=self.device)
        )
        self.register_buffer(
            "x_std", torch.ones(1, max(self.x_dim, 1), device=self.device)
        )
        self.register_buffer("y_mean", torch.zeros(1, self.out_dim, device=self.device))
        self.register_buffer("y_std", torch.ones(1, self.out_dim, device=self.device))

    # ----- helpers -----
    def _normalize_x(self, X: Tensor) -> Tensor:
        return (X - self.x_mean) / self.x_std

    @staticmethod
    def _median_heuristic_ell(Xn: Tensor) -> Tensor:
        # Xn: (N, Dn) normalized
        if Xn.shape[0] > 4096:
            Xn = Xn[torch.randperm(Xn.shape[0], device=Xn.device)[:4096]]
        # pairwise L2 distances; approximate per-dim ℓ via overall median
        dists = torch.cdist(Xn, Xn, p=2)
        med = (
            torch.median(dists[dists > 0.0])
            if (dists > 0).any()
            else torch.tensor(1.0, device=Xn.device)
        )
        ell = med / math.sqrt(2.0)
        return torch.full((Xn.shape[1],), ell.item(), device=Xn.device)

    def _kernel(self, X: Tensor, Y: Tensor) -> Tensor:
        # floors to avoid degeneracy
        ell = F.softplus(self.log_ell) + 1e-5  # (D,)
        sf2 = F.softplus(self.log_sf2) + 1e-8  # ()
        X_ = X / ell
        Y_ = Y / ell
        x2 = (X_**2).sum(-1, keepdim=True)
        y2 = (Y_**2).sum(-1, keepdim=True).transpose(0, 1)
        dist2 = x2 + y2 - 2 * (X_ @ Y_.T)
        K = sf2 * torch.exp(-0.5 * dist2)
        return K

    def forward(self, parents: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        X = _cat_parents(parents)
        if X is None:
            X = torch.zeros(
                (next(iter(parents.values())).shape[0] if parents else 1, 1),
                device=self.device,
            )
        else:
            X = X.to(self.device).float()

        # normalize X
        Xn = self._normalize_x(X)

        # predictive mean = mean_lin(X) + GP contribution
        mean_f = self.mean_lin(Xn)

        Kzz = self._kernel(self.Z, self.Z)
        L = _stable_cholesky(Kzz, jitter_init=1e-6)
        Kxz = self._kernel(Xn, self.Z)
        A = torch.cholesky_solve(Kxz.T, L).T  # (N, M)

        mu_gp = A @ self.m  # (N, out_dim)

        # variance
        Kxx_diag = (F.softplus(self.log_sf2) + 1e-6).expand(Xn.shape[0])
        AKzx = (A @ self._kernel(self.Z, Xn)).diagonal(dim1=-2, dim2=-1)  # diag term
        var_shared = (Kxx_diag - AKzx).unsqueeze(-1)  # (N,1)

        Sdiag = F.softplus(self.log_Sdiag) + 1e-6
        var_q = (A**2) @ Sdiag  # (N,out_dim)
        sn2 = F.softplus(self.log_sn2) + 1e-8
        var = (var_shared + var_q + sn2).clamp_min(1e-9)  # (N,out_dim)

        # denormalize outputs
        mu = mean_f + mu_gp
        mu = mu * self.y_std + self.y_mean
        var = var * (self.y_std**2)

        return mu, var

    def log_prob(self, y: Tensor, parents: Dict[str, Tensor]) -> Tensor:
        mu, var = self.forward(parents)
        resid = y.to(self.device) - mu
        return -0.5 * (resid.pow(2) / var).sum(-1) - 0.5 * (
            var.log().sum(-1) + self.out_dim * math.log(2 * math.pi)
        )

    def elbo(self, y: Tensor, parents: Dict[str, Tensor]) -> Tensor:
        # Compute in standardized Y space for stable nll
        X = _cat_parents(parents)
        if X is None:
            X = torch.zeros((y.shape[0], 1), device=self.device)
        else:
            X = X.to(self.device).float()
        Xn = self._normalize_x(X)

        # GP predictive stats in standardized space
        mean_f = self.mean_lin(Xn)  # standardized mean part BEFORE scaling
        Kzz = self._kernel(self.Z, self.Z)
        L = _stable_cholesky(Kzz, jitter_init=1e-6)
        Kxz = self._kernel(Xn, self.Z)
        A = torch.cholesky_solve(Kxz.T, L).T
        mu_gp = A @ self.m

        Kxx_diag = (F.softplus(self.log_sf2) + 1e-6).expand(Xn.shape[0])
        AKzx = (A @ self._kernel(self.Z, Xn)).diagonal(dim1=-2, dim2=-1)
        var_shared = (Kxx_diag - AKzx).unsqueeze(-1)
        Sdiag = F.softplus(self.log_Sdiag) + 1e-6
        var_q = (A**2) @ Sdiag
        sn2 = F.softplus(self.log_sn2)
        mu_std = mean_f + mu_gp
        var_std = (var_shared + var_q + sn2).clamp_min(1e-9)

        # standardize y
        y_std = (y.to(self.device) - self.y_mean) / self.y_std

        # Gaussian nll in standardized space
        nll = 0.5 * (((y_std - mu_std) ** 2) / var_std + var_std.log()).sum(
            -1
        ) + 0.5 * self.out_dim * math.log(2 * math.pi)

        # KL[q(u)||p(u)] with diagonal S
        Kzz = self._kernel(self.Z, self.Z)
        L = _stable_cholesky(Kzz, jitter_init=1e-6)
        kl = 0.0
        for j in range(self.out_dim):
            Sj = torch.diag(Sdiag[:, j])
            tr_term = torch.trace(torch.cholesky_solve(Sj, L))
            m_term = (
                self.m[:, j : j + 1].T @ torch.cholesky_solve(self.m[:, j : j + 1], L)
            ).squeeze()
            logdet_K = 2 * torch.log(torch.diagonal(L)).sum()
            logdet_S = torch.log(torch.diagonal(Sj)).sum()
            kl = kl + 0.5 * (tr_term + m_term - self.M + logdet_K - logdet_S)

        return -(-nll.mean() - kl / y.shape[0])  # return positive loss

    def training_loss(self, y: Tensor, parents: Dict[str, Tensor]) -> Tensor:
        return self.elbo(y, parents)

    @torch.no_grad()
    def fit(self, parents: Dict[str, Tensor], y: Tensor) -> None:
        # set standardization stats
        ym = y.to(self.device).mean(dim=0, keepdim=True)
        ys = y.to(self.device).std(dim=0, keepdim=True).clamp_min(1e-6)
        self.y_mean.copy_(ym)
        self.y_std.copy_(ys)

        X = _cat_parents(parents)
        if X is None:
            X = torch.zeros((y.shape[0], 1), device=self.device)
        else:
            X = X.to(self.device).float()

        xm = X.mean(dim=0, keepdim=True)
        xs = X.std(dim=0, keepdim=True).clamp_min(1e-6)
        self.x_mean.copy_(xm)
        self.x_std.copy_(xs)
        Xn = self._normalize_x(X)

        # init inducing locations on normalized X
        N = Xn.shape[0]
        if N >= self.M:
            idx = torch.randperm(N, device=self.device)[: self.M]
            self.Z.data.copy_(Xn[idx])
        else:
            reps = math.ceil(self.M / N)
            self.Z.data.copy_(Xn.repeat(reps, 1)[: self.M])

        self.Z.data.add_(1e-6 * torch.randn_like(self.Z))

        # init mean to linear regression on normalized X
        try:
            Xb = torch.cat([Xn, torch.ones_like(Xn[:, :1])], -1)
            theta = torch.linalg.lstsq(Xb, ((y - ym) / ys)).solution  # (Dn+1, out_dim)
            W, b = theta[:-1].T, theta[-1]
            with torch.no_grad():
                self.mean_lin.weight.copy_(W)
                self.mean_lin.bias.copy_(b)
        except Exception:
            self.mean_lin.reset_parameters()

        # median heuristic for ℓ
        ell0 = self._median_heuristic_ell(Xn)
        self.log_ell.data.copy_(torch.log(torch.clamp(ell0, min=1e-3)))

        # init q(u)
        self.m.data.zero_()
        self.log_Sdiag.data.fill_(math.log(1e-2))

    @torch.no_grad()
    def update(
        self, parents: Dict[str, Tensor], y: Tensor, alpha: float = 5e-3, steps: int = 3
    ) -> None:
        """Small online ELBO step(s) to assimilate a new batch."""
        self.train()
        # keep a tiny optimizer around across updates to avoid re-init each call (optional)
        if not hasattr(self, "_upd_opt"):
            self._upd_opt = torch.optim.Adam(self.parameters(), lr=alpha)
        opt = self._upd_opt
        # even if caller disabled grads, re-enable here

        with torch.enable_grad():
            for _ in range(int(steps)):
                opt.zero_grad(set_to_none=True)
                loss = self.elbo(y, parents)  # must track params
                if not loss.requires_grad:
                    raise RuntimeError(
                        "SVGP.update(): ELBO has no grad; check for outer no_grad/inference_mode."
                    )
                loss.backward()
                opt.step()

    @torch.no_grad()
    def sample(self, parents: Dict[str, Tensor], n_samples: int) -> Tensor:
        mu, var = self.forward(parents)
        eps = torch.randn((mu.shape[0], n_samples, mu.shape[-1]), device=self.device)
        return mu.unsqueeze(1) + eps * var.sqrt().unsqueeze(0)
