from __future__ import annotations

from typing import Optional

import torch
from torch import nn, Tensor
from vbn.learning.base import ImplicitGenerator


def _feat(X: Tensor) -> Tensor:
    if X.dim() == 2 and X.shape[1] == 0:
        return torch.ones((X.shape[0], 1), device=X.device, dtype=X.dtype)
    if X.dim() == 1:
        X = X.unsqueeze(0)
    return X


def _rbf_kernel(x: Tensor, y: Tensor, sigma: float) -> Tensor:
    # x:[N,D], y:[M,D] -> [N,M]
    x2 = (x * x).sum(-1, keepdim=True)  # [N,1]
    y2 = (y * y).sum(-1, keepdim=True).T  # [1,M]
    dist2 = (x2 - 2 * x @ y.T + y2).clamp_min(0)
    return torch.exp(-dist2 / (2.0 * sigma * sigma))


def _mmd_unbiased(x: Tensor, y: Tensor, sigma: float) -> Tensor:
    # x,y: [B,D]
    Kxx = _rbf_kernel(x, x, sigma)
    Kyy = _rbf_kernel(y, y, sigma)
    Kxy = _rbf_kernel(x, y, sigma)
    n = x.shape[0]
    if n > 1:
        mmd = (
            (Kxx.sum() - Kxx.diag().sum()) / (n * (n - 1))
            + (Kyy.sum() - Kyy.diag().sum()) / (n * (n - 1))
            - 2.0 * Kxy.mean()
        )
    else:
        mmd = Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()
    return mmd


class CondMMDGenerator(ImplicitGenerator):
    """
    Conditional implicit generator trained by MMD on the JOINT [x,y]:
      minimize  MMD^2( [x,y], [x,G(x,z)] )
    - Stable and simple (no discriminator).
    - For continuous targets.
    """

    def __init__(
        self,
        name: str,
        in_dim: int,
        out_dim: int,
        hidden: int = 128,
        z_dim: int = 8,
        lr: float = 1e-3,
        epochs: int = 20,
        batch_size: int | None = 1024,
        sigma: float = 1.0,  # RBF bandwidth on concatenated [x,y]
        weight_decay: float = 0.0,
    ):
        super().__init__(
            name,
            {
                "in_dim": in_dim,
                "out_dim": out_dim,
                "hidden": hidden,
                "z_dim": z_dim,
                "lr": lr,
                "epochs": epochs,
                "batch_size": batch_size,
                "sigma": sigma,
                "weight_decay": weight_decay,
            },
        )
        p = max(1, in_dim)
        H = hidden
        self.gen = nn.Sequential(
            nn.Linear(p + z_dim, H),
            nn.ReLU(),
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Linear(H, out_dim),
        )
        self._optim = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        # buffers for update sampling (not required, but handy)
        self.register_buffer("X_train", torch.empty(0, in_dim))
        self.register_buffer("Y_train", torch.empty(0, out_dim))

    def _iter(self, X: Tensor, Y: Tensor, bs: Optional[int]):
        N = X.shape[0]
        if bs is None or bs <= 0 or bs >= N:
            yield X, Y
            return
        perm = torch.randperm(N, device=X.device)
        for s in range(0, N, bs):
            idx = perm[s : s + bs]
            yield X[idx], Y[idx]

    def _train_epochs(self, X: Tensor, Y: Tensor, epochs: int):
        Xf = _feat(X)
        Yf = Y
        self.train()
        for _ in range(epochs):
            for xb, yb in self._iter(Xf, Yf, self.cfg["batch_size"]):
                z = torch.randn(
                    xb.shape[0], self.cfg["z_dim"], device=xb.device, dtype=xb.dtype
                )
                yhat = self.gen(torch.cat([xb, z], dim=-1))
                # MMD on joint [x,y]
                real = torch.cat([xb, yb], dim=-1)
                fake = torch.cat([xb, yhat], dim=-1)
                loss = _mmd_unbiased(real, fake, self.cfg["sigma"])
                self._optim.zero_grad(set_to_none=True)
                loss.backward()
                self._optim.step()

    def fit(self, X: Tensor, y: Tensor) -> None:
        X, y = _feat(X), y
        self.X_train, self.Y_train = X.detach(), y.detach()
        self._train_epochs(X, y, epochs=self.cfg["epochs"])

    @torch.no_grad()
    def update(self, X: Tensor, y: Tensor) -> None:
        X, y = _feat(X), y
        if self.X_train.numel() == 0:
            self.X_train, self.Y_train = X.detach(), y.detach()
        else:
            self.X_train = torch.cat([self.X_train, X.detach()], dim=0)
            self.Y_train = torch.cat([self.Y_train, y.detach()], dim=0)
        with torch.enable_grad():
            self._train_epochs(X, y, epochs=max(1, self.cfg["epochs"] // 2))

    def log_prob(self, X: Tensor, y: Tensor) -> Tensor:
        raise NotImplementedError("Implicit generator: log_prob is unavailable.")

    @torch.no_grad()
    def sample(self, X: Tensor, n: int = 1) -> Tensor:
        Xf = _feat(X)
        B = Xf.shape[0]
        z = torch.randn(n, B, self.cfg["z_dim"], device=Xf.device, dtype=Xf.dtype)
        Xrep = Xf.unsqueeze(0).expand(n, B, Xf.shape[1])
        y = self.gen(torch.cat([Xrep, z], dim=-1))  # [n,B,D]
        return y
