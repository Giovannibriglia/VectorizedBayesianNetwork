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


def _grad_penalty(
    critic: nn.Module, x: Tensor, y_real: Tensor, y_fake: Tensor, lam: float = 10.0
) -> Tensor:
    # interpolate
    eps = torch.rand(y_real.shape[0], 1, device=y_real.device, dtype=y_real.dtype)
    yi = eps * y_real + (1 - eps) * y_fake
    yi.requires_grad_(True)
    scores = critic(torch.cat([x, yi], dim=-1))
    grad = torch.autograd.grad(
        outputs=scores.sum(),
        inputs=yi,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gp = ((grad.norm(2, dim=-1) - 1.0) ** 2).mean() * lam
    return gp


class CondWGANGenerator(ImplicitGenerator):
    """
    Conditional WGAN-GP:
      Critic C(x,y) trained with Wasserstein loss + gradient penalty;
      Generator G(x,z) tries to fool C on pairs [x,G(x,z)].
    For continuous targets.
    """

    def __init__(
        self,
        name: str,
        in_dim: int,
        out_dim: int,
        hidden: int = 128,
        z_dim: int = 8,
        g_lr: float = 1e-4,
        c_lr: float = 2e-4,
        epochs: int = 20,
        batch_size: int | None = 1024,
        critic_steps: int = 5,
        gp_lambda: float = 10.0,
        weight_decay: float = 0.0,
    ):
        super().__init__(
            name,
            {
                "in_dim": in_dim,
                "out_dim": out_dim,
                "hidden": hidden,
                "z_dim": z_dim,
                "g_lr": g_lr,
                "c_lr": c_lr,
                "epochs": epochs,
                "batch_size": batch_size,
                "critic_steps": critic_steps,
                "gp_lambda": gp_lambda,
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
        self.critic = nn.Sequential(
            nn.Linear(p + out_dim, H),
            nn.LeakyReLU(0.2),
            nn.Linear(H, H),
            nn.LeakyReLU(0.2),
            nn.Linear(H, 1),
        )
        self._g_opt = torch.optim.Adam(
            self.gen.parameters(), lr=g_lr, betas=(0.5, 0.9), weight_decay=weight_decay
        )
        self._c_opt = torch.optim.Adam(
            self.critic.parameters(),
            lr=c_lr,
            betas=(0.5, 0.9),
            weight_decay=weight_decay,
        )
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
        Xf, Yf = _feat(X), Y
        for _ in range(epochs):
            for xb, yb in self._iter(Xf, Yf, self.cfg["batch_size"]):
                # --- Critic steps ---
                for _ in range(self.cfg["critic_steps"]):
                    z = torch.randn(
                        xb.shape[0], self.cfg["z_dim"], device=xb.device, dtype=xb.dtype
                    )
                    y_fake = self.gen(torch.cat([xb, z], dim=-1)).detach()
                    real_in = torch.cat([xb, yb], dim=-1)
                    fake_in = torch.cat([xb, y_fake], dim=-1)
                    c_real = self.critic(real_in).mean()
                    c_fake = self.critic(fake_in).mean()
                    gp = _grad_penalty(
                        self.critic, xb, yb, y_fake, lam=self.cfg["gp_lambda"]
                    )
                    c_loss = -(c_real - c_fake) + gp
                    self._c_opt.zero_grad(set_to_none=True)
                    c_loss.backward()
                    self._c_opt.step()

                # --- Generator step ---
                z = torch.randn(
                    xb.shape[0], self.cfg["z_dim"], device=xb.device, dtype=xb.dtype
                )
                y_fake = self.gen(torch.cat([xb, z], dim=-1))
                g_loss = -self.critic(torch.cat([xb, y_fake], dim=-1)).mean()
                self._g_opt.zero_grad(set_to_none=True)
                g_loss.backward()
                self._g_opt.step()

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
        return self.gen(torch.cat([Xrep, z], dim=-1))  # [n,B,D]
