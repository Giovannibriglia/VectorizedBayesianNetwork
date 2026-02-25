from __future__ import annotations

import math
from typing import Optional

import torch

from vbn.core.base import BaseCPD
from vbn.core.registry import register_cpd
from vbn.core.utils import broadcast_samples, ensure_2d, flatten_samples


@register_cpd("linear_gaussian")
class LinearGaussianCPD(BaseCPD):
    """Linear Gaussian CPD with closed-form ridge regression fit."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        device: torch.device,
        seed: Optional[int] = None,
        ridge: float = 1e-6,
        min_scale: float = 1e-3,
    ) -> None:
        super().__init__(
            input_dim=input_dim, output_dim=output_dim, device=device, seed=seed
        )
        self.ridge = float(ridge)
        self.min_scale = float(min_scale)
        self.register_buffer(
            "_weight",
            torch.zeros(self.input_dim, self.output_dim, device=self.device),
        )
        self.register_buffer("_bias", torch.zeros(self.output_dim, device=self.device))
        self.register_buffer("_var", torch.ones(self.output_dim, device=self.device))

    def get_init_kwargs(self) -> dict:
        return {"ridge": self.ridge, "min_scale": self.min_scale}

    def _prepare_training_tensors(
        self, parents: Optional[torch.Tensor], x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if x.dim() == 2:
            x_flat = x
        elif x.dim() == 3:
            x_flat = x.reshape(-1, x.shape[-1])
        else:
            raise ValueError(f"Expected x with 1D/2D/3D shape, got {tuple(x.shape)}")

        if parents is None:
            if self.input_dim != 0:
                raise ValueError("parents cannot be None when input_dim > 0")
            parents = torch.zeros(x_flat.shape[0], 0, device=self.device)
        else:
            if parents.dim() == 1:
                parents = ensure_2d(parents)
            if parents.dim() == 2:
                if x.dim() == 3:
                    parents = broadcast_samples(parents, x.shape[1])
                    parents = parents.reshape(-1, parents.shape[-1])
            elif parents.dim() == 3:
                parents = parents.reshape(-1, parents.shape[-1])
            else:
                raise ValueError(
                    f"Expected parents with 1D/2D/3D shape, got {tuple(parents.shape)}"
                )

        parents = parents.to(device=self.device, dtype=x_flat.dtype)
        x_flat = x_flat.to(device=self.device)
        if parents.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected parents_dim {self.input_dim}, got {parents.shape[-1]}"
            )
        return parents, x_flat

    def _fit_closed_form(
        self, parents: torch.Tensor, x: torch.Tensor, ridge: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if parents.shape[0] == 0:
            raise ValueError("Cannot fit LinearGaussianCPD with zero rows")
        n = parents.shape[0]
        ones = torch.ones(n, 1, device=self.device, dtype=x.dtype)
        x_aug = torch.cat([parents, ones], dim=1)
        xtx = x_aug.T @ x_aug
        reg = float(ridge)
        if reg < 0:
            raise ValueError("ridge must be >= 0")
        eye = torch.eye(xtx.shape[0], device=self.device, dtype=x.dtype)
        eye[-1, -1] = 0.0  # Do not penalize the bias term.
        xtx_reg = xtx + reg * eye
        xty = x_aug.T @ x
        try:
            w_aug = torch.linalg.solve(xtx_reg, xty)
        except RuntimeError:
            w_aug = torch.linalg.pinv(xtx_reg) @ xty

        weight = w_aug[:-1]
        bias = w_aug[-1]
        resid = x - x_aug @ w_aug
        var = (resid**2).mean(dim=0)
        min_var = float(self.min_scale) ** 2
        var = torch.clamp(var, min=min_var)
        return weight, bias, var

    def fit(
        self,
        parents: Optional[torch.Tensor],
        x: torch.Tensor,
        epochs: int = 1,
        lr: float = 1e-3,
        batch_size: int = 128,
        weight_decay: float = 0.0,
        ridge: Optional[float] = None,
        **kwargs,
    ) -> None:
        del epochs, lr, batch_size, weight_decay, kwargs
        parents, x = self._prepare_training_tensors(parents, x)
        ridge = self.ridge if ridge is None else float(ridge)
        weight, bias, var = self._fit_closed_form(parents, x, ridge)
        self._weight.copy_(weight)
        self._bias.copy_(bias)
        self._var.copy_(var)

    def update(
        self,
        parents: Optional[torch.Tensor],
        x: torch.Tensor,
        lr: float = 1e-3,
        n_steps: int = 1,
        batch_size: int = 128,
        weight_decay: float = 0.0,
        ridge: Optional[float] = None,
        **kwargs,
    ) -> None:
        del lr, n_steps, batch_size, weight_decay, kwargs
        self.fit(parents, x, ridge=ridge)

    def _scale(self) -> torch.Tensor:
        min_var = float(self.min_scale) ** 2
        return torch.sqrt(self._var.clamp(min=min_var))

    def _params(
        self, parents: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.input_dim == 0:
            loc = self._bias
            scale = self._scale()
            return loc, scale

        if parents is None:
            raise ValueError("parents cannot be None when input_dim > 0")
        if parents.dim() == 2:
            parents = parents.unsqueeze(1)
        flat, b, s = flatten_samples(parents)
        mu = flat @ self._weight + self._bias
        loc = mu.reshape(b, s, self.output_dim)
        scale = self._scale().view(1, 1, -1).expand(b, s, -1)
        return loc, scale

    def sample(self, parents: Optional[torch.Tensor], n_samples: int) -> torch.Tensor:
        if self.input_dim == 0:
            b = 1 if parents is None else parents.shape[0]
            loc = self._bias.view(1, 1, -1).expand(b, n_samples, -1)
            scale = self._scale().view(1, 1, -1).expand(b, n_samples, -1)
        else:
            if parents is None:
                raise ValueError("parents cannot be None when input_dim > 0")
            parents = broadcast_samples(parents, n_samples)
            loc, scale = self._params(parents)
        eps = torch.randn_like(scale)
        return loc + eps * scale

    def log_prob(
        self, x: torch.Tensor, parents: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if x.dim() <= 2:
            x = ensure_2d(x)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if self.input_dim == 0:
            b, s, _ = x.shape
            loc = self._bias.view(1, 1, -1).expand(b, s, -1)
            scale = self._scale().view(1, 1, -1).expand(b, s, -1)
        else:
            if parents is None:
                raise ValueError("parents cannot be None when input_dim > 0")
            parents = broadcast_samples(parents, x.shape[1])
            loc, scale = self._params(parents)
        var = scale**2
        return -0.5 * (
            ((x - loc) ** 2) / var + 2 * torch.log(scale) + math.log(2 * math.pi)
        ).sum(dim=-1)
