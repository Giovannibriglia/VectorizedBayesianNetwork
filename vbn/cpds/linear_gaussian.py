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
        if self.input_dim == 0:
            mean = x.mean(dim=0)
            std = x.std(dim=0, unbiased=False).clamp_min(1e-6)
            weight = torch.zeros(0, self.output_dim, device=self.device, dtype=x.dtype)
            bias = mean
            var = std**2
            return weight, bias, var

        n = parents.shape[0]
        ones = torch.ones(n, 1, device=self.device, dtype=x.dtype)
        x_aug = torch.cat([parents, ones], dim=1)
        reg = float(ridge)
        if reg < 0:
            raise ValueError("ridge must be >= 0")
        if reg > 0:
            eye = torch.eye(self.input_dim, device=self.device, dtype=x.dtype)
            reg_block = torch.cat(
                [
                    math.sqrt(reg) * eye,
                    torch.zeros(self.input_dim, 1, device=self.device, dtype=x.dtype),
                ],
                dim=1,
            )
            x_reg = torch.cat([x_aug, reg_block], dim=0)
            y_reg = torch.cat(
                [
                    x,
                    torch.zeros(
                        self.input_dim, x.shape[1], device=self.device, dtype=x.dtype
                    ),
                ],
                dim=0,
            )
        else:
            x_reg = x_aug
            y_reg = x

        theta = torch.linalg.lstsq(x_reg, y_reg).solution
        weight = theta[:-1]
        bias = theta[-1]
        residual = x - x_aug @ theta
        sigma2 = residual.var(dim=0).clamp_min(1e-6)
        return weight, bias, sigma2

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
        self.theta = torch.cat([weight, bias.unsqueeze(0)], dim=0)
        self.sigma = torch.sqrt(var)

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
