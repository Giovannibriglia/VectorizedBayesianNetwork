from __future__ import annotations

import math
from typing import Optional

import torch

from vbn.core.base import BaseCPD
from vbn.core.registry import register_cpd
from vbn.core.utils import broadcast_samples, ensure_2d, flatten_samples


@register_cpd("rff_gaussian")
class RFFGaussianCPD(BaseCPD):
    """Random Fourier Features Gaussian CPD (closed-form ridge regression)."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        device: torch.device,
        seed: Optional[int] = None,
        n_features: int = 256,
        lengthscale: float = 1.0,
        ridge: float = 1e-6,
        min_scale: float = 1e-3,
        use_bias: bool = True,
    ) -> None:
        super().__init__(
            input_dim=input_dim, output_dim=output_dim, device=device, seed=seed
        )
        self.n_features = int(n_features)
        if self.n_features <= 0:
            raise ValueError("n_features must be >= 1")
        self.lengthscale = float(lengthscale)
        if self.lengthscale <= 0:
            raise ValueError("lengthscale must be > 0")
        self.ridge = float(ridge)
        self.min_scale = float(min_scale)
        self.use_bias = bool(use_bias)

        self.register_buffer("mean_x", torch.zeros(self.input_dim, device=self.device))
        self.register_buffer("std_x", torch.ones(self.input_dim, device=self.device))
        self.register_buffer("mean_y", torch.zeros(self.output_dim, device=self.device))
        self.register_buffer("std_y", torch.ones(self.output_dim, device=self.device))
        self.register_buffer("_stats_ready", torch.tensor(False, device=self.device))

        if self.input_dim == 0:
            w = torch.empty(self.n_features, 0, device=self.device)
            b = torch.empty(self.n_features, device=self.device)
        else:
            scale = max(self.lengthscale, 1e-6)
            w = torch.randn(self.n_features, self.input_dim, device=self.device) / scale
            b = 2 * math.pi * torch.rand(self.n_features, device=self.device)
        self.register_buffer("_rff_w", w)
        self.register_buffer("_rff_b", b)

        self.register_buffer(
            "_coef",
            torch.zeros(self.n_features, self.output_dim, device=self.device),
        )
        self.register_buffer("_bias", torch.zeros(self.output_dim, device=self.device))
        self.register_buffer("_var", torch.ones(self.output_dim, device=self.device))

    def get_init_kwargs(self) -> dict:
        return {
            "n_features": self.n_features,
            "lengthscale": self.lengthscale,
            "ridge": self.ridge,
            "min_scale": self.min_scale,
            "use_bias": self.use_bias,
        }

    def _check_fitted(self) -> None:
        if not bool(self._stats_ready.item()):
            raise RuntimeError("RFFGaussianCPD is not fitted yet.")

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

    def _update_standardization(self, parents: torch.Tensor, x: torch.Tensor) -> None:
        if parents.numel() == 0:
            mean_x = torch.zeros(self.input_dim, device=self.device, dtype=x.dtype)
            std_x = torch.ones(self.input_dim, device=self.device, dtype=x.dtype)
        else:
            mean_x = parents.mean(dim=0)
            std_x = parents.std(dim=0, unbiased=False).clamp_min(1e-6)
        mean_y = x.mean(dim=0)
        std_y = x.std(dim=0, unbiased=False).clamp_min(1e-6)
        self.mean_x.copy_(mean_x.to(self.mean_x.device, dtype=self.mean_x.dtype))
        self.std_x.copy_(std_x.to(self.std_x.device, dtype=self.std_x.dtype))
        self.mean_y.copy_(mean_y.to(self.mean_y.device, dtype=self.mean_y.dtype))
        self.std_y.copy_(std_y.to(self.std_y.device, dtype=self.std_y.dtype))
        self._stats_ready.fill_(True)

    def _normalize_parents(self, parents: torch.Tensor) -> torch.Tensor:
        if self.input_dim == 0:
            return parents
        mean_x = self.mean_x.to(device=parents.device, dtype=parents.dtype).view(1, -1)
        std_x = self.std_x.to(device=parents.device, dtype=parents.dtype).view(1, -1)
        return (parents - mean_x) / std_x

    def _features(self, parents: torch.Tensor) -> torch.Tensor:
        if self.input_dim == 0:
            return parents.new_zeros(parents.shape[0], 0)
        parents = self._normalize_parents(parents)
        w = self._rff_w.to(device=parents.device, dtype=parents.dtype)
        b = self._rff_b.to(device=parents.device, dtype=parents.dtype)
        proj = parents @ w.t() + b
        scale = math.sqrt(2.0 / float(self.n_features))
        return scale * torch.cos(proj)

    def _fit_closed_form(
        self, features: torch.Tensor, targets: torch.Tensor, ridge: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if features.shape[0] == 0:
            raise ValueError("Cannot fit RFFGaussianCPD with zero rows")
        if ridge < 0:
            raise ValueError("ridge must be >= 0")
        if self.use_bias:
            ones = torch.ones(
                features.shape[0], 1, device=features.device, dtype=features.dtype
            )
            feats = torch.cat([features, ones], dim=1)
        else:
            feats = features
        if ridge > 0:
            eye = torch.eye(
                feats.shape[1], device=features.device, dtype=features.dtype
            )
            theta = torch.linalg.solve(
                feats.t() @ feats + ridge * eye, feats.t() @ targets
            )
        else:
            theta = torch.linalg.lstsq(feats, targets).solution
        if self.use_bias:
            coef = theta[:-1]
            bias = theta[-1]
        else:
            coef = theta
            bias = torch.zeros(
                targets.shape[1], device=targets.device, dtype=targets.dtype
            )
        residual = targets - (features @ coef + bias)
        var = residual.var(dim=0, unbiased=False).clamp_min(1e-6)
        return coef, bias, var

    def _scale(self) -> torch.Tensor:
        min_var = float(self.min_scale) ** 2
        return torch.sqrt(self._var.clamp(min=min_var))

    def _params(
        self, parents: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.input_dim == 0:
            loc = self.mean_y
            scale = self._scale()
            return loc, scale

        if parents is None:
            raise ValueError("parents cannot be None when input_dim > 0")
        if parents.dim() == 2:
            parents = parents.unsqueeze(1)
        flat, b, s = flatten_samples(parents)
        feats = self._features(flat)
        loc_norm = feats @ self._coef + self._bias
        loc_norm = loc_norm.reshape(b, s, self.output_dim)
        mean_y = self.mean_y.to(device=loc_norm.device, dtype=loc_norm.dtype).view(
            1, 1, -1
        )
        std_y = self.std_y.to(device=loc_norm.device, dtype=loc_norm.dtype).view(
            1, 1, -1
        )
        loc = loc_norm * std_y + mean_y
        scale = self._scale().view(1, 1, -1).expand(b, s, -1)
        return loc, scale

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
        parents, x_flat = self._prepare_training_tensors(parents, x)
        self._update_standardization(parents, x_flat)
        if self.input_dim == 0:
            var = (self.std_y**2).clamp_min(1e-6)
            self._coef.zero_()
            self._bias.zero_()
            self._var.copy_(var)
            return

        x_norm = (x_flat - self.mean_y) / self.std_y
        features = self._features(parents)
        ridge = self.ridge if ridge is None else float(ridge)
        coef, bias, var_norm = self._fit_closed_form(features, x_norm, ridge)
        self._coef.copy_(coef)
        self._bias.copy_(bias)
        var = var_norm * (self.std_y**2)
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

    def sample(self, parents: Optional[torch.Tensor], n_samples: int) -> torch.Tensor:
        self._check_fitted()
        if self.input_dim == 0:
            b = 1 if parents is None else parents.shape[0]
            loc = self.mean_y.view(1, 1, -1).expand(b, n_samples, -1)
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
        self._check_fitted()
        if x.dim() <= 2:
            x = ensure_2d(x)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if self.input_dim == 0:
            b, s, _ = x.shape
            loc = self.mean_y.view(1, 1, -1).expand(b, s, -1)
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
