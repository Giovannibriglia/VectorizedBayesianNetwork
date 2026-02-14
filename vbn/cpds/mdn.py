from __future__ import annotations

import math
from typing import Iterable, Optional

import torch
from torch import nn

from vbn.core.base import BaseCPD
from vbn.core.registry import register_cpd
from vbn.core.utils import broadcast_samples, ensure_2d, flatten_samples


def _build_mlp(
    input_dim: int, hidden_dims: Iterable[int], output_dim: int, activation: str
) -> nn.Sequential:
    act_map = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
        "elu": nn.ELU,
    }
    if activation not in act_map:
        raise ValueError(f"Unknown activation '{activation}'")
    layers = []
    last = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(last, h))
        layers.append(act_map[activation]())
        last = h
    layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)


@register_cpd("mdn")
class MDNCPD(BaseCPD):
    """Mixture Density Network CPD (Gaussian mixtures)."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        device: torch.device,
        seed: Optional[int] = None,
        n_components: int = 5,
        hidden_dims: Iterable[int] = (32, 32),
        activation: str = "relu",
        min_scale: float = 1e-3,
    ) -> None:
        super().__init__(
            input_dim=input_dim, output_dim=output_dim, device=device, seed=seed
        )
        self.n_components = int(n_components)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.activation = str(activation)
        self.min_scale = float(min_scale)
        if self.input_dim == 0:
            self._logits = nn.Parameter(
                torch.zeros(self.n_components, device=self.device)
            )
            self._loc = nn.Parameter(
                torch.zeros(self.n_components, self.output_dim, device=self.device)
            )
            self._log_scale = nn.Parameter(
                torch.zeros(self.n_components, self.output_dim, device=self.device)
            )
            self.net = None
        else:
            out_dim = self.n_components * (2 * self.output_dim) + self.n_components
            self.net = _build_mlp(
                self.input_dim, self.hidden_dims, out_dim, self.activation
            ).to(self.device)
        self._optimizer: Optional[torch.optim.Optimizer] = None

    def get_init_kwargs(self) -> dict:
        return {
            "n_components": self.n_components,
            "hidden_dims": self.hidden_dims,
            "activation": self.activation,
            "min_scale": self.min_scale,
        }

    def _params(
        self, parents: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.input_dim == 0:
            logits = self._logits
            loc = self._loc
            scale = torch.nn.functional.softplus(self._log_scale) + self.min_scale
            return logits, loc, scale

        if parents is None:
            raise ValueError("parents cannot be None when input_dim > 0")
        if parents.dim() == 2:
            parents = parents.unsqueeze(1)
        flat, b, s = flatten_samples(parents)
        out = self.net(flat)
        out = out.reshape(b, s, -1)
        logits = out[..., : self.n_components]
        rest = out[..., self.n_components :]
        rest = rest.reshape(b, s, self.n_components, 2 * self.output_dim)
        loc = rest[..., : self.output_dim]
        log_scale = rest[..., self.output_dim :]
        scale = torch.nn.functional.softplus(log_scale) + self.min_scale
        return logits, loc, scale

    def sample(self, parents: Optional[torch.Tensor], n_samples: int) -> torch.Tensor:
        if self.input_dim == 0:
            b = 1 if parents is None else parents.shape[0]
            logits = self._logits.view(1, 1, -1).expand(b, n_samples, -1)
            loc = self._loc.view(1, 1, self.n_components, self.output_dim).expand(
                b, n_samples, -1, -1
            )
            scale = (
                (torch.nn.functional.softplus(self._log_scale) + self.min_scale)
                .view(1, 1, self.n_components, self.output_dim)
                .expand(b, n_samples, -1, -1)
            )
        else:
            if parents is None:
                raise ValueError("parents cannot be None when input_dim > 0")
            parents = broadcast_samples(parents, n_samples)
            logits, loc, scale = self._params(parents)
        probs = torch.softmax(logits, dim=-1)
        b, s, k = probs.shape
        comp = torch.multinomial(probs.reshape(b * s, k), num_samples=1).reshape(b, s)
        comp_expanded = (
            comp.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, self.output_dim)
        )
        loc_sel = torch.gather(loc, dim=2, index=comp_expanded).squeeze(2)
        scale_sel = torch.gather(scale, dim=2, index=comp_expanded).squeeze(2)
        eps = torch.randn_like(scale_sel)
        return loc_sel + eps * scale_sel

    def log_prob(
        self, x: torch.Tensor, parents: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if x.dim() <= 2:
            x = ensure_2d(x)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if self.input_dim == 0:
            b, s, _ = x.shape
            logits = self._logits.view(1, 1, -1).expand(b, s, -1)
            loc = self._loc.view(1, 1, self.n_components, self.output_dim).expand(
                b, s, -1, -1
            )
            scale = (
                (torch.nn.functional.softplus(self._log_scale) + self.min_scale)
                .view(1, 1, self.n_components, self.output_dim)
                .expand(b, s, -1, -1)
            )
        else:
            if parents is None:
                raise ValueError("parents cannot be None when input_dim > 0")
            parents = broadcast_samples(parents, x.shape[1])
            logits, loc, scale = self._params(parents)
        x_exp = x.unsqueeze(2)  # [B, S, 1, Dx]
        var = scale**2
        log_prob = -0.5 * (
            (x_exp - loc) ** 2 / var + 2 * torch.log(scale) + math.log(2 * math.pi)
        )
        log_prob = log_prob.sum(dim=-1)  # [B, S, K]
        log_mix = torch.log_softmax(logits, dim=-1)
        return torch.logsumexp(log_mix + log_prob, dim=-1)

    def _get_optimizer(self, lr: float, weight_decay: float) -> torch.optim.Optimizer:
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(
                self.parameters(), lr=lr, weight_decay=weight_decay
            )
        return self._optimizer

    def fit(
        self,
        parents: Optional[torch.Tensor],
        x: torch.Tensor,
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 128,
        weight_decay: float = 0.0,
        **kwargs,
    ) -> None:
        if parents is None:
            parents = torch.zeros(x.shape[0], 0, device=self.device)
        parents = ensure_2d(parents)
        x = ensure_2d(x)
        dataset = torch.utils.data.TensorDataset(parents, x)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        optimizer = self._get_optimizer(float(lr), weight_decay)
        for _ in range(int(epochs)):
            for batch_parents, batch_x in loader:
                log_prob = self.log_prob(batch_x, batch_parents)
                loss = -log_prob.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def update(
        self,
        parents: Optional[torch.Tensor],
        x: torch.Tensor,
        n_steps: int = 1,
        lr: float = 1e-3,
        batch_size: int = 128,
        weight_decay: float = 0.0,
        **kwargs,
    ) -> None:
        if parents is None:
            parents = torch.zeros(x.shape[0], 0, device=self.device)
        parents = ensure_2d(parents)
        x = ensure_2d(x)
        dataset = torch.utils.data.TensorDataset(parents, x)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        optimizer = self._get_optimizer(lr, weight_decay)
        for _ in range(int(n_steps)):
            for batch_parents, batch_x in loader:
                log_prob = self.log_prob(batch_x, batch_parents)
                loss = -log_prob.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
