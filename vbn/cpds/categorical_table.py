from __future__ import annotations

from typing import Optional

import torch

from vbn.core.base import BaseCPD
from vbn.core.registry import register_cpd
from vbn.core.utils import broadcast_samples, ensure_2d, flatten_samples


def _map_values_to_indices(values: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    support = support.contiguous()
    values = values.contiguous()
    idx = torch.searchsorted(support, values)
    if torch.any(idx < 0) or torch.any(idx >= support.shape[0]):
        raise ValueError("Found values outside support.")
    if not torch.all(support[idx] == values):
        raise ValueError("Found values outside support.")
    return idx


@register_cpd("categorical_table")
class CategoricalTableCPD(BaseCPD):
    """Tabular categorical CPD with Dirichlet/Laplace smoothing."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        device: torch.device,
        seed: Optional[int] = None,
        n_classes: int = 0,
        alpha: float = 1.0,
    ) -> None:
        super().__init__(
            input_dim=input_dim, output_dim=output_dim, device=device, seed=seed
        )
        self.n_classes = int(n_classes)
        self.alpha = float(alpha)
        if self.alpha < 0:
            raise ValueError("alpha must be >= 0")

        self._parent_values: Optional[list[torch.Tensor]] = None
        self._parent_cards: Optional[list[int]] = None
        self._parent_strides: Optional[list[int]] = None

        self.register_buffer("_class_values", torch.zeros(1, 1, device=self.device))
        self.register_buffer("_sample_values", torch.zeros(1, 1, device=self.device))
        self.register_buffer(
            "_class_mask", torch.zeros(1, 1, device=self.device, dtype=torch.bool)
        )
        self.register_buffer("_counts", torch.zeros(1, 1, 1, device=self.device))
        self.register_buffer("_stats_ready", torch.tensor(False, device=self.device))

    def get_init_kwargs(self) -> dict:
        return {"n_classes": self.n_classes, "alpha": self.alpha}

    def get_extra_state(self) -> Optional[dict]:
        return {
            "parent_values": self._parent_values,
            "parent_cards": self._parent_cards,
            "parent_strides": self._parent_strides,
        }

    def set_extra_state(self, state: Optional[dict]) -> None:
        if not state:
            self._parent_values = None
            self._parent_cards = None
            self._parent_strides = None
            self._stats_ready.fill_(False)
            return
        self._parent_values = state.get("parent_values")
        self._parent_cards = state.get("parent_cards")
        self._parent_strides = state.get("parent_strides")

    def _ensure_ready(self) -> None:
        if not bool(self._stats_ready.item()):
            raise RuntimeError("CategoricalTableCPD is not fitted yet.")

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

    def _infer_parent_support(self, parents: torch.Tensor) -> None:
        parent_values = []
        parent_cards = []
        for d in range(self.input_dim):
            uniq = torch.unique(parents[:, d], sorted=True)
            parent_values.append(uniq)
            parent_cards.append(int(uniq.numel()))
        strides = []
        stride = 1
        for card in reversed(parent_cards):
            strides.append(stride)
            stride *= card
        strides = list(reversed(strides))
        self._parent_values = parent_values
        self._parent_cards = parent_cards
        self._parent_strides = strides

    def _infer_class_support(
        self, x_flat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_classes = self.n_classes if self.n_classes > 0 else None
        class_values = []
        class_mask = []
        max_classes = 0
        for d in range(self.output_dim):
            uniq = torch.unique(x_flat[:, d], sorted=True)
            max_classes = max(max_classes, int(uniq.numel()))
            class_values.append(uniq)
        if n_classes is None:
            n_classes = max_classes
        self.n_classes = int(n_classes)
        for d in range(self.output_dim):
            uniq = class_values[d]
            if int(uniq.numel()) > n_classes:
                raise ValueError(
                    f"Found {int(uniq.numel())} classes for dim {d}, "
                    f"but n_classes={n_classes}."
                )
            mask = torch.zeros(n_classes, device=self.device, dtype=torch.bool)
            mask[: int(uniq.numel())] = True
            padded = torch.zeros(n_classes, device=self.device, dtype=x_flat.dtype)
            if uniq.numel() > 0:
                padded[: int(uniq.numel())] = uniq.to(
                    device=padded.device, dtype=padded.dtype
                )
            class_mask.append(mask)
            class_values[d] = padded
        class_values_tensor = torch.stack(class_values, dim=0)
        class_mask_tensor = torch.stack(class_mask, dim=0)
        return class_values_tensor, class_mask_tensor

    def _parents_to_index(self, parents: torch.Tensor) -> torch.Tensor:
        if self.input_dim == 0:
            return torch.zeros(
                parents.shape[0], device=parents.device, dtype=torch.long
            )
        if self._parent_values is None or self._parent_strides is None:
            raise RuntimeError("Parent support not initialized.")
        idx = torch.zeros(parents.shape[0], device=parents.device, dtype=torch.long)
        for d, support in enumerate(self._parent_values):
            support = support.to(device=parents.device, dtype=parents.dtype)
            indices = _map_values_to_indices(parents[:, d], support)
            idx = idx + indices * int(self._parent_strides[d])
        return idx

    def _targets_to_index(self, x_flat: torch.Tensor) -> torch.Tensor:
        class_values = self._class_values.to(device=x_flat.device, dtype=x_flat.dtype)
        class_mask = self._class_mask.to(device=x_flat.device)
        target_idx = torch.zeros(
            x_flat.shape[0], self.output_dim, device=x_flat.device, dtype=torch.long
        )
        for d in range(self.output_dim):
            support = class_values[d][class_mask[d]]
            target_idx[:, d] = _map_values_to_indices(x_flat[:, d], support)
        return target_idx

    def fit(
        self,
        parents: Optional[torch.Tensor],
        x: torch.Tensor,
        epochs: int = 1,
        lr: float = 1e-3,
        batch_size: int = 128,
        weight_decay: float = 0.0,
        **kwargs,
    ) -> None:
        del epochs, lr, batch_size, weight_decay, kwargs
        parents, x_flat = self._prepare_training_tensors(parents, x)

        self._infer_parent_support(parents)
        class_values, class_mask = self._infer_class_support(x_flat)
        n_classes = int(class_values.shape[1])

        self._class_values = class_values.to(
            device=self.device, dtype=self._class_values.dtype
        )
        self._sample_values = self._class_values
        self._class_mask = class_mask.to(
            device=self.device, dtype=self._class_mask.dtype
        )

        parent_state_count = 1
        for card in self._parent_cards or []:
            parent_state_count *= int(card)
        counts = torch.full(
            (self.output_dim, parent_state_count, n_classes),
            float(self.alpha),
            device=self.device,
            dtype=x_flat.dtype,
        )
        if class_mask.numel() > 0:
            invalid = ~class_mask.to(device=counts.device)
            counts = counts.masked_fill(invalid.unsqueeze(1), 0.0)

        parent_idx = self._parents_to_index(parents)
        target_idx = self._targets_to_index(x_flat)
        ones = torch.ones(parent_idx.shape[0], device=self.device, dtype=counts.dtype)
        for d in range(self.output_dim):
            flat = parent_idx * n_classes + target_idx[:, d]
            counts_d = counts[d].view(-1)
            counts_d.scatter_add_(0, flat, ones)

        self._counts = counts
        self._stats_ready.fill_(True)

    def update(
        self,
        parents: Optional[torch.Tensor],
        x: torch.Tensor,
        lr: float = 1e-3,
        n_steps: int = 1,
        batch_size: int = 128,
        weight_decay: float = 0.0,
        **kwargs,
    ) -> None:
        del lr, n_steps, batch_size, weight_decay, kwargs
        self.fit(parents, x)

    def _logits_from_parents(self, parents: torch.Tensor) -> torch.Tensor:
        if parents.dim() == 2:
            parents = parents.unsqueeze(1)
        flat, b, s = flatten_samples(parents)
        parent_idx = self._parents_to_index(flat)
        counts = self._counts
        probs = counts[:, parent_idx, :]
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        logits = torch.log(probs.clamp_min(1e-12))
        logits = logits.permute(1, 0, 2).reshape(b, s, self.output_dim, -1)
        return logits

    def _params(self, parents: Optional[torch.Tensor]) -> torch.Tensor:
        if parents is None:
            counts = self._counts
            probs = counts / counts.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            logits = torch.log(probs.clamp_min(1e-12)).view(1, 1, self.output_dim, -1)
            return logits
        return self._logits_from_parents(parents)

    def sample(self, parents: Optional[torch.Tensor], n_samples: int) -> torch.Tensor:
        self._ensure_ready()
        if self.input_dim == 0:
            b = 1 if parents is None else parents.shape[0]
            logits = self._params(None).expand(b, n_samples, -1, -1)
        else:
            if parents is None:
                raise ValueError("parents cannot be None when input_dim > 0")
            parents = broadcast_samples(parents, n_samples)
            logits = self._logits_from_parents(parents)
        dist = torch.distributions.Categorical(logits=logits)
        indices = dist.sample()
        values = self._class_values.to(device=indices.device, dtype=logits.dtype)
        values = values.view(1, 1, self.output_dim, -1).expand(
            indices.shape[0], indices.shape[1], -1, -1
        )
        samples = values.gather(-1, indices.unsqueeze(-1)).squeeze(-1)
        return samples

    def log_prob(
        self, x: torch.Tensor, parents: Optional[torch.Tensor]
    ) -> torch.Tensor:
        self._ensure_ready()
        if x.dim() <= 2:
            x = ensure_2d(x)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if self.input_dim == 0:
            logits = self._params(None).expand(x.shape[0], x.shape[1], -1, -1)
        else:
            if parents is None:
                raise ValueError("parents cannot be None when input_dim > 0")
            parents = broadcast_samples(parents, x.shape[1])
            logits = self._logits_from_parents(parents)
        log_probs = torch.log_softmax(logits, dim=-1)
        targets = self._targets_to_index(x.reshape(-1, x.shape[-1]))
        targets = targets.reshape(x.shape[0], x.shape[1], self.output_dim)
        logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return logp.sum(dim=-1)
