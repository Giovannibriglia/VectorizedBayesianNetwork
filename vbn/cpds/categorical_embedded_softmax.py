from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from torch import nn

from vbn.config_cast import coerce_numbers, UPDATE_SCHEMA
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


def _map_values_to_indices(values: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    support = support.contiguous()
    values = values.contiguous()
    idx = torch.searchsorted(support, values)
    if torch.any(idx < 0) or torch.any(idx >= support.shape[0]):
        raise ValueError("Found values outside support.")
    if not torch.all(support[idx] == values):
        raise ValueError("Found values outside support.")
    return idx


@register_cpd("categorical_embedded_softmax")
class CategoricalEmbeddedSoftmaxCPD(BaseCPD):
    """Categorical CPD with embeddings for discrete parent variables."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        device: torch.device,
        seed: Optional[int] = None,
        n_classes: int = 0,
        embedding_dim: int = 8,
        hidden_dims: Iterable[int] = (64, 64),
        activation: str = "relu",
        label_smoothing: float = 0.0,
        class_weighting: str = "none",
        max_grad_norm: Optional[float] = None,
    ) -> None:
        super().__init__(
            input_dim=input_dim, output_dim=output_dim, device=device, seed=seed
        )
        self.n_classes = int(n_classes)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.activation = str(activation)
        self.label_smoothing = float(label_smoothing)
        self.class_weighting = str(class_weighting).lower().strip()
        self.max_grad_norm = max_grad_norm

        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be >= 1")
        if self.class_weighting not in {"none", "inverse_freq"}:
            raise ValueError("class_weighting must be 'none' or 'inverse_freq'")

        self._parent_values: Optional[list[torch.Tensor]] = None
        self._parent_cards: Optional[list[int]] = None

        self.register_buffer("_class_values", torch.zeros(1, 1, device=self.device))
        self.register_buffer("_sample_values", torch.zeros(1, 1, device=self.device))
        self.register_buffer(
            "_class_mask", torch.zeros(1, 1, device=self.device, dtype=torch.bool)
        )
        self.register_buffer("_stats_ready", torch.tensor(False, device=self.device))

        if self.input_dim == 0:
            self._logits = nn.Parameter(
                torch.zeros(self.output_dim, max(self.n_classes, 1), device=self.device)
            )
            self.embeddings = None
            self.net = None
        else:
            self._logits = None
            self.embeddings = None
            self.net = None

        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._current_classes: Optional[int] = None

    def get_init_kwargs(self) -> dict:
        return {
            "n_classes": self.n_classes,
            "embedding_dim": self.embedding_dim,
            "hidden_dims": self.hidden_dims,
            "activation": self.activation,
            "label_smoothing": self.label_smoothing,
            "class_weighting": self.class_weighting,
            "max_grad_norm": self.max_grad_norm,
        }

    def get_extra_state(self) -> Optional[dict]:
        return {
            "parent_values": self._parent_values,
            "parent_cards": self._parent_cards,
        }

    def set_extra_state(self, state: Optional[dict]) -> None:
        if not state:
            self._parent_values = None
            self._parent_cards = None
            self._stats_ready.fill_(False)
            return
        self._parent_values = state.get("parent_values")
        self._parent_cards = state.get("parent_cards")

    def _ensure_ready(self) -> None:
        if not bool(self._stats_ready.item()):
            raise RuntimeError("CategoricalEmbeddedSoftmaxCPD is not fitted yet.")

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
        self._parent_values = parent_values
        self._parent_cards = parent_cards

    def _infer_class_support(
        self, x_flat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
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
        return class_values_tensor, class_mask_tensor, n_classes

    def _build_modules(self, n_classes: int) -> None:
        if self.input_dim == 0:
            self._logits = nn.Parameter(
                torch.zeros(self.output_dim, n_classes, device=self.device)
            )
            self.embeddings = None
            self.net = None
            self._optimizer = None
            return
        if self._parent_cards is None:
            raise RuntimeError("Parent support not initialized.")
        self.embeddings = nn.ModuleList(
            [nn.Embedding(card, self.embedding_dim) for card in self._parent_cards]
        ).to(self.device)
        total_dim = self.embedding_dim * int(self.input_dim)
        out_dim = int(self.output_dim) * int(n_classes)
        self.net = _build_mlp(total_dim, self.hidden_dims, out_dim, self.activation).to(
            self.device
        )
        self._optimizer = None

    def _parents_to_indices(self, parents: torch.Tensor) -> torch.Tensor:
        if self.input_dim == 0:
            return torch.zeros(
                parents.shape[0], 0, device=parents.device, dtype=torch.long
            )
        if self._parent_values is None:
            raise RuntimeError("Parent support not initialized.")
        idx = torch.zeros(
            parents.shape[0], self.input_dim, device=parents.device, dtype=torch.long
        )
        for d, support in enumerate(self._parent_values):
            support = support.to(device=parents.device, dtype=parents.dtype)
            idx[:, d] = _map_values_to_indices(parents[:, d], support)
        return idx

    def _targets_to_indices(self, x_flat: torch.Tensor) -> torch.Tensor:
        class_values = self._class_values.to(device=x_flat.device, dtype=x_flat.dtype)
        class_mask = self._class_mask.to(device=x_flat.device)
        target_idx = torch.zeros(
            x_flat.shape[0], self.output_dim, device=x_flat.device, dtype=torch.long
        )
        for d in range(self.output_dim):
            support = class_values[d][class_mask[d]]
            target_idx[:, d] = _map_values_to_indices(x_flat[:, d], support)
        return target_idx

    def _embed_parents(self, parent_idx: torch.Tensor) -> torch.Tensor:
        if self.input_dim == 0:
            return parent_idx.new_zeros(parent_idx.shape[0], 0)
        if self.embeddings is None:
            raise RuntimeError("Embeddings not initialized.")
        parts = []
        for d, emb in enumerate(self.embeddings):
            parts.append(emb(parent_idx[:, d]))
        return torch.cat(parts, dim=-1)

    def _logits_from_parents(self, parents: torch.Tensor) -> torch.Tensor:
        if parents.dim() == 2:
            parents = parents.unsqueeze(1)
        flat, b, s = flatten_samples(parents)
        parent_idx = self._parents_to_indices(flat)
        if self.input_dim == 0:
            logits = self._logits.view(1, 1, self.output_dim, -1).expand(b, s, -1, -1)
        else:
            feats = self._embed_parents(parent_idx)
            out = self.net(feats).reshape(b, s, self.output_dim, -1)
            logits = out
        mask = self._class_mask.to(device=logits.device)
        logits = logits.masked_fill(~mask.view(1, 1, self.output_dim, -1), -1e9)
        return logits

    def _train_loop(
        self,
        parents: Optional[torch.Tensor],
        x: torch.Tensor,
        epochs: int = 1,
        lr: float = 1e-3,
        batch_size: int = 128,
        weight_decay: float = 0.0,
        n_steps: Optional[int] = None,
        max_grad_norm: Optional[float] = None,
    ) -> None:
        params = {
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
        }
        if max_grad_norm is not None:
            params["max_grad_norm"] = max_grad_norm
        if n_steps is not None:
            params["n_steps"] = n_steps
        params = coerce_numbers(params, UPDATE_SCHEMA | {"epochs": int})
        parents, x_flat = self._prepare_training_tensors(parents, x)

        self._infer_parent_support(parents)
        class_values, class_mask, n_classes = self._infer_class_support(x_flat)
        self._class_values = class_values.to(
            device=self.device, dtype=self._class_values.dtype
        )
        self._sample_values = self._class_values
        self._class_mask = class_mask.to(
            device=self.device, dtype=self._class_mask.dtype
        )
        rebuild = self._current_classes != n_classes or self.net is None
        if self.input_dim == 0:
            rebuild = rebuild or self._logits is None
        if self.embeddings is not None and self._parent_cards is not None:
            if len(self.embeddings) != len(self._parent_cards):
                rebuild = True
            else:
                for emb, card in zip(self.embeddings, self._parent_cards):
                    if int(emb.num_embeddings) != int(card):
                        rebuild = True
                        break
        if rebuild:
            self._build_modules(n_classes)
            self._current_classes = n_classes

        parent_idx = self._parents_to_indices(parents)
        targets = self._targets_to_indices(x_flat)

        dataset = torch.utils.data.TensorDataset(parent_idx, targets)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        optimizer = getattr(self, "_optimizer", None)
        if optimizer is None:
            optimizer = torch.optim.Adam(
                self.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
            )
            self._optimizer = optimizer

        weights = None
        if self.class_weighting == "inverse_freq":
            flat_targets = targets.reshape(-1)
            counts = torch.bincount(flat_targets, minlength=n_classes).float()
            weights = counts.sum() / counts.clamp_min(1.0)
            weights = weights / weights.mean().clamp_min(1e-12)

        steps = int(params.get("n_steps", n_steps) or params["epochs"])
        for _ in range(steps):
            for batch_parents, batch_targets in loader:
                if self.input_dim == 0:
                    logits = self._logits.view(1, self.output_dim, n_classes).expand(
                        batch_targets.shape[0], -1, -1
                    )
                else:
                    feats = self._embed_parents(batch_parents)
                    logits = self.net(feats).reshape(-1, self.output_dim, n_classes)
                mask = self._class_mask.to(device=logits.device)
                logits = logits.masked_fill(~mask.view(1, self.output_dim, -1), -1e9)
                logits_flat = logits.reshape(-1, n_classes)
                targets_flat = batch_targets.reshape(-1)
                loss = F.cross_entropy(
                    logits_flat,
                    targets_flat,
                    weight=weights,
                    label_smoothing=float(self.label_smoothing),
                )
                optimizer.zero_grad()
                loss.backward()
                max_norm = params.get("max_grad_norm", max_grad_norm)
                if max_norm is not None and max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)
                optimizer.step()

        self._stats_ready.fill_(True)

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
        self._train_loop(
            parents,
            x,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            weight_decay=weight_decay,
            max_grad_norm=self.max_grad_norm,
        )

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
        self._train_loop(
            parents,
            x,
            lr=lr,
            batch_size=batch_size,
            weight_decay=weight_decay,
            n_steps=n_steps,
            max_grad_norm=self.max_grad_norm,
        )

    def sample(self, parents: Optional[torch.Tensor], n_samples: int) -> torch.Tensor:
        self._ensure_ready()
        if self.input_dim == 0:
            b = 1 if parents is None else parents.shape[0]
            logits = self._logits.view(1, 1, self.output_dim, -1).expand(
                b, n_samples, -1, -1
            )
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
            logits = self._logits.view(1, 1, self.output_dim, -1).expand(
                x.shape[0], x.shape[1], -1, -1
            )
        else:
            if parents is None:
                raise ValueError("parents cannot be None when input_dim > 0")
            parents = broadcast_samples(parents, x.shape[1])
            logits = self._logits_from_parents(parents)
        log_probs = torch.log_softmax(logits, dim=-1)
        targets = self._targets_to_indices(x.reshape(-1, x.shape[-1]))
        targets = targets.reshape(x.shape[0], x.shape[1], self.output_dim)
        logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return logp.sum(dim=-1)
