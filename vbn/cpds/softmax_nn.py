from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch import nn

from vbn.config_cast import coerce_numbers, UPDATE_SCHEMA
from vbn.core.base import BaseCPD
from vbn.core.registry import register_cpd
from vbn.core.utils import broadcast_samples, ensure_2d, flatten_samples

_BINNING_IDS = {"uniform": 0, "gaussian": 1, "quantile": 2}
_WITHIN_BIN_OPTIONS = {"uniform", "triangular", "gaussian"}
_MODE_WHEN_NOT_DISCRETE_OPTIONS = {"binned"}


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


@register_cpd("softmax_nn")
class SoftmaxNNCPD(BaseCPD):
    """Categorical CPD with discrete and continuous modes.

    If a target dimension has exactly `n_classes` unique values, it is treated as
    discrete and modeled as a categorical distribution over those class values.
    Otherwise the target is treated as continuous: the model predicts a categorical
    distribution over bins and a within-bin distribution to yield continuous samples
    and a proper continuous density. Training expects targets shaped [B, D] and
    handles [B, 1, D] inputs by squeezing the sample dimension.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        device: torch.device,
        seed: Optional[int] = None,
        n_classes: int = 8,
        hidden_dims: Iterable[int] = (32, 32),
        activation: str = "relu",
        label_smoothing: float = 0.0,
        min_bin_width: float = 1e-12,
        binning: str = "uniform",
        within_bin: str = "uniform",
        within_bin_scale: float = 0.25,
        within_bin_clip: bool = False,
        mode_when_not_discrete: str = "binned",
        class_weighting: str = "none",
        debug: bool = False,
        debug_every: int = 0,
    ) -> None:
        super().__init__(
            input_dim=input_dim, output_dim=output_dim, device=device, seed=seed
        )
        self.n_classes = int(n_classes)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.activation = str(activation)
        self.label_smoothing = float(label_smoothing)
        self.min_bin_width = float(min_bin_width)
        self.binning = str(binning).lower().strip()
        self.within_bin = str(within_bin).lower().strip()
        self.within_bin_scale = float(within_bin_scale)
        self.within_bin_clip = bool(within_bin_clip)
        self.mode_when_not_discrete = str(mode_when_not_discrete).lower().strip()
        self.class_weighting = str(class_weighting).lower().strip()
        self.debug = bool(debug)
        self.debug_every = int(debug_every)
        if self.n_classes <= 0:
            raise ValueError("n_classes must be >= 1")
        if self.binning not in _BINNING_IDS:
            raise ValueError(f"Unknown binning '{binning}'")
        if self.within_bin not in _WITHIN_BIN_OPTIONS:
            raise ValueError(f"Unknown within_bin '{within_bin}'")
        if self.mode_when_not_discrete not in _MODE_WHEN_NOT_DISCRETE_OPTIONS:
            raise ValueError(
                f"Unknown mode_when_not_discrete '{mode_when_not_discrete}'"
            )
        if self.class_weighting not in {"none", "inverse_freq"}:
            raise ValueError(f"Unknown class_weighting '{class_weighting}'")
        if self.debug_every < 0:
            raise ValueError("debug_every must be >= 0")

        if self.input_dim == 0:
            self._logits = nn.Parameter(
                torch.zeros(self.output_dim, self.n_classes, device=self.device)
            )
            self.net = None
        else:
            self.net = _build_mlp(
                self.input_dim,
                self.hidden_dims,
                self.output_dim * self.n_classes,
                self.activation,
            ).to(self.device)

        self.register_buffer("_vmin", torch.zeros(self.output_dim, device=self.device))
        self.register_buffer("_vmax", torch.zeros(self.output_dim, device=self.device))
        self.register_buffer(
            "_bin_edges",
            torch.zeros(self.output_dim, self.n_classes + 1, device=self.device),
        )
        self.register_buffer(
            "_bin_centers",
            torch.zeros(self.output_dim, self.n_classes, device=self.device),
        )
        self.register_buffer(
            "_class_values",
            torch.zeros(self.output_dim, self.n_classes, device=self.device),
        )
        self.register_buffer(
            "_sample_values",
            torch.zeros(self.output_dim, self.n_classes, device=self.device),
        )
        self.register_buffer(
            "_is_discrete",
            torch.zeros(self.output_dim, device=self.device, dtype=torch.bool),
        )
        self.register_buffer(
            "_n_classes",
            torch.tensor(self.n_classes, device=self.device, dtype=torch.long),
        )
        self.register_buffer(
            "_binning_id",
            torch.tensor(
                _BINNING_IDS[self.binning], device=self.device, dtype=torch.long
            ),
        )
        self.register_buffer("_bins_ready", torch.tensor(False, device=self.device))

        self._optimizer: Optional[torch.optim.Optimizer] = None

    def get_init_kwargs(self) -> dict:
        return {
            "n_classes": self.n_classes,
            "hidden_dims": self.hidden_dims,
            "activation": self.activation,
            "label_smoothing": self.label_smoothing,
            "min_bin_width": self.min_bin_width,
            "binning": self.binning,
            "within_bin": self.within_bin,
            "within_bin_scale": self.within_bin_scale,
            "within_bin_clip": self.within_bin_clip,
            "mode_when_not_discrete": self.mode_when_not_discrete,
            "class_weighting": self.class_weighting,
            "debug": self.debug,
            "debug_every": self.debug_every,
        }

    def _bins_initialized(self) -> bool:
        return bool(self._bins_ready.item())

    def _ensure_bins_ready(self) -> None:
        if not self._bins_initialized():
            raise RuntimeError("Bins not initialized. Call fit(...) before sampling.")

    def _flatten_training_x(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if x.dim() == 2:
            return x
        if x.dim() == 3:
            return x.reshape(-1, x.shape[-1])
        raise ValueError(f"Expected x with 1D, 2D, or 3D shape, got {tuple(x.shape)}")

    def _prepare_training_tensors(
        self, parents: Optional[torch.Tensor], x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_flat = self._flatten_training_x(x)
        if parents is None:
            if self.input_dim != 0:
                raise ValueError("parents cannot be None when input_dim > 0")
            parents = torch.zeros(x_flat.shape[0], 0, device=self.device)
            return parents, x_flat

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
                f"Expected parents with 2D or 3D shape, got {tuple(parents.shape)}"
            )
        return parents, x_flat

    def _compute_data_min_max(
        self, x_flat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return x_flat.amin(dim=0), x_flat.amax(dim=0)

    def _normalize_range(
        self, vmin: torch.Tensor, vmax: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vmin = vmin.to(device=self.device)
        vmax = vmax.to(device=self.device)
        min_range = float(self.min_bin_width) * float(self.n_classes)
        if min_range > 0:
            span = vmax - vmin
            vmax = torch.where(span < min_range, vmin + min_range, vmax)
        return vmin, vmax

    def _enforce_min_bin_width(self, edges: torch.Tensor) -> torch.Tensor:
        if self.min_bin_width <= 0:
            return edges
        out = edges.clone()
        for i in range(1, out.shape[1]):
            min_edge = out[:, i - 1] + self.min_bin_width
            out[:, i] = torch.maximum(out[:, i], min_edge)
        return out

    def _compute_bin_edges(
        self, x_flat: torch.Tensor, vmin: torch.Tensor, vmax: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = x_flat.dtype
        device = x_flat.device
        q = torch.linspace(0.0, 1.0, self.n_classes + 1, device=device, dtype=dtype)
        if self.binning == "uniform":
            width = (vmax - vmin) / float(self.n_classes)
            width = torch.clamp(width, min=self.min_bin_width)
            edges = vmin[:, None] + width[:, None] * q[None, :]
        elif self.binning == "gaussian":
            mean = x_flat.mean(dim=0)
            std = x_flat.std(dim=0, unbiased=False)
            std = torch.clamp(std, min=self.min_bin_width)
            dist = torch.distributions.Normal(mean, std)
            eps = 1e-6
            q_safe = q.clamp(eps, 1.0 - eps)
            edges = dist.icdf(q_safe.unsqueeze(0))
            edges[:, 0] = vmin
            edges[:, -1] = vmax
        elif self.binning == "quantile":
            edges = torch.quantile(x_flat, q, dim=0).transpose(0, 1)
            edges[:, 0] = vmin
            edges[:, -1] = vmax
        else:
            raise ValueError(f"Unknown binning '{self.binning}'")

        edges = self._enforce_min_bin_width(edges)
        centers = 0.5 * (edges[:, :-1] + edges[:, 1:])
        return edges, centers

    def _detect_discrete_classes(
        self, x_flat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        is_discrete = torch.zeros(self.output_dim, device=self.device, dtype=torch.bool)
        class_values = torch.zeros(
            self.output_dim, self.n_classes, device=self.device, dtype=x_flat.dtype
        )
        for d in range(self.output_dim):
            uniq = torch.unique(x_flat[:, d], sorted=True)
            if uniq.numel() == self.n_classes:
                is_discrete[d] = True
                class_values[d] = uniq
        return is_discrete, class_values

    def _set_bin_state(
        self,
        vmin: torch.Tensor,
        vmax: torch.Tensor,
        edges: torch.Tensor,
        centers: torch.Tensor,
        class_values: torch.Tensor,
        sample_values: torch.Tensor,
        is_discrete: torch.Tensor,
    ) -> None:
        self._vmin.copy_(vmin.to(self._vmin.device, dtype=self._vmin.dtype))
        self._vmax.copy_(vmax.to(self._vmax.device, dtype=self._vmax.dtype))
        self._bin_edges.copy_(
            edges.to(self._bin_edges.device, dtype=self._bin_edges.dtype)
        )
        self._bin_centers.copy_(
            centers.to(self._bin_centers.device, dtype=self._bin_centers.dtype)
        )
        self._class_values.copy_(
            class_values.to(self._class_values.device, dtype=self._class_values.dtype)
        )
        self._sample_values.copy_(
            sample_values.to(
                self._sample_values.device, dtype=self._sample_values.dtype
            )
        )
        self._is_discrete.copy_(
            is_discrete.to(self._is_discrete.device, dtype=self._is_discrete.dtype)
        )
        self._bins_ready.fill_(True)

    def _merge_sample_values(
        self,
        centers: torch.Tensor,
        class_values: torch.Tensor,
        is_discrete: torch.Tensor,
    ) -> torch.Tensor:
        return torch.where(is_discrete[:, None], class_values, centers)

    def _check_discrete_membership(self, x_flat: torch.Tensor) -> None:
        if not self._is_discrete.any():
            return
        class_values = self._class_values.to(device=x_flat.device, dtype=x_flat.dtype)
        match = x_flat.unsqueeze(-1) == class_values.unsqueeze(0)
        has_match = match.any(dim=-1)
        missing = (~has_match) & self._is_discrete.unsqueeze(0)
        if missing.any():
            raise ValueError("Found values outside discrete class set during update.")

    def _update_bins_from_data(
        self,
        x_flat: torch.Tensor,
        *,
        allow_expand: bool,
        force: bool = False,
    ) -> None:
        data_min, data_max = self._compute_data_min_max(x_flat)
        if force or not self._bins_initialized():
            is_discrete, class_values = self._detect_discrete_classes(x_flat)
            vmin, vmax = self._normalize_range(data_min, data_max)
            edges, centers = self._compute_bin_edges(x_flat, vmin, vmax)
            sample_values = self._merge_sample_values(
                centers, class_values, is_discrete
            )
            self._set_bin_state(
                vmin=vmin,
                vmax=vmax,
                edges=edges,
                centers=centers,
                class_values=class_values,
                sample_values=sample_values,
                is_discrete=is_discrete,
            )
            return

        self._check_discrete_membership(x_flat)
        if not allow_expand:
            return

        new_vmin = torch.minimum(
            self._vmin.to(data_min.device, data_min.dtype), data_min
        )
        new_vmax = torch.maximum(
            self._vmax.to(data_max.device, data_max.dtype), data_max
        )
        if torch.any(new_vmin < self._vmin) or torch.any(new_vmax > self._vmax):
            vmin, vmax = self._normalize_range(new_vmin, new_vmax)
            edges, centers = self._compute_bin_edges(x_flat, vmin, vmax)
            class_values = self._class_values.to(
                device=centers.device, dtype=centers.dtype
            )
            sample_values = self._merge_sample_values(
                centers,
                class_values,
                self._is_discrete,
            )
            self._set_bin_state(
                vmin=vmin,
                vmax=vmax,
                edges=edges,
                centers=centers,
                class_values=class_values,
                sample_values=sample_values,
                is_discrete=self._is_discrete,
            )

    def _train_loop(
        self,
        parents: Optional[torch.Tensor],
        x: torch.Tensor,
        epochs: int = 1,
        lr: float = 1e-3,
        batch_size: int = 128,
        weight_decay: float = 0.0,
        n_steps: Optional[int] = None,
        *,
        allow_expand: bool = False,
        force_bins: bool = False,
    ) -> None:
        params = {
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
        }
        if n_steps is not None:
            params["n_steps"] = n_steps
        params = coerce_numbers(params, UPDATE_SCHEMA | {"epochs": int})
        parents, x_flat = self._prepare_training_tensors(parents, x)
        self._update_bins_from_data(x_flat, allow_expand=allow_expand, force=force_bins)
        epochs = params["epochs"]
        lr = params["lr"]
        batch_size = params["batch_size"]
        weight_decay = params["weight_decay"]
        n_steps = params.get("n_steps", n_steps)

        dataset = torch.utils.data.TensorDataset(parents, x_flat)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        optimizer = getattr(self, "_optimizer", None)
        if optimizer is None:
            optimizer = torch.optim.Adam(
                self.parameters(), lr=lr, weight_decay=weight_decay
            )
            self._optimizer = optimizer
        weights = None
        if self.class_weighting == "inverse_freq":
            all_targets = self._x_to_bin(x_flat)
            if all_targets.dim() == 3 and all_targets.shape[1] == 1:
                all_targets = all_targets.squeeze(1)
            t_all = all_targets.reshape(-1).long()
            counts = torch.bincount(t_all, minlength=self.n_classes).float()
            weights = counts.sum() / counts.clamp_min(1.0)
            weights = weights / weights.mean().clamp_min(1e-12)
        loss_fn = nn.CrossEntropyLoss(
            weight=weights, label_smoothing=self.label_smoothing
        )
        steps = int(n_steps) if n_steps is not None else int(epochs)
        global_step = 0
        for _ in range(steps):
            for batch_parents, batch_x in loader:
                if batch_x.dim() == 3 and batch_x.shape[1] == 1:
                    batch_x = batch_x.squeeze(1)
                if self.input_dim == 0:
                    logits = self._logits.unsqueeze(0).expand(batch_x.shape[0], -1, -1)
                else:
                    if batch_parents.dim() == 2:
                        batch_parents = batch_parents.unsqueeze(1)
                    logits = self._logits_from_parents(batch_parents)
                    if logits.dim() == 4 and logits.shape[1] == 1:
                        logits = logits.squeeze(1)
                targets = self._x_to_bin(batch_x).long()
                if targets.dim() == 3 and targets.shape[1] == 1:
                    targets = targets.squeeze(1)
                if self.debug:
                    assert logits.shape == (
                        batch_x.shape[0],
                        self.output_dim,
                        self.n_classes,
                    ), f"logits shape {tuple(logits.shape)}"
                    assert targets.shape == (
                        batch_x.shape[0],
                        self.output_dim,
                    ), f"targets shape {tuple(targets.shape)}"
                logits_flat = logits.reshape(-1, self.n_classes)
                targets_flat = targets.reshape(-1)
                loss = loss_fn(logits_flat, targets_flat)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if self.debug and self.debug_every > 0:
                    if global_step % self.debug_every == 0:
                        with torch.no_grad():
                            t = targets_flat
                            counts = torch.bincount(t, minlength=self.n_classes).float()
                            probs = counts / counts.sum().clamp_min(1.0)
                            pred = logits.argmax(dim=-1).reshape(-1)
                            pred_counts = torch.bincount(
                                pred, minlength=self.n_classes
                            ).float()
                            pred_probs = pred_counts / pred_counts.sum().clamp_min(1.0)
                            max_show = min(8, self.n_classes)
                            edges0 = (
                                self._bin_edges[0, : min(5, self.n_classes + 1)]
                                .detach()
                                .cpu()
                                .tolist()
                            )
                            print(
                                "[softmax_nn debug]"
                                f" step={global_step}"
                                f" target_hist[:{max_show}]={probs[:max_show].cpu().tolist()}"
                                f" pred_hist[:{max_show}]={pred_probs[:max_show].cpu().tolist()}"
                                f" pred_minmax=({int(pred.min().item())},{int(pred.max().item())})"
                                f" vmin={float(self._vmin[0].item())}"
                                f" vmax={float(self._vmax[0].item())}"
                                f" edges0={edges0}"
                            )
                global_step += 1

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
            allow_expand=False,
            force_bins=True,
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
            allow_expand=True,
            force_bins=False,
        )

    def _logits_from_parents(self, parents: torch.Tensor) -> torch.Tensor:
        if parents.dim() == 2:
            parents = parents.unsqueeze(1)
        flat, b, s = flatten_samples(parents)
        out = self.net(flat)
        out = out.reshape(b, s, self.output_dim, self.n_classes)
        return out

    def _gather_bin_edges(
        self, indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        edges = self._bin_edges.to(device=indices.device)
        if indices.dim() == 3:
            d_idx = (
                torch.arange(self.output_dim, device=indices.device)
                .view(1, 1, -1)
                .expand(indices.shape[0], indices.shape[1], -1)
            )
        else:
            d_idx = (
                torch.arange(self.output_dim, device=indices.device)
                .view(1, -1)
                .expand(indices.shape[0], -1)
            )
        idx = indices.clamp(min=0, max=self.n_classes - 1)
        left = edges[d_idx, idx]
        right = edges[d_idx, (idx + 1).clamp(max=self.n_classes)]
        width = torch.clamp(right - left, min=self.min_bin_width)
        center = 0.5 * (left + right)
        return left, right, width, center

    def _x_to_bin(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_bins_ready()
        edges = self._bin_edges.to(device=x.device, dtype=x.dtype)
        flat = x.reshape(-1, x.shape[-1])
        cont_bins = (flat.unsqueeze(-1) >= edges.unsqueeze(0)).sum(dim=-1) - 1
        cont_bins = cont_bins.clamp(min=0, max=self.n_classes - 1)
        if self._is_discrete.any():
            class_values = self._class_values.to(device=x.device, dtype=x.dtype)
            match = flat.unsqueeze(-1) == class_values.unsqueeze(0)
            disc_bins = match.long().argmax(dim=-1)
            has_match = match.any(dim=-1)
            missing = (~has_match) & self._is_discrete.unsqueeze(0)
            if missing.any():
                raise ValueError("Found values outside discrete class set.")
            mask = self._is_discrete.unsqueeze(0)
            bins = torch.where(mask, disc_bins, cont_bins)
        else:
            bins = cont_bins
        return bins.reshape(*x.shape)

    def sample(self, parents: Optional[torch.Tensor], n_samples: int) -> torch.Tensor:
        self._ensure_bins_ready()
        if self.input_dim == 0:
            b = 1 if parents is None else parents.shape[0]
            logits = self._logits.view(1, 1, self.output_dim, self.n_classes).expand(
                b, n_samples, -1, -1
            )
        else:
            if parents is None:
                raise ValueError("parents cannot be None when input_dim > 0")
            parents = broadcast_samples(parents, n_samples)
            logits = self._logits_from_parents(parents)
        dist = torch.distributions.Categorical(logits=logits)
        indices = dist.sample()
        if self.mode_when_not_discrete != "binned" and (~self._is_discrete).any():
            raise NotImplementedError(
                f"mode_when_not_discrete='{self.mode_when_not_discrete}' is unsupported"
            )

        values = self._sample_values.to(device=indices.device, dtype=logits.dtype)
        values = values.view(1, 1, self.output_dim, self.n_classes).expand(
            indices.shape[0], indices.shape[1], -1, -1
        )
        disc_values = values.gather(-1, indices.unsqueeze(-1)).squeeze(-1)

        left, right, width, center = self._gather_bin_edges(indices)
        if self.within_bin == "uniform":
            u = torch.rand_like(center)
            cont_values = left + u * width
        elif self.within_bin == "triangular":
            u = torch.rand_like(center)
            left_vals = left + width * torch.sqrt(torch.clamp(u * 0.5, min=0.0))
            right_vals = right - width * torch.sqrt(
                torch.clamp((1.0 - u) * 0.5, min=0.0)
            )
            cont_values = torch.where(u < 0.5, left_vals, right_vals)
        elif self.within_bin == "gaussian":
            sigma = torch.clamp(self.within_bin_scale * width, min=self.min_bin_width)
            cont_values = center + torch.randn_like(center) * sigma
        else:
            raise ValueError(f"Unknown within_bin '{self.within_bin}'")

        if self.within_bin_clip:
            cont_values = cont_values.clamp(min=left, max=right)

        if self._is_discrete.any():
            mask = self._is_discrete.to(device=indices.device).view(1, 1, -1)
            return torch.where(mask, disc_values, cont_values)
        return cont_values

    def log_prob(
        self, x: torch.Tensor, parents: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if x.dim() <= 2:
            x = ensure_2d(x)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if self.input_dim == 0:
            b, s, _ = x.shape
            logits = self._logits.view(1, 1, self.output_dim, self.n_classes).expand(
                b, s, -1, -1
            )
        else:
            if parents is None:
                raise ValueError("parents cannot be None when input_dim > 0")
            parents = broadcast_samples(parents, x.shape[1])
            logits = self._logits_from_parents(parents)
        bins = self._x_to_bin(x).long()
        log_probs = torch.log_softmax(logits, dim=-1)
        log_bin = log_probs.gather(-1, bins.unsqueeze(-1)).squeeze(-1)

        left, right, width, center = self._gather_bin_edges(bins)
        if self.within_bin_clip:
            x_use = x.clamp(min=left, max=right)
        else:
            x_use = x

        if self.within_bin == "uniform":
            log_within = -torch.log(width)
            if not self.within_bin_clip:
                inside = (x >= left) & (x <= right)
                log_within = torch.where(
                    inside,
                    log_within,
                    torch.full_like(log_within, float("-inf")),
                )
        elif self.within_bin == "triangular":
            denom_left = torch.clamp(width * (center - left), min=self.min_bin_width**2)
            denom_right = torch.clamp(
                width * (right - center), min=self.min_bin_width**2
            )
            left_pdf = 2.0 * (x_use - left) / denom_left
            right_pdf = 2.0 * (right - x_use) / denom_right
            pdf = torch.where(x_use <= center, left_pdf, right_pdf)
            pdf = torch.clamp(pdf, min=0.0)
            log_within = torch.log(torch.clamp(pdf, min=1e-12))
            if not self.within_bin_clip:
                inside = (x >= left) & (x <= right)
                log_within = torch.where(
                    inside,
                    log_within,
                    torch.full_like(log_within, float("-inf")),
                )
        elif self.within_bin == "gaussian":
            sigma = torch.clamp(self.within_bin_scale * width, min=self.min_bin_width)
            log_within = torch.distributions.Normal(center, sigma).log_prob(x_use)
        else:
            raise ValueError(f"Unknown within_bin '{self.within_bin}'")

        if self._is_discrete.any():
            mask = (~self._is_discrete).to(device=x.device).view(1, 1, -1)
            log_within = torch.where(mask, log_within, torch.zeros_like(log_within))

        return (log_bin + log_within).sum(dim=-1)

    def debug_mode(self) -> dict:
        return {
            "n_classes": int(self.n_classes),
            "is_discrete": self._is_discrete.detach().cpu().tolist(),
            "vmin": self._vmin.detach().cpu().tolist(),
            "vmax": self._vmax.detach().cpu().tolist(),
            "within_bin": self.within_bin,
            "binning": self.binning,
            "mode_when_not_discrete": self.mode_when_not_discrete,
        }
