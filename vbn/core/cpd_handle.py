from __future__ import annotations

from typing import Optional

import torch

from vbn.core.base import CPDOutput
from vbn.core.registry import CPD_REGISTRY
from vbn.core.utils import ensure_2d, ensure_tensor, to_serializable


def _resolve_cpd_name(cpd) -> str:
    for key, cls in CPD_REGISTRY.items():
        if isinstance(cpd, cls):
            return key
    return type(cpd).__name__


def _clone_extra_state(state, detach: bool = True):
    if state is None:
        return None
    if isinstance(state, torch.Tensor):
        out = state.clone()
        return out.detach() if detach else out
    if isinstance(state, dict):
        return {k: _clone_extra_state(v, detach=detach) for k, v in state.items()}
    if isinstance(state, (list, tuple)):
        return [_clone_extra_state(v, detach=detach) for v in state]
    return state


def _ensure_3d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 2:
        return tensor.unsqueeze(1)
    if tensor.dim() == 3:
        return tensor
    raise ValueError(f"Expected 2D or 3D tensor, got shape {tuple(tensor.shape)}")


def _extract_normal_params(cpd, parents_tensor: Optional[torch.Tensor]):
    if hasattr(cpd, "_weight") and hasattr(cpd, "_bias") and hasattr(cpd, "_var"):
        if parents_tensor is None:
            loc = cpd._bias.view(1, 1, -1)
            if hasattr(cpd, "_scale") and callable(cpd._scale):
                scale = cpd._scale().view(1, 1, -1)
            else:
                scale = torch.sqrt(cpd._var.clamp_min(1e-12)).view(1, 1, -1)
            return loc, scale
        parents = _ensure_3d(parents_tensor)
        loc = parents @ cpd._weight + cpd._bias
        if hasattr(cpd, "_scale") and callable(cpd._scale):
            scale = cpd._scale().view(1, 1, -1).expand_as(loc)
        else:
            scale = torch.sqrt(cpd._var.clamp_min(1e-12)).view(1, 1, -1).expand_as(loc)
        return loc, scale

    if hasattr(cpd, "n_components"):
        return None
    if hasattr(cpd, "_params") and callable(cpd._params):
        try:
            loc, scale = None, None
            if parents_tensor is None:
                loc, scale = cpd._params(None)
            else:
                loc, scale = cpd._params(parents_tensor)
            return loc, scale
        except Exception:
            return None
    return None


def _extract_mixture_params(cpd, parents_tensor: Optional[torch.Tensor]):
    if not hasattr(cpd, "n_components"):
        return None
    if not hasattr(cpd, "_params") or not callable(cpd._params):
        return None
    try:
        logits, loc, scale = (
            cpd._params(None) if parents_tensor is None else cpd._params(parents_tensor)
        )
    except Exception:
        return None
    if logits.dim() == 1:
        logits = logits.view(1, 1, -1)
        loc = loc.view(1, 1, *loc.shape)
        scale = scale.view(1, 1, *scale.shape)
    weights = torch.softmax(logits, dim=-1)
    return weights, loc, scale


def _extract_categorical_probs(cpd, parents_tensor: Optional[torch.Tensor]):
    if not hasattr(cpd, "n_classes"):
        return None
    if hasattr(cpd, "_ensure_bins_ready") and callable(cpd._ensure_bins_ready):
        cpd._ensure_bins_ready()
    try:
        if parents_tensor is None:
            if hasattr(cpd, "_root_ready") and bool(cpd._root_ready.item()):
                logits = cpd._root_log_probs
            else:
                logits = cpd._logits
            logits = logits.view(1, 1, cpd.output_dim, cpd.n_classes)
        else:
            logits = cpd._logits_from_parents(parents_tensor)
        probs = torch.softmax(logits, dim=-1)
    except Exception:
        return None
    support = getattr(cpd, "_sample_values", None)
    return probs, support


class CPDHandle:
    def __init__(self, vbn, node: str) -> None:
        if node not in vbn.nodes:
            raise ValueError(f"Unknown node '{node}'.")
        self._vbn = vbn
        self._node = node
        self._cpd = vbn.nodes[node]
        self._parents = list(vbn.dag.parents(node))

    @property
    def node(self) -> str:
        return self._node

    @property
    def name(self) -> str:
        return self._node

    @property
    def cpd(self):
        return self._cpd

    @property
    def cpd_name(self) -> str:
        return _resolve_cpd_name(self._cpd)

    @property
    def cpd_type(self) -> str:
        return type(self._cpd).__name__

    @property
    def parents(self) -> list[str]:
        return list(self._parents)

    @property
    def is_fitted(self) -> bool:
        if hasattr(self._cpd, "is_fitted"):
            try:
                return bool(self._cpd.is_fitted)
            except Exception:
                pass
        for flag in ("_targets", "_bins_ready", "_stats_ready"):
            if hasattr(self._cpd, flag):
                value = getattr(self._cpd, flag)
                try:
                    if value is not None and bool(
                        value.item() if isinstance(value, torch.Tensor) else value
                    ):
                        return True
                except Exception:
                    continue
        return bool(self._cpd.state_dict())

    @property
    def device(self) -> torch.device:
        if hasattr(self._cpd, "device"):
            return torch.device(getattr(self._cpd, "device"))
        return self._vbn.device

    @property
    def x_dim(self) -> int:
        return int(self._cpd.output_dim)

    @property
    def output_dim(self) -> int:
        return self.x_dim

    @property
    def parents_dim(self) -> int:
        return int(self._cpd.input_dim)

    @property
    def input_dim(self) -> int:
        return self.parents_dim

    def _parents_tensor(self, parents) -> Optional[torch.Tensor]:
        if self.parents_dim == 0:
            if parents is None:
                return None
            if isinstance(parents, dict) and not parents:
                return None
            if isinstance(parents, torch.Tensor):
                t = ensure_tensor(parents, device=self._vbn.device)
                if t.dim() == 2 and t.shape[-1] == 0:
                    return t
            raise ValueError(f"Node '{self._node}' has no parents.")

        if parents is None:
            raise ValueError(f"Parents required for node '{self._node}'.")

        if isinstance(parents, dict):
            tensors = []
            for parent in self._parents:
                if parent not in parents:
                    raise ValueError(
                        f"Missing parent '{parent}' for node '{self._node}'."
                    )
                t = ensure_tensor(parents[parent], device=self._vbn.device)
                t = ensure_2d(t)
                tensors.append(t)
            if not tensors:
                return None
            parents_tensor = torch.cat(tensors, dim=-1)
            if parents_tensor.shape[-1] != self.parents_dim:
                raise ValueError(
                    f"Expected parents_dim {self.parents_dim}, got {parents_tensor.shape[-1]}"
                )
            return parents_tensor

        if isinstance(parents, torch.Tensor):
            t = ensure_tensor(parents, device=self._vbn.device)
            if t.dim() == 1:
                t = ensure_2d(t)
            if t.dim() not in (2, 3):
                raise ValueError(
                    f"Expected parents with 2D or 3D shape, got {tuple(t.shape)}"
                )
            if t.shape[-1] != self.parents_dim:
                raise ValueError(
                    f"Expected parents_dim {self.parents_dim}, got {t.shape[-1]}"
                )
            return t

        raise TypeError("parents must be a tensor or dict")

    def _x_tensor(self, x) -> torch.Tensor:
        t = ensure_tensor(x, device=self._vbn.device)
        if t.dim() == 1:
            t = ensure_2d(t)
        if t.dim() not in (2, 3):
            raise ValueError(f"Expected x with 2D or 3D shape, got {tuple(t.shape)}")
        return t

    def sample(self, parents, n_samples: int) -> torch.Tensor:
        parents_tensor = self._parents_tensor(parents)
        samples = self._cpd.sample(parents_tensor, int(n_samples))
        return samples.detach()

    def log_prob(self, x, parents) -> torch.Tensor:
        x_tensor = self._x_tensor(x)
        parents_tensor = self._parents_tensor(parents)
        log_prob = self._cpd.log_prob(x_tensor, parents_tensor)
        return log_prob.detach()

    def pdf(self, x, parents) -> torch.Tensor:
        return torch.exp(self.log_prob(x, parents))

    def forward(self, parents, n_samples: int) -> CPDOutput:
        parents_tensor = self._parents_tensor(parents)
        output = self._cpd.forward(parents_tensor, int(n_samples))
        return CPDOutput(
            samples=output.samples.detach(),
            log_prob=output.log_prob.detach(),
            pdf=output.pdf.detach(),
        )

    def summary(self) -> dict:
        return {
            "node": self.node,
            "parents": self.parents,
            "cpd_name": self.cpd_name,
            "cpd_type": self.cpd_type,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "device": str(self.device),
            "is_fitted": self.is_fitted,
        }

    def export_config(self) -> dict:
        init_kwargs = {}
        if hasattr(self._cpd, "get_init_kwargs") and callable(
            self._cpd.get_init_kwargs
        ):
            init_kwargs = self._cpd.get_init_kwargs() or {}
        extra_state = None
        if hasattr(self._cpd, "get_extra_state") and callable(
            self._cpd.get_extra_state
        ):
            extra_state = self._cpd.get_extra_state()
        return {
            "node": self.node,
            "parents": self.parents,
            "cpd_name": self.cpd_name,
            "cpd_type": self.cpd_type,
            "init_kwargs": to_serializable(init_kwargs),
            "extra_state": to_serializable(extra_state),
        }

    def state_dict(self) -> dict:
        if hasattr(self._cpd, "state_dict") and callable(self._cpd.state_dict):
            return self._cpd.state_dict()
        raise NotImplementedError("CPD does not expose state_dict().")

    def clone_cpd(self, detach: bool = True):
        if not hasattr(self._cpd, "get_init_kwargs") or not callable(
            self._cpd.get_init_kwargs
        ):
            raise ValueError(
                f"CPD '{self.cpd_type}' does not support get_init_kwargs()."
            )
        init_kwargs = self._cpd.get_init_kwargs() or {}
        try:
            new_cpd = type(self._cpd)(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                device=self.device,
                **init_kwargs,
            )
        except Exception as exc:
            raise ValueError(
                f"Could not reconstruct CPD '{self.cpd_type}': {exc}"
            ) from exc

        if hasattr(self._cpd, "state_dict") and callable(self._cpd.state_dict):
            new_cpd.load_state_dict(self._cpd.state_dict())
        if hasattr(self._cpd, "get_extra_state") and callable(
            self._cpd.get_extra_state
        ):
            extra_state = self._cpd.get_extra_state()
            if extra_state is not None and hasattr(new_cpd, "set_extra_state"):
                new_cpd.set_extra_state(_clone_extra_state(extra_state, detach=detach))
        if detach:
            for param in new_cpd.parameters():
                param.detach_()
            for buf in new_cpd.buffers():
                buf.detach_()
        return new_cpd

    def conditional(self, parents, *, n_samples: int = 1024) -> dict:
        parents_tensor = self._parents_tensor(parents)
        base = {
            "node": self.node,
            "parents": self.parents,
            "cpd_name": self.cpd_name,
            "cpd_type": self.cpd_type,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "conditioning": to_serializable(parents_tensor),
        }

        normal = _extract_normal_params(self._cpd, parents_tensor)
        if normal is not None:
            loc, scale = normal
            return {
                **base,
                "format": "normal_params",
                "mean": to_serializable(loc),
                "std": to_serializable(scale),
            }

        mixture = _extract_mixture_params(self._cpd, parents_tensor)
        if mixture is not None:
            weights, loc, scale = mixture
            return {
                **base,
                "format": "mixture_params",
                "weights": to_serializable(weights),
                "loc": to_serializable(loc),
                "scale": to_serializable(scale),
            }

        categorical = _extract_categorical_probs(self._cpd, parents_tensor)
        if categorical is not None:
            probs, support = categorical
            return {
                **base,
                "format": "categorical_probs",
                "probs": to_serializable(probs),
                "k": int(getattr(self._cpd, "n_classes", probs.shape[-1])),
                "support": to_serializable(support),
            }

        samples = self._cpd.sample(parents_tensor, int(n_samples)).detach()
        mean = samples.mean(dim=1, keepdim=False)
        std = samples.std(dim=1, unbiased=False, keepdim=False)
        return {
            **base,
            "format": "empirical_samples",
            "samples": to_serializable(samples),
            "mean": to_serializable(mean),
            "std": to_serializable(std),
            "n_samples": int(n_samples),
        }

    def conditional_samples(self, parents, n_samples: int = 1024) -> torch.Tensor:
        return self.sample(parents, n_samples)

    def conditional_log_prob(self, x, parents) -> torch.Tensor:
        return self.log_prob(x, parents)

    def conditional_pdf(self, x, parents) -> torch.Tensor:
        return self.pdf(x, parents)

    def conditional_mean_std(self, parents, n_samples: int = 1024) -> dict:
        parents_tensor = self._parents_tensor(parents)
        normal = _extract_normal_params(self._cpd, parents_tensor)
        if normal is not None:
            loc, scale = normal
            return {
                "format": "normal_params",
                "mean": loc.detach(),
                "std": scale.detach(),
            }
        samples = self._cpd.sample(parents_tensor, int(n_samples)).detach()
        return {
            "format": "empirical_samples",
            "mean": samples.mean(dim=1, keepdim=False),
            "std": samples.std(dim=1, unbiased=False, keepdim=False),
        }
