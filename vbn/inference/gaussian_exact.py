from __future__ import annotations

import math
from typing import Optional

import torch

from vbn.core.base import Query
from vbn.core.registry import INFERENCE_REGISTRY, register_inference
from vbn.inference._core import get_inference_state, prepare_fixed_values, resolve_dtype
from vbn.utils import infer_batch_size


@register_inference("gaussian_exact")
class GaussianExact:
    """Exact Gaussian posterior on a fixed support grid when possible."""

    def __init__(
        self,
        n_samples: int = 200,
        stddevs: float = 4.0,
        min_scale: float = 1e-6,
        fallback: str = "likelihood_weighting",
        **kwargs,
    ) -> None:
        self.n_samples = int(n_samples)
        self.stddevs = float(stddevs)
        self.min_scale = float(min_scale)
        self.fallback = (
            str(fallback).strip().lower() if fallback is not None else "none"
        )
        self._cache = {}
        self._fallback = None
        if self.fallback != "none":
            if self.fallback not in INFERENCE_REGISTRY:
                raise ValueError(
                    f"Unknown fallback inference '{fallback}'. Available: {list(INFERENCE_REGISTRY.keys())}"
                )
            if self.fallback == "gaussian_exact":
                raise ValueError("fallback cannot be 'gaussian_exact'")
            fallback_kwargs = dict(kwargs)
            fallback_kwargs.setdefault("n_samples", self.n_samples)
            self._fallback = INFERENCE_REGISTRY[self.fallback](**fallback_kwargs)

    def _fallback_infer(self, vbn, query: Query, **kwargs):
        if self._fallback is None:
            raise RuntimeError(
                "gaussian_exact cannot handle this query and has no fallback"
            )
        return self._fallback.infer_posterior(vbn, query, **kwargs)

    def _to_3d(self, tensor: torch.Tensor, batch_size: int) -> Optional[torch.Tensor]:
        if tensor.dim() == 1:
            tensor = tensor.view(1, 1, -1)
        elif tensor.dim() == 2:
            tensor = tensor.unsqueeze(1)
        elif tensor.dim() != 3:
            return None
        if tensor.shape[0] == 1 and batch_size > 1:
            tensor = tensor.expand(batch_size, -1, -1)
        if tensor.shape[0] != batch_size:
            return None
        return tensor

    def _gaussian_params(
        self,
        cpd,
        parents_tensor: Optional[torch.Tensor],
        batch_size: int,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        if hasattr(cpd, "n_classes") or hasattr(cpd, "n_components"):
            return None

        try:
            if (
                hasattr(cpd, "_weight")
                and hasattr(cpd, "_bias")
                and hasattr(cpd, "_var")
            ):
                if parents_tensor is None:
                    loc = cpd._bias.view(1, 1, -1)
                    if hasattr(cpd, "_scale") and callable(cpd._scale):
                        scale = cpd._scale().view(1, 1, -1)
                    else:
                        scale = torch.sqrt(
                            cpd._var.clamp_min(self.min_scale**2)
                        ).view(1, 1, -1)
                else:
                    parents = parents_tensor
                    if parents.dim() == 2:
                        parents = parents.unsqueeze(1)
                    loc = parents @ cpd._weight + cpd._bias
                    if hasattr(cpd, "_scale") and callable(cpd._scale):
                        scale = cpd._scale().view(1, 1, -1).expand_as(loc)
                    else:
                        scale = torch.sqrt(
                            cpd._var.clamp_min(self.min_scale**2)
                        ).view(1, 1, -1)
                        scale = scale.expand_as(loc)
            elif hasattr(cpd, "_params") and callable(cpd._params):
                loc, scale = (
                    cpd._params(None)
                    if parents_tensor is None
                    else cpd._params(parents_tensor)
                )
            else:
                return None
        except Exception:
            return None

        loc = self._to_3d(loc, batch_size)
        scale = self._to_3d(scale, batch_size)
        if loc is None or scale is None:
            return None

        if loc.shape[-1] != 1:
            return None
        if scale.shape[-1] != 1:
            return None

        if scale.shape[1] == 1 and loc.shape[1] > 1:
            scale = scale.expand(-1, loc.shape[1], -1)
        elif loc.shape[1] == 1 and scale.shape[1] > 1:
            loc = loc.expand(-1, scale.shape[1], -1)
        elif loc.shape[1] != scale.shape[1]:
            return None

        scale = torch.nan_to_num(
            scale, nan=self.min_scale, posinf=self.min_scale, neginf=self.min_scale
        ).abs()
        scale = scale.clamp_min(self.min_scale)
        return loc[:, :1, :], scale[:, :1, :]

    def infer_posterior(self, vbn, query: Query, **kwargs):
        n_samples = max(1, int(kwargs.get("n_samples", self.n_samples)))
        b = infer_batch_size(query.evidence, query.do)
        state = get_inference_state(vbn, query, self._cache)
        device = vbn.device
        dtype = resolve_dtype(vbn, query)

        fixed_values = prepare_fixed_values(query, state, device, dtype, clamp_obs=True)
        target_idx = state.target_idx
        target_slice = state.node_slices[target_idx]
        target_cpd = vbn.nodes[state.topo_order[target_idx]]

        if target_slice.stop - target_slice.start != 1:
            return self._fallback_infer(vbn, query, **kwargs)

        target_fixed = fixed_values[target_idx]
        if target_fixed is not None:
            target_value = target_fixed.unsqueeze(1).expand(b, 1, -1)
            weights = torch.ones(b, 1, device=device, dtype=dtype)
            return weights.detach(), target_value.detach()

        parents_idx = state.parent_idx[target_idx]
        parents_observed = all(fixed_values[p] is not None for p in parents_idx)
        if not parents_observed:
            return self._fallback_infer(vbn, query, **kwargs)

        if parents_idx:
            parent_tensor = torch.cat(
                [fixed_values[p].unsqueeze(1).expand(b, 1, -1) for p in parents_idx],
                dim=-1,
            )
        else:
            parent_tensor = None

        params = self._gaussian_params(target_cpd, parent_tensor, b)
        if params is None:
            return self._fallback_infer(vbn, query, **kwargs)

        loc, scale = params
        z = torch.linspace(
            -self.stddevs, self.stddevs, n_samples, device=device, dtype=dtype
        )
        z = z.view(1, n_samples, 1)
        samples = loc + scale * z

        log_pdf = -0.5 * (
            z**2 + 2.0 * torch.log(scale) + math.log(2.0 * math.pi)
        ).sum(dim=-1)
        pdf = torch.exp(log_pdf)
        return pdf.detach(), samples.detach()
