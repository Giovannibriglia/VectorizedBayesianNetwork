from __future__ import annotations

from typing import Optional

import torch

from vbn.core.base import Query
from vbn.core.registry import INFERENCE_REGISTRY, register_inference
from vbn.inference._core import get_inference_state, prepare_fixed_values, resolve_dtype
from vbn.utils import infer_batch_size


@register_inference("categorical_exact")
class CategoricalExact:
    """Exact categorical posterior when the target CPD is categorical and parents observed."""

    def __init__(self, fallback: str = "likelihood_weighting", **kwargs) -> None:
        self.fallback = (
            str(fallback).strip().lower() if fallback is not None else "none"
        )
        self._fallback = None
        if self.fallback != "none":
            if self.fallback not in INFERENCE_REGISTRY:
                raise ValueError(
                    f"Unknown fallback inference '{fallback}'. Available: {list(INFERENCE_REGISTRY.keys())}"
                )
            if self.fallback == "categorical_exact":
                raise ValueError("fallback cannot be 'categorical_exact'")
            self._fallback = INFERENCE_REGISTRY[self.fallback](**kwargs)

    def _fallback_infer(self, vbn, query: Query, **kwargs):
        if self._fallback is None:
            raise RuntimeError(
                "categorical_exact cannot handle this query and has no fallback"
            )
        return self._fallback.infer_posterior(vbn, query, **kwargs)

    def _categorical_probs(
        self,
        cpd,
        parents_tensor: Optional[torch.Tensor],
        batch_size: int,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        if not hasattr(cpd, "n_classes"):
            return None

        temperature = float(getattr(cpd, "temperature", 1.0))
        if parents_tensor is None:
            if hasattr(cpd, "_root_ready") and bool(getattr(cpd, "_root_ready").item()):
                logits = cpd._root_log_probs
                logits = torch.log_softmax(logits / temperature, dim=-1)
            else:
                logits = cpd._logits / temperature
            probs = torch.softmax(logits, dim=-1).view(1, -1, cpd.n_classes)
            probs = probs.expand(batch_size, -1, -1)
        else:
            if parents_tensor.dim() == 2:
                parents_tensor = parents_tensor.unsqueeze(1)
            logits = cpd._logits_from_parents(parents_tensor)
            probs = torch.softmax(logits, dim=-1)
            if probs.dim() == 4 and probs.shape[1] == 1:
                probs = probs.squeeze(1)

        if probs.dim() != 3:
            return None
        if probs.shape[1] != 1:
            return None

        probs = probs[:, 0, :]
        if hasattr(cpd, "_sample_values"):
            support = cpd._sample_values[0].to(device=probs.device, dtype=probs.dtype)
        else:
            support = torch.arange(
                probs.shape[-1], device=probs.device, dtype=probs.dtype
            )
        return probs, support

    def infer_posterior(self, vbn, query: Query, **kwargs):
        b = infer_batch_size(query.evidence, query.do)
        state = get_inference_state(vbn, query, {})
        device = vbn.device
        dtype = resolve_dtype(vbn, query)

        fixed_values = prepare_fixed_values(query, state, device, dtype, clamp_obs=True)
        target_idx = state.target_idx
        target_slice = state.node_slices[target_idx]
        target_cpd = vbn.nodes[state.topo_order[target_idx]]

        target_fixed = fixed_values[target_idx]
        if target_fixed is not None:
            target_value = target_fixed.unsqueeze(1).expand(b, 1, -1)
            weights = torch.ones(b, 1, device=device, dtype=dtype)
            return weights, target_value

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

        cat = self._categorical_probs(target_cpd, parent_tensor, b)
        if cat is None:
            return self._fallback_infer(vbn, query, **kwargs)

        probs, support = cat
        if target_slice.stop - target_slice.start != 1:
            return self._fallback_infer(vbn, query, **kwargs)

        samples = support.view(1, -1, 1).expand(b, -1, 1)
        return probs.detach(), samples.detach()
