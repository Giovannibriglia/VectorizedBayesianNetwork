from __future__ import annotations

from typing import Optional

import torch

from vbn.core.base import Query
from vbn.core.registry import register_inference
from vbn.inference._core import get_inference_state, prepare_fixed_values, resolve_dtype
from vbn.utils import infer_batch_size


@register_inference("resampled_importance_sampling")
class ResampledImportanceSampling:
    """Sequential importance resampling with optional ESS-triggered resampling."""

    def __init__(
        self,
        n_samples: int = 512,
        ess_threshold: float = 0.5,
        resample: bool = True,
        clamp_obs: bool = True,
        **kwargs,
    ) -> None:
        self.n_samples = int(n_samples)
        self.ess_threshold = float(ess_threshold)
        self.resample = bool(resample)
        self.clamp_obs = bool(clamp_obs)
        self._cache = {}
        self._last_ess: Optional[torch.Tensor] = None
        self._last_resampled = False

    def _resample(
        self, samples: torch.Tensor, log_weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(log_weights, dim=1)
        b, n = weights.shape
        idx = torch.multinomial(weights, num_samples=n, replacement=True)
        batch_idx = torch.arange(b, device=weights.device).unsqueeze(1)
        resampled = samples[batch_idx, idx]
        return resampled, torch.zeros_like(log_weights)

    def infer_posterior(self, vbn, query: Query, **kwargs):
        n_samples = int(kwargs.get("n_samples", self.n_samples))
        ess_threshold = float(kwargs.get("ess_threshold", self.ess_threshold))
        resample = bool(kwargs.get("resample", self.resample))
        clamp_obs = bool(kwargs.get("clamp_obs", self.clamp_obs))

        b = infer_batch_size(query.evidence, query.do)
        state = get_inference_state(vbn, query, self._cache)
        device = vbn.device
        dtype = resolve_dtype(vbn, query)

        samples = torch.zeros(b, n_samples, state.total_dim, device=device, dtype=dtype)
        log_weights = torch.zeros(b, n_samples, device=device, dtype=dtype)
        fixed_values = prepare_fixed_values(
            query, state, device, dtype, clamp_obs=clamp_obs
        )
        nodes = state.topo_order
        cpds = [vbn.nodes[node] for node in nodes]

        def ess_threshold_value() -> float:
            if ess_threshold <= 1.0:
                return max(1.0, ess_threshold * float(n_samples))
            return float(ess_threshold)

        threshold = ess_threshold_value()
        self._last_resampled = False
        for idx, node in enumerate(nodes):
            node_slice = state.node_slices[idx]
            fixed = fixed_values[idx]
            if fixed is not None:
                value = fixed.unsqueeze(1).expand(b, n_samples, -1)
                samples[..., node_slice] = value
                if state.evidence_mask[idx]:
                    parent_slices = state.parent_slices[idx]
                    if parent_slices:
                        parent_tensor = torch.cat(
                            [samples[..., sl] for sl in parent_slices], dim=-1
                        )
                    else:
                        parent_tensor = None
                    log_weights = log_weights + cpds[idx].log_prob(value, parent_tensor)
                    if resample:
                        weights = torch.softmax(log_weights, dim=1)
                        ess = 1.0 / (weights**2).sum(dim=1)
                        self._last_ess = ess
                        if torch.any(ess < threshold):
                            samples, log_weights = self._resample(samples, log_weights)
                            self._last_resampled = True
                continue

            parent_slices = state.parent_slices[idx]
            if parent_slices:
                parent_tensor = torch.cat(
                    [samples[..., sl] for sl in parent_slices], dim=-1
                )
            else:
                parent_tensor = None
            samples[..., node_slice] = cpds[idx].sample(parent_tensor, n_samples)

        weights = torch.softmax(log_weights, dim=1)
        target_slice = state.node_slices[state.target_idx]
        target_samples = samples[..., target_slice]
        return weights.detach(), target_samples.detach()
