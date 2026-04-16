from __future__ import annotations

from typing import Optional

import torch

from vbn.core.base import Query
from vbn.core.registry import register_inference
from vbn.inference._core import get_inference_state, prepare_fixed_values, resolve_dtype
from vbn.inference.likelihood_weighting import LikelihoodWeighting
from vbn.utils import infer_batch_size


@register_inference("importance_sampling")
class ImportanceSampling:
    def __init__(self, n_samples: int = 200, **kwargs) -> None:
        self.n_samples = int(n_samples)
        self.ess_threshold = 0.1
        self._cache = {}
        self._lw = LikelihoodWeighting(n_samples=self.n_samples)
        self._last_fallback = False
        self._last_ess: Optional[torch.Tensor] = None

    def infer_posterior(self, vbn, query: Query, **kwargs):
        n_samples = int(kwargs.get("n_samples", self.n_samples))
        b = infer_batch_size(query.evidence, query.do)
        state = get_inference_state(vbn, query, self._cache)
        device = vbn.device
        dtype = resolve_dtype(vbn, query)

        samples = torch.zeros(b, n_samples, state.total_dim, device=device, dtype=dtype)
        log_weights = torch.zeros(b, n_samples, device=device, dtype=dtype)
        fixed_values = prepare_fixed_values(query, state, device, dtype)
        nodes = state.topo_order
        cpds = [vbn.nodes[node] for node in nodes]

        def _sample_node(cpd, parent_tensor):
            if b <= 1:
                return cpd.sample(parent_tensor, n_samples)
            batches = []
            for batch_idx in range(b):
                if parent_tensor is None:
                    parent_i = None
                else:
                    parent_i = parent_tensor[batch_idx : batch_idx + 1]
                sample_i = cpd.sample(parent_i, n_samples)
                if sample_i.dim() == 2:
                    sample_i = sample_i.unsqueeze(0)
                if sample_i.dim() != 3 or sample_i.shape[0] != 1:
                    raise ValueError(
                        "CPD.sample must return [B,S,D] or [S,D] for inference"
                    )
                batches.append(sample_i)
            return torch.cat(batches, dim=0)

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
                continue

            parent_slices = state.parent_slices[idx]
            if parent_slices:
                parent_tensor = torch.cat(
                    [samples[..., sl] for sl in parent_slices], dim=-1
                )
            else:
                parent_tensor = None
            samples[..., node_slice] = _sample_node(cpds[idx], parent_tensor)

        weights = torch.softmax(log_weights, dim=1)
        ess = 1.0 / (weights**2).sum(dim=1)
        self._last_ess = ess
        threshold = max(1.0, self.ess_threshold * float(n_samples))
        if torch.any(ess < threshold):
            self._last_fallback = True
            return self._lw.infer_posterior(vbn, query, n_samples=n_samples)
        self._last_fallback = False

        target_slice = state.node_slices[state.target_idx]
        target_samples = samples[..., target_slice]
        return weights, target_samples
