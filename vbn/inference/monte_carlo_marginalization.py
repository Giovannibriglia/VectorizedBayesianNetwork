from __future__ import annotations

import torch

from vbn.core.base import Query
from vbn.core.registry import register_inference
from vbn.inference._core import get_inference_state, prepare_fixed_values, resolve_dtype
from vbn.utils import infer_batch_size
from vbn.utils.interventions import is_intervened


@register_inference("monte_carlo_marginalization")
class MonteCarloMarginalization:
    def __init__(self, n_samples: int = 200, **kwargs) -> None:
        self.n_samples = int(n_samples)
        self._cache = {}

    def infer_posterior(self, vbn, query: Query, **kwargs):
        n_samples = int(kwargs.get("n_samples", self.n_samples))
        b = infer_batch_size(query.evidence, query.do)
        state = get_inference_state(vbn, query, self._cache)
        device = vbn.device
        dtype = resolve_dtype(vbn, query)

        fixed_values = prepare_fixed_values(query, state, device, dtype)
        target_idx = state.target_idx
        target_slice = state.node_slices[target_idx]
        target_cpd = vbn.nodes[state.topo_order[target_idx]]

        parents_idx = state.parent_idx[target_idx]
        parents_observed = all(fixed_values[p] is not None for p in parents_idx)

        if is_intervened(query.target, query):
            target_value = fixed_values[target_idx]
            target_samples = target_value.unsqueeze(1).expand(b, n_samples, -1)
            pdf = torch.ones(b, n_samples, device=device, dtype=dtype)
            return pdf, target_samples

        if parents_observed:
            if parents_idx:
                parent_tensor = torch.cat(
                    [
                        fixed_values[p].unsqueeze(1).expand(b, n_samples, -1)
                        for p in parents_idx
                    ],
                    dim=-1,
                )
            else:
                parent_tensor = None
            if fixed_values[target_idx] is not None:
                target_samples = (
                    fixed_values[target_idx].unsqueeze(1).expand(b, n_samples, -1)
                )
            else:
                target_samples = target_cpd.sample(parent_tensor, n_samples)
            log_prob = target_cpd.log_prob(target_samples, parent_tensor)
            pdf = torch.exp(log_prob)
            return pdf, target_samples

        samples = torch.zeros(b, n_samples, state.total_dim, device=device, dtype=dtype)
        nodes = state.topo_order
        cpds = [vbn.nodes[node] for node in nodes]

        for idx, node in enumerate(nodes):
            node_slice = state.node_slices[idx]
            fixed = fixed_values[idx]
            if fixed is not None:
                value = fixed.unsqueeze(1).expand(b, n_samples, -1)
                samples[..., node_slice] = value
                continue
            parent_slices = state.parent_slices[idx]
            if parent_slices:
                parent_tensor = torch.cat(
                    [samples[..., sl] for sl in parent_slices], dim=-1
                )
            else:
                parent_tensor = None
            samples[..., node_slice] = cpds[idx].sample(parent_tensor, n_samples)

        target_samples = samples[..., target_slice]
        log_prob = target_cpd.log_prob(
            target_samples,
            (
                torch.cat(
                    [samples[..., sl] for sl in state.parent_slices[target_idx]], dim=-1
                )
                if state.parent_slices[target_idx]
                else None
            ),
        )
        pdf = torch.exp(log_prob)
        return pdf, target_samples
