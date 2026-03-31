from __future__ import annotations

import torch

from vbn.core.base import Query
from vbn.core.registry import register_inference
from vbn.inference._core import get_inference_state, prepare_fixed_values, resolve_dtype
from vbn.utils import infer_batch_size


@register_inference("likelihood_weighting")
class LikelihoodWeighting:
    def __init__(
        self,
        n_samples: int = 512,
        eps: float = 1e-12,
        normalize: bool = True,
        **kwargs,
    ) -> None:
        self.n_samples = int(n_samples)
        self.eps = float(eps)
        self.normalize = bool(normalize)

    def infer_posterior(self, vbn, query: Query, **kwargs):
        n_samples = int(kwargs.get("n_samples", self.n_samples))
        normalize = bool(kwargs.get("normalize", self.normalize))
        eps = float(kwargs.get("eps", self.eps))

        b = infer_batch_size(query.evidence, query.do)
        if not hasattr(self, "_cache"):
            self._cache = {}
        state = get_inference_state(vbn, query, self._cache)
        device = vbn.device
        dtype = resolve_dtype(vbn, query)

        samples = torch.zeros(b, n_samples, state.total_dim, device=device, dtype=dtype)
        log_weights = torch.zeros(b, n_samples, device=device, dtype=dtype)
        fixed_values = prepare_fixed_values(query, state, device, dtype, clamp_obs=True)
        nodes = state.topo_order
        cpds = [vbn.nodes[node] for node in nodes]

        for idx, node in enumerate(nodes):
            node_slice = state.node_slices[idx]
            fixed = fixed_values[idx]
            if fixed is not None:
                fixed_value = fixed.unsqueeze(1).expand(b, n_samples, -1)
                samples[..., node_slice] = fixed_value
                if state.evidence_mask[idx]:
                    parent_slices = state.parent_slices[idx]
                    if parent_slices:
                        parent_tensor = torch.cat(
                            [samples[..., sl] for sl in parent_slices], dim=-1
                        )
                    else:
                        parent_tensor = None
                    cpd = cpds[idx]
                    if not hasattr(cpd, "log_prob") or not callable(cpd.log_prob):
                        raise NotImplementedError(
                            f"CPD for node '{node}' does not implement log_prob."
                        )
                    log_weights = log_weights + cpd.log_prob(fixed_value, parent_tensor)
                continue

            parent_slices = state.parent_slices[idx]
            if parent_slices:
                parent_tensor = torch.cat(
                    [samples[..., sl] for sl in parent_slices], dim=-1
                )
            else:
                parent_tensor = None
            samples[..., node_slice] = cpds[idx].sample(parent_tensor, n_samples)

        target_samples = samples[..., state.node_slices[state.target_idx]]

        if normalize:
            weights = torch.softmax(log_weights, dim=1)
        else:
            # Stabilize exponentiation while keeping unnormalized scale.
            log_weights = log_weights - log_weights.max(dim=-1, keepdim=True).values
            weights = torch.exp(log_weights).clamp_min(eps)

        return weights.detach(), target_samples.detach()
