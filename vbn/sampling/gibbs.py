from __future__ import annotations

import torch

from vbn.core.base import Query
from vbn.core.registry import register_sampling
from vbn.inference._core import get_inference_state, prepare_fixed_values, resolve_dtype
from vbn.sampling.ancestral import _ancestral_sample_tensor
from vbn.utils import infer_batch_size


@register_sampling("gibbs")
class GibbsSampler:
    def __init__(
        self, n_samples: int = 200, burn_in: int = 10, n_steps: int = 1, **kwargs
    ) -> None:
        self.n_samples = int(n_samples)
        self.burn_in = int(burn_in)
        self.n_steps = int(n_steps)
        self.n_candidates = 8
        self._cache = {}

    def sample(self, vbn, query: Query, n_samples: int | None = None, **kwargs):
        n_samples = int(n_samples or self.n_samples)
        b = infer_batch_size(query.evidence, query.do)
        state = get_inference_state(vbn, query, self._cache)
        device = vbn.device
        dtype = resolve_dtype(vbn, query)

        fixed_values = prepare_fixed_values(query, state, device, dtype)
        current = _ancestral_sample_tensor(vbn, query, n_samples=1)
        latent_idx = [
            idx for idx in range(len(state.topo_order)) if fixed_values[idx] is None
        ]
        cpds = [vbn.nodes[node] for node in state.topo_order]

        total_steps = self.burn_in + n_samples * max(self.n_steps, 1)
        collected = []
        for step in range(total_steps):
            for idx in latent_idx:
                node_slice = state.node_slices[idx]
                parent_slices = state.parent_slices[idx]
                if parent_slices:
                    parent_tensor = torch.cat(
                        [current[..., sl] for sl in parent_slices], dim=-1
                    )
                else:
                    parent_tensor = None
                if (
                    parent_tensor is not None
                    and parent_tensor.shape[1] != self.n_candidates
                ):
                    parent_tensor = parent_tensor.expand(b, self.n_candidates, -1)
                candidates = cpds[idx].sample(parent_tensor, self.n_candidates)
                log_score = cpds[idx].log_prob(candidates, parent_tensor)
                for child_idx, child_parents in zip(
                    state.children_idx[idx], state.child_parent_idx[idx]
                ):
                    child_slice = state.node_slices[child_idx]
                    child_value = current[..., child_slice].expand(
                        b, self.n_candidates, -1
                    )
                    parent_parts = []
                    for p in child_parents:
                        if p == idx:
                            parent_parts.append(candidates)
                        else:
                            parent_parts.append(
                                current[..., state.node_slices[p]].expand(
                                    b, self.n_candidates, -1
                                )
                            )
                    child_parent_tensor = (
                        torch.cat(parent_parts, dim=-1) if parent_parts else None
                    )
                    log_score = log_score + cpds[child_idx].log_prob(
                        child_value, child_parent_tensor
                    )
                weights = torch.softmax(log_score, dim=1)
                choice = torch.multinomial(weights, num_samples=1).squeeze(1)
                chosen = candidates[torch.arange(b, device=device), choice]
                current[..., node_slice] = chosen.unsqueeze(1)
            if step >= self.burn_in and (
                (step - self.burn_in) % max(self.n_steps, 1) == 0
            ):
                target_slice = state.node_slices[state.target_idx]
                collected.append(current[..., target_slice])

        if not collected:
            target_slice = state.node_slices[state.target_idx]
            return current[..., target_slice]
        return torch.cat(collected, dim=1)
