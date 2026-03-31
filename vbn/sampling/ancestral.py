from __future__ import annotations

from typing import Dict

import torch

from vbn.core.base import Query
from vbn.core.registry import register_sampling
from vbn.inference._core import get_inference_state, prepare_fixed_values, resolve_dtype
from vbn.utils import infer_batch_size


def _ancestral_sample_tensor(vbn, query: Query, n_samples: int) -> torch.Tensor:
    b = infer_batch_size(query.evidence, query.do)
    if not hasattr(vbn, "_sampling_cache"):
        vbn._sampling_cache = {}
    state = get_inference_state(vbn, query, vbn._sampling_cache)
    device = vbn.device
    dtype = resolve_dtype(vbn, query)

    samples = torch.zeros(b, n_samples, state.total_dim, device=device, dtype=dtype)
    fixed_values = prepare_fixed_values(query, state, device, dtype)
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
    return samples


def _ancestral_sample_joint(
    vbn, query: Query, n_samples: int
) -> Dict[str, torch.Tensor]:
    if not hasattr(vbn, "_sampling_cache"):
        vbn._sampling_cache = {}
    state = get_inference_state(vbn, query, vbn._sampling_cache)
    samples_tensor = _ancestral_sample_tensor(vbn, query, n_samples)
    out: Dict[str, torch.Tensor] = {}
    for idx, node in enumerate(state.topo_order):
        out[node] = samples_tensor[..., state.node_slices[idx]]
    return out


@register_sampling("ancestral")
class AncestralSampler:
    def __init__(self, n_samples: int = 200, **kwargs) -> None:
        self.n_samples = int(n_samples)

    def sample(self, vbn, query: Query, n_samples: int | None = None, **kwargs):
        n_samples = int(n_samples or self.n_samples)
        samples = _ancestral_sample_joint(vbn, query, n_samples)
        return samples[query.target] if query.target else samples
