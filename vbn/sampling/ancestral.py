from __future__ import annotations

from typing import Dict

import torch

from vbn.core.base import Query
from vbn.core.registry import register_sampling
from vbn.core.utils import ensure_2d
from vbn.utils import infer_batch_size
from vbn.utils.interventions import get_fixed_value


def _ancestral_sample_joint(
    vbn, query: Query, n_samples: int
) -> Dict[str, torch.Tensor]:
    b = infer_batch_size(query.evidence, query.do)
    samples: Dict[str, torch.Tensor] = {}
    for node in vbn.dag.topological_order():
        fixed = get_fixed_value(node, query)
        if fixed is not None:
            value = ensure_2d(fixed).to(vbn.device)
            value = value.unsqueeze(1).expand(b, n_samples, -1)
        else:
            parents = vbn.dag.parents(node)
            if parents:
                parent_tensor = torch.cat([samples[p] for p in parents], dim=-1)
            else:
                parent_tensor = torch.zeros(b, 0, device=vbn.device)
            value = vbn.nodes[node].sample(parent_tensor, n_samples)
        samples[node] = value
    return samples


@register_sampling("ancestral")
class AncestralSampler:
    def __init__(self, n_samples: int = 200, **kwargs) -> None:
        self.n_samples = int(n_samples)

    def sample(self, vbn, query: Query, n_samples: int | None = None, **kwargs):
        n_samples = int(n_samples or self.n_samples)
        samples = _ancestral_sample_joint(vbn, query, n_samples)
        return samples[query.target] if query.target else samples
