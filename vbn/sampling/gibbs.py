from __future__ import annotations

import torch

from vbn.core.base import Query
from vbn.core.registry import register_sampling
from vbn.core.utils import ensure_2d
from vbn.sampling.ancestral import _ancestral_sample_joint
from vbn.utils import infer_batch_size
from vbn.utils.interventions import get_fixed_value


@register_sampling("gibbs")
class GibbsSampler:
    def __init__(
        self, n_samples: int = 200, burn_in: int = 10, n_steps: int = 1, **kwargs
    ) -> None:
        self.n_samples = int(n_samples)
        self.burn_in = int(burn_in)
        self.n_steps = int(n_steps)

    def sample(self, vbn, query: Query, n_samples: int | None = None, **kwargs):
        n_samples = int(n_samples or self.n_samples)
        b = infer_batch_size(query.evidence, query.do)
        current = _ancestral_sample_joint(vbn, query, n_samples=1)
        collected = []
        total_steps = self.burn_in + n_samples
        for step in range(total_steps):
            for node in vbn.dag.topological_order():
                fixed = get_fixed_value(node, query)
                if fixed is not None:
                    value = ensure_2d(fixed).to(vbn.device)
                    current[node] = value.unsqueeze(1)
                    continue
                parents = vbn.dag.parents(node)
                if parents:
                    parent_tensor = torch.cat([current[p] for p in parents], dim=-1)
                else:
                    parent_tensor = torch.zeros(b, 0, device=vbn.device)
                current[node] = vbn.nodes[node].sample(parent_tensor, 1)
            if step >= self.burn_in:
                collected.append(current[query.target])
        return torch.cat(collected, dim=1)
