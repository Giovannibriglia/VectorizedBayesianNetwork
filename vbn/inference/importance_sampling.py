from __future__ import annotations

from typing import Dict

import torch

from vbn.core.base import Query
from vbn.core.registry import register_inference
from vbn.core.utils import ensure_2d
from vbn.utils import infer_batch_size


def _ancestral_sample_prior(
    vbn, batch_size: int, n_samples: int
) -> Dict[str, torch.Tensor]:
    samples: Dict[str, torch.Tensor] = {}
    for node in vbn.dag.topological_order():
        parents = vbn.dag.parents(node)
        if parents:
            parent_tensor = torch.cat([samples[p] for p in parents], dim=-1)
        else:
            parent_tensor = torch.zeros(batch_size, 0, device=vbn.device)
        samples[node] = vbn.nodes[node].sample(parent_tensor, n_samples)
    return samples


def _concat_parent_samples(
    samples: Dict[str, torch.Tensor], parents: list[str]
) -> torch.Tensor | None:
    if not parents:
        return None
    return torch.cat([samples[p] for p in parents], dim=-1)


@register_inference("importance_sampling")
class ImportanceSampling:
    def __init__(self, n_samples: int = 200, **kwargs) -> None:
        self.n_samples = int(n_samples)

    def infer_posterior(self, vbn, query: Query, **kwargs):
        n_samples = int(kwargs.get("n_samples", self.n_samples))
        b = infer_batch_size(query.evidence)
        samples = _ancestral_sample_prior(vbn, b, n_samples)
        log_weights = torch.zeros(b, n_samples, device=vbn.device)
        for node, value in query.evidence.items():
            evidence_value = ensure_2d(value).to(vbn.device)
            evidence_value = evidence_value.unsqueeze(1).expand(b, n_samples, -1)
            parents = vbn.dag.parents(node)
            parent_tensor = _concat_parent_samples(samples, parents)
            log_w = vbn.nodes[node].log_prob(evidence_value, parent_tensor)
            log_weights = log_weights + log_w
        weights = torch.softmax(log_weights, dim=-1)
        if query.target in query.evidence:
            target_samples = (
                ensure_2d(query.evidence[query.target])
                .to(vbn.device)
                .unsqueeze(1)
                .expand(b, n_samples, -1)
            )
        else:
            target_samples = samples[query.target]
        return weights, target_samples
