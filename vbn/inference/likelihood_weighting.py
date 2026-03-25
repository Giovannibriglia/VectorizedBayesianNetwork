from __future__ import annotations

from typing import Dict

import torch

from vbn.core.base import Query
from vbn.core.registry import register_inference
from vbn.core.utils import ensure_2d
from vbn.utils import infer_batch_size
from vbn.utils.interventions import get_fixed_value


def _concat_parent_samples(
    samples: Dict[str, torch.Tensor], parents: list[str]
) -> torch.Tensor | None:
    if not parents:
        return None
    return torch.cat([samples[p] for p in parents], dim=-1)


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
        samples: Dict[str, torch.Tensor] = {}
        log_weights = torch.zeros(b, n_samples, device=vbn.device)

        for node in vbn.dag.topological_order():
            parents = vbn.dag.parents(node)
            parent_tensor = _concat_parent_samples(samples, parents)
            fixed = get_fixed_value(node, query)
            if fixed is not None:
                fixed_value = ensure_2d(fixed).to(vbn.device)
                fixed_value = fixed_value.unsqueeze(1).expand(b, n_samples, -1)
                samples[node] = fixed_value
                if node in query.evidence:
                    # Evidence contributes likelihood; do-interventions do not.
                    cpd = vbn.nodes[node]
                    if not hasattr(cpd, "log_prob") or not callable(cpd.log_prob):
                        raise NotImplementedError(
                            f"CPD for node '{node}' does not implement log_prob."
                        )
                    log_weights = log_weights + cpd.log_prob(fixed_value, parent_tensor)
            else:
                samples[node] = vbn.nodes[node].sample(parent_tensor, n_samples)

        target_samples = samples[query.target]

        if normalize:
            weights = torch.softmax(log_weights, dim=-1)
        else:
            # Stabilize exponentiation while keeping unnormalized scale.
            log_weights = log_weights - log_weights.max(dim=-1, keepdim=True).values
            weights = torch.exp(log_weights).clamp_min(eps)

        return weights.detach(), target_samples.detach()
