from __future__ import annotations

from vbn.core.base import Query
from vbn.core.registry import register_sampling
from vbn.sampling.ancestral import _ancestral_sample_joint


@register_sampling("hmc")
class HMCSampler:
    """Placeholder for HMC sampling. Uses ancestral sampling as a fallback."""

    def __init__(self, n_samples: int = 200, **kwargs) -> None:
        self.n_samples = int(n_samples)

    def sample(self, vbn, query: Query, n_samples: int | None = None, **kwargs):
        n_samples = int(n_samples or self.n_samples)
        samples = _ancestral_sample_joint(vbn, query, n_samples)
        return samples[query.target] if query.target else samples
