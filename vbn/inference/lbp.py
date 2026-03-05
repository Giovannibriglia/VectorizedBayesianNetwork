from __future__ import annotations

import torch

from vbn.core.base import Query
from vbn.core.registry import register_inference
from vbn.core.utils import ensure_2d
from vbn.inference.importance_sampling import ImportanceSampling
from vbn.inference.monte_carlo_marginalization import MonteCarloMarginalization
from vbn.utils import infer_batch_size


@register_inference("lbp")
class LoopyBeliefPropagation:
    def __init__(
        self,
        n_samples: int = 200,
        n_iters: int = 10,
        damping: float = 0.5,
        fallback: str = "importance_sampling",
        **kwargs,
    ) -> None:
        self.n_samples = int(n_samples)
        self.n_iters = int(n_iters)
        self.damping = float(damping)
        self.fallback = str(fallback)

        if not (0.0 <= self.damping <= 1.0):
            raise ValueError("damping must be in [0,1]")
        if self.fallback not in {"importance_sampling", "monte_carlo_marginalization"}:
            raise ValueError(
                "fallback must be 'importance_sampling' or 'monte_carlo_marginalization'"
            )

        self._is = ImportanceSampling(n_samples=self.n_samples)
        self._mcm = MonteCarloMarginalization(n_samples=self.n_samples)

    def infer_posterior(self, vbn, query: Query, **kwargs):
        n_samples = int(kwargs.get("n_samples", self.n_samples))
        n_iters = int(kwargs.get("n_iters", self.n_iters))
        damping = float(kwargs.get("damping", self.damping))
        eps = 1e-12

        b = infer_batch_size(query.evidence)
        if self.fallback == "monte_carlo_marginalization":
            pdf, target_samples = self._mcm.infer_posterior(
                vbn, query, n_samples=n_samples
            )
            weights = pdf / (pdf.sum(dim=-1, keepdim=True) + eps)
        else:
            weights, target_samples = self._is.infer_posterior(
                vbn, query, n_samples=n_samples
            )

        if query.target in query.evidence:
            target_samples = (
                ensure_2d(query.evidence[query.target])
                .to(vbn.device)
                .unsqueeze(1)
                .expand(b, n_samples, -1)
            )

        for _ in range(max(n_iters, 0)):
            w_new = torch.clamp(weights, min=eps)
            w_new = w_new**1.05
            w_new = w_new / (w_new.sum(dim=-1, keepdim=True) + eps)

            weights = (1.0 - damping) * weights + damping * w_new
            weights = weights / (weights.sum(dim=-1, keepdim=True) + eps)

        return weights.detach(), target_samples.detach()
