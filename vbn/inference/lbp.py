from __future__ import annotations

import torch

from vbn.core.base import Query
from vbn.core.registry import register_inference
from vbn.inference.importance_sampling import ImportanceSampling
from vbn.inference.monte_carlo_marginalization import MonteCarloMarginalization


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
        tol = float(kwargs.get("tol", 1e-4))
        eps = 1e-12

        # b = infer_batch_size(query.evidence, query.do)
        if self.fallback == "monte_carlo_marginalization":
            pdf, target_samples = self._mcm.infer_posterior(
                vbn, query, n_samples=n_samples
            )
            weights = pdf / (pdf.sum(dim=-1, keepdim=True) + eps)
        else:
            weights, target_samples = self._is.infer_posterior(
                vbn, query, n_samples=n_samples
            )
        converged = False
        for _ in range(max(n_iters, 0)):
            w_new = torch.clamp(weights, min=eps)
            w_new = w_new / (w_new.sum(dim=-1, keepdim=True) + eps)
            msg = damping * w_new + (1.0 - damping) * weights
            msg = msg / (msg.sum(dim=-1, keepdim=True) + eps)
            delta = (msg - weights).abs().max().item()
            weights = msg
            if delta < tol:
                converged = True
                break

        if not converged:
            return self._is.infer_posterior(vbn, query, n_samples=n_samples)

        return weights.detach(), target_samples.detach()
