from __future__ import annotations

from vbn.core.base import Query
from vbn.core.registry import register_inference
from vbn.inference.monte_carlo_marginalization import MonteCarloMarginalization


@register_inference("svgp")
class SVGPInference:
    """Placeholder for SVGP-based inference. Uses Monte Carlo as a fallback."""

    def __init__(self, n_samples: int = 200, **kwargs) -> None:
        self._fallback = MonteCarloMarginalization(n_samples=n_samples)

    def infer_posterior(self, vbn, query: Query, **kwargs):
        return self._fallback.infer_posterior(vbn, query, **kwargs)
