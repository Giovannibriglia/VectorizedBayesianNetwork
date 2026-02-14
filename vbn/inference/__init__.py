from vbn.core.registry import INFERENCE_REGISTRY
from vbn.inference.importance_sampling import ImportanceSampling
from vbn.inference.monte_carlo_marginalization import MonteCarloMarginalization
from vbn.inference.svgp import SVGPInference

__all__ = [
    "INFERENCE_REGISTRY",
    "MonteCarloMarginalization",
    "ImportanceSampling",
    "SVGPInference",
]
