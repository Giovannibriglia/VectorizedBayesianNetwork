from vbn.core.dags import BaseDAG, DynamicDAG, StaticDAG, TemporalDAG
from vbn.core.registry import (
    CPD_REGISTRY,
    INFERENCE_REGISTRY,
    LEARNING_REGISTRY,
    SAMPLING_REGISTRY,
    UPDATE_REGISTRY,
)

_TORCH_AVAILABLE = True
try:
    import torch  # noqa: F401
except Exception:
    _TORCH_AVAILABLE = False

if _TORCH_AVAILABLE:
    from vbn.core.base import (
        BaseCPD,
        BaseInference,
        BaseLearning,
        BaseSampling,
        BaseUpdatePolicy,
        CPDOutput,
        Query,
    )

    __all__ = [
        "BaseCPD",
        "BaseInference",
        "BaseLearning",
        "BaseSampling",
        "BaseUpdatePolicy",
        "CPDOutput",
        "Query",
        "BaseDAG",
        "StaticDAG",
        "TemporalDAG",
        "DynamicDAG",
        "CPD_REGISTRY",
        "LEARNING_REGISTRY",
        "INFERENCE_REGISTRY",
        "SAMPLING_REGISTRY",
        "UPDATE_REGISTRY",
    ]
else:
    __all__ = [
        "BaseDAG",
        "StaticDAG",
        "TemporalDAG",
        "DynamicDAG",
        "CPD_REGISTRY",
        "LEARNING_REGISTRY",
        "INFERENCE_REGISTRY",
        "SAMPLING_REGISTRY",
        "UPDATE_REGISTRY",
    ]
