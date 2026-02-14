# Ensure registries are populated on import.
from vbn import (  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401
    cpds as _cpds,
    inference as _inference,
    learning as _learning,
    sampling as _sampling,
    update as _update,
)
from vbn.core.registry import (
    CPD_REGISTRY,
    INFERENCE_REGISTRY,
    LEARNING_REGISTRY,
    SAMPLING_REGISTRY,
    UPDATE_REGISTRY,
)
from vbn.defaults import defaults
from vbn.vbn import VBN

__all__ = [
    "VBN",
    "CPD_REGISTRY",
    "LEARNING_REGISTRY",
    "INFERENCE_REGISTRY",
    "SAMPLING_REGISTRY",
    "UPDATE_REGISTRY",
    "defaults",
]
