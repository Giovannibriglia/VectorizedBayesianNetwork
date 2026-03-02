from vbn.core.registry import (
    CPD_REGISTRY,
    INFERENCE_REGISTRY,
    LEARNING_REGISTRY,
    SAMPLING_REGISTRY,
    UPDATE_REGISTRY,
)
from vbn.defaults import defaults

_TORCH_AVAILABLE = True
try:
    import torch  # noqa: F401
except Exception:
    _TORCH_AVAILABLE = False

if _TORCH_AVAILABLE:
    # Ensure registries are populated on import.
    from vbn import (
        cpds as _cpds,
        inference as _inference,
        learning as _learning,
        sampling as _sampling,
        update as _update,
    )
    from vbn.vbn import VBN
else:

    class VBN:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for learning and inference. "
                "Please install torch separately before using VBN."
            )


__all__ = [
    "VBN",
    "CPD_REGISTRY",
    "LEARNING_REGISTRY",
    "INFERENCE_REGISTRY",
    "SAMPLING_REGISTRY",
    "UPDATE_REGISTRY",
    "defaults",
]
