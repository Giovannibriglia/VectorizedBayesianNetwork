from __future__ import annotations

from vbn.core.registry import register_learning


@register_learning("amortized")
class AmortizedLearner:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def fit(self, vbn, data, **kwargs):
        raise NotImplementedError(
            "Amortized learning is a placeholder for future work."
        )
