from __future__ import annotations

from typing import Dict, Type

from .base import BaseBenchmarkModel

BENCHMARK_MODEL_REGISTRY: Dict[str, Type[BaseBenchmarkModel]] = {}


def register_benchmark_model(
    cls: Type[BaseBenchmarkModel],
) -> Type[BaseBenchmarkModel]:
    if not getattr(cls, "name", None):
        raise ValueError("Benchmark model must define a non-empty 'name'.")
    BENCHMARK_MODEL_REGISTRY[cls.name] = cls
    return cls


def get_benchmark_model(name: str) -> Type[BaseBenchmarkModel]:
    if name not in BENCHMARK_MODEL_REGISTRY:
        available = ", ".join(sorted(BENCHMARK_MODEL_REGISTRY)) or "<none>"
        raise KeyError(f"Unknown benchmark model '{name}'. Available: {available}")
    return BENCHMARK_MODEL_REGISTRY[name]


def list_benchmark_models() -> list[str]:
    return sorted(BENCHMARK_MODEL_REGISTRY)
