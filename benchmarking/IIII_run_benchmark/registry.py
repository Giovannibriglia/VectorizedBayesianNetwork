from __future__ import annotations

from typing import Dict, Type

from .base import BaseBenchmarkRunner

BENCHMARK_RUNNER_REGISTRY: Dict[str, Type[BaseBenchmarkRunner]] = {}


def register_benchmark_runner(
    cls: Type[BaseBenchmarkRunner],
) -> Type[BaseBenchmarkRunner]:
    if not getattr(cls, "generator", None):
        raise ValueError("Benchmark runner class must define 'generator'.")
    BENCHMARK_RUNNER_REGISTRY[cls.generator] = cls
    return cls


def get_benchmark_runner(name: str) -> Type[BaseBenchmarkRunner]:
    if name not in BENCHMARK_RUNNER_REGISTRY:
        available = ", ".join(sorted(BENCHMARK_RUNNER_REGISTRY)) or "<none>"
        raise KeyError(f"Unknown benchmark runner '{name}'. Available: {available}")
    return BENCHMARK_RUNNER_REGISTRY[name]


def list_benchmark_runners() -> list[str]:
    return sorted(BENCHMARK_RUNNER_REGISTRY)
