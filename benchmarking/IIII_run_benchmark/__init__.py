from __future__ import annotations

import pkgutil
from importlib import import_module

from .base import BaseBenchmarkRunner
from .registry import (
    BENCHMARK_RUNNER_REGISTRY,
    get_benchmark_runner,
    list_benchmark_runners,
    register_benchmark_runner,
)


def _auto_import_runners() -> None:
    for module in pkgutil.iter_modules(__path__):
        name = module.name
        if name in {"base", "registry", "__init__"}:
            continue
        import_module(f"{__name__}.{name}")


_auto_import_runners()

__all__ = [
    "BaseBenchmarkRunner",
    "BENCHMARK_RUNNER_REGISTRY",
    "get_benchmark_runner",
    "list_benchmark_runners",
    "register_benchmark_runner",
]
