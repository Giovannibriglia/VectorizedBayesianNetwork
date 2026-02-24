from __future__ import annotations

import pkgutil
from importlib import import_module

from .base import BaseBenchmarkModel
from .registry import (
    BENCHMARK_MODEL_REGISTRY,
    get_benchmark_model,
    list_benchmark_models,
    register_benchmark_model,
)


def _auto_import_models() -> None:
    for module in pkgutil.iter_modules(__path__):
        name = module.name
        if name in {"base", "registry", "config", "presets", "__init__"}:
            continue
        import_module(f"{__name__}.{name}")


_auto_import_models()

__all__ = [
    "BaseBenchmarkModel",
    "BENCHMARK_MODEL_REGISTRY",
    "get_benchmark_model",
    "list_benchmark_models",
    "register_benchmark_model",
]
