from __future__ import annotations

import pkgutil
from importlib import import_module

from .base import BaseDataGenerator
from .registry import DATA_GENERATOR_REGISTRY, get_generator, register_generator


def _auto_import_generators() -> None:
    for module in pkgutil.iter_modules(__path__):
        name = module.name
        if name in {"base", "registry", "__init__"}:
            continue
        import_module(f"{__name__}.{name}")


_auto_import_generators()

__all__ = [
    "BaseDataGenerator",
    "DATA_GENERATOR_REGISTRY",
    "get_generator",
    "register_generator",
]
