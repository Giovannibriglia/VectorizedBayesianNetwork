from __future__ import annotations

import pkgutil
from importlib import import_module

from .base import BaseQueryGenerator
from .registry import (
    get_query_generator,
    QUERY_GENERATOR_REGISTRY,
    register_query_generator,
)


def _auto_import_generators() -> None:
    for module in pkgutil.iter_modules(__path__):
        name = module.name
        if name in {"base", "registry", "__init__"}:
            continue
        import_module(f"{__name__}.{name}")


_auto_import_generators()

__all__ = [
    "BaseQueryGenerator",
    "QUERY_GENERATOR_REGISTRY",
    "get_query_generator",
    "register_query_generator",
]
