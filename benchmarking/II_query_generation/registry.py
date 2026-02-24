from __future__ import annotations

from typing import Dict, Type

from .base import BaseQueryGenerator

QUERY_GENERATOR_REGISTRY: Dict[str, Type[BaseQueryGenerator]] = {}


def register_query_generator(cls: Type[BaseQueryGenerator]) -> Type[BaseQueryGenerator]:
    if not getattr(cls, "name", None):
        raise ValueError("Query generator class must define a non-empty 'name'.")
    QUERY_GENERATOR_REGISTRY[cls.name] = cls
    return cls


def get_query_generator(name: str) -> Type[BaseQueryGenerator]:
    if name not in QUERY_GENERATOR_REGISTRY:
        available = ", ".join(sorted(QUERY_GENERATOR_REGISTRY)) or "<none>"
        raise KeyError(f"Unknown query generator '{name}'. Available: {available}")
    return QUERY_GENERATOR_REGISTRY[name]
