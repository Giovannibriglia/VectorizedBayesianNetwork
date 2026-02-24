from __future__ import annotations

from typing import Dict, Type

from .base import BaseDataGenerator

DATA_GENERATOR_REGISTRY: Dict[str, Type[BaseDataGenerator]] = {}


def register_data_generator(cls: Type[BaseDataGenerator]) -> Type[BaseDataGenerator]:
    if not getattr(cls, "name", None):
        raise ValueError("Data generator class must define a non-empty 'name'.")
    DATA_GENERATOR_REGISTRY[cls.name] = cls
    return cls


def get_data_generator(name: str) -> Type[BaseDataGenerator]:
    if name not in DATA_GENERATOR_REGISTRY:
        available = ", ".join(sorted(DATA_GENERATOR_REGISTRY)) or "<none>"
        raise KeyError(f"Unknown data generator '{name}'. Available: {available}")
    return DATA_GENERATOR_REGISTRY[name]
