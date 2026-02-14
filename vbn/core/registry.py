from __future__ import annotations

from typing import Callable, Dict, Type, TypeVar

T = TypeVar("T")

CPD_REGISTRY: Dict[str, Type] = {}
LEARNING_REGISTRY: Dict[str, Type] = {}
INFERENCE_REGISTRY: Dict[str, Type] = {}
SAMPLING_REGISTRY: Dict[str, Type] = {}
UPDATE_REGISTRY: Dict[str, Type] = {}


def _register(registry: Dict[str, Type], name: str) -> Callable[[Type[T]], Type[T]]:
    key = name.lower().strip()

    def decorator(cls: Type[T]) -> Type[T]:
        if key in registry:
            raise ValueError(f"Duplicate registry key '{key}' for {cls.__name__}")
        registry[key] = cls
        return cls

    return decorator


def register_cpd(name: str) -> Callable[[Type[T]], Type[T]]:
    return _register(CPD_REGISTRY, name)


def register_learning(name: str) -> Callable[[Type[T]], Type[T]]:
    return _register(LEARNING_REGISTRY, name)


def register_inference(name: str) -> Callable[[Type[T]], Type[T]]:
    return _register(INFERENCE_REGISTRY, name)


def register_sampling(name: str) -> Callable[[Type[T]], Type[T]]:
    return _register(SAMPLING_REGISTRY, name)


def register_update(name: str) -> Callable[[Type[T]], Type[T]]:
    return _register(UPDATE_REGISTRY, name)
