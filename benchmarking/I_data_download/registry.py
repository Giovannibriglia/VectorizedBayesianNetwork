from __future__ import annotations

from typing import Dict, Type

from .base import BaseDataDownloader

DATA_DOWNLOADER_REGISTRY: Dict[str, Type[BaseDataDownloader]] = {}


def register_downloader(cls: Type[BaseDataDownloader]) -> Type[BaseDataDownloader]:
    if not getattr(cls, "name", None):
        raise ValueError("Downloader class must define a non-empty 'name'.")
    DATA_DOWNLOADER_REGISTRY[cls.name] = cls
    return cls


def get_downloader(name: str) -> Type[BaseDataDownloader]:
    if name not in DATA_DOWNLOADER_REGISTRY:
        available = ", ".join(sorted(DATA_DOWNLOADER_REGISTRY)) or "<none>"
        raise KeyError(f"Unknown downloader '{name}'. Available: {available}")
    return DATA_DOWNLOADER_REGISTRY[name]
