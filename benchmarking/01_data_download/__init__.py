from __future__ import annotations

import pkgutil
from importlib import import_module

from .base import BaseDataDownloader
from .registry import DATA_DOWNLOADER_REGISTRY, get_downloader, register_downloader


def _auto_import_downloaders() -> None:
    for module in pkgutil.iter_modules(__path__):
        name = module.name
        if name in {"base", "registry", "__init__"}:
            continue
        import_module(f"{__name__}.{name}")


_auto_import_downloaders()

__all__ = [
    "BaseDataDownloader",
    "DATA_DOWNLOADER_REGISTRY",
    "get_downloader",
    "register_downloader",
]
