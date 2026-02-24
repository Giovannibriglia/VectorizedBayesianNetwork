from __future__ import annotations

from pathlib import Path
from typing import Optional


def get_benchmarking_root(root_path: Optional[Path] = None) -> Path:
    if root_path is None:
        return Path(__file__).resolve().parent
    return Path(root_path).resolve() / "benchmarking"


def get_static_metadata_dir(root_path: Optional[Path] = None) -> Path:
    return get_benchmarking_root(root_path) / "metadata"


def get_data_dir(root_path: Optional[Path] = None) -> Path:
    return get_benchmarking_root(root_path) / "data"


def get_datasets_dir(root_path: Optional[Path] = None) -> Path:
    return get_data_dir(root_path) / "datasets"


def get_generator_datasets_dir(root_path: Optional[Path], generator: str) -> Path:
    return get_datasets_dir(root_path) / generator


def get_dataset_dir(root_path: Optional[Path], generator: str, problem: str) -> Path:
    return get_generator_datasets_dir(root_path, generator) / problem


def get_metadata_dir_generated(root_path: Optional[Path] = None) -> Path:
    return get_data_dir(root_path) / "metadata"


def get_generator_metadata_dir_generated(
    root_path: Optional[Path], generator: str
) -> Path:
    return get_metadata_dir_generated(root_path) / generator


def get_dataset_metadata_dir_generated(
    root_path: Optional[Path], generator: str, problem: str
) -> Path:
    return get_generator_metadata_dir_generated(root_path, generator) / problem


def get_dataset_download_metadata_path(
    root_path: Optional[Path], generator: str, problem: str
) -> Path:
    return (
        get_dataset_metadata_dir_generated(root_path, generator, problem)
        / "download.json"
    )


def get_dataset_encoding_metadata_path(
    root_path: Optional[Path], generator: str, problem: str
) -> Path:
    return (
        get_dataset_metadata_dir_generated(root_path, generator, problem)
        / "encoding.json"
    )


def get_dataset_domain_metadata_path(
    root_path: Optional[Path], generator: str, problem: str
) -> Path:
    return (
        get_dataset_metadata_dir_generated(root_path, generator, problem)
        / "domain.json"
    )


def get_dataset_data_generation_metadata_path(
    root_path: Optional[Path], generator: str, problem: str
) -> Path:
    return (
        get_dataset_metadata_dir_generated(root_path, generator, problem)
        / "data_generation.json"
    )


def get_queries_dir(root_path: Optional[Path] = None) -> Path:
    return get_data_dir(root_path) / "queries"


def get_generator_queries_dir(root_path: Optional[Path], generator: str) -> Path:
    return get_queries_dir(root_path) / generator


def get_dataset_queries_dir(
    root_path: Optional[Path], generator: str, problem: str
) -> Path:
    return get_generator_queries_dir(root_path, generator) / problem


def get_queries_log_dir(root_path: Optional[Path] = None) -> Path:
    return get_queries_dir(root_path) / "log"


def get_generator_queries_log_dir(root_path: Optional[Path], generator: str) -> Path:
    return get_queries_log_dir(root_path) / generator


def get_datasets_log_dir(root_path: Optional[Path] = None) -> Path:
    return get_datasets_dir(root_path) / "log"


def get_generator_datasets_log_dir(root_path: Optional[Path], generator: str) -> Path:
    return get_datasets_log_dir(root_path) / generator


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
