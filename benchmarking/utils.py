from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# ----------------------------
# Path helpers
# ----------------------------


def get_benchmarking_root(root_path: Optional[Path] = None) -> Path:
    if root_path is None:
        return Path(__file__).resolve().parent
    return Path(root_path).resolve() / "benchmarking"


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


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


def get_out_dir(root_path: Optional[Path] = None) -> Path:
    return get_benchmarking_root(root_path) / "out"


def get_generator_out_dir(root_path: Optional[Path], generator: str) -> Path:
    return get_out_dir(root_path) / generator


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ----------------------------
# JSON helpers
# ----------------------------


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, obj: Any, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=indent, sort_keys=True))


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


# ----------------------------
# DataFrame helpers
# ----------------------------


def _has_parquet_support() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except Exception:
        pass
    try:
        import fastparquet  # noqa: F401

        return True
    except Exception:
        return False


def read_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".pkl":
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported dataset format: {path}")


def save_dataframe(
    df: pd.DataFrame,
    path_prefix: Path,
    prefer: str = "parquet",
) -> tuple[Path, str]:
    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    if prefer == "csv":
        order = ["csv", "parquet", "pkl"]
    elif prefer == "pkl":
        order = ["pkl", "parquet", "csv"]
    else:
        order = ["parquet", "csv", "pkl"]

    last_error: Exception | None = None
    for fmt in order:
        try:
            if fmt == "parquet":
                if not _has_parquet_support():
                    continue
                path = path_prefix.with_suffix(".parquet")
                df.to_parquet(path, index=False)
                return path, "parquet"
            if fmt == "csv":
                path = path_prefix.with_suffix(".csv")
                df.to_csv(path, index=False, float_format="%.10g")
                return path, "csv"
            path = path_prefix.with_suffix(".pkl")
            df.to_pickle(path)
            return path, "pkl"
        except Exception as exc:  # pragma: no cover - best-effort fallback
            last_error = exc
            continue
    if last_error is None:
        raise RuntimeError("Failed to save dataframe")
    raise RuntimeError(
        f"Failed to save dataframe: {type(last_error).__name__}: {last_error}"
    )


# ----------------------------
# Dataset selection
# ----------------------------


def select_data_file(dataset_dir: Path, seed: int, logger) -> Path | None:
    pattern = re.compile(
        r"data_(?P<strategy>.+)_n(?P<n>\d+)_seed(?P<seed>\d+)\.(parquet|csv|pkl)$"
    )
    candidates = []
    for path in sorted(dataset_dir.iterdir()):
        if not path.is_file():
            continue
        match = pattern.match(path.name)
        if not match:
            continue
        try:
            n_samples = int(match.group("n"))
            file_seed = int(match.group("seed"))
        except Exception:
            continue
        candidates.append((path, n_samples, file_seed))

    seed_candidates = [c for c in candidates if c[2] == int(seed)]
    if seed_candidates:
        seed_candidates.sort(key=lambda item: (item[1], item[0].name))
        chosen = seed_candidates[-1][0]
        logger.info("Selected data file for seed %s: %s", seed, chosen)
        return chosen

    data_files = [
        p
        for p in sorted(dataset_dir.iterdir())
        if p.is_file()
        and p.name.startswith("data_")
        and p.suffix in {".parquet", ".csv", ".pkl"}
    ]
    if not data_files:
        return None
    data_files.sort(key=lambda p: (p.stat().st_mtime, p.name))
    chosen = data_files[-1]
    logger.info("No seed-specific data found; selected most recent file: %s", chosen)
    return chosen


# ----------------------------
# Timing
# ----------------------------


def timed_call(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    end = time.perf_counter()
    return result, (end - start) * 1000.0


# ----------------------------
# BIF helpers
# ----------------------------


def parse_bif_structure(path: Path) -> tuple[list[str], dict]:
    from benchmarking.III_data_generation import bnlearn as bnlearn_data

    nodes, _, _, _, parents_map, _ = bnlearn_data._parse_bif(path)
    return nodes, parents_map


__all__ = [
    "get_benchmarking_root",
    "get_project_root",
    "get_static_metadata_dir",
    "get_data_dir",
    "get_datasets_dir",
    "get_generator_datasets_dir",
    "get_dataset_dir",
    "get_metadata_dir_generated",
    "get_generator_metadata_dir_generated",
    "get_dataset_metadata_dir_generated",
    "get_dataset_download_metadata_path",
    "get_dataset_encoding_metadata_path",
    "get_dataset_domain_metadata_path",
    "get_dataset_data_generation_metadata_path",
    "get_queries_dir",
    "get_generator_queries_dir",
    "get_dataset_queries_dir",
    "get_queries_log_dir",
    "get_generator_queries_log_dir",
    "get_datasets_log_dir",
    "get_generator_datasets_log_dir",
    "get_out_dir",
    "get_generator_out_dir",
    "ensure_dir",
    "read_json",
    "write_json",
    "read_dataframe",
    "save_dataframe",
    "select_data_file",
    "timed_call",
    "parse_bif_structure",
]
