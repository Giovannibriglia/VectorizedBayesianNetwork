from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from benchmarking.paths import (
    ensure_dir,
    get_dataset_data_generation_metadata_path,
    get_dataset_metadata_dir_generated,
    get_datasets_dir,
    get_generator_datasets_dir,
    get_generator_datasets_log_dir,
)


@dataclass(frozen=True)
class DataGenResult:
    data_path: Path | None
    format: Literal["parquet", "csv", "pkl"] | None
    schema: dict | None
    capabilities: dict
    notes: dict
    domain_path: Path | None = None
    skipped: bool = False
    reason: str | None = None


def stable_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


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


def save_dataframe(
    df: pd.DataFrame,
    path_prefix: Path,
    prefer: Literal["parquet", "csv", "pkl"] = "parquet",
) -> tuple[Path, Literal["parquet", "csv", "pkl"]]:
    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    order: list[str]
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


class BaseDataGenerator(ABC):
    name: str

    def __init__(
        self,
        root_path: Path,
        seed: int,
        n_samples: int,
        generation_strategy: str = "default",
        generator_kwargs: dict | None = None,
        **kwargs: Any,
    ) -> None:
        self.root_path = Path(root_path).resolve()
        self.seed = int(seed)
        self.n_samples = int(n_samples)
        if self.n_samples <= 0:
            raise ValueError("n_samples must be a positive integer")
        self.generation_strategy = generation_strategy or "default"
        if not getattr(self, "name", None):
            raise ValueError("Data generator class must define a non-empty 'name'.")
        self.generator_kwargs: dict = dict(generator_kwargs or {})
        for key, value in kwargs.items():
            if key not in self.generator_kwargs:
                self.generator_kwargs[key] = value

        self.datasets_dir = ensure_dir(
            get_generator_datasets_dir(self.root_path, self.name)
        )
        self.generator_log_dir = ensure_dir(
            get_generator_datasets_log_dir(self.root_path, self.name)
        )

    def list_dataset_dirs(self) -> list[Path]:
        if not self.datasets_dir.exists():
            self._warn_legacy_layout()
            return []
        dataset_dirs = sorted([p for p in self.datasets_dir.iterdir() if p.is_dir()])
        if not dataset_dirs:
            self._warn_legacy_layout()
        return dataset_dirs

    def _warn_legacy_layout(self) -> None:
        legacy_root = get_datasets_dir(self.root_path)
        if not legacy_root.exists():
            return
        legacy_entries = [
            p
            for p in legacy_root.iterdir()
            if p.is_dir() and p.name.startswith(f"{self.name}__")
        ]
        if legacy_entries:
            logging.getLogger(__name__).warning(
                "Legacy dataset layout detected under %s. Run 'python -m benchmarking.migrate_data_layout' or re-run 01_download_data.",
                legacy_root,
            )

    def _stable_seed(self, key: str) -> int:
        h = hashlib.sha256(key.encode("utf-8")).digest()
        return (int(self.seed) + int.from_bytes(h[:4], "little")) % (2**32)

    def _logger_for(self, dataset_id: str) -> logging.Logger:
        log_path = self.generator_log_dir / f"{dataset_id}_seed{self.seed}.log"
        logger_name = f"benchmarking.data.{self.name}.{dataset_id}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.handlers.clear()
        handler = logging.FileHandler(log_path)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _metadata_path(self, dataset_id: str) -> Path:
        return get_dataset_data_generation_metadata_path(
            self.root_path, self.name, dataset_id
        )

    def _relative_path(self, path: Path | None) -> str | None:
        if path is None:
            return None
        try:
            return str(path.relative_to(self.root_path))
        except ValueError:
            return str(path)

    def _write_metadata(self, dataset_id: str, result: DataGenResult) -> Path:
        meta_path = self._metadata_path(dataset_id)
        ensure_dir(meta_path.parent)

        hashes = None
        if result.data_path is not None and result.data_path.exists():
            hashes = {"sha256": stable_sha256(result.data_path)}

        payload = {
            "dataset_id": dataset_id,
            "generator": self.name,
            "generation_strategy": self.generation_strategy,
            "seed": int(self.seed),
            "n_samples": int(self.n_samples),
            "generator_kwargs": dict(sorted(self.generator_kwargs.items())),
            "data_path": self._relative_path(result.data_path),
            "format": result.format,
            "hashes": hashes,
            "schema": result.schema,
            "domain_path": self._relative_path(result.domain_path),
            "capabilities": result.capabilities,
            "notes": result.notes,
            "skipped": bool(result.skipped),
            "reason": result.reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if isinstance(result.notes, dict):
            if "approx_domain" in result.notes:
                payload["approx_domain"] = result.notes.get("approx_domain")
            if "approx_domain_reason" in result.notes:
                payload["approx_domain_reason"] = result.notes.get(
                    "approx_domain_reason"
                )

        meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return meta_path

    def generate_all(self) -> list[Path]:
        outputs: list[Path] = []
        dataset_dirs = self.list_dataset_dirs()
        root_logger = logging.getLogger(__name__)
        for dataset_dir in dataset_dirs:
            dataset_id = dataset_dir.name
            logger = self._logger_for(dataset_id)
            meta_dir = ensure_dir(
                get_dataset_metadata_dir_generated(
                    self.root_path, self.name, dataset_id
                )
            )
            out_dir = ensure_dir(dataset_dir)
            root_logger.info(
                "Generating data for dataset %s (n_samples=%s, strategy=%s)",
                dataset_id,
                self.n_samples,
                self.generation_strategy,
            )
            logger.info(
                "Generating data for dataset %s (n_samples=%s, strategy=%s)",
                dataset_id,
                self.n_samples,
                self.generation_strategy,
            )
            result = self.generate(dataset_id, dataset_dir, out_dir, meta_dir, logger)
            if result is None:
                result = DataGenResult(
                    data_path=None,
                    format=None,
                    schema=None,
                    capabilities={"can_generate_data": False},
                    notes={"approx_on_manifold": None, "approx_reason": "Skipped"},
                    skipped=True,
                    reason="Generator returned no result",
                )
                logger.warning("No data generated for dataset %s", dataset_id)
                root_logger.warning("No data generated for dataset %s", dataset_id)
            meta_path = self._write_metadata(dataset_id, result)
            logger.info("Wrote data generation metadata to %s", meta_path)
            root_logger.info("Wrote data generation metadata to %s", meta_path)
            outputs.append(meta_path)
        return outputs

    @abstractmethod
    def generate(
        self,
        dataset_id: str,
        dataset_dir: Path,
        out_dir: Path,
        meta_dir: Path,
        logger: logging.Logger,
    ) -> DataGenResult | None:
        """
        Generate a dataset for the given dataset directory.
        Return a DataGenResult or None to skip.
        """
