from __future__ import annotations

import hashlib
import json
import logging
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from benchmarking.paths import (
    ensure_dir,
    get_generator_datasets_dir,
    get_generator_queries_dir,
    get_generator_queries_log_dir,
)


class BaseQueryGenerator(ABC):
    name: str

    def __init__(
        self,
        root_path: Path,
        seed: int,
        n_queries: int | dict | None = None,
        n_queries_cpds: int | None = None,
        n_queries_inference: int | None = None,
        n_queries_by_type: dict | None = None,
        n_mc: int | None = None,
        generator_kwargs: dict | None = None,
        **kwargs: Any,
    ) -> None:
        self.root_path = Path(root_path).resolve()
        self.seed = int(seed)
        if not getattr(self, "name", None):
            raise ValueError("Query generator class must define a non-empty 'name'.")

        cpds, inference = self._resolve_query_counts(
            n_queries=n_queries,
            n_queries_cpds=n_queries_cpds,
            n_queries_inference=n_queries_inference,
            n_queries_by_type=n_queries_by_type,
        )
        self.n_queries_cpds = cpds
        self.n_queries_inference = inference

        self.generator_kwargs: dict = dict(generator_kwargs or {})
        for key, value in kwargs.items():
            if key not in self.generator_kwargs:
                self.generator_kwargs[key] = value

        if n_mc is None:
            n_mc = self.generator_kwargs.get("n_mc", 32)
        self.n_mc = int(n_mc)
        if self.n_mc <= 0:
            raise ValueError("n_mc must be a positive integer")

        self.datasets_dir = ensure_dir(
            get_generator_datasets_dir(self.root_path, self.name)
        )
        self.queries_dir = ensure_dir(
            get_generator_queries_dir(self.root_path, self.name)
        )
        self.generator_log_dir = ensure_dir(
            get_generator_queries_log_dir(self.root_path, self.name)
        )

    @staticmethod
    def _resolve_query_counts(
        *,
        n_queries: int | dict | None,
        n_queries_cpds: int | None,
        n_queries_inference: int | None,
        n_queries_by_type: dict | None,
    ) -> tuple[int, int]:
        if n_queries_by_type is not None:
            n_queries = n_queries_by_type

        if isinstance(n_queries, dict):
            cpds = n_queries.get("cpds")
            inference = n_queries.get("inference")
            if cpds is None or inference is None:
                raise ValueError("n_queries dict must include 'cpds' and 'inference'.")
            return int(cpds), int(inference)

        if n_queries_cpds is not None or n_queries_inference is not None:
            if n_queries_cpds is None or n_queries_inference is None:
                raise ValueError(
                    "Both n_queries_cpds and n_queries_inference must be provided."
                )
            return int(n_queries_cpds), int(n_queries_inference)

        if n_queries is None:
            raise ValueError(
                "Provide n_queries or both n_queries_cpds and n_queries_inference."
            )

        total = int(n_queries)
        if total <= 0:
            raise ValueError("n_queries must be a positive integer")
        cpds = total // 2
        inference = total - cpds
        return int(cpds), int(inference)

    def list_dataset_dirs(self) -> list[Path]:
        if not self.datasets_dir.exists():
            self._warn_legacy_layout()
            return []
        dataset_dirs = sorted([p for p in self.datasets_dir.iterdir() if p.is_dir()])
        if not dataset_dirs:
            self._warn_legacy_layout()
        return dataset_dirs

    def _warn_legacy_layout(self) -> None:
        from benchmarking.paths import get_datasets_dir

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
        logger_name = f"benchmarking.query.{self.name}.{dataset_id}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.handlers.clear()
        handler = logging.FileHandler(log_path)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _output_path(self, dataset_id: str) -> Path:
        dataset_dir = ensure_dir(self.queries_dir / dataset_id)
        return dataset_dir / "queries.json"

    def generate_all(self) -> list[Path]:
        outputs: list[Path] = []
        dataset_dirs = self.list_dataset_dirs()
        root_logger = logging.getLogger(__name__)
        for dataset_dir in dataset_dirs:
            dataset_id = dataset_dir.name
            logger = self._logger_for(dataset_id)
            dataset_seed = self._stable_seed(dataset_id)
            rng = random.Random(dataset_seed)
            root_logger.info("Generating queries for dataset %s", dataset_id)
            logger.info(
                "Generating queries for dataset %s (cpds=%s, inference=%s, n_mc=%s)",
                dataset_id,
                self.n_queries_cpds,
                self.n_queries_inference,
                self.n_mc,
            )
            payload = self.generate_payload(dataset_id, dataset_dir, rng, logger)
            if payload is None:
                logger.info("Skipping dataset %s", dataset_id)
                root_logger.warning("Skipping dataset %s", dataset_id)
                continue
            out_path = self._output_path(dataset_id)
            out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
            logger.info("Wrote query payload to %s", out_path)
            root_logger.info("Wrote query payload to %s", out_path)
            outputs.append(out_path)
        return outputs

    @abstractmethod
    def generate_payload(
        self,
        dataset_id: str,
        dataset_dir: Path,
        rng: random.Random,
        logger: logging.Logger,
    ) -> dict | None:
        """
        Return a JSON-serializable payload for this dataset, or None to skip.
        """
