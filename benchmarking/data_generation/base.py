from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm


class BaseDataGenerator(ABC):
    name: str
    test_networks: list[str] | None = None

    def __init__(self, root_path: Path, seed: int = 42, **kwargs: Any) -> None:
        self.root_path = Path(root_path).resolve()
        self.seed = int(seed)
        if not getattr(self, "name", None):
            raise ValueError("Generator class must define a non-empty 'name'.")
        self.dataset_path = self.root_path / "benchmarking" / "data" / self.name
        self.dataset_path.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def generate(
        self,
        n_samples: int,
        networks: list[str] | None = None,
        force: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Main generation method.
        Must create:
            - observational data in parquet format
            - metadata.json
        inside self.dataset_path
        """

    def dataset_file(self, network: str) -> Path:
        return self.dataset_path / f"{network}.parquet"

    def inspect_parquet(self, path: Path) -> tuple[int | None, list[str] | None]:
        try:
            import pyarrow.parquet as pq  # type: ignore

            pf = pq.ParquetFile(path)
            rows = int(pf.metadata.num_rows)
            cols = list(pf.schema.names)
            return rows, cols
        except Exception:
            try:
                import pandas as pd

                df = pd.read_parquet(path)
                return int(len(df)), list(df.columns)
            except Exception:
                return None, None

    def should_skip(self, network: str, n_samples: int | None, force: bool) -> bool:
        if force:
            return False
        path = self.dataset_file(network)
        if not path.exists():
            return False
        if n_samples is None:
            return True
        rows, _cols = self.inspect_parquet(path)
        return rows == int(n_samples)

    def progress(self, iterable: Iterable[str], desc: str) -> Iterable[str]:
        return tqdm(iterable, desc=desc, unit="net")

    def build_metadata(
        self,
        n_samples: int,
        networks: list[str],
        variables: dict[str, list[str]],
        encoding: dict[str, dict[str, dict[str, float]]],
        **extra: Any,
    ) -> dict:
        metadata = {
            "generator": self.name,
            "n_samples": int(n_samples),
            "seed": int(self.seed),
            "networks": list(networks),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "variables": variables,
            "encoding": encoding,
        }
        metadata.update(extra)
        return metadata

    def save_metadata(self, metadata: dict) -> None:
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.now(timezone.utc).isoformat()
        metadata_path = self.dataset_path / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))
