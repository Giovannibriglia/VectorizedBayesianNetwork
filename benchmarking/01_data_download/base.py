from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - best-effort fallback

    def tqdm(iterable: Iterable[str], **kwargs: Any) -> Iterable[str]:
        return iterable


from benchmarking.paths import (
    ensure_dir,
    get_dataset_metadata_dir_generated,
    get_generator_datasets_dir,
    get_generator_metadata_dir_generated,
)


class BaseDataDownloader(ABC):
    name: str
    test_datasets: list[str] | None = None

    def __init__(self, root_path: Path, seed: int = 42, **kwargs: Any) -> None:
        self.root_path = Path(root_path).resolve()
        self.seed = int(seed)
        if not getattr(self, "name", None):
            raise ValueError("Downloader class must define a non-empty 'name'.")
        self.datasets_dir = ensure_dir(
            get_generator_datasets_dir(self.root_path, self.name)
        )
        self.generated_metadata_dir = ensure_dir(
            get_generator_metadata_dir_generated(self.root_path, self.name)
        )

    @abstractmethod
    def download(
        self,
        datasets: list[str] | None = None,
        force: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Main download method.
        Must create dataset artifacts under benchmarking/data/datasets/<generator>/<problem>/
        and write generated metadata under benchmarking/data/metadata/<generator>/.
        """

    def dataset_id(self, dataset_name: str) -> str:
        return dataset_name

    def dataset_dir(self, dataset_id: str) -> Path:
        return ensure_dir(self.datasets_dir / dataset_id)

    def progress(self, iterable: Iterable[str], desc: str) -> Iterable[str]:
        return tqdm(iterable, desc=desc, unit="dataset")

    def build_metadata(self, datasets: list[dict], **extra: Any) -> dict:
        metadata = {
            "generator": self.name,
            "seed": int(self.seed),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "datasets": datasets,
        }
        metadata.update(extra)
        return metadata

    def save_metadata(self, metadata: dict, filename: str | None = None) -> Path:
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.now(timezone.utc).isoformat()
        name = filename or f"{self.name}.json"
        metadata_path = self.generated_metadata_dir / name
        metadata_path.write_text(json.dumps(metadata, indent=2))
        return metadata_path

    def write_dataset_manifest(self, dataset_dir: Path, manifest: dict) -> Path:
        manifest_path = dataset_dir / "dataset.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        return manifest_path

    def write_dataset_metadata(
        self, dataset_id: str, filename: str, payload: dict
    ) -> Path:
        metadata_dir = ensure_dir(
            get_dataset_metadata_dir_generated(self.root_path, self.name, dataset_id)
        )
        path = metadata_dir / filename
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return path
