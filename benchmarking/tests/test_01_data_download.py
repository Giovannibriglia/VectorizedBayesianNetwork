from __future__ import annotations

import gzip
import importlib
import json
import sys
from pathlib import Path

import pytest

from benchmarking.paths import (
    get_generator_datasets_dir,
    get_generator_metadata_dir_generated,
    get_static_metadata_dir,
)

BIF_TEMPLATE = """network \"{network}\" {{
}}

variable A {{
  type discrete [2] {{ a0, a1 }};
}}

variable B {{
  type discrete [2] {{ b0, b1 }};
}}

probability (A) {{
  table 0.6, 0.4;
}}

probability (B | A) {{
  (a0) 0.7, 0.3;
  (a1) 0.2, 0.8;
}}
"""


def _write_bif(path: Path, network: str) -> None:
    path.write_text(BIF_TEMPLATE.format(network=network))


def _gzip_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as f_in, gzip.open(dst, "wb") as f_out:
        f_out.write(f_in.read())


def _prepare_bnlearn_static_metadata(root: Path, networks: list[str]) -> None:
    static_metadata_dir = get_static_metadata_dir(root)
    static_metadata_dir.mkdir(parents=True, exist_ok=True)

    source_root = root / "bnlearn_source"
    source_root.mkdir(parents=True, exist_ok=True)

    metadata = {}
    for network in networks:
        bif_path = source_root / f"{network}.bif"
        bif_gz_path = source_root / f"{network}.bif.gz"
        _write_bif(bif_path, network)
        _gzip_file(bif_path, bif_gz_path)
        metadata[network] = {
            "name": network.upper(),
            "type": "discrete",
            "category": "small",
            "urls": {"bif_gz": bif_gz_path.as_uri()},
            "stats": {"nodes": 2, "arcs": 1, "parameters": 4},
        }

    (static_metadata_dir / "bnlearn.json").write_text(json.dumps(metadata, indent=2))


def _prepare_root_for_downloader(name: str, root: Path) -> list[str]:
    if name == "bnlearn":
        networks = ["asia", "alarm"]
        _prepare_bnlearn_static_metadata(root, networks)
        return networks
    raise AssertionError(
        f"No test setup for downloader '{name}'. Add a test fixture for it."
    )


def test_01_download_data_pipeline(tmp_path: Path) -> None:
    print("\n[Benchmark Test] Step 1: Data Download")
    module = importlib.import_module("benchmarking.01_data_download")
    registry = module.DATA_DOWNLOADER_REGISTRY
    assert registry, "No data downloaders registered"

    for name in registry:
        root = tmp_path / name
        networks = _prepare_root_for_downloader(name, root)

        downloader_cls = registry[name]
        downloader = downloader_cls(root_path=root, seed=123)
        downloader.download(datasets=networks, force=True)

        datasets_dir = get_generator_datasets_dir(root, name)
        assert datasets_dir.is_dir()

        for network in networks:
            dataset_id = network
            dataset_dir = datasets_dir / dataset_id
            assert dataset_dir.is_dir()
            assert (dataset_dir / "model.bif").exists()
            manifest_path = dataset_dir / "dataset.json"
            assert manifest_path.exists()
            manifest = json.loads(manifest_path.read_text())
            assert manifest["dataset_id"] == dataset_id

        metadata_path = (
            get_generator_metadata_dir_generated(root, name) / f"{name}.json"
        )
        assert metadata_path.exists()
        metadata = json.loads(metadata_path.read_text())
        assert metadata["generator"] == name
        assert metadata["seed"] == 123
        assert "n_samples" not in metadata
        assert len(metadata["datasets"]) == len(networks)


def test_01_download_data_cli_rejects_n_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("benchmarking.scripts.01_download_data")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "01_download_data.py",
            "--generator",
            "bnlearn",
            "--n_samples",
            "10",
        ],
    )
    with pytest.raises(SystemExit):
        module.main()
