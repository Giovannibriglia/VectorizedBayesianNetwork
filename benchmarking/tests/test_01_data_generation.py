from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

import pandas as pd
import pytest

from benchmarking.data_generation import DATA_GENERATOR_REGISTRY

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


def _prepare_bnlearn_root(root: Path, networks: list[str]) -> None:
    bnlearn_root = root / "benchmarking" / "bnlearn_data"
    source_root = root / "bnlearn_source"
    bnlearn_root.mkdir(parents=True, exist_ok=True)
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

    (bnlearn_root / "metadata.json").write_text(json.dumps(metadata, indent=2))


def _prepare_root_for_generator(name: str, root: Path) -> list[str]:
    if name == "bnlearn":
        networks = ["asia", "alarm"]
        _prepare_bnlearn_root(root, networks)
        return networks
    raise AssertionError(
        f"No test setup for generator '{name}'. Add a test fixture for it."
    )


def _run_generation(
    root: Path, generator_name: str, networks: list[str], seed: int
) -> Path:
    generator_cls = DATA_GENERATOR_REGISTRY[generator_name]
    generator = generator_cls(root_path=root, seed=seed)
    generator.generate(n_samples=50, networks=networks, force=True)
    return root / "benchmarking" / "data" / generator_name


def _bnlearn_available() -> bool:
    try:
        from pgmpy.readwrite import BIFReader  # noqa: F401

        return True
    except Exception:
        try:
            import bnlearn  # noqa: F401

            return True
        except Exception:
            return False


def test_01_data_generation_pipeline(tmp_path: Path) -> None:
    print("\n[Benchmark Test] Step 1: Data Generation")
    assert DATA_GENERATOR_REGISTRY, "No data generators registered"

    for name in DATA_GENERATOR_REGISTRY:
        if name == "bnlearn" and not _bnlearn_available():
            pytest.skip("bnlearn data generator requires pgmpy or bnlearn.")
        root = tmp_path / name
        networks = _prepare_root_for_generator(name, root)

        dataset_dir = _run_generation(root, name, networks, seed=123)
        assert dataset_dir.is_dir()

        for network in networks:
            data_path = dataset_dir / f"{network}.parquet"
            assert data_path.exists()
            df = pd.read_parquet(data_path)
            assert len(df) == 50

        metadata = json.loads((dataset_dir / "metadata.json").read_text())
        assert metadata["generator"] == name
        assert metadata["n_samples"] == 50
        assert metadata["seed"] == 123
        assert set(metadata["networks"]) == set(networks)
        assert set(metadata["variables"].keys()) == set(networks)
        assert set(metadata["encoding"].keys()) == set(networks)

        df_before = pd.read_parquet(dataset_dir / f"{networks[0]}.parquet")
        _run_generation(root, name, networks, seed=123)
        df_after = pd.read_parquet(dataset_dir / f"{networks[0]}.parquet")
        assert df_before.equals(df_after)

        generator_cls = DATA_GENERATOR_REGISTRY[name]
        generator = generator_cls(root_path=root, seed=0)
        generator.generate(n_samples=50, networks=networks, force=True)

        data_path = dataset_dir / f"{networks[0]}.parquet"
        mtime_before = data_path.stat().st_mtime
        generator.generate(n_samples=50, networks=networks, force=False)
        mtime_after = data_path.stat().st_mtime
        assert mtime_after == mtime_before

        time.sleep(1.1)
        generator.generate(n_samples=50, networks=networks, force=True)
        mtime_forced = data_path.stat().st_mtime
        assert mtime_forced > mtime_after
