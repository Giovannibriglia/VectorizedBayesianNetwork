from __future__ import annotations

import importlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest


def _bnlearn_module():
    return importlib.import_module("benchmarking.03_data_generation.bnlearn")


def _read_dataset(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".pkl":
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported data format: {path}")


def test_alarm_bif_states_inferred() -> None:
    module = _bnlearn_module()
    repo_root = Path(__file__).resolve().parents[2]
    bif_path = repo_root / "benchmarking/data/datasets/bnlearn/alarm/model.bif"
    if not bif_path.exists():
        pytest.skip("alarm BIF not available in repo")

    (
        nodes,
        node_states,
        node_types,
        node_sources,
        parents_map,
        cpds,
    ) = module._parse_bif(bif_path)

    inferred, missing = module._infer_missing_states(
        nodes, node_states, node_types, node_sources, cpds, parents_map
    )

    for node in nodes:
        if node_types.get(node, "discrete") == "discrete":
            assert node_states.get(node), f"Missing states for {node}"

    assert not missing
    for node in inferred:
        assert node_sources.get(node) == "inferred_from_cpd"


def test_generate_alarm_small(tmp_path: Path) -> None:
    module = _bnlearn_module()
    repo_root = Path(__file__).resolve().parents[2]
    src_dataset = repo_root / "benchmarking/data/datasets/bnlearn/alarm"
    if not src_dataset.exists():
        pytest.skip("alarm dataset not available in repo")

    dst_dataset = tmp_path / "benchmarking/data/datasets/bnlearn/alarm"
    dst_dataset.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_dataset, dst_dataset)

    src_meta = repo_root / "benchmarking/data/metadata/bnlearn/alarm/download.json"
    dst_meta_dir = tmp_path / "benchmarking/data/metadata/bnlearn/alarm"
    dst_meta_dir.mkdir(parents=True, exist_ok=True)
    if src_meta.exists():
        shutil.copy2(src_meta, dst_meta_dir / "download.json")

    generator = module.BNLearnDataGenerator(root_path=tmp_path, seed=0, n_samples=10)
    outputs = generator.generate_all()
    assert outputs

    data_files = list(dst_dataset.glob("data_default_n10_seed0.*"))
    assert data_files
    df = _read_dataset(data_files[0])
    assert len(df) == 10

    domain_path = tmp_path / "benchmarking/data/metadata/bnlearn/alarm/domain.json"
    assert domain_path.exists()
    domain = json.loads(domain_path.read_text())
    expected_cols = set(domain.get("nodes", {}).keys())
    assert set(df.columns) == expected_cols

    gen_meta_path = (
        tmp_path / "benchmarking/data/metadata/bnlearn/alarm/data_generation.json"
    )
    assert gen_meta_path.exists()
