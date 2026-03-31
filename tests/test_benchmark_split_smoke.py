from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
from benchmarking.II_query_generation import (
    get_cpd_query_generator,
    get_inference_query_generator,
)
from benchmarking.models.base import BaseBenchmarkModel
from benchmarking.models.registry import BENCHMARK_MODEL_REGISTRY

BIF_TEMPLATE = """network \"{network}\" {{
}}

variable A {{
  type discrete [2] {{ a0, a1 }};
}}

variable B {{
  type discrete [2] {{ b0, b1 }};
}}

variable C {{
  type discrete [2] {{ c0, c1 }};
}}

probability (A) {{
  table 0.6, 0.4;
}}

probability (B | A) {{
  (a0) 0.7, 0.3;
  (a1) 0.2, 0.8;
}}

probability (C | A, B) {{
  (a0, b0) 0.5, 0.5;
  (a0, b1) 0.4, 0.6;
  (a1, b0) 0.3, 0.7;
  (a1, b1) 0.2, 0.8;
}}
"""


class DummyBenchmarkModel(BaseBenchmarkModel):
    name = "dummy"
    family = "test"
    version = "0"

    def fit(self, data_df: pd.DataFrame, *, progress: bool = True, **kwargs) -> None:
        return None

    def answer_cpd_query(self, query: dict) -> dict:
        return {
            "ok": True,
            "error": None,
            "timing_ms": 0.01,
            "result": {
                "format": "categorical_probs",
                "k": 2,
                "probs": [0.5, 0.5],
                "support": [0, 1],
            },
        }

    def answer_inference_query(self, query: dict) -> dict:
        return {
            "ok": True,
            "error": None,
            "timing_ms": 0.01,
            "result": {
                "format": "categorical_probs",
                "k": 2,
                "probs": [0.5, 0.5],
                "support": [0, 1],
            },
        }


def _load_script(path: Path):
    module_name = f"benchmarking_script_{path.name.replace('.', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prepare_dataset(project_root: Path, network: str, seed: int) -> Path:
    datasets_dir = project_root / "benchmarking" / "data" / "datasets" / "bnlearn"
    dataset_dir = datasets_dir / network
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "model.bif").write_text(BIF_TEMPLATE.format(network=network))
    data = pd.DataFrame(
        {
            "A": [0, 1, 0, 1, 0],
            "B": [1, 0, 1, 0, 1],
            "C": [0, 0, 1, 1, 0],
        }
    )
    path = dataset_dir / f"data_default_n5_seed{seed}.csv"
    data.to_csv(path, index=False)
    return dataset_dir


def _write_preset(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def test_benchmark_split_smoke(tmp_path, monkeypatch) -> None:
    project_root = Path(__file__).resolve().parents[1]
    seed = 0
    network = f"smoke_{tmp_path.name.replace('-', '_').replace('.', '_')}"

    dataset_dir = _prepare_dataset(project_root, network, seed)
    metadata_dir = (
        project_root / "benchmarking" / "data" / "metadata" / "bnlearn" / network
    )

    cpds_queries_dir = tmp_path / "queries_cpds"
    inf_queries_dir = tmp_path / "queries_inference"
    cpds_results_dir = tmp_path / "results_cpds"
    inf_results_dir = tmp_path / "results_inference"

    cpds_script = (
        project_root / "benchmarking" / "scripts" / "02_generate_cpds_queries.py"
    )
    inf_script = (
        project_root / "benchmarking" / "scripts" / "02_generate_inference_queries.py"
    )
    cpds_runner_script = (
        project_root / "benchmarking" / "scripts" / "04_run_cpds_benchmark.py"
    )
    inf_runner_script = (
        project_root / "benchmarking" / "scripts" / "04_run_inference_benchmark.py"
    )

    BENCHMARK_MODEL_REGISTRY["dummy"] = DummyBenchmarkModel

    try:
        assert get_cpd_query_generator("bnlearn")
        assert get_inference_query_generator("bnlearn")

        cpds_module = _load_script(cpds_script)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                str(cpds_script),
                "--networks",
                network,
                "--generator",
                "bnlearn",
                "--strategy",
                "parents",
                "--budget",
                "2",
                "--seed",
                str(seed),
                "--out_dir",
                str(cpds_queries_dir),
                "--log-level",
                "ERROR",
            ],
        )
        cpds_module.main()
        cpd_files = list((cpds_queries_dir / network).glob("*.jsonl"))
        assert cpd_files
        first_cpd = json.loads(cpd_files[0].read_text().splitlines()[0])
        assert first_cpd.get("kind") == "cpd"

        inf_module = _load_script(inf_script)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                str(inf_script),
                "--networks",
                network,
                "--generator",
                "bnlearn",
                "--strategy",
                "balanced",
                "--budget",
                "3",
                "--seed",
                str(seed),
                "--out_dir",
                str(inf_queries_dir),
                "--log-level",
                "ERROR",
            ],
        )
        inf_module.main()
        inf_files = list((inf_queries_dir / network).glob("*.jsonl"))
        assert inf_files
        first_inf = json.loads(inf_files[0].read_text().splitlines()[0])
        assert first_inf.get("kind") == "inference"

        cpds_preset_path = tmp_path / "cpds_preset.json"
        _write_preset(
            cpds_preset_path,
            {
                "models": {
                    "dummy": {
                        "learning": {"name": "none", "kwargs": {}},
                        "cpd": {"name": "none", "kwargs": {}},
                    }
                }
            },
        )

        cpds_runner_module = _load_script(cpds_runner_script)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                str(cpds_runner_script),
                "--networks",
                network,
                "--cpd_models",
                "dummy",
                "--cpd_preset",
                str(cpds_preset_path),
                "--queries_dir",
                str(cpds_queries_dir),
                "--results_dir",
                str(cpds_results_dir),
                "--seed",
                str(seed),
                "--log-level",
                "ERROR",
            ],
        )
        cpds_runner_module.main()
        cpd_runs = list((cpds_results_dir / "dummy" / network).glob("run_*.jsonl"))
        assert cpd_runs

        inf_preset_path = tmp_path / "inference_preset.json"
        _write_preset(
            inf_preset_path,
            {
                "methods": {
                    "dummy_inf": {
                        "model": "dummy",
                        "inference": {"name": "none", "kwargs": {}},
                    }
                }
            },
        )

        inf_runner_module = _load_script(inf_runner_script)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                str(inf_runner_script),
                "--networks",
                network,
                "--cpd_models",
                "dummy",
                "--inference_methods",
                "dummy_inf",
                "--cpd_preset",
                str(cpds_preset_path),
                "--inference_preset",
                str(inf_preset_path),
                "--queries_dir",
                str(inf_queries_dir),
                "--results_dir",
                str(inf_results_dir),
                "--seed",
                str(seed),
                "--log-level",
                "ERROR",
            ],
        )
        inf_runner_module.main()
        inf_runs = list(
            (inf_results_dir / "dummy" / "dummy_inf" / network).glob("run_*.jsonl")
        )
        assert inf_runs
    finally:
        BENCHMARK_MODEL_REGISTRY.pop("dummy", None)
        shutil.rmtree(dataset_dir, ignore_errors=True)
        shutil.rmtree(metadata_dir, ignore_errors=True)
