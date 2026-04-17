from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
from benchmarking.bundles import BenchmarkBundle
from benchmarking.II_query_generation import get_query_generator
from benchmarking.models.base import BaseBenchmarkModel
from benchmarking.models.config import make_component, ModelBenchmarkConfig
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
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _prepare_dataset(bundle: BenchmarkBundle, network: str, seed: int) -> Path:
    dataset_dir = bundle.paths.datasets / "bnlearn" / network
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


def _dummy_config(model: str, mode: str, config_id: str) -> ModelBenchmarkConfig:
    del mode
    return ModelBenchmarkConfig(
        model=model,
        config_id=config_id,
        learning=make_component("learning", "none", kwargs={}),
        cpd=make_component("cpd", "none", kwargs={}),
        inference=make_component("inference", "none", kwargs={}),
    )


def test_benchmark_split_smoke(tmp_path, monkeypatch) -> None:
    project_root = Path(__file__).resolve().parents[1]
    seed = 0
    network = f"smoke_{tmp_path.name.replace('-', '_').replace('.', '_')}"
    bundle_root = tmp_path / "bundles"
    bundle_cpds = BenchmarkBundle.create(
        mode="cpds",
        generator="bnlearn",
        seed=seed,
        root=bundle_root,
        timestamp="cpds_000001",
    )
    bundle_inference = BenchmarkBundle.create(
        mode="inference",
        generator="bnlearn",
        seed=seed,
        root=bundle_root,
        timestamp="inf_000001",
    )
    _prepare_dataset(bundle_cpds, network, seed)
    _prepare_dataset(bundle_inference, network, seed)

    generate_script = (
        project_root / "benchmarking" / "scripts" / "02_generate_benchmark_queries.py"
    )
    run_script = project_root / "benchmarking" / "scripts" / "04_run_benchmark.py"
    out_root = tmp_path / "benchmark_out"

    BENCHMARK_MODEL_REGISTRY["dummy"] = DummyBenchmarkModel

    try:
        assert get_query_generator("bnlearn")

        import benchmarking.IIII_run_benchmark.base as runner_base

        monkeypatch.setattr(
            runner_base, "get_preset_config", lambda m, mo, c: _dummy_config(m, mo, c)
        )
        monkeypatch.setattr(
            runner_base, "get_generator_out_dir", lambda root, gen: out_root / gen
        )

        generate_module = _load_script(generate_script)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                str(generate_script),
                "--generator",
                "bnlearn",
                "--mode",
                "cpds",
                "--seed",
                str(seed),
                "--n_queries_cpds",
                "2",
                "--bundle_dir",
                str(bundle_cpds.paths.root),
                "--log-level",
                "ERROR",
            ],
        )
        generate_module.main()
        cpds_path = bundle_cpds.problem_paths(network).cpds_path
        assert cpds_path.exists()
        first_cpd = json.loads(cpds_path.read_text().splitlines()[0])
        assert first_cpd.get("query_type") == "cpd"

        monkeypatch.setattr(
            sys,
            "argv",
            [
                str(generate_script),
                "--generator",
                "bnlearn",
                "--mode",
                "inference",
                "--seed",
                str(seed),
                "--n_queries_inference",
                "3",
                "--bundle_dir",
                str(bundle_inference.paths.root),
                "--log-level",
                "ERROR",
            ],
        )
        generate_module.main()
        inf_path = bundle_inference.problem_paths(network).inference_path
        assert inf_path.exists()
        first_inf = json.loads(inf_path.read_text().splitlines()[0])
        assert first_inf.get("query_type") == "inference"

        run_module = _load_script(run_script)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                str(run_script),
                "--generator",
                "bnlearn",
                "--seed",
                str(seed),
                "--mode",
                "cpds",
                "--models",
                "dummy",
                "--bundle_dir",
                str(bundle_cpds.paths.root),
                "--log-level",
                "ERROR",
            ],
        )
        run_module.main()
        cpds_runs = sorted((out_root / "bnlearn").glob("benchmark_cpds_*"))
        assert cpds_runs
        cpds_result_files = list(
            (cpds_runs[-1] / "results" / network).glob("dummy*.jsonl")
        )
        assert cpds_result_files
        cpd_record = json.loads(cpds_result_files[0].read_text().splitlines()[0])
        assert cpd_record.get("result", {}).get("timing_ms") == 0.01

        monkeypatch.setattr(
            sys,
            "argv",
            [
                str(run_script),
                "--generator",
                "bnlearn",
                "--seed",
                str(seed),
                "--mode",
                "inference",
                "--models",
                "dummy",
                "--bundle_dir",
                str(bundle_inference.paths.root),
                "--log-level",
                "ERROR",
            ],
        )
        run_module.main()
        inf_runs = sorted((out_root / "bnlearn").glob("benchmark_inference_*"))
        assert inf_runs
        inf_result_files = list(
            (inf_runs[-1] / "results" / network).glob("dummy*.jsonl")
        )
        assert inf_result_files
        inf_record = json.loads(inf_result_files[0].read_text().splitlines()[0])
        assert inf_record.get("result", {}).get("timing_ms") == 0.01
    finally:
        BENCHMARK_MODEL_REGISTRY.pop("dummy", None)
