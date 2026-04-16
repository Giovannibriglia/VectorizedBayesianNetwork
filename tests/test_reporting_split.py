from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

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


def _load_script(path: Path):
    module_name = f"benchmarking_script_{path.name.replace('.', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, sort_keys=True) + "\n")


def _prepare_dataset(project_root: Path, problem: str) -> Path:
    dataset_dir = (
        project_root / "benchmarking" / "data" / "datasets" / "bnlearn" / problem
    )
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "model.bif").write_text(BIF_TEMPLATE.format(network=problem))
    return dataset_dir


def test_reporting_split_smoke(tmp_path, monkeypatch) -> None:
    problem = f"report_smoke_{tmp_path.name.replace('-', '_')}"
    project_root = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "bnlearn" / "benchmark_cpds_000000"
    dataset_dir = _prepare_dataset(project_root, problem)

    try:
        result_record = {
            "mode": "cpds",
            "run": {
                "seed": 0,
                "generator": "bnlearn",
                "run_id": "benchmark_cpds_000000",
                "timestamp_utc": "2020-01-01T00:00:00Z",
            },
            "model": {
                "name": "dummy",
                "config_id": "cfg",
                "config_hash": "sha256:dummy",
                "components": {
                    "learning": {"name": "none"},
                    "cpd": {"name": "none"},
                    "inference": {"name": "none"},
                },
            },
            "problem": {"id": problem, "n_nodes": 3, "n_edges": 2},
            "query": {
                "id": "q_cpd",
                "type": "cpd",
                "index": 0,
                "target": "C",
                "target_category": "markov_blanket",
                "evidence_strategy": "paths",
                "evidence": {"vars": ["A"], "mode": "on_manifold"},
            },
            "result": {
                "ok": True,
                "timing_ms": 1.5,
                "output": {
                    "format": "categorical_probs",
                    "k": 2,
                    "probs": [0.6, 0.4],
                    "support": [0, 1],
                },
                "ground_truth": {
                    "output": {
                        "format": "categorical_probs",
                        "k": 2,
                        "probs": [0.5, 0.5],
                        "support": [0, 1],
                    }
                },
            },
        }
        _write_jsonl(run_dir / "results" / problem / "dummy.jsonl", result_record)
        (run_dir / "run_metadata.json").write_text(
            json.dumps(
                {
                    "mode": "cpds",
                    "generator": "bnlearn",
                    "seed": 0,
                    "bundle_dir": "",
                    "datasets_run": [problem],
                    "models": [{"alias": "dummy", "backend": "dummy", "preset": "cfg"}],
                },
                sort_keys=True,
            )
        )

        report_out = tmp_path / "reports"
        report_script = (
            project_root / "benchmarking" / "scripts" / "05_report_results.py"
        )
        report_module = _load_script(report_script)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                str(report_script),
                "--run_dir",
                str(run_dir),
                "--out_dir",
                str(report_out),
                "--gt_source",
                "embedded",
                "--log-level",
                "ERROR",
            ],
        )
        report_module.main()

        common_overall = (
            report_out / "aggregate" / "common" / "tables" / "overall_by_model.csv"
        )
        all_overall = (
            report_out / "aggregate" / "all" / "tables" / "overall_by_model.csv"
        )
        overall_path = common_overall if common_overall.exists() else all_overall
        assert overall_path.exists()
        header = overall_path.read_text().splitlines()[0]
        assert "kl_iqm" in header
        assert "wass_iqm" in header
        figure_candidates = list(
            (report_out / "aggregate" / "common" / "figures").glob("*.png")
        )
        if not figure_candidates:
            figure_candidates = list(
                (report_out / "aggregate" / "all" / "figures").glob("*.png")
            )
        assert figure_candidates
    finally:
        shutil.rmtree(dataset_dir, ignore_errors=True)
