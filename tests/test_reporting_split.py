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
    project_root = Path(__file__).resolve().parents[1]
    problem = f"report_smoke_{tmp_path.name.replace('-', '_')}"

    dataset_dir = _prepare_dataset(project_root, problem)

    try:
        results_root = tmp_path / "results"
        cpds_root = results_root / "cpds" / "dummy" / problem
        inf_root = results_root / "inference" / "dummy" / "dummy_inf" / problem

        base_record = {
            "model": {
                "name": "dummy",
                "config_id": "cfg",
                "components": {
                    "learning": {"name": "none"},
                    "cpd": {"name": "none"},
                    "inference": {"name": "none"},
                },
            },
            "problem": {"id": problem, "n_nodes": 3, "n_edges": 2},
            "result": {
                "output": {
                    "format": "categorical_probs",
                    "probs": [0.6, 0.4],
                },
                "ground_truth": {"output": {"probs": [0.5, 0.5]}},
                "timing_ms": 1.5,
            },
            "run": {
                "seed": 0,
                "generator": "bnlearn",
                "timestamp_utc": "2020-01-01T00:00:00Z",
            },
        }

        cpd_record = {
            **base_record,
            "query": {
                "id": "q_cpd",
                "type": "cpd",
                "index": 0,
                "target": "C",
                "target_category": "markov_blanket",
                "evidence_strategy": "paths",
                "evidence": {"vars": ["A"], "mode": "on_manifold"},
            },
        }
        inf_record = {
            **base_record,
            "query": {
                "id": "q_inf",
                "type": "inference",
                "index": 1,
                "target": "B",
                "target_category": "central_hub",
                "task": "prediction",
                "evidence": {
                    "vars": ["A", "C"],
                    "mode": "on_manifold",
                    "skeleton_id": "sk1",
                },
            },
        }

        _write_jsonl(cpds_root / "run_000.jsonl", cpd_record)
        _write_jsonl(inf_root / "run_000.jsonl", inf_record)

        cpds_report_out = tmp_path / "reports" / "cpds"
        inf_report_out = tmp_path / "reports" / "inference"

        cpds_script = (
            project_root / "benchmarking" / "scripts" / "05_report_cpds_results.py"
        )
        inf_script = (
            project_root / "benchmarking" / "scripts" / "05_report_inference_results.py"
        )

        cpds_module = _load_script(cpds_script)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                str(cpds_script),
                "--results_root",
                str(results_root / "cpds"),
                "--out_dir",
                str(cpds_report_out),
                "--gt_source",
                "embedded",
                "--generator",
                "bnlearn",
                "--log-level",
                "ERROR",
            ],
        )
        cpds_module.main()
        overall_path = cpds_report_out / "tables" / "overall_by_model.csv"
        assert overall_path.exists()
        header = overall_path.read_text().splitlines()[0]
        assert "kl_iqm" in header
        assert "wass_iqm" in header
        assert list((cpds_report_out / "figures").glob("*.png"))

        inf_module = _load_script(inf_script)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                str(inf_script),
                "--results_root",
                str(results_root / "inference"),
                "--out_dir",
                str(inf_report_out),
                "--gt_source",
                "embedded",
                "--generator",
                "bnlearn",
                "--log-level",
                "ERROR",
            ],
        )
        inf_module.main()
        overall_path = inf_report_out / "tables" / "overall_by_model.csv"
        assert overall_path.exists()
        header = overall_path.read_text().splitlines()[0]
        assert "kl_iqm" in header
        assert "wass_iqm" in header
        assert list((inf_report_out / "figures").glob("*.png"))
    finally:
        shutil.rmtree(dataset_dir, ignore_errors=True)
