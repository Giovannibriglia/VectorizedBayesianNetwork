from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarking.bundles import BenchmarkBundle


def test_report_results_single_and_aggregate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_root = tmp_path / "benchmarking" / "data" / "benchmarks"
    bundle = BenchmarkBundle.create(
        mode="cpds",
        generator="bnlearn",
        seed=0,
        root=bundle_root,
        bundle_id="benchmark_cpds_test",
    )
    bundle.set_dataset_ids(["asia"])
    bundle.save_metadata()

    run_dir = tmp_path / "benchmarking" / "out" / "bnlearn" / "benchmark_cpds_test"
    results_dir = run_dir / "results" / "asia"
    results_dir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "mode": "cpds",
        "generator": "bnlearn",
        "bundle_dir": str(bundle.paths.root),
        "run_id": "benchmark_cpds_test",
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))

    record = {
        "mode": "cpds",
        "run": {
            "run_id": "benchmark_cpds_test",
            "seed": 0,
            "generator": "bnlearn",
            "timestamp_utc": "2025-01-01T00:00:00Z",
            "mode": "cpds",
        },
        "model": {
            "name": "dummy",
            "config_id": "default",
            "config_hash": "deadbeef",
            "components": {
                "cpd": {"name": "dummy_cpd"},
                "inference": {"name": "dummy_inf"},
                "learning": {"name": "dummy_learn"},
            },
        },
        "problem": {"id": "asia", "n_nodes": 2, "n_edges": 1},
        "query": {
            "type": "cpd",
            "target": "A",
            "target_category": "parent_set",
            "index": 0,
            "evidence": {"vars": ["B"], "strategy": "paths"},
        },
        "result": {
            "ok": True,
            "timing_ms": 1.0,
            "output": {
                "format": "categorical_probs",
                "probs": [0.6, 0.4],
                "support": [0, 1],
            },
            "ground_truth": {"output": {"probs": [0.6, 0.4]}},
        },
    }
    (results_dir / "dummy.jsonl").write_text(json.dumps(record) + "\n")

    module = importlib.import_module("benchmarking.scripts.05_report_results")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "05_report_results.py",
            "--run_dir",
            str(run_dir),
            "--gt_source",
            "embedded",
            "--summary_style",
            "mean",
        ],
    )
    module.main()

    report_root = run_dir / "report"
    assert (report_root / "aggregate" / "all").is_dir()
    assert (report_root / "single" / "asia" / "all").is_dir()
    assert (report_root / "by_category" / "small" / "all").is_dir()
    assert (report_root / "aggregate" / "partition_inventory.csv").exists()
    assert (report_root / "single" / "asia" / "index.md").exists()
    root_index = (report_root / "index.md").read_text()
    assert "(by_category/small/)" in root_index


def test_report_robust_summary_clips_non_negative_metrics() -> None:
    module = importlib.import_module("benchmarking.scripts.05_report_results")
    style = module.SUMMARY_STYLES["robust"]
    summary = module.summarize([-0.4, 0.1, 0.3], style, metric="jsd")
    assert summary["iqm"] >= 0.0
    assert summary["min"] >= 0.0
    assert summary["max"] >= summary["min"]
    assert summary["q1"] >= 0.0
    assert summary["iqr"] >= 0.0
    assert summary["iqm_pm_iqr"] is not None
    assert summary["iqm_low_iqrstd_clipped"] >= summary["min"]
    assert summary["iqm_high_iqrstd_clipped"] <= summary["max"]
    assert summary["iqm_range_iqrstd_clipped"] is not None


def test_report_aggregate_time_table_stores_total_and_per_query() -> None:
    module = importlib.import_module("benchmarking.scripts.05_report_results")
    style = module.SUMMARY_STYLES["robust"]
    df = pd.DataFrame(
        {
            "method_id": ["m1", "m1", "m1"],
            "time": [1.0, 2.0, 3.0],
        }
    )
    table = module.aggregate_time_table(df, ["method_id"], summary_style=style)
    row = table.iloc[0]
    assert "time_min" in table.columns
    assert "time_max" in table.columns
    assert "time_iqm_low_iqrstd_clipped" in table.columns
    assert "time_iqm_high_iqrstd_clipped" in table.columns
    assert int(row["n_queries"]) == 3
    assert int(row["n_timed_queries"]) == 3
    assert row["time_sum_ms"] == pytest.approx(6.0)
    assert row["time_per_query_ms"] == pytest.approx(2.0)


def test_metric_errorbars_robust_are_asymmetric_and_clipped() -> None:
    module = importlib.import_module("benchmarking.scripts.05_report_results")
    style = module.SUMMARY_STYLES["robust"]
    df = pd.DataFrame(
        {
            "kl_iqm": [1.5, 2.0],
            "kl_iqr": [4.0, 5.0],  # should be ignored for robust error bars
            "kl_iqm_low_iqrstd_clipped": [1.0, 1.6],
            "kl_iqm_high_iqrstd_clipped": [2.2, 2.5],
        }
    )
    yerr = module._metric_errorbars(df, metric="kl", summary_style=style)
    assert isinstance(yerr, np.ndarray)
    assert yerr.shape == (2, 2)
    assert yerr[0, 0] == pytest.approx(0.5)
    assert yerr[1, 0] == pytest.approx(0.7)
