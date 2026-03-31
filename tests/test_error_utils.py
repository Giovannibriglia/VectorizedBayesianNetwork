import importlib
import json

import numpy as np
from benchmarking.utils_errors import classify_error, normalize_error_signature


def test_error_signature_normalization_and_oom():
    msg = "State '1_1' not found in pgmpy output for 'N6_d_g'"
    info = classify_error("ValueError", msg)
    assert (
        "state <state> not found in pgmpy output for <var>" in info["error_signature"]
    )
    assert info["is_oom"] is False

    oom_info = classify_error("RuntimeError", "CUDA out of memory. Tried to allocate")
    assert oom_info["is_oom"] is True

    sig = normalize_error_signature("ValueError", "Failed at step 123456")
    assert "<num>" in sig


def test_success_rate_aggregation(tmp_path):
    report = importlib.import_module("benchmarking.scripts.05_report_results")

    run_dir = tmp_path / "benchmark_cpds_20250101"
    cpds_dir = run_dir / "cpds"
    inf_dir = run_dir / "inference"
    cpds_dir.mkdir(parents=True)
    inf_dir.mkdir(parents=True)

    base_record = {
        "mode": "cpds",
        "run": {
            "run_id": "r1",
            "seed": 1,
            "generator": "gen",
            "timestamp_utc": "2025-01-01T00:00:00Z",
        },
        "model": {
            "name": "m",
            "alias": "m",
            "backend": "m",
            "config_id": "default",
            "config_hash": "abcd",
        },
        "problem": {"id": "p1", "n_nodes": 10, "n_edges": 20},
        "query": {
            "type": "cpd",
            "index": 0,
            "id": "p1::cpd::0",
            "target": "A",
            "target_category": "markov_blanket",
            "evidence": {"vars": [], "mode": "empty"},
        },
    }

    records = []
    ok_record = dict(base_record)
    ok_record["result"] = {"ok": True}
    records.append(ok_record)

    err_record = dict(base_record)
    err_record["query"] = dict(base_record["query"])
    err_record["query"]["index"] = 1
    err_record["query"]["id"] = "p1::cpd::1"
    err_record["result"] = {
        "ok": False,
        "error_type": "ValueError",
        "error_msg": "boom",
    }
    records.append(err_record)

    with (cpds_dir / "m.jsonl").open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec) + "\n")

    attempts = report._build_attempts(
        run_dir=run_dir, max_records=None, model_filter=None
    )
    assert len(attempts) == 2

    summary = report._aggregate_success(attempts, ["method_id", "query_type"])
    row = summary[summary["query_type"] == "cpd"].iloc[0]
    assert int(row["n_total"]) == 2
    assert int(row["n_ok"]) == 1
    assert np.isclose(float(row["success_rate"]), 0.5)
