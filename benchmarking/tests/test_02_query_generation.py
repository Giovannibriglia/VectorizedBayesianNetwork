from __future__ import annotations

import importlib
import json
from pathlib import Path

from benchmarking.paths import (
    get_dataset_domain_metadata_path,
    get_generator_datasets_dir,
    get_queries_dir,
    get_queries_log_dir,
)

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


def _prepare_dataset(root: Path, generator: str, problem: str) -> Path:
    datasets_dir = get_generator_datasets_dir(root, generator)
    dataset_dir = datasets_dir / problem
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "model.bif").write_text(BIF_TEMPLATE.format(network=problem))
    return dataset_dir


def test_02_query_generation_registry_and_determinism(tmp_path: Path) -> None:
    print("\n[Benchmark Test] Step 2: Query Generation")
    module = importlib.import_module("benchmarking.II_query_generation")
    registry = module.QUERY_GENERATOR_REGISTRY
    assert "bnlearn" in registry

    root = tmp_path / "queries"
    generator_name = "bnlearn"
    dataset_id = "toy"
    _prepare_dataset(root, generator_name, dataset_id)

    generator_cls = registry["bnlearn"]
    generator = generator_cls(
        root_path=root,
        seed=123,
        n_queries={"cpds": 2, "inference": 3},
        generator_kwargs={"n_mc": 4, "foo": "bar"},
    )
    outputs = generator.generate_all()
    assert outputs

    output_path = get_queries_dir(root) / generator_name / dataset_id / "queries.json"
    assert output_path.exists()
    payload_first = json.loads(output_path.read_text())
    domain_path = get_dataset_domain_metadata_path(root, generator_name, dataset_id)
    assert domain_path.exists()
    domain_meta = json.loads(domain_path.read_text())
    assert domain_meta["dataset_id"] == dataset_id
    assert "nodes" in domain_meta
    assert payload_first["generator"] == "bnlearn"
    assert payload_first["seed"] == 123
    assert payload_first["dataset_id"] == dataset_id
    assert payload_first["n_queries"]["cpds"] == 2
    assert payload_first["n_queries"]["inference"] == 3
    assert payload_first["n_mc"] == 4
    assert payload_first["generator_kwargs"]["foo"] == "bar"
    assert payload_first["generator_kwargs"]["n_mc"] == 4
    assert len(payload_first["cpd_queries"]) == 2
    assert len(payload_first["inference_queries"]) == 3

    sample_cpd = payload_first["cpd_queries"][0]
    assert sample_cpd["query_type"] == "cpd"
    assert "target" in sample_cpd
    assert "target_category" in sample_cpd
    assert "evidence_strategy" in sample_cpd
    assert isinstance(sample_cpd["evidence_vars"], list)
    assert isinstance(sample_cpd["evidence_values"], dict)

    gt_path = get_queries_dir(root) / generator_name / dataset_id / "ground_truth.jsonl"
    assert gt_path.exists()
    assert "ground_truth" in payload_first
    assert payload_first["ground_truth"]["path"]

    sample_inf = payload_first["inference_queries"][0]
    assert sample_inf["query_type"] == "inference"
    assert sample_inf["task"] in {"prediction", "diagnosis"}
    assert sample_inf["generator"] == "bnlearn"
    assert "generator_kwargs" in sample_inf
    assert sample_inf["evidence_mode"] in {"empty", "on_manifold", "off_manifold"}
    evidence = sample_inf["evidence"]
    assert evidence["mode"] in {"empty", "on_manifold", "off_manifold"}
    assert "vars" in evidence
    assert "values" in evidence
    assert "skeleton_id" in sample_inf
    assert "mc_id" in sample_inf
    assert sample_inf["generator"] == "bnlearn"
    assert sample_inf["generator_kwargs"]["foo"] == "bar"

    modes = {q["evidence"]["mode"] for q in payload_first["inference_queries"]}
    assert modes <= {"empty", "on_manifold", "off_manifold"}

    for query in payload_first["inference_queries"]:
        mode = query["evidence"]["mode"]
        values = query["evidence"]["values"]
        assert query["mc_id"] is not None
        assert isinstance(values, dict)
        if mode == "empty":
            assert query["evidence"]["vars"] == []
            assert values == {}
        else:
            for value in values.values():
                assert isinstance(value, (int, float))

    coverage = payload_first["coverage"]
    assert "cpds" in coverage and "inference" in coverage
    assert "evidence" in coverage["cpds"]
    assert "evidence" in coverage["inference"]
    assert "mode_counts" in coverage["inference"]["evidence"]
    assert coverage["inference"]["evidence"]["n_instantiated"] == 3
    assert coverage["inference"]["evidence"]["n_skeletons"] == 1
    assert sum(coverage["inference"]["evidence"]["mode_counts"].values()) == 3

    log_path = get_queries_log_dir(root) / "bnlearn" / f"{dataset_id}_seed123.log"
    assert log_path.exists()

    generator_again = generator_cls(
        root_path=root,
        seed=123,
        n_queries={"cpds": 2, "inference": 3},
        generator_kwargs={"n_mc": 4, "foo": "bar"},
    )
    generator_again.generate_all()
    payload_second = json.loads(output_path.read_text())
    assert payload_first == payload_second
