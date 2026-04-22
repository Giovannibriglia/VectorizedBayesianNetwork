from __future__ import annotations

import pandas as pd
import torch

from benchmarking.models.config import make_component, ModelBenchmarkConfig
from benchmarking.models.pyro import PyroBenchmarkModel


class _FakeDag:
    def __init__(
        self, nodes: list[str] | None = None, edges: list[tuple[str, str]] | None = None
    ) -> None:
        self._nodes = list(nodes or ["A", "B"])
        self._edges = list(edges or [("A", "B")])

    def nodes(self) -> list[str]:
        return list(self._nodes)

    def predecessors(self, node: str) -> list[str]:
        return [u for u, v in self._edges if v == node]


class _FakePyro:
    @staticmethod
    def set_rng_seed(seed: int) -> None:
        torch.manual_seed(int(seed))


def _config() -> ModelBenchmarkConfig:
    return ModelBenchmarkConfig(
        model="pyro",
        config_id="test_pyro",
        learning=make_component("learning", "dirichlet_table", kwargs={"alpha": 1.0}),
        cpd=make_component("cpd", "dirichlet_table", kwargs={"alpha": 1.0}),
        inference=make_component("inference", "likelihood_weighting", kwargs={}),
    )


def test_pyro_model_fit_and_queries(monkeypatch) -> None:
    import benchmarking.models.pyro as module

    monkeypatch.setattr(
        module,
        "_require_pyro",
        lambda: (torch, _FakePyro, torch.distributions),
    )

    domain = {
        "nodes": {
            "A": {"type": "discrete", "states": ["0", "1"]},
            "B": {"type": "discrete", "states": ["0", "1"]},
        }
    }
    data = pd.DataFrame(
        {
            "A": [0, 0, 0, 1, 1, 1, 1, 0],
            "B": [0, 0, 1, 1, 1, 1, 0, 0],
        }
    )

    model = PyroBenchmarkModel(
        dag=_FakeDag(),
        seed=0,
        domain=domain,
        benchmark_config=_config(),
    )
    model.fit(data)

    cpd = model.answer_cpd_query({"target": "B", "evidence_values": {"A": 1}})
    assert cpd["ok"] is True
    cpd_result = cpd["result"]
    assert cpd_result["format"] == "categorical_probs"
    assert cpd_result["k"] == 2
    assert abs(float(sum(cpd_result["probs"])) - 1.0) < 1e-8

    inf = model.answer_inference_query(
        {"target": "B", "evidence_values": {"A": 0}, "n_mc": 128}
    )
    assert inf["ok"] is True
    inf_result = inf["result"]
    assert inf_result["format"] == "categorical_probs"
    assert inf_result["k"] == 2
    assert abs(float(sum(inf_result["probs"])) - 1.0) < 1e-8


def test_pyro_model_batch_inference_deduplicates(monkeypatch) -> None:
    import benchmarking.models.pyro as module

    monkeypatch.setattr(
        module,
        "_require_pyro",
        lambda: (torch, _FakePyro, torch.distributions),
    )

    domain = {
        "nodes": {
            "A": {"type": "discrete", "states": ["0", "1"]},
            "B": {"type": "discrete", "states": ["0", "1"]},
        }
    }
    data = pd.DataFrame(
        {
            "A": [0, 0, 0, 1, 1, 1, 1, 0],
            "B": [0, 0, 1, 1, 1, 1, 0, 0],
        }
    )

    model = PyroBenchmarkModel(
        dag=_FakeDag(),
        seed=0,
        domain=domain,
        benchmark_config=_config(),
    )
    model.fit(data)
    assert model.supports_batched_inference_queries is True

    calls = {"n": 0}
    original = model._infer_distribution

    def _wrapped(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "_infer_distribution", _wrapped)
    queries = [
        {"target": "B", "evidence_values": {"A": 0}, "n_mc": 16},
        {"target": "B", "evidence_values": {"A": 0}, "n_mc": 16},
        {"target": "B", "evidence_values": {"A": 1}, "n_mc": 16},
    ]
    responses = model.answer_inference_queries(queries)
    assert len(responses) == len(queries)
    assert all(bool(r.get("ok")) for r in responses)
    assert calls["n"] == 2


def test_pyro_model_supports_continuous_nodes(monkeypatch) -> None:
    import benchmarking.models.pyro as module

    monkeypatch.setattr(
        module,
        "_require_pyro",
        lambda: (torch, _FakePyro, torch.distributions),
    )

    domain = {
        "nodes": {
            "A": {"type": "discrete", "states": ["0", "1"]},
            "Y": {"type": "continuous"},
        }
    }
    data = pd.DataFrame(
        {
            "A": [0, 0, 0, 1, 1, 1, 0, 1],
            "Y": [-1.2, -0.8, -0.5, 1.1, 1.4, 0.9, -0.7, 1.2],
        }
    )

    model = PyroBenchmarkModel(
        dag=_FakeDag(nodes=["A", "Y"], edges=[("A", "Y")]),
        seed=0,
        domain=domain,
        benchmark_config=_config(),
    )
    model.fit(data)

    cpd_cont = model.answer_cpd_query({"target": "Y", "evidence_values": {"A": 1}})
    assert cpd_cont["ok"] is True
    assert cpd_cont["result"]["format"] == "normal_params"

    inf_cont = model.answer_inference_query(
        {"target": "Y", "evidence_values": {"A": 0}, "n_mc": 128}
    )
    assert inf_cont["ok"] is True
    assert inf_cont["result"]["format"] == "normal_params"

    inf_disc = model.answer_inference_query(
        {"target": "A", "evidence_values": {"Y": 1.0}, "n_mc": 128}
    )
    assert inf_disc["ok"] is True
    assert inf_disc["result"]["format"] == "categorical_probs"
    assert abs(float(sum(inf_disc["result"]["probs"])) - 1.0) < 1e-8
