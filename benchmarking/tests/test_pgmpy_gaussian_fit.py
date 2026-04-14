from __future__ import annotations

import numpy as np
import pandas as pd

from benchmarking.models.config import make_component, ModelBenchmarkConfig
from benchmarking.models.pgmpy import PgmpyBenchmarkModel


class _FakeDag:
    def __init__(self) -> None:
        self._nodes = ["x1", "x2"]
        self._edges = [("x1", "x2")]

    def nodes(self) -> list[str]:
        return list(self._nodes)

    def edges(self) -> list[tuple[str, str]]:
        return list(self._edges)

    def predecessors(self, node: str) -> list[str]:
        return [u for u, v in self._edges if v == node]


class _FakeBN:
    def __init__(self, edges) -> None:
        self.edges = list(edges)
        self.nodes_list: list[str] = []

    def add_nodes_from(self, nodes) -> None:
        self.nodes_list.extend(list(nodes))


class _FakeLG:
    last_instance = None

    def __init__(self, edges) -> None:
        self.edges = list(edges)
        self.nodes_list: list[str] = []
        self.fit_calls: list[pd.DataFrame] = []
        _FakeLG.last_instance = self

    def add_nodes_from(self, nodes) -> None:
        self.nodes_list.extend(list(nodes))

    def fit(self, df: pd.DataFrame) -> None:
        self.fit_calls.append(df.copy())

    def check_model(self) -> bool:
        return True

    def to_joint_gaussian(self):
        n_vars = len(self.nodes_list)
        return type(
            "_Joint",
            (),
            {
                "variables": list(self.nodes_list),
                "mean": np.zeros(n_vars, dtype=float),
                "covariance": np.eye(n_vars, dtype=float),
            },
        )()


def _gaussian_config() -> ModelBenchmarkConfig:
    return ModelBenchmarkConfig(
        model="pgmpy",
        config_id="test_gaussian",
        learning=make_component("learning", "gaussian", kwargs={}),
        cpd=make_component("cpd", "gaussian", kwargs={}),
        inference=make_component("inference", "gaussian_exact", kwargs={}),
    )


def test_pgmpy_gaussian_fit_uses_model_fit(monkeypatch) -> None:
    import benchmarking.models.pgmpy as module

    def _fake_require_pgmpy():
        return (_FakeBN, object, object, object, _FakeLG, object)

    monkeypatch.setattr(module, "_require_pgmpy", _fake_require_pgmpy)

    domain = {
        "nodes": {
            "x1": {"type": "continuous", "states": []},
            "x2": {"type": "continuous", "states": []},
        }
    }
    df = pd.DataFrame({"x1": [0.0, 1.0, 2.0], "x2": [1.0, 2.0, 3.0]})

    model = PgmpyBenchmarkModel(
        dag=_FakeDag(), seed=0, domain=domain, benchmark_config=_gaussian_config()
    )
    model.fit(df)

    lg = _FakeLG.last_instance
    assert lg is not None
    assert len(lg.fit_calls) == 1
    assert list(lg.fit_calls[0].columns) == ["x1", "x2"]
    assert model._continuous is True
    assert model._lg_model is lg
    assert set(model._lg_order) == {"x1", "x2"}


def test_pgmpy_gaussian_can_be_forced_on_discrete_domain(monkeypatch) -> None:
    import benchmarking.models.pgmpy as module

    def _fake_require_pgmpy():
        return (_FakeBN, object, object, object, _FakeLG, object)

    monkeypatch.setattr(module, "_require_pgmpy", _fake_require_pgmpy)

    config = _gaussian_config()

    domain = {
        "nodes": {
            "x1": {"type": "discrete", "states": ["0", "1"]},
            "x2": {"type": "discrete", "states": ["0", "1"]},
        }
    }
    df = pd.DataFrame({"x1": [0, 1, 0, 1], "x2": [1, 1, 0, 0]})

    model = PgmpyBenchmarkModel(
        dag=_FakeDag(), seed=0, domain=domain, benchmark_config=config
    )
    model.fit(df)

    response = model.answer_cpd_query({"target": "x1", "evidence_values": {"x2": 1.0}})
    assert response["ok"] is True
    result = response["result"]
    assert isinstance(result, dict)
    assert result.get("format") == "normal_params"
