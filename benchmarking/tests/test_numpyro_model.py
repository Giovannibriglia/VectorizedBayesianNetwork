from __future__ import annotations

import numpy as np
import pandas as pd

from benchmarking.models.config import make_component, ModelBenchmarkConfig
from benchmarking.models.numpyro import NumpyroBenchmarkModel


class _FakeDag:
    def __init__(self) -> None:
        self._nodes = ["A", "B"]
        self._edges = [("A", "B")]

    def nodes(self) -> list[str]:
        return list(self._nodes)

    def predecessors(self, node: str) -> list[str]:
        return [u for u, v in self._edges if v == node]


class _FakeJnp:
    float32 = np.float32

    @staticmethod
    def asarray(x, dtype=None):
        return np.asarray(x, dtype=dtype)


class _FakeRandom:
    @staticmethod
    def PRNGKey(seed: int) -> int:
        return int(seed)

    @staticmethod
    def split(key: int):
        return int(key) + 1, int(key) + 2


class _FakeCategorical:
    def __init__(self, *, probs) -> None:
        arr = np.asarray(probs, dtype=float).reshape(-1)
        total = float(arr.sum())
        if total <= 0:
            arr = np.ones_like(arr) / float(len(arr))
        else:
            arr = arr / total
        self._probs = arr

    def sample(self, key: int) -> int:
        rng = np.random.default_rng(int(key))
        return int(rng.choice(len(self._probs), p=self._probs))

    def log_prob(self, value) -> float:
        idx = int(np.asarray(value).reshape(-1)[0])
        prob = float(self._probs[idx])
        return float(np.log(max(prob, 1e-12)))


class _FakeDist:
    Categorical = _FakeCategorical


def _config() -> ModelBenchmarkConfig:
    return ModelBenchmarkConfig(
        model="numpyro",
        config_id="test_numpyro",
        learning=make_component("learning", "dirichlet_table", kwargs={"alpha": 1.0}),
        cpd=make_component("cpd", "dirichlet_table", kwargs={"alpha": 1.0}),
        inference=make_component("inference", "likelihood_weighting", kwargs={}),
    )


def test_numpyro_model_fit_and_queries(monkeypatch) -> None:
    import benchmarking.models.numpyro as module

    monkeypatch.setattr(
        module,
        "_require_numpyro",
        lambda: (_FakeJnp, _FakeRandom, _FakeDist),
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

    model = NumpyroBenchmarkModel(
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
