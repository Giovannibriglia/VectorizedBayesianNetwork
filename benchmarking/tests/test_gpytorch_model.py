from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import torch

from benchmarking.models.config import make_component, ModelBenchmarkConfig
from benchmarking.models.gpytorch import GpytorchBenchmarkModel


class _FakeDag:
    def __init__(self) -> None:
        self._nodes = ["A", "B"]
        self._edges = [("A", "B")]

    def nodes(self) -> list[str]:
        return list(self._nodes)

    def predecessors(self, node: str) -> list[str]:
        return [u for u, v in self._edges if v == node]


class _FakeLikelihood:
    def __init__(self) -> None:
        self.noise_covar = SimpleNamespace(noise=1e-4)

    def train(self):
        return self

    def eval(self):
        return self

    def __call__(self, output):
        return output


class _FakeMll:
    def __init__(self, likelihood, model) -> None:
        del likelihood
        self.model = model

    def __call__(self, output, target):
        del output, target
        return -(self.model.param**2).sum()


class _FakeFastPredVar:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        return False


class _FakeSettings:
    @staticmethod
    def fast_pred_var():
        return _FakeFastPredVar()


class _FakePredictive:
    def __init__(self, mean, variance) -> None:
        self.mean = mean
        self.variance = variance


class _FakeModel:
    def __init__(self) -> None:
        self.param = torch.nn.Parameter(torch.tensor([0.1], dtype=torch.float32))

    def train(self):
        return self

    def eval(self):
        return self

    def parameters(self):
        return [self.param]

    def __call__(self, x):
        mean = x.mean(dim=1) + self.param[0]
        variance = torch.ones_like(mean) * (0.2 + torch.abs(self.param[0]))
        return _FakePredictive(mean, variance)


def _config() -> ModelBenchmarkConfig:
    return ModelBenchmarkConfig(
        model="gpytorch",
        config_id="test_gpytorch",
        learning=make_component(
            "learning",
            "exact_gp",
            kwargs={
                "kernel": "rbf",
                "training_steps": 3,
                "lr": 0.05,
                "max_train_size": 64,
                "min_std": 1e-3,
            },
        ),
        cpd=make_component("cpd", "gp_posterior", kwargs={}),
        inference=make_component("inference", "gp_forward_sample", kwargs={}),
    )


def test_gpytorch_model_fit_and_queries(monkeypatch) -> None:
    import benchmarking.models.gpytorch as module

    fake_gpytorch = SimpleNamespace(
        likelihoods=SimpleNamespace(GaussianLikelihood=_FakeLikelihood),
        mlls=SimpleNamespace(ExactMarginalLogLikelihood=_FakeMll),
        settings=_FakeSettings,
    )
    monkeypatch.setattr(module, "_require_gpytorch", lambda: (torch, fake_gpytorch))
    monkeypatch.setattr(
        module, "_make_exact_gp_model", lambda *args, **kwargs: _FakeModel()
    )

    domain = {
        "nodes": {
            "A": {"type": "discrete", "states": ["0", "1"]},
            "B": {"type": "discrete", "states": ["0", "1"]},
        }
    }
    data = pd.DataFrame(
        {
            "A": [0, 0, 1, 1, 0, 1, 0, 1],
            "B": [0, 1, 1, 1, 0, 1, 0, 1],
        }
    )

    model = GpytorchBenchmarkModel(
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
