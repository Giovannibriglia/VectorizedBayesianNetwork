import os

import networkx as nx
import pytest
import torch
from tqdm.auto import tqdm
from vbn import VBN
from vbn.core.registry import INFERENCE_REGISTRY


def _make_vbn():
    g = nx.DiGraph()
    g.add_edge("x", "y")
    vbn = VBN(g, seed=0, device="cpu")
    vbn.set_learning_method(
        "node_wise",
        nodes_cpds={
            "x": {"cpd": "gaussian_nn"},
            "y": {"cpd": "mdn", "n_components": 2},
        },
    )
    data = {"x": torch.randn(30, 1), "y": torch.randn(30, 1)}
    vbn.fit(data)
    return vbn


def test_inference_methods_suite():
    vbn = _make_vbn()
    bar = tqdm(
        list(INFERENCE_REGISTRY.keys()), desc="Inference", disable=bool(os.getenv("CI"))
    )
    for method in bar:
        bar.set_description(f"Testing Inference: {method}")
        vbn.set_inference_method(method, n_samples=5)
        query = {"target": "y", "evidence": {"x": torch.randn(4, 1)}}
        pdf, samples = vbn.infer_posterior(query)
        assert pdf.shape == (4, 5)
        assert samples.shape == (4, 5, 1)


def test_inference_do_only_shapes():
    vbn = _make_vbn()
    vbn.set_inference_method("likelihood_weighting", n_samples=6)
    query = {"target": "y", "do": {"x": torch.tensor([[0.5], [-0.5]])}}
    pdf, samples = vbn.infer_posterior(query)
    assert pdf.shape == (2, 6)
    assert samples.shape == (2, 6, 1)


def test_inference_rejects_do_evidence_overlap():
    vbn = _make_vbn()
    vbn.set_inference_method("monte_carlo_marginalization", n_samples=5)
    query = {
        "target": "y",
        "evidence": {"x": torch.randn(2, 1)},
        "do": {"x": torch.randn(2, 1)},
    }
    with pytest.raises(ValueError):
        vbn.infer_posterior(query)


def _make_categorical_vbn():
    g = nx.DiGraph()
    g.add_edge("x", "y")
    vbn = VBN(g, seed=0, device="cpu")
    vbn.set_learning_method(
        "node_wise",
        nodes_cpds={
            "x": {"cpd": "categorical_table", "n_classes": 3},
            "y": {"cpd": "categorical_table", "n_classes": 3},
        },
    )
    x = torch.randint(0, 3, (128, 1)).float()
    noise = torch.randint(0, 3, (128, 1)).float()
    y = (x + noise).remainder(3.0)
    vbn.fit({"x": x, "y": y})
    return vbn


def test_categorical_exact_supports_categorical_table_root():
    vbn = _make_categorical_vbn()
    vbn.set_inference_method("categorical_exact", fallback="none")
    pdf, samples = vbn.infer_posterior({"target": "x"})
    assert pdf.shape == (1, 3)
    assert samples.shape == (1, 3, 1)
    assert torch.isfinite(pdf).all()
    assert torch.allclose(pdf.sum(dim=1), torch.ones(1), atol=1e-6)
