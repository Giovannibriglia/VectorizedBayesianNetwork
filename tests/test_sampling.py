import os

import networkx as nx
import torch
from tqdm.auto import tqdm
from vbn import VBN
from vbn.core.registry import SAMPLING_REGISTRY


def _make_vbn():
    g = nx.DiGraph()
    g.add_edge("x", "y")
    vbn = VBN(g, seed=0, device="cpu")
    vbn.set_learning_method(
        "node_wise",
        nodes_cpds={
            "x": {"cpd": "gaussian_nn"},
            "y": {"cpd": "gaussian_nn"},
        },
    )
    data = {"x": torch.randn(30, 1), "y": torch.randn(30, 1)}
    vbn.fit(data)
    return vbn


def _make_linear_vbn():
    g = nx.DiGraph()
    g.add_edge("x", "y")
    vbn = VBN(g, seed=0, device="cpu")
    vbn.set_learning_method(
        "node_wise",
        nodes_cpds={
            "x": {"cpd": "linear_gaussian"},
            "y": {"cpd": "linear_gaussian"},
        },
    )
    torch.manual_seed(0)
    x = torch.linspace(-2.0, 2.0, steps=200).unsqueeze(-1)
    noise = 0.05 * torch.randn_like(x)
    y = 2.0 * x + noise
    data = {"x": x, "y": y}
    vbn.fit(data)
    return vbn


def test_sampling_methods_suite():
    vbn = _make_vbn()
    bar = tqdm(
        list(SAMPLING_REGISTRY.keys()), desc="Sampling", disable=bool(os.getenv("CI"))
    )
    for method in bar:
        bar.set_description(f"Testing Sampling: {method}")
        vbn.set_sampling_method(method, n_samples=5)
        query = {"target": "y", "evidence": {"x": torch.randn(3, 1)}}
        samples = vbn.sample(query, n_samples=5)
        assert samples.shape == (3, 5, 1)


def test_sampling_do_clamps_and_affects_descendant():
    vbn = _make_linear_vbn()
    vbn.set_sampling_method("ancestral", n_samples=200)

    do_value = torch.tensor([[0.7]])
    samples_x = vbn.sample({"target": "x", "do": {"x": do_value}}, n_samples=200)
    expected = do_value.view(1, 1, 1).expand_as(samples_x)
    assert torch.allclose(samples_x, expected)

    samples_y_pos = vbn.sample(
        {"target": "y", "do": {"x": torch.tensor([[1.0]])}}, n_samples=200
    )
    samples_y_neg = vbn.sample(
        {"target": "y", "do": {"x": torch.tensor([[-1.0]])}}, n_samples=200
    )
    mean_diff = samples_y_pos.mean().item() - samples_y_neg.mean().item()
    assert mean_diff > 0.5
