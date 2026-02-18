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
