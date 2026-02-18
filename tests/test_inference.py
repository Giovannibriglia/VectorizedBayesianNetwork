import os

import networkx as nx
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
