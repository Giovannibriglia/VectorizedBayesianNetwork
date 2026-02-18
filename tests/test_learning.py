import os

import networkx as nx
import pytest
import torch
from tqdm.auto import tqdm
from vbn import VBN
from vbn.core.registry import LEARNING_REGISTRY


def test_learning_methods_suite():
    g = nx.DiGraph()
    g.add_edges_from([("x", "y"), ("z", "y")])
    data = {
        "x": torch.randn(20, 1),
        "z": torch.randn(20, 1),
        "y": torch.randn(20, 1),
    }

    bar = tqdm(
        list(LEARNING_REGISTRY.keys()), desc="Learning", disable=bool(os.getenv("CI"))
    )
    for method in bar:
        bar.set_description(f"Testing Learning: {method}")
        vbn = VBN(g, seed=0, device="cpu")
        if method == "amortized":
            learner_cls = LEARNING_REGISTRY[method]
            learner = learner_cls()
            with pytest.raises(NotImplementedError):
                learner.fit(vbn, data)
            continue
        vbn.set_learning_method(
            method,
            nodes_cpds={
                "x": {"cpd": "gaussian_nn"},
                "z": {"cpd": "gaussian_nn"},
                "y": {"cpd": "mdn", "n_components": 3},
            },
        )
        vbn.fit(data)
        assert set(vbn.nodes.keys()) == {"x", "y", "z"}
        for node, cpd in vbn.nodes.items():
            assert cpd.device.type == "cpu"
