import os

import networkx as nx
import torch
from tqdm.auto import tqdm
from vbn import VBN
from vbn.core.registry import UPDATE_REGISTRY


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
    data = {"x": torch.randn(40, 1), "y": torch.randn(40, 1)}
    vbn.fit(data)
    return vbn


def test_update_methods_suite(monkeypatch):
    bar = tqdm(
        list(UPDATE_REGISTRY.keys()), desc="Update", disable=bool(os.getenv("CI"))
    )
    for method in bar:
        bar.set_description(f"Testing Update: {method}")
        vbn = _make_vbn()
        for cpd in vbn.nodes.values():

            def _raise(*args, **kwargs):
                raise AssertionError("fit should not be called during update")

            monkeypatch.setattr(cpd, "fit", _raise)
        update_data = {"x": torch.randn(10, 1), "y": torch.randn(10, 1)}
        vbn.update(update_data, update_method=method)


def test_update_keeps_optimizer_state():
    vbn = _make_vbn()
    update_data = {"x": torch.randn(10, 1), "y": torch.randn(10, 1)}
    vbn.update(update_data, update_method="online_sgd")
    first_opt = next(iter(vbn.nodes.values()))._optimizer
    vbn.update(update_data, update_method="online_sgd")
    second_opt = next(iter(vbn.nodes.values()))._optimizer
    assert first_opt is second_opt
