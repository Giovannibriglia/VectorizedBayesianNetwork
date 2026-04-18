import json

import networkx as nx
import pytest
import torch
from vbn import VBN


def _make_vbn():
    g = nx.DiGraph()
    g.add_edge("x", "y")
    vbn = VBN(g, seed=0, device="cpu")
    vbn.set_learning_method(
        "node_wise",
        nodes_cpds={"x": {"cpd": "linear_gaussian"}, "y": {"cpd": "linear_gaussian"}},
    )
    torch.manual_seed(0)
    x = torch.randn(64, 1)
    y = 0.7 * x + 0.1 * torch.randn_like(x)
    vbn.fit({"x": x, "y": y})
    return vbn


def test_get_cpd_returns_handle():
    vbn = _make_vbn()
    handle = vbn.get_cpd("x")
    assert handle.node == "x"
    assert handle.parents == []


def test_get_cpds_returns_all_nodes():
    vbn = _make_vbn()
    handles = vbn.get_cpds()
    assert set(handles.keys()) == set(vbn.dag.nodes())


def test_summary_is_serializable():
    vbn = _make_vbn()
    handle = vbn.get_cpd("x")
    summary = handle.summary()
    json.dumps(summary)


def test_export_config_is_serializable():
    vbn = _make_vbn()
    handle = vbn.get_cpd("x")
    exported = handle.export_config()
    json.dumps(exported)


def test_clone_cpd():
    vbn = _make_vbn()
    handle = vbn.get_cpd("x")
    clone = handle.clone_cpd()
    assert type(clone) is type(handle.cpd)
    assert set(clone.state_dict().keys()) == set(handle.state_dict().keys())
    samples = clone.sample(None, 3)
    assert samples.shape[-1] == handle.output_dim


def test_conditional_returns_valid_format():
    vbn = _make_vbn()
    handle = vbn.get_cpd("y")
    cond = handle.conditional({"x": torch.tensor([[0.2]])}, n_samples=16)
    assert cond["format"] in {
        "normal_params",
        "mixture_params",
        "categorical_probs",
        "empirical_samples",
    }


def test_conditional_mean_std():
    vbn = _make_vbn()
    handle = vbn.get_cpd("y")
    out = handle.conditional_mean_std({"x": torch.tensor([[0.2]])}, n_samples=32)
    assert torch.isfinite(out["mean"]).all()
    assert torch.isfinite(out["std"]).all()


def test_conditional_log_prob_alias():
    vbn = _make_vbn()
    handle = vbn.get_cpd("y")
    x = torch.tensor([[0.4]])
    parents = {"x": torch.tensor([[0.2]])}
    assert torch.allclose(
        handle.conditional_log_prob(x, parents), handle.log_prob(x, parents)
    )


def test_unknown_node_raises():
    vbn = _make_vbn()
    with pytest.raises(ValueError):
        vbn.get_cpd("missing")


def test_missing_parent_raises():
    vbn = _make_vbn()
    handle = vbn.get_cpd("y")
    with pytest.raises(ValueError):
        handle.sample({}, n_samples=1)


def test_root_categorical_table_conditional_probs_format():
    g = nx.DiGraph()
    g.add_node("x")
    vbn = VBN(g, seed=0, device="cpu")
    vbn.set_learning_method(
        "node_wise",
        nodes_cpds={"x": {"cpd": "categorical_table", "n_classes": 3}},
    )
    x = torch.randint(0, 3, (64, 1)).float()
    vbn.fit({"x": x})

    handle = vbn.get_cpd("x")
    cond = handle.conditional(None, n_samples=16)
    assert cond["format"] == "categorical_probs"
    assert int(cond["k"]) == 3
