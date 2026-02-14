import networkx as nx
from vbn import defaults, VBN
from vbn.core.utils import to_plain_dict
from vbn.vbn import _load_configs, _serialize_nodes_cpds


def test_to_plain_dict_returns_copy_for_dict():
    conf = {"a": 1}
    out = to_plain_dict(conf)
    assert out == conf
    assert out is not conf


def test_to_plain_dict_config_item_includes_cpd_key():
    configs = _load_configs()
    conf = configs.cpds.softmax_nn
    out = to_plain_dict(conf)
    assert out["cpd"] == conf.name


def test_serialize_nodes_cpds_accepts_config_item_string_dict():
    configs = _load_configs()
    nodes_cpds = {
        "x": configs.cpds.softmax_nn,
        "y": "mdn",
        "z": {"cpd": "mdn", "n_components": 3},
    }
    serialized = _serialize_nodes_cpds(nodes_cpds)
    assert serialized["x"] == defaults.cpd("softmax_nn")
    assert serialized["y"] == defaults.cpd("mdn")
    assert serialized["z"] == {**defaults.cpd("mdn"), "n_components": 3}


def test_defaults_learning_dict_accepted():
    g = nx.DiGraph()
    g.add_edge("x", "y")
    vbn = VBN(g, seed=0, device="cpu")
    vbn.set_learning_method(
        method=defaults.learning("node_wise"),
        nodes_cpds={
            "x": defaults.cpd("softmax_nn"),
            "y": defaults.cpd("softmax_nn"),
        },
    )
    assert vbn._learning_config["name"] == "node_wise"


def test_config_items_in_set_learning_method():
    g = nx.DiGraph()
    g.add_edge("x", "y")
    vbn = VBN(g, seed=0, device="cpu")
    vbn.set_learning_method(
        method=vbn.config.learning.node_wise,
        nodes_cpds={"x": vbn.config.cpds.softmax_nn, "y": "mdn"},
    )
    assert vbn._learning_config["name"] == "node_wise"
