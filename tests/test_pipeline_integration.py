import networkx as nx
import torch
from vbn import VBN


def test_full_pipeline():
    g = nx.DiGraph()
    g.add_edges_from([("x", "z"), ("y", "z")])
    vbn = VBN(g, seed=0, device="cpu")
    vbn.set_learning_method(
        "node_wise",
        nodes_cpds={
            "x": {"cpd": "softmax_nn"},
            "y": {"cpd": "softmax_nn"},
            "z": {"cpd": "mdn", "n_components": 2},
        },
    )

    data = {
        "x": torch.randn(50, 1),
        "y": torch.randn(50, 1),
        "z": torch.randn(50, 1),
    }
    vbn.fit(data)

    vbn.set_inference_method("monte_carlo_marginalization", n_samples=5)
    query = {
        "target": "z",
        "evidence": {"x": torch.randn(4, 1), "y": torch.randn(4, 1)},
    }
    pdf1, samples1 = vbn.infer_posterior(query)
    assert pdf1.shape == (4, 5)
    assert samples1.shape == (4, 5, 1)

    vbn.set_sampling_method("ancestral", n_samples=5)
    samples2 = vbn.sample(query, n_samples=5)
    assert samples2.shape == (4, 5, 1)

    update_data = {
        "x": torch.randn(10, 1),
        "y": torch.randn(10, 1),
        "z": torch.randn(10, 1),
    }
    vbn.update(update_data, update_method="online_sgd", n_steps=1)

    pdf2, samples3 = vbn.infer_posterior(query)
    assert pdf2.shape == (4, 5)
    assert samples3.shape == (4, 5, 1)
