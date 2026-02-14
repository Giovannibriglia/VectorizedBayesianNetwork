import networkx as nx
import torch
from vbn import VBN


def test_save_load_roundtrip(tmp_path):
    g = nx.DiGraph()
    g.add_edge("x", "y")
    vbn = VBN(g, seed=0, device="cpu")
    vbn.set_learning_method(
        "node_wise",
        nodes_cpds={
            "x": {"cpd": "kde"},
            "y": {"cpd": "softmax_nn"},
        },
    )
    data = {"x": torch.randn(30, 1), "y": torch.randn(30, 1)}
    vbn.fit(data)

    vbn.set_inference_method("monte_carlo_marginalization", n_samples=4)
    vbn.set_sampling_method("ancestral", n_samples=4)

    path = tmp_path / "model.pt"
    vbn.save(str(path))
    loaded = VBN.load(str(path), map_location="cpu")

    query = {"target": "y", "evidence": {"x": torch.randn(2, 1)}}
    pdf, samples = loaded.infer_posterior(query)
    assert pdf.shape == (2, 4)
    assert samples.shape == (2, 4, 1)

    samples2 = loaded.sample(query, n_samples=4)
    assert samples2.shape == (2, 4, 1)

    x_samples = loaded.nodes["x"].sample(None, 3)
    assert x_samples.shape == (1, 3, 1)
