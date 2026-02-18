import networkx as nx
import torch
from vbn import VBN


def test_device_consistency_cpu():
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
    data = {"x": torch.randn(20, 1), "y": torch.randn(20, 1)}
    vbn.fit(data)
    assert all(cpd.device.type == "cpu" for cpd in vbn.nodes.values())

    vbn.set_inference_method("monte_carlo_marginalization", n_samples=4)
    query = {"target": "y", "evidence": {"x": torch.tensor([[0.5]])}}
    pdf, samples = vbn.infer_posterior(query)
    assert pdf.device.type == "cpu"
    assert samples.device.type == "cpu"


def test_sample_output_shape():
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
    data = {"x": torch.randn(20, 1), "y": torch.randn(20, 1)}
    vbn.fit(data)
    vbn.set_sampling_method("ancestral", n_samples=8)
    query = {"target": "y", "evidence": {"x": torch.tensor([[1.0]])}}
    samples = vbn.sample(query, n_samples=8)
    assert samples.shape == (1, 8, 1)
