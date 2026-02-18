import networkx as nx
import torch
from vbn import VBN


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
    data = {"x": torch.randn(20, 1), "y": torch.randn(20, 1)}
    vbn.fit(data)
    vbn.set_inference_method("monte_carlo_marginalization", n_samples=4)
    vbn.set_sampling_method("ancestral", n_samples=4)
    return vbn


def test_infer_posterior_outputs_detached():
    vbn = _make_vbn()
    query = {"target": "y", "evidence": {"x": torch.randn(3, 1)}}
    pdf, samples = vbn.infer_posterior(query)
    assert not pdf.requires_grad
    assert pdf.grad_fn is None
    assert not samples.requires_grad
    assert samples.grad_fn is None


def test_sample_outputs_detached():
    vbn = _make_vbn()
    query = {"target": "y", "evidence": {"x": torch.randn(3, 1)}}
    samples = vbn.sample(query, n_samples=4)
    assert not samples.requires_grad
    assert samples.grad_fn is None


def test_cpd_handle_outputs_detached():
    vbn = _make_vbn()
    handle = vbn.cpd("y")
    parents = {"x": torch.randn(2, 1)}
    samples = handle.sample(parents, 3)
    assert not samples.requires_grad
    assert samples.grad_fn is None

    log_prob = handle.log_prob(samples, parents)
    pdf = handle.pdf(samples, parents)
    assert not log_prob.requires_grad
    assert log_prob.grad_fn is None
    assert not pdf.requires_grad
    assert pdf.grad_fn is None

    out = handle.forward(parents, 3)
    assert not out.samples.requires_grad
    assert not out.log_prob.requires_grad
    assert not out.pdf.requires_grad
