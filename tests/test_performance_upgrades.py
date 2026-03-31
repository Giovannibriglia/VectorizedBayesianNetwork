import networkx as nx
import torch
from vbn import VBN
from vbn.core.registry import CPD_REGISTRY


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


def test_batched_vs_single_inference_consistency():
    vbn = _make_vbn()
    vbn.set_inference_method("importance_sampling", n_samples=8)
    evidence = torch.randn(2, 1)
    torch.manual_seed(0)
    pdf_batch, samples_batch = vbn.infer_posterior(
        {"target": "y", "evidence": {"x": evidence}}
    )
    torch.manual_seed(0)
    pdf_single, samples_single = vbn.infer_posterior(
        {"target": "y", "evidence": {"x": evidence[:1]}}
    )
    assert torch.allclose(pdf_batch[0], pdf_single[0])
    assert torch.allclose(samples_batch[0], samples_single[0])


def test_no_nans_in_inference_outputs():
    vbn = _make_vbn()
    vbn.set_inference_method("importance_sampling", n_samples=10)
    pdf, samples = vbn.infer_posterior(
        {"target": "y", "evidence": {"x": torch.randn(3, 1)}}
    )
    assert torch.isfinite(pdf).all()
    assert torch.isfinite(samples).all()


def test_kde_chunk_matches_full():
    device = torch.device("cpu")
    cpd = CPD_REGISTRY["kde"](
        input_dim=1,
        output_dim=1,
        device=device,
        bandwidth=0.4,
        parent_bandwidth=0.3,
        max_points=32,
    )
    parents = torch.linspace(-1.0, 1.0, steps=7, device=device).unsqueeze(1)
    x = torch.sin(parents)
    cpd.fit(parents, x)
    cpd._chunk_size = 2
    xq = torch.tensor([[-0.2], [0.4]], device=device)
    pq = torch.tensor([[-0.3], [0.5]], device=device)
    log_prob_chunk = cpd.log_prob(xq, pq)

    xq_exp = xq.unsqueeze(1)
    pq_exp = pq.unsqueeze(1)
    targets = cpd._targets
    parents_data = cpd._parents
    n = targets.shape[0]
    diff_y = xq_exp.unsqueeze(2) - targets.view(1, 1, n, 1)
    log_ky = cpd._kernel_log_prob(diff_y, cpd.bandwidth).sum(dim=-1)
    diff_p = pq_exp.unsqueeze(2) - parents_data.view(1, 1, n, 1)
    log_kp = cpd._kernel_log_prob(diff_p, cpd.parent_bandwidth).sum(dim=-1)
    log_prob_full = torch.logsumexp(log_kp + log_ky, dim=2) - torch.logsumexp(
        log_kp, dim=2
    )
    assert torch.allclose(log_prob_chunk, log_prob_full, atol=1e-6, rtol=1e-5)


def test_importance_sampling_ess_fallback():
    vbn = _make_vbn()
    vbn.set_inference_method("importance_sampling", n_samples=12)
    vbn._inference.ess_threshold = 1.1
    vbn.infer_posterior({"target": "y", "evidence": {"x": torch.randn(2, 1)}})
    assert vbn._inference._last_fallback is True


def test_gaussian_nn_root_fast_path():
    device = torch.device("cpu")
    cpd = CPD_REGISTRY["gaussian_nn"](input_dim=0, output_dim=1, device=device)
    x = torch.randn(64, 1, device=device)
    cpd.fit(None, x, epochs=1, batch_size=16)
    samples = cpd.sample(None, 5)
    assert cpd._root_dist is not None
    assert samples.shape == (1, 5, 1)
