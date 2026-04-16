import networkx as nx
import torch
from vbn import VBN


def _weighted_grid_mean(pdf: torch.Tensor, samples: torch.Tensor) -> torch.Tensor:
    if pdf.dim() != 2 or samples.dim() != 3:
        raise ValueError("Unexpected tensor shapes for posterior summary.")
    weights = pdf / pdf.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return (weights.unsqueeze(-1) * samples).sum(dim=1)


def _make_linear_chain(seed: int = 0) -> VBN:
    g = nx.DiGraph()
    g.add_edges_from([("x", "y"), ("y", "z")])
    model = VBN(g, seed=seed, device="cpu")
    model.set_learning_method(
        "node_wise",
        nodes_cpds={
            "x": {"cpd": "linear_gaussian"},
            "y": {"cpd": "linear_gaussian"},
            "z": {"cpd": "linear_gaussian"},
        },
    )

    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(1024, 1, generator=gen)
    y = 1.2 * x + 0.3 + 0.2 * torch.randn(1024, 1, generator=gen)
    z = -0.7 * y + 0.1 + 0.2 * torch.randn(1024, 1, generator=gen)
    model.fit({"x": x, "y": y, "z": z})
    return model


def test_rb_marginalizes_missing_parents_for_linear_gaussian():
    model = _make_linear_chain(seed=0)
    model.set_inference_method(
        "rao_blackwellized_marginalization",
        n_samples=81,
        n_particles=256,
    )

    pdf, samples = model.infer_posterior(
        {"target": "y", "evidence": {"x": torch.tensor([[0.4]])}}
    )
    assert pdf.shape == (1, 81)
    assert samples.shape == (1, 81, 1)
    assert torch.isfinite(pdf).all()
    assert torch.isfinite(samples).all()
    assert model._inference._last_fallback is False

    rb_mean = _weighted_grid_mean(pdf, samples).squeeze()
    y_cpd = model.nodes["y"]
    expected_mean = (torch.tensor([0.4]) @ y_cpd._weight + y_cpd._bias).squeeze()
    assert torch.allclose(rb_mean, expected_mean, atol=0.25, rtol=0.15)


def test_rb_falls_back_when_descendant_evidence_is_present():
    model = _make_linear_chain(seed=1)
    model.set_inference_method(
        "rao_blackwellized_marginalization",
        n_samples=9,
        n_particles=128,
        fallback="likelihood_weighting",
    )
    pdf, samples = model.infer_posterior(
        {"target": "y", "evidence": {"z": torch.tensor([[0.2]])}}
    )
    assert pdf.shape == (1, 9)
    assert samples.shape == (1, 9, 1)
    assert model._inference._last_fallback is True
