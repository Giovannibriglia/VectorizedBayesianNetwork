import networkx as nx
import torch
from vbn import VBN


def _make_linear_vbn(seed: int = 0) -> VBN:
    g = nx.DiGraph()
    g.add_edge("x", "y")
    model = VBN(g, seed=seed, device="cpu")
    model.set_learning_method(
        "node_wise",
        nodes_cpds={
            "x": {"cpd": "linear_gaussian"},
            "y": {"cpd": "linear_gaussian"},
        },
    )

    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(512, 1, generator=gen)
    noise = 0.15 * torch.randn(512, 1, generator=gen)
    y = 1.5 * x + 0.4 + noise
    model.fit({"x": x, "y": y})
    return model


def test_gaussian_exact_inference_shapes_and_finite_values():
    model = _make_linear_vbn(seed=0)
    model.set_inference_method("gaussian_exact", n_samples=31, stddevs=3.5)

    query = {"target": "y", "evidence": {"x": torch.tensor([[0.2], [-0.8]])}}
    pdf, samples = model.infer_posterior(query)

    assert pdf.shape == (2, 31)
    assert samples.shape == (2, 31, 1)
    assert torch.isfinite(pdf).all()
    assert (pdf >= 0).all()
    assert torch.isfinite(samples).all()


def test_infer_relative_outputs_expected_shapes():
    model = _make_linear_vbn(seed=1)
    model.set_inference_method("gaussian_exact", n_samples=41, stddevs=4.0)

    out = model.infer_relative(
        query={"target": "y", "evidence": {"x": torch.tensor([[1.0], [1.5]])}},
        reference_query={"target": "y", "evidence": {"x": torch.tensor([[0.0]])}},
        n_samples=41,
    )

    assert out["target"] == "y"
    assert out["delta_mean"].shape == (2, 1)
    assert out["delta_std"].shape == (2, 1)
    assert out["relative_mean_change"].shape == (2, 1)
    assert out["relative_std_change"].shape == (2, 1)
    assert torch.isfinite(out["relative_mean_change"]).all()
    assert torch.isfinite(out["relative_std_change"]).all()
    assert (out["delta_mean"] > 0).all()
