from __future__ import annotations

import pytest
import torch
from vbn.core.registry import CPD_REGISTRY


def _make_int_data(n=800, seed=0):
    gen = torch.Generator().manual_seed(seed)
    x1 = torch.randn(n, generator=gen)
    x2 = torch.randn(n, generator=gen)
    k_true = 5
    slopes = torch.linspace(0.0, 4.0, k_true)
    x1_boundaries = torch.tensor([-2.0, -0.5, 0.8, 2.0])
    intercepts = torch.zeros(k_true)
    for i in range(k_true - 1):
        intercepts[i + 1] = intercepts[i] - x1_boundaries[i]
    x2_weights = torch.tensor([0.3, -0.2, 0.0, 0.2, -0.3])
    logits = (
        x1[:, None] * slopes[None, :]
        + x2[:, None] * x2_weights[None, :]
        + intercepts[None, :]
    )
    y = torch.distributions.Categorical(logits=logits).sample()
    parents = torch.stack([x1, x2], dim=1)
    return parents, y.unsqueeze(1).float()


def _make_cont_data(n=600, seed=0):
    gen = torch.Generator().manual_seed(seed)
    x1 = torch.randn(n, generator=gen)
    x2 = torch.randn(n, generator=gen)
    y = 0.8 * x1 - 0.3 * x2 + 0.2 * torch.randn(n, generator=gen)
    parents = torch.stack([x1, x2], dim=1)
    return parents, y.unsqueeze(1)


def test_softmax_mismatch_continuous_bins():
    device = torch.device("cpu")
    parents, y = _make_int_data(n=800, seed=0)
    cpd = CPD_REGISTRY["softmax_nn"](
        input_dim=2,
        output_dim=1,
        device=device,
        n_classes=8,
        within_bin="uniform",
        seed=0,
    )
    cpd.fit(parents, y, epochs=10, batch_size=128)

    assert bool(cpd._is_discrete.item()) is False

    parents_grid = torch.tensor([[-2.0, 0.0], [2.0, 0.0]], device=device)
    samples = cpd.sample(parents_grid, n_samples=200)
    frac = (samples - samples.round()).abs()
    assert (frac > 1e-3).any()

    log_prob = cpd.log_prob(samples, parents_grid)
    assert torch.isfinite(log_prob).all()


def test_softmax_continuous_within_bin_density():
    device = torch.device("cpu")
    parents, y = _make_cont_data(n=600, seed=0)
    cpd = CPD_REGISTRY["softmax_nn"](
        input_dim=2,
        output_dim=1,
        device=device,
        n_classes=6,
        within_bin="triangular",
        seed=0,
    )
    cpd.fit(parents, y, epochs=10, batch_size=128)

    parents_grid = torch.tensor([[-1.0, 0.0], [1.0, 0.0]], device=device)
    samples = cpd.sample(parents_grid, n_samples=200)
    assert samples.std().item() > 0.0

    log_prob = cpd.log_prob(samples, parents_grid)
    assert torch.isfinite(log_prob).all()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_softmax_device_consistency(device: str):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    dev = torch.device(device)
    parents, y = _make_cont_data(n=200, seed=1)
    parents = parents.to(dev)
    y = y.to(dev)
    cpd = CPD_REGISTRY["softmax_nn"](
        input_dim=2,
        output_dim=1,
        device=dev,
        n_classes=4,
        within_bin="gaussian",
        seed=1,
    )
    cpd.fit(parents, y, epochs=5, batch_size=64)
    samples = cpd.sample(parents[:4], n_samples=10)
    assert samples.device == dev
