import os

import torch
import vbn  # noqa: F401
from tqdm.auto import tqdm
from vbn.core.registry import CPD_REGISTRY


def test_cpds_registry_suite():
    device = torch.device("cpu")
    batch = 8
    in_dim = 2
    out_dim = 1
    bar = tqdm(list(CPD_REGISTRY.items()), desc="CPDs", disable=bool(os.getenv("CI")))
    for name, cpd_cls in bar:
        bar.set_description(f"Testing CPD: {name}")
        parents = torch.randn(batch, in_dim, device=device)
        x = torch.randn(batch, out_dim, device=device)
        cpd = cpd_cls(input_dim=in_dim, output_dim=out_dim, device=device)

        cpd.fit(parents, x, epochs=1, batch_size=4)

        n_samples = 5
        samples = cpd.sample(parents, n_samples)
        assert samples.shape == (batch, n_samples, out_dim)
        assert samples.device == device

        log_prob = cpd.log_prob(samples, parents)
        assert log_prob.shape == (batch, n_samples)

        out = cpd.forward(parents, n_samples)
        assert out.samples.shape == (batch, n_samples, out_dim)
        assert out.log_prob.shape == (batch, n_samples)
        assert out.pdf.shape == (batch, n_samples)

        log_prob_2d = cpd.log_prob(x, parents)
        assert log_prob_2d.shape == (batch, 1)

        if name in {"softmax_nn", "gaussian_nn", "mdn"}:
            loss = -cpd.log_prob(x, parents).mean()
            loss.backward()
            grads = [p.grad for p in cpd.parameters() if p.requires_grad]
            assert any(g is not None for g in grads)


def test_softmax_nn_detects_discrete_classes():
    device = torch.device("cpu")
    n_classes = 3
    x = torch.tensor([0.0, 1.0, 2.0, 1.0, 0.0], device=device).unsqueeze(1)
    cpd = CPD_REGISTRY["softmax_nn"](
        input_dim=0, output_dim=1, device=device, n_classes=n_classes
    )
    cpd.fit(None, x, epochs=1, batch_size=2)
    assert bool(cpd._is_discrete.item()) is True
    class_values = cpd._class_values.squeeze(0)
    expected = torch.tensor([0.0, 1.0, 2.0], device=device)
    assert torch.allclose(class_values, expected)
    samples = cpd.sample(None, 10)
    assert torch.isin(samples.flatten(), class_values).all()


def test_softmax_nn_continuous_binning_clamps():
    device = torch.device("cpu")
    n_classes = 4
    x = torch.linspace(0.0, 1.0, steps=9, device=device).unsqueeze(1)
    cpd = CPD_REGISTRY["softmax_nn"](
        input_dim=0,
        output_dim=1,
        device=device,
        n_classes=n_classes,
        binning="uniform",
    )
    cpd.fit(None, x, epochs=1, batch_size=3)
    assert bool(cpd._is_discrete.item()) is False
    x_out = torch.tensor([[-1.0], [2.0]], device=device)
    bins = cpd._x_to_bin(x_out)
    assert int(bins.min().item()) == 0
    assert int(bins.max().item()) == n_classes - 1
