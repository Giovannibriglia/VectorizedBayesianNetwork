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

        if name in {"softmax_nn", "mdn"}:
            loss = -cpd.log_prob(x, parents).mean()
            loss.backward()
            grads = [p.grad for p in cpd.parameters() if p.requires_grad]
            assert any(g is not None for g in grads)
