#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Dict, List

import torch

# Your integrated VBN with parallel fit support
from vbn import VBN


def make_synthetic_dataset(N: int, device: str) -> Dict[str, torch.Tensor]:
    """
    Ground truth:
      S ∈ {0,...,4} (uniform)
      X | S = s  ~  Normal(mu_s, 0.4^2),   mu_s = 0.8*s - 1.0
      R | X = x  =  f(x) + ε,  with f(x) = sin(2x) + 0.25*x^2,  ε ~ Normal(0, 0.10^2)
    """
    S = torch.randint(0, 5, (N,), device=device)
    mu_s = 0.8 * S.float() - 1.0
    X = mu_s + 0.4 * torch.randn(N, device=device)
    fX = torch.sin(2 * X) + 0.25 * (X**2)
    R = fX + 0.10 * torch.randn(N, device=device)

    # shapes: S:(N,), X:(N,1), R:(N,1)
    return {"S": S, "X": X.unsqueeze(-1), "R": R.unsqueeze(-1)}


@torch.no_grad()
def true_E_R_do_S(
    s_vals: List[int], n_mc: int = 200000, device: str = "cpu"
) -> Dict[int, float]:
    """Monte-Carlo ground truth for E[R | do(S=s)] under the known generative process."""
    out = {}
    for s in s_vals:
        mu_s = 0.8 * torch.tensor(float(s), device=device) - 1.0
        X = mu_s + 0.4 * torch.randn(n_mc, device=device)
        fX = torch.sin(2 * X) + 0.25 * (X**2)
        R = fX + 0.10 * torch.randn(n_mc, device=device)
        out[s] = float(R.mean().item())
    return out


@torch.no_grad()
def bn_E_R_do_S(bn: VBN, s_vals: List[int], n_mc: int = 16384) -> Dict[int, float]:
    """Estimate E[R | do(S=s)] by sampling from the BN."""
    out = {}
    for s in s_vals:
        do = {"S": torch.tensor([s], device=bn.device)}
        samples = bn.sample(n_samples=n_mc, do=do)
        # samples["R"]: (n_mc, 1) or (n_mc,) depending on your sampler — handle both
        R = samples["R"].reshape(n_mc, -1)
        out[s] = float(R.mean().item())
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)

    # ── Define BN structure: S → X → R
    nodes = {
        "S": {"type": "discrete", "card": 5},
        "X": {"type": "gaussian", "dim": 1},
        "R": {"type": "gaussian", "dim": 1},
    }
    parents = {"S": [], "X": ["S"], "R": ["X"]}

    # ── Create datasets
    train = make_synthetic_dataset(N=50_000, device=device)
    test = make_synthetic_dataset(N=10_000, device=device)

    # ── Prepare ground-truth interventions E[R | do(S=s)]
    s_vals = list(range(5))
    gt = true_E_R_do_S(s_vals, n_mc=250_000, device=device)

    # ── Build three models that differ ONLY in R|X:
    #     (all use S ~ mle, X|S ~ linear_gaussian)
    configs = {
        "R|X: linear_gaussian": {"R": "linear_gaussian"},
        "R|X: kde": {"R": "kde"},
        "R|X: gp_svgp": {"R": "gp_svgp"},
    }

    results = {}
    for label, learner_map_R in configs.items():
        # Compose full learner map: S->mle, X->linear_gaussian, R->(variant)
        learner_map = {"S": "mle", "X": "linear_gaussian", **learner_map_R}

        bn = VBN(
            nodes=nodes,
            parents=parents,
            device=device,
            learner_map=learner_map,
            # defaults: linear/mle do 0 extra steps; kde/gp_svgp do minibatch steps automatically
            default_steps_svgp_kde=600,
            default_steps_others=0,
            default_batch_size=4096,
            default_lr=1e-3,
        )

        # Parallel learning inside VBN.fit()
        bn.fit(train)  # offline init + (optional) minibatch joint optimization

        # Interventional means
        est = bn_E_R_do_S(bn, s_vals, n_mc=32_768)

        # Simple error metric vs ground truth curve
        rmse = math.sqrt(sum((est[s] - gt[s]) ** 2 for s in s_vals) / len(s_vals))

        # Also report test NLL proxy: Monte-Carlo estimate of -log p(R|X) on test (for comparison).
        # We estimate via per-sample log_prob at each node that involves R (here only R|X).
        # (This is a rough cross-model comparison; KDE returns conditional log-densities.)
        with torch.no_grad():
            par_R = {"X": test["X"]}
            y_R = test["R"]
            nll_R = -bn.cpd["R"].log_prob(y_R, par_R).mean().item()

        results[label] = {"curve": est, "rmse_vs_true_do": rmse, "test_nll_R": nll_R}

    # ── Pretty print comparison
    print("\n=== Ground-truth E[R | do(S=s)] ===")
    for s in s_vals:
        print(f"s={s}: {gt[s]: .4f}")

    print("\n=== Model comparison ===")
    for label, res in results.items():
        print(f"\n[{label}]")
        for s in s_vals:
            print(f"  s={s}:  E_hat={res['curve'][s]: .4f}   (true {gt[s]: .4f})")
        print(f"  RMSE vs true do-curve: {res['rmse_vs_true_do']:.4f}")
        print(f"  Test NLL for R|X (lower is better): {res['test_nll_R']:.4f}")

    # ── Optional: demonstrate streaming update (partial_fit)
    # simulate a tiny new batch; update and re-evaluate one intervention
    new_batch = make_synthetic_dataset(N=4_096, device=device)
    # Pick a model to update, e.g., KDE:
    bn_kde = VBN(
        nodes,
        parents,
        device=device,
        learner_map={"S": "mle", "X": "linear_gaussian", "R": "kde"},
    )
    bn_kde.fit(train)
    before = bn_E_R_do_S(bn_kde, [2], n_mc=16_384)[2]
    bn_kde.partial_fit(new_batch)  # online update
    after = bn_E_R_do_S(bn_kde, [2], n_mc=16_384)[2]
    print(
        f"\n[KDE] streaming update demo at do(S=2): before={before:.4f}, after={after:.4f}"
    )


if __name__ == "__main__":
    main()
