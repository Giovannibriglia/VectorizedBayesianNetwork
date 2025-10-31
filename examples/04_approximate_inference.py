from __future__ import annotations

from typing import Dict, List, Optional

import torch

from vbn import VBN

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _normalize_probs(p: torch.Tensor) -> torch.Tensor:
    return p / p.sum().clamp_min(1e-12)


def batched_posterior_discrete(
    bn: VBN,
    query: str,
    evidences: List[Optional[Dict[str, torch.Tensor]]],
    interventions: List[Optional[Dict[str, torch.Tensor]]],
    method: str,
    n_samples: int = 4096,
) -> torch.Tensor:
    """
    Returns a [B, K] tensor with posterior probs over K categories for `query`.
    Each batch item can have its own evidence and do(·).
    """
    assert len(evidences) == len(interventions)
    out = []
    for ev, do in zip(evidences, interventions):
        post = bn.posterior(
            [query], evidence=ev, do=do, method=method, n_samples=n_samples
        )
        probs = post[query]
        probs = _normalize_probs(probs).detach()
        out.append(probs)
    return torch.stack(out, dim=0)  # [B, K]


def batched_posterior_cont_mean(
    bn: VBN,
    query: str,
    interventions: List[Optional[Dict[str, torch.Tensor]]],
    method: str,
    n_samples: int = 8192,
) -> torch.Tensor:
    """
    Returns a [B] tensor with posterior means for a scalar Gaussian-like variable `query`.
    For LW/SMC backends that return dict with {'mean','var'}, we take 'mean';
    if they return samples, we average them.
    """
    means: List[torch.Tensor] = []
    for do in interventions:
        post = bn.posterior([query], do=do, method=method, n_samples=n_samples)[query]
        if isinstance(post, dict) and "mean" in post:
            m = post["mean"]
            if torch.is_tensor(m):
                m = m.view(-1).detach()
                means.append(m[0] if m.numel() == 1 else m.mean())
            else:
                means.append(torch.tensor(float(m), device=bn.device))
        elif torch.is_tensor(post):
            means.append(post.detach().float().mean())
        else:
            # Fallback to direct sampling under do(·)
            s = bn.sample(n_samples=n_samples, do=do)[query].float()
            means.append(s.mean())
    return torch.stack(means, dim=0)  # [B]


# ─────────────────────────────────────────────────────────────────────────────
# 1) Discrete BN: Gibbs & LBP batched queries
# ─────────────────────────────────────────────────────────────────────────────


def demo_discrete_gibbs_lbp(device: str):
    print("\n=== Approximate (Discrete): ParallelGibbs & LoopyBP ===")
    nodes = {
        "A": {"type": "discrete", "card": 2},
        "B": {"type": "discrete", "card": 3},
        "C": {"type": "discrete", "card": 2},
    }
    parents = {"A": [], "B": ["A"], "C": ["B"]}

    # synthetic training data
    N = 8000
    A = torch.distributions.Bernoulli(0.55).sample((N,)).long().to(device)
    B = ((A + torch.randint(0, 3, (N,), device=device)) % 3).long()
    C = ((B % 2) ^ (torch.rand(N, device=device) < 0.2).long()).long()
    data = {"A": A, "B": B, "C": C}

    # build & fit BN
    bn = VBN(nodes, parents, device=device)
    bn.set_learners({"A": "mle", "B": "mle", "C": "mle"})
    bn.fit(data)

    # Batch of evidence/do scenarios
    evidences = [
        {"A": torch.tensor(0, device=device)},
        {"A": torch.tensor(1, device=device)},
        {"B": torch.tensor(0, device=device)},
        {"B": torch.tensor(2, device=device)},
    ]
    interventions = [
        None,
        None,
        {"A": torch.tensor(1, device=device)},
        {"A": torch.tensor(0, device=device)},
    ]

    # Gibbs
    probs_gibbs = batched_posterior_discrete(
        bn,
        query="C",
        evidences=evidences,
        interventions=interventions,
        method="gibbs",
        n_samples=4096,
    )  # [B, 2]
    print("Gibbs  P(C | batch):\n", probs_gibbs)

    # Loopy BP
    probs_lbp = batched_posterior_discrete(
        bn,
        query="C",
        evidences=evidences,
        interventions=interventions,
        method="lbp",
        n_samples=0,  # n_samples unused in LBP
    )  # [B, 2]
    print("LBP    P(C | batch):\n", probs_lbp)


# ─────────────────────────────────────────────────────────────────────────────
# 2) Mixed BN (continuous): LW & SMC batched queries
# ─────────────────────────────────────────────────────────────────────────────


def _f_nl(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(2 * x) + 0.3 * x


def demo_mixed_lw_smc(device: str):
    print("\n=== Approximate (Mixed): LW & SMC (KDE on R|X) ===")
    nodes = {"X": {"type": "gaussian", "dim": 1}, "R": {"type": "gaussian", "dim": 1}}
    parents = {"X": [], "R": ["X"]}

    # Build BN: X ~ N(0,1) via LG root; R|X via KDE
    bn = VBN(
        nodes,
        parents,
        device=device,
        learner_map={"X": "linear_gaussian", "R": "kde"},
        default_steps_svgp_kde=1000,
        default_lr=1e-3,
    )

    # Train batch 1
    N = 60000
    X1 = torch.randn(N, 1, device=device)
    R1 = _f_nl(X1) + 0.15 * torch.randn_like(X1)
    bn.fit({"X": X1, "R": R1}, steps=1000, lr=1e-3)

    # Batched do( X = x* ) scenarios
    X_tests = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], device=device)
    dos = [{"X": xt.view(1)} for xt in X_tests]  # per-batch dicts

    # LW (weighted moment)
    means_lw = batched_posterior_cont_mean(
        bn, query="R", interventions=dos, method="lw", n_samples=20000
    )  # [B]
    print("LW    E[R | do(X)] batch:\n", means_lw)

    # SMC (resampling)
    means_smc = batched_posterior_cont_mean(
        bn, query="R", interventions=dos, method="smc", n_samples=20000
    )  # [B]
    print("SMC   E[R | do(X)] batch:\n", means_smc)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    demo_discrete_gibbs_lbp(device)
    demo_mixed_lw_smc(device)


if __name__ == "__main__":
    main()
