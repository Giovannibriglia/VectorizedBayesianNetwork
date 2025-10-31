#!/usr/bin/env python3
from __future__ import annotations

import torch

from vbn import VBN

# causal modules you shared
from vbn.inference.causal_inference.backdoor import BackdoorAdjuster
from vbn.inference.causal_inference.counterfactual_lg import CounterfactualLG
from vbn.inference.causal_inference.frontdoor import FrontdoorAdjuster


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _to_dev(x, device):
    return x.to(device) if torch.is_tensor(x) else torch.tensor(x, device=device)


def _batch(xs):
    return xs if isinstance(xs, (list, tuple)) else [xs]


def batched_backdoor(
    adj: BackdoorAdjuster,
    x_var: str,
    y_var: str,
    z_vars,
    x_values,
    *,
    z_support=None,
    mc_samples_z: int = 0,
):
    outs = []
    for xv in _batch(x_values):
        mu = adj.effect(
            x_var,
            y_var,
            z_vars,
            _to_dev(xv, adj.device),
            z_support=z_support,
            mc_samples_z=mc_samples_z,
        )
        outs.append(mu)
    return torch.stack(outs)


def batched_frontdoor(
    adj: FrontdoorAdjuster,
    x_var: str,
    m_var: str,
    y_var: str,
    x_support,
    m_support,
    x_values,
):
    outs = []
    for xv in _batch(x_values):
        mu = adj.effect(
            x_var, m_var, y_var, x_support, m_support, _to_dev(xv, adj.device)
        )
        outs.append(mu)
    return torch.stack(outs)


# ─────────────────────────────────────────────────────────────────────────────
# 1) Back-door: discrete Z (exact sum) and continuous Z (MC)
#    Graph: Z → X → Y and Z → Y
# ─────────────────────────────────────────────────────────────────────────────
def demo_backdoor(device: str):
    print("\n=== Back-door adjustment ===")
    # (a) Discrete case: all categorical (MLE)
    nodes_d = {
        "Z": {"type": "discrete", "card": 2},
        "X": {"type": "discrete", "card": 2},
        "Y": {"type": "discrete", "card": 3},
    }
    parents_d = {"Z": [], "X": ["Z"], "Y": ["X", "Z"]}

    bn_d = VBN(nodes_d, parents_d, device=device)
    bn_d.set_learners({"Z": "mle", "X": "mle", "Y": "mle"})

    N = 20000
    Z = torch.distributions.Bernoulli(0.6).sample((N,)).long().to(device)
    X = Z ^ (torch.rand(N, device=device) < 0.25).long()  # P(X|Z) depends on Z
    Y = (X + Z + torch.randint(0, 3, (N,), device=device)) % 3  # Y depends on (X,Z)
    bn_d.fit({"Z": Z, "X": X, "Y": Y})

    adj_d = BackdoorAdjuster(bn_d, device=device, exact_method="ve")
    z_support = {"Z": [torch.tensor(0, device=device), torch.tensor(1, device=device)]}

    xs = [0, 1]
    mu_disc = batched_backdoor(adj_d, "X", "Y", ["Z"], xs, z_support=z_support)
    print("Discrete Z   E[Y | do(X=0/1)] ~ (index-weighted):", mu_disc.tolist())

    # (b) Continuous Z: LG + MLE Y for demo; backdoor via MC over Z
    nodes_c = {
        "Z": {"type": "gaussian", "dim": 1},
        "X": {"type": "gaussian", "dim": 1},
        "Y": {"type": "gaussian", "dim": 1},
    }
    parents_c = {"Z": [], "X": ["Z"], "Y": ["X", "Z"]}

    bn_c = VBN(
        nodes_c,
        parents_c,
        device=device,
        learner_map={
            "Z": "linear_gaussian",
            "X": "linear_gaussian",
            "Y": "linear_gaussian",
        },
    )

    N2 = 30000
    Zc = torch.randn(N2, 1, device=device)  # Z ~ N(0,1)
    Xc = 0.8 * Zc + 0.1 * torch.randn_like(Zc)  # X|Z
    Yc = 0.5 * Xc + 0.7 * Zc + 0.2 * torch.randn_like(Zc)  # Y|X,Z
    bn_c.fit({"Z": Zc, "X": Xc, "Y": Yc})

    adj_c = BackdoorAdjuster(bn_c, device=device, exact_method="gaussian")
    xs_cont = torch.tensor([-1.0, 0.0, 1.0], device=device)
    mu_cont = batched_backdoor(
        adj_c, "X", "Y", ["Z"], xs_cont, z_support=None, mc_samples_z=8192
    )
    print("Continuous Z E[Y | do(X=-1,0,1)]:", mu_cont.tolist())


# ─────────────────────────────────────────────────────────────────────────────
# 2) Front-door: discrete mediator M
#    Graph: X → M → Y, with (possible) unobserved confounding X ↔ Y
# ─────────────────────────────────────────────────────────────────────────────
def demo_frontdoor(device: str):
    print("\n=== Front-door adjustment (discrete mediator) ===")
    nodes = {
        "X": {"type": "discrete", "card": 2},
        "M": {"type": "discrete", "card": 3},
        "Y": {"type": "discrete", "card": 2},
    }
    parents = {
        "X": [],
        "M": ["X"],
        "Y": ["M", "X"],
    }  # we model Y also conditioned on X, front-door will adjust it

    bn = VBN(nodes, parents, device=device)
    bn.set_learners({"X": "mle", "M": "mle", "Y": "mle"})

    N = 25000
    X = torch.distributions.Bernoulli(0.5).sample((N,)).long().to(device)
    # M|X
    M = (X + torch.randint(0, 3, (N,), device=device)) % 3
    # Y|M,X  (keep a mild dependence on X to emulate confounding path handled by FD)
    Y = ((M % 2) ^ (X & (torch.rand(N, device=device) < 0.2)).long()).long()
    bn.fit({"X": X, "M": M, "Y": Y})

    adj = FrontdoorAdjuster(bn, device=device, exact_method="ve")

    x_support = [torch.tensor(0, device=device), torch.tensor(1, device=device)]
    m_support = [
        torch.tensor(0, device=device),
        torch.tensor(1, device=device),
        torch.tensor(2, device=device),
    ]

    xs = [0, 1]
    mu = batched_frontdoor(adj, "X", "M", "Y", x_support, m_support, xs)
    print("E[Y | do(X=0/1)] via Front-door:", mu.tolist())


# ─────────────────────────────────────────────────────────────────────────────
# 3) Counterfactual (Linear-Gaussian) via Abduction–Action–Prediction
#    Single node Y with parents (e.g., X,Z)
# ─────────────────────────────────────────────────────────────────────────────
def demo_counterfactual_lg(device: str):
    print("\n=== Counterfactual (Linear-Gaussian, A–A–P) ===")
    # Y = 1.2*X + 0.5*Z + b + ε
    nodes = {
        "X": {"type": "gaussian", "dim": 1},
        "Z": {"type": "gaussian", "dim": 1},
        "Y": {"type": "gaussian", "dim": 1},
    }
    parents = {"X": [], "Z": [], "Y": ["X", "Z"]}

    bn = VBN(
        nodes,
        parents,
        device=device,
        learner_map={
            "X": "linear_gaussian",
            "Z": "linear_gaussian",
            "Y": "linear_gaussian",
        },
    )

    N = 30000
    X = torch.randn(N, 1, device=device)
    Z = 0.6 * torch.randn(N, 1, device=device) + 0.2
    Y = 1.2 * X + 0.5 * Z + 0.3 + 0.1 * torch.randn_like(X)
    bn.fit({"X": X, "Z": Z, "Y": Y})

    cf = CounterfactualLG(bn, device=device)

    # One factual instance:
    x_f = torch.tensor(0.7, device=device)
    z_f = torch.tensor(-0.4, device=device)
    y_obs = (
        1.2 * x_f + 0.5 * z_f + 0.3 + 0.1 * torch.randn((), device=device)
    )  # observed Y

    y_cf = cf.y_cf(
        y="Y",
        factual_parents={"X": x_f.view(1, 1), "Z": z_f.view(1, 1)},
        factual_y=y_obs.view(1, 1),
        intervened_parents={
            "X": torch.tensor(1.5, device=device).view(1, 1),
            "Z": z_f.view(1, 1),
        },
    )
    print(f"Y factual={y_obs.item():.3f}  ⇒  Y^{'{do(X=1.5)}'} ≈ {y_cf.item():.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    print("device:", device)

    demo_backdoor(device)
    demo_frontdoor(device)
    demo_counterfactual_lg(device)


if __name__ == "__main__":
    main()
