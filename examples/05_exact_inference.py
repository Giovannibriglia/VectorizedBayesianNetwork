#!/usr/bin/env python3
# examples/05_exact_inference.py
from __future__ import annotations

import torch

# Adjust these imports if your package layout differs
from vbn.core import VBN


def discrete_demo(device: str):
    """
    Discrete BN: A -> B -> C
    Learners: MLE categorical for each node.
    Inference: Variable Elimination ("ve").
    """
    print("\n=== Discrete exact inference (VE) ===")

    nodes = {
        "A": {"type": "discrete", "card": 2},
        "B": {"type": "discrete", "card": 3},
        "C": {"type": "discrete", "card": 2},
    }
    parents = {
        "A": [],
        "B": ["A"],
        "C": ["B"],
    }

    # ----- synthetic training data -----
    N = 6000
    A = (
        torch.distributions.Bernoulli(probs=torch.tensor(0.55, device=device))
        .sample((N,))
        .long()
    )
    # P(B|A): let B = (A + noise) % 3 with some bias
    noise = torch.randint(0, 3, (N,), device=device)
    B = (A + noise) % 3
    # P(C|B): parity of B + small flip noise
    flip = (torch.rand(N, device=device) < 0.2).long()
    C = ((B % 2) ^ flip).long()

    data = {"A": A, "B": B, "C": C}

    # ----- build & fit BN -----
    bn = VBN(nodes, parents, device=device, seed=0)
    bn.set_learners({"A": "mle", "B": "mle", "C": "mle"})
    bn.fit(data)

    # ----- queries (evidence & do) -----
    # 1) P(C | A = 1)
    out = bn.posterior(
        query=["C"], evidence={"A": torch.tensor(1, device=device)}, method="ve"
    )
    pC = out["C"]
    print("P(C | A=1):", (pC / pC.sum()).tolist())

    # 2) P(B, C | A = 0) joint over (B,C)
    out = bn.posterior(
        query=["B", "C"], evidence={"A": torch.tensor(0, device=device)}, method="ve"
    )
    joint_BC = out[
        "joint"
    ]  # flattened over scope order (parents then child within current factorization)
    # reshape to [card(B), card(C)] for readability
    joint_BC = joint_BC.view(nodes["B"]["card"], nodes["C"]["card"])
    joint_BC = joint_BC / joint_BC.sum()
    print("P(B,C | A=0):\n", joint_BC)

    # 3) P(C | do(B = 2))
    out = bn.posterior(
        query=["C"], do={"B": torch.tensor(2, device=device)}, method="ve"
    )
    pC_do = out["C"]
    print("P(C | do(B=2)):", (pC_do / pC_do.sum()).tolist())


def gaussian_demo(device: str):
    """
    Linear-Gaussian BN: X -> R  (scalar nodes)
    Learners: linear_gaussian for each node.
    Inference: Gaussian exact ("gaussian").
    """
    print("\n=== Gaussian exact inference (linear-Gaussian) ===")

    nodes = {
        "X": {"type": "gaussian", "dim": 1},
        "R": {"type": "gaussian", "dim": 1},
    }
    parents = {"X": [], "R": ["X"]}

    # ----- synthetic training data -----
    # R = w*X + b + eps, eps ~ N(0, sigma2)
    N = 6000
    w_true, b_true, sigma2_true = 0.9, 0.1, 0.05
    X = torch.randn(N, device=device)
    R = w_true * X + b_true + (sigma2_true**0.5) * torch.randn(N, device=device)

    data = {"X": X, "R": R}

    # ----- build & fit BN -----
    bn = VBN(nodes, parents, device=device, seed=0)
    bn.set_learners({"X": "linear_gaussian", "R": "linear_gaussian"})
    bn.fit(data)

    # ----- queries (evidence & do) -----
    # 1) E[R | X = 1.0]
    out = bn.posterior(
        query=["R"], evidence={"X": torch.tensor(1.0, device=device)}, method="gaussian"
    )
    m, v = float(out["R"]["mean"]), float(out["R"]["var"])
    print(f"E[R | X=1.0] = {m:.4f},  Var[R | X=1.0] = {v:.4f}")

    # 2) E[R | do(X = 1.0)]
    out_do = bn.posterior(
        query=["R"], do={"X": torch.tensor(1.0, device=device)}, method="gaussian"
    )
    m_do, v_do = float(out_do["R"]["mean"]), float(out_do["R"]["var"])
    print(f"E[R | do(X=1.0)] = {m_do:.4f},  Var[R | do(X=1.0)] = {v_do:.4f}")

    # 3) Marginal mean/var of R
    out_m = bn.posterior(query=["R"], method="gaussian")
    print(
        f"E[R] = {float(out_m['R']['mean']):.4f},  Var[R] = {float(out_m['R']['var']):.4f}"
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    discrete_demo(device)
    gaussian_demo(device)


if __name__ == "__main__":
    main()
