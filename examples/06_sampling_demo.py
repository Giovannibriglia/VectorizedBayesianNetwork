from __future__ import annotations

from typing import Tuple

import torch
from vbn import VBN

torch.set_printoptions(precision=4, sci_mode=False)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def hist_from_samples(idx: torch.Tensor, K: int) -> torch.Tensor:
    """Weighted-uniform histogram from class indices [N] → probs [K]."""
    idx = idx.view(-1).long()
    hist = torch.bincount(idx, minlength=K).float()
    return (hist / hist.sum().clamp_min(1e-9)).detach()


def print_probs(label: str, p: torch.Tensor):
    print(f"{label} [{', '.join(f'{x:.4f}' for x in p.tolist())}]")


def analytic_truths() -> Tuple[float, float]:
    # Matches construction in mixed BN below:  X = 2A + ε_x,   R = -1 + 0.5X + 1.0A + ε_r
    # ⇒ E[R | do(A=a)] = -1 + 0.5*E[X|do(A=a)] + a = -1 + 0.5*(2a) + a = -1 + 2a
    return -1.0, +1.0


# ─────────────────────────────────────────────────────────────────────────────
# Discrete demo: A -> B -> C
# ─────────────────────────────────────────────────────────────────────────────
def build_discrete_bn(device: str) -> VBN:
    nodes = {
        "A": {"type": "discrete", "card": 2},
        "B": {"type": "discrete", "card": 3},
        "C": {"type": "discrete", "card": 2},
    }
    parents = {"A": [], "B": ["A"], "C": ["B"]}

    bn = VBN(nodes, parents, device=device)
    # Learners: simple categorical MLE everywhere
    bn.set_learners({"A": "mle", "B": "mle", "C": "mle"})

    # Synthesize a dataset consistent with the structure
    N = 40000
    A = torch.randint(0, 2, (N, 1), device=device)
    # B depends on A:  B = (A + noise) mod 3 (skew slightly)
    B = ((A + torch.randint(0, 3, (N, 1), device=device)) % 3).long()
    # C depends on B:  p(C=1|B=2) high, else balanced
    C = torch.where(
        (B == 2),
        (torch.rand(N, 1, device=device) < 0.8).long(),
        (torch.rand(N, 1, device=device) < 0.5).long(),
    )
    bn.fit({"A": A, "B": B, "C": C})
    return bn


def demo_discrete(bn: VBN):
    print("\n=== Discrete sampling: A -> B -> C ===")

    # Unconditional P(C) via ancestral
    samp = bn.sample(n_samples=20000, method="ancestral")
    pC = hist_from_samples(samp["C"], K=2)
    print_probs("Ancestral   P(C):", pC)

    # Interventional P(C | do(B=2)) via ancestral
    samp_do = bn.sample(
        n_samples=20000,
        method="ancestral",
        do={"B": torch.tensor([2], device=bn.device)},
    )
    pC_do = hist_from_samples(samp_do["C"], K=2)
    print_probs("Ancestral   P(C | do(B=2)):", pC_do)

    # Conditional P(C | A=0/1) via SMC
    for a in [0, 1]:
        post = bn.sample_conditional(
            evidence={"A": torch.tensor([0], device=bn.device)}, n_samples=20000
        )

        p = hist_from_samples(post["C"], K=2)
        print_probs(f"SMC         P(C | A={a}):", p)


# ─────────────────────────────────────────────────────────────────────────────
# Mixed linear-Gaussian demo: A -> X -> R, A -> R
# ─────────────────────────────────────────────────────────────────────────────
def build_mixed_bn(device: str) -> VBN:
    # A: discrete; X,R: gaussian
    nodes = {
        "A": {"type": "discrete", "card": 2},
        "X": {"type": "gaussian", "dim": 1},
        "R": {"type": "gaussian", "dim": 1},
    }
    parents = {"A": [], "X": ["A"], "R": ["A", "X"]}

    bn = VBN(nodes, parents, device=device)
    bn.set_learners(
        {
            "A": "mle",  # categorical root
            "X": "linear_gaussian",
            "R": "linear_gaussian",
        }
    )

    # Generate data:
    # A ~ Bernoulli(0.5)
    # X = 2*A + ε_x, ε_x ~ N(0, 0.25)
    # R = -1 + 0.5*X + 1.0*A + ε_r, ε_r ~ N(0, 0.36)
    N = 60_000
    with torch.no_grad():
        A = torch.randint(0, 2, (N, 1), device=device)
        X = 2.0 * A.float() + 0.5 * torch.randn(N, 1, device=device)
        R = -1.0 + 0.5 * X + 1.0 * A.float() + 0.6 * torch.randn(N, 1, device=device)

    bn.fit({"A": A, "X": X, "R": R})
    return bn


def demo_mixed_sampling(bn: VBN):
    mu0, mu1 = analytic_truths()
    print("\n=== Mixed sampling: A -> X -> R, A -> R (linear-Gaussian) ===")
    print(f"Analytic truths:  E[R|do(A=0)] = {mu0:+.1f},   E[R|do(A=1)] = {mu1:+.1f}\n")

    # (1) RB sampler: MC vs QMC for E[R | do(A=1)]
    print("— RB sampler: MC vs QMC (E[R | do(A=1)]) —")
    for n in (512, 2048, 8192):
        # MC
        s_mc = bn.sample(
            n_samples=n,
            method="rb",
            qmc=False,
            do={"A": torch.tensor([1], device=bn.device)},
        )
        mu_mc = s_mc["R"].mean().item()
        std_mc = s_mc["R"].std(unbiased=False).item()

        # QMC
        s_qmc = bn.sample(
            n_samples=n,
            method="rb",
            qmc=True,
            qmc_seed=777,
            do={"A": torch.tensor([1], device=bn.device)},
        )
        mu_qmc = s_qmc["R"].mean().item()
        std_qmc = s_qmc["R"].std(unbiased=False).item()

        print(
            f"n={n:>5}  MC:  {mu_mc:+.4f} ± {std_mc:.4f}   |   QMC: {mu_qmc:+.4f} ± {std_qmc:.4f}"
        )

    # (2) RB sampler returning Gaussian params (single run, under do(A=1))
    print("\n— RB sampler with returned Gaussian params (single run) —")
    rb_params = bn.sample(
        n_samples=4096,
        method="rb",
        return_gaussian_params=True,
        qmc=True,
        qmc_seed=999,
        do={"A": torch.tensor([1], device=bn.device)},
    )
    Rm, Rv = rb_params["R"]["mean"], rb_params["R"]["var"]  # [n,1] each
    print("R mean   (first 5):", Rm[:5, 0].tolist())
    print("R var    (first 5):", Rv[:5, 0].tolist())
    print(f"E[R|do(A=1)] via params ≈ {Rm.mean().item():+.4f}")

    # (3) Conditional SMC: MC vs QMC for E[R | A=0]
    print("\n— SMC conditional: MC vs QMC (E[R | A=0]) —")

    def cond_mean(n, qmc, seed):
        samp = bn.sample_conditional(
            evidence={"A": torch.tensor([0], device=bn.device)},
            n_samples=n,
            qmc=qmc,
            qmc_seed=seed,
        )
        return float(samp["R"].mean().item())

    repeats = 20
    for n in (512, 2048, 8192):
        mc = torch.tensor([cond_mean(n, False, 0) for _ in range(repeats)])
        qmc = torch.tensor([cond_mean(n, True, 9876 + r) for r in range(repeats)])
        print(
            f"n={n:5d}  MC mean={mc.mean():+7.4f}, std={mc.std(False):.4f}  |  "
            f"QMC mean={qmc.mean():+7.4f}, std={qmc.std(False):.4f}"
        )

    # When comparing MC vs QMC, vary the QMC seed per repeat
    def mean_under_do(n, qmc, seed):
        samp = bn.sample(n_samples=n, method="rb", qmc=qmc, qmc_seed=seed)
        return float(samp["R"].mean().item())

    for n in (512, 2048, 8192):
        mc_means = torch.tensor([mean_under_do(n, False, 0) for _ in range(repeats)])
        qmc_means = torch.tensor(
            [mean_under_do(n, True, 1234 + r) for r in range(repeats)]
        )
        print(
            f"n={n:5d}  MC: {mc_means.mean():+7.4f} ± {mc_means.std(unbiased=False):.4f} "
            f"| QMC: {qmc_means.mean():+7.4f} ± {qmc_means.std(unbiased=False):.4f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # Discrete part
    bn_disc = build_discrete_bn(device)
    demo_discrete(bn_disc)

    # Mixed (linear-Gaussian + discrete) part with QMC comparisons
    bn_mix = build_mixed_bn(device)
    demo_mixed_sampling(bn_mix)


if __name__ == "__main__":
    main()
