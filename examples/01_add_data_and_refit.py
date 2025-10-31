from __future__ import annotations

import torch
from vbn import VBN


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def posterior_mean(value) -> float:
    """
    Robust mean extractor from VBN.posterior()[node] across backends/types.
    Supports:
      • torch.Tensor of samples
      • dict with keys: 'mean' (and optionally 'var')
      • dict with key: 'samples'  (Tensor)
      • numeric histogram dict {numeric_value -> count/prob}
    """
    import torch as _torch

    # Case 1: tensor of samples
    if isinstance(value, _torch.Tensor):
        return float(value.float().mean().item())

    # Case 2: dict payloads
    if isinstance(value, dict):
        # 2a) explicit mean
        if "mean" in value:
            m = value["mean"]
            if isinstance(m, _torch.Tensor):
                return float(m.detach().float().mean().item())
            return float(m)

        # 2b) explicit samples
        if "samples" in value:
            s = value["samples"]
            if isinstance(s, _torch.Tensor):
                return float(s.detach().float().mean().item())
            # if it’s a list/iterable of numerics
            try:
                return float(sum(float(x) for x in s) / max(len(s), 1))
            except Exception:
                pass

        # 2c) numeric histogram {v: weight}
        try:
            total = 0.0
            exp = 0.0
            for k, v in value.items():
                # skip non-numeric keys (e.g., 'var', 'std', etc.)
                if isinstance(k, str):
                    continue
                kk = float(k if not isinstance(k, _torch.Tensor) else k.item())
                vv = float(v if not isinstance(v, _torch.Tensor) else v.item())
                total += vv
                exp += kk * vv
            if total > 0:
                return float(exp / total)
        except Exception:
            pass

        # If we get here, we don't recognize the structure
        raise TypeError(f"Unsupported posterior dict keys: {list(value.keys())}")

    # Fallback
    raise TypeError(f"Unexpected posterior value type: {type(value)}")


def posterior_expectation(bn, node: str, do: dict, n_samples: int = 10240) -> float:
    post = bn.posterior([node], do=do, n_samples=n_samples)
    return posterior_mean(post[node])


# ─────────────────────────────────────────────────────────────────────────────
# 1) MLE (Categorical): A -> B  (fit + partial_fit)
# ─────────────────────────────────────────────────────────────────────────────


def demo_mle(device: str):
    print("\n=== MLE (categorical) :: A -> B ===")
    nodes = {"A": {"type": "discrete", "card": 2}, "B": {"type": "discrete", "card": 3}}
    parents = {"A": [], "B": ["A"]}

    bn = VBN(nodes, parents, device=device, learner_map={"A": "mle", "B": "mle"})

    # batch 1
    N = 4096
    A1 = torch.randint(0, 2, (N, 1), device=device)
    B1 = (A1 + torch.randint(0, 3, (N, 1), device=device)) % 3
    bn.fit({"A": A1, "B": B1})

    # posterior P(B | do(A=1)) — print normalized histogram
    post1 = bn.posterior(
        ["B"], do={"A": torch.tensor([1], device=device)}, n_samples=4096
    )["B"]
    if isinstance(post1, torch.Tensor):
        # treat as samples → histogram
        hist = torch.bincount(post1.view(-1).long(), minlength=3).float()
        print("MLE 1) P(B|do(A=1)) ≈", (hist / hist.sum()).tolist())
    else:
        # dict histogram already
        tot = sum(float(v) for v in post1.values())
        probs = [float(post1.get(i, 0.0)) / max(tot, 1e-12) for i in range(3)]
        print("MLE 1) P(B|do(A=1)) ≈", probs)

    # batch 2 (update) — bias towards class 2
    A2 = torch.randint(0, 2, (N, 1), device=device)
    B2 = ((A2 + torch.randint(0, 3, (N, 1), device=device)) % 3).clamp_max(2)
    bn.partial_fit({"A": A2, "B": B2})

    post2 = bn.posterior(
        ["B"], do={"A": torch.tensor([1], device=device)}, n_samples=4096
    )["B"]
    if isinstance(post2, torch.Tensor):
        hist = torch.bincount(post2.view(-1).long(), minlength=3).float()
        print("MLE 2) P(B|do(A=1)) ≈", (hist / hist.sum()).tolist())
    else:
        tot = sum(float(v) for v in post2.values())
        probs = [float(post2.get(i, 0.0)) / max(tot, 1e-12) for i in range(3)]
        print("MLE 2) P(B|do(A=1)) ≈", probs)


# ─────────────────────────────────────────────────────────────────────────────
# 2) Linear-Gaussian: X -> R (fit + partial_fit)
# ─────────────────────────────────────────────────────────────────────────────


def demo_linear_gaussian(device: str):
    print("\n=== Linear-Gaussian :: X -> R ===")
    nodes = {"X": {"type": "gaussian", "dim": 1}, "R": {"type": "gaussian", "dim": 1}}
    parents = {"X": [], "R": ["X"]}

    bn = VBN(
        nodes,
        parents,
        device=device,
        learner_map={"X": "linear_gaussian", "R": "linear_gaussian"},
    )

    # batch 1: R ≈ 0.7*X + 0.2 + ε
    N = 30000
    X1 = torch.randn(N, 1, device=device)
    R1 = 0.7 * X1 + 0.2 + 0.1 * torch.randn_like(X1)
    bn.fit({"X": X1, "R": R1})

    m1 = posterior_expectation(
        bn, "R", do={"X": torch.tensor([1.0], device=device)}, n_samples=10240
    )
    print(f"LG  1) E[R|do(X=1)] ≈ {m1:.4f}")

    # batch 2: slope shifts
    X2 = torch.randn(N, 1, device=device)
    R2 = 1.1 * X2 + 0.2 + 0.1 * torch.randn_like(X2)
    bn.partial_fit({"X": X2, "R": R2})

    m2 = posterior_expectation(
        bn, "R", do={"X": torch.tensor([1.0], device=device)}, n_samples=10240
    )
    print(f"LG  2) E[R|do(X=1)] ≈ {m2:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 3) KDE: X -> R (nonlinear), fit + update (+short retune)
# ─────────────────────────────────────────────────────────────────────────────


def _f_nl1(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(2 * x) + 0.2 * x


def _f_nl2(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(2 * x) + 0.5 * x + 0.2 * torch.cos(3 * x)


def demo_kde(device: str):
    print("\n=== KDE :: X -> R (nonlinear) ===")
    nodes = {"X": {"type": "gaussian", "dim": 1}, "R": {"type": "gaussian", "dim": 1}}
    parents = {"X": [], "R": ["X"]}

    bn = VBN(
        nodes,
        parents,
        device=device,
        learner_map={"X": "linear_gaussian", "R": "kde"},
        default_steps_svgp_kde=1200,
        default_lr=1e-3,
    )

    # batch 1
    N = 60000
    X1 = torch.randn(N, 1, device=device)
    R1 = _f_nl1(X1) + 0.1 * torch.randn_like(X1)
    bn.fit({"X": X1, "R": R1}, steps=1200, lr=1e-3)

    m1 = posterior_expectation(
        bn, "R", do={"X": torch.tensor([1.0], device=device)}, n_samples=20000
    )
    print(f"KDE 1) E[R|do(X=1)] ≈ {m1:.4f}")

    # batch 2 (update + quick bandwidth retune)
    X2 = torch.randn(N, 1, device=device)
    R2 = _f_nl2(X2) + 0.1 * torch.randn_like(X2)
    bn.partial_fit({"X": X2, "R": R2})
    bn.fit({"X": X2, "R": R2}, steps=800, lr=1e-3)

    m2 = posterior_expectation(
        bn, "R", do={"X": torch.tensor([1.0], device=device)}, n_samples=20000
    )
    print(f"KDE 2) E[R|do(X=1)] ≈ {m2:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 4) SVGP: X -> R (nonlinear), fit + update (ELBO)
# ─────────────────────────────────────────────────────────────────────────────


def _g_true(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(1.5 * x) + 0.25 * (x**2)


def demo_svgp(device: str):
    print("\n=== SVGP :: X -> R (nonlinear) ===")
    nodes = {"X": {"type": "gaussian", "dim": 1}, "R": {"type": "gaussian", "dim": 1}}
    parents = {"X": [], "R": ["X"]}

    # ELBO-trained SVGP for R|X; LG for X
    bn = VBN(
        nodes,
        parents,
        device=device,
        learner_map={"X": "linear_gaussian", "R": "gp_svgp"},
        default_steps_svgp_kde=2000,
        default_lr=2e-3,
    )

    # batch 1
    N = 10240
    X1 = 1.2 * torch.randn(N, 1, device=device)
    R1 = _g_true(X1) + 0.1 * torch.randn_like(X1)
    bn.fit({"X": X1, "R": R1}, steps=2000, lr=2e-3)

    m1 = posterior_expectation(
        bn, "R", do={"X": torch.tensor([1.0], device=device)}, n_samples=2048
    )
    print(f"SVGP 1) E[R|do(X=1)] ≈ {m1:.4f}")

    # batch 2: mild drift
    X2 = 1.2 * torch.randn(N, 1, device=device)
    R2 = (_g_true(X2) + 0.3 * X2) + 0.1 * torch.randn_like(X2)
    bn.partial_fit({"X": X2, "R": R2})
    bn.fit({"X": X2, "R": R2}, steps=1500, lr=2e-3)

    m2 = posterior_expectation(
        bn, "R", do={"X": torch.tensor([1.0], device=device)}, n_samples=2048
    )
    print(f"SVGP 2) E[R|do(X=1)] ≈ {m2:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Run all demos
# ─────────────────────────────────────────────────────────────────────────────


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    demo_mle(device)
    demo_linear_gaussian(device)
    demo_kde(device)
    demo_svgp(device)


if __name__ == "__main__":
    main()
