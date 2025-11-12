#!/usr/bin/env python3
from __future__ import annotations

import random
import time
from typing import Dict, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from vbn import VBN

# ──────────────────────────────────────────────────────────────────────────────
# Repro utils
# ──────────────────────────────────────────────────────────────────────────────


def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic data generators
# ──────────────────────────────────────────────────────────────────────────────


def gen_discrete_bn(n: int, pA=0.5, pB=0.6) -> Tuple[nx.DiGraph, pd.DataFrame, Dict]:
    """A → C ← B with fixed noisy table for P(C=1|A,B)."""
    dag = nx.DiGraph()
    dag.add_nodes_from(["A", "B", "C"])
    dag.add_edges_from([("A", "C"), ("B", "C")])

    table = torch.tensor([[0.05, 0.55], [0.70, 0.95]], dtype=torch.float32)  # [A,B]

    A = torch.bernoulli(torch.full((n,), float(pA)))
    B = torch.bernoulli(torch.full((n,), float(pB)))
    idxA = A.long()
    idxB = B.long()
    pC1 = table[idxA, idxB]
    C = torch.bernoulli(pC1)

    df = pd.DataFrame(
        {
            "A": A.numpy().astype(np.int64),
            "B": B.numpy().astype(np.int64),
            "C": C.numpy().astype(np.int64),
        }
    )
    params = {"pA": float(pA), "pB": float(pB), "table": table}  # keep for GT
    return dag, df, params


def gen_linear_gaussian(n: int) -> Tuple[nx.DiGraph, pd.DataFrame, Dict[str, float]]:
    # A ~ N(0,1), B ~ N(0,1), C = 1.2*A + 0.8*B + eps, eps ~ N(0, 0.5^2)
    dag = nx.DiGraph()
    dag.add_nodes_from(["A", "B", "C"])
    dag.add_edges_from([("A", "C"), ("B", "C")])

    A = torch.randn(n)
    B = torch.randn(n)
    sigma_eps = 0.5
    C = 1.2 * A + 0.8 * B + sigma_eps * torch.randn(n)

    df = pd.DataFrame(
        {
            "A": A.numpy().astype(np.float32),
            "B": B.numpy().astype(np.float32),
            "C": C.numpy().astype(np.float32),
        }
    )
    return dag, df, {"wA": 1.2, "wB": 0.8, "sigma_eps": sigma_eps}


# ──────────────────────────────────────────────────────────────────────────────
# Empirical references for sanity
# ──────────────────────────────────────────────────────────────────────────────


def ref_discrete_P_A1_given_C1(df: pd.DataFrame) -> float:
    sub = df[df["C"] > 0.5]
    return float((sub["A"] > 0.5).mean())


def ref_gaussian_EA_given_C_close(
    df: pd.DataFrame, c0: float, bandwidth: float = 0.15
) -> float:
    mask = (df["C"] > c0 - bandwidth) & (df["C"] < c0 + bandwidth)
    sub = df[mask]
    if len(sub) < 5:
        # fallback to global linear regression if band too small
        x = torch.tensor(df["C"].values, dtype=torch.float32)
        y = torch.tensor(df["A"].values, dtype=torch.float32)
        X = torch.stack([torch.ones_like(x), x], dim=1)
        beta = torch.linalg.lstsq(X, y).solution
        EA = (beta[0] + beta[1] * c0).item()
        return EA
    return float(sub["A"].mean())


def gt_discrete_P_A1_given_C1(pA: float, pB: float, table: torch.Tensor) -> float:
    """
    Exact P(A=1 | C=1) under the generative model:
      P(A=1,C=1) = sum_b P(C=1|A=1,b) P(A=1) P(b)
      P(C=1)     = sum_{a,b} P(C=1|a,b) P(a) P(b)
    """
    pA1, pA0 = pA, (1 - pA)
    pB1, pB0 = pB, (1 - pB)
    # table[a,b] = P(C=1|a,b) for a,b in {0,1}
    pC1 = (
        table[0, 0] * pA0 * pB0
        + table[0, 1] * pA0 * pB1
        + table[1, 0] * pA1 * pB0
        + table[1, 1] * pA1 * pB1
    )
    pA1C1 = (table[1, 0] * pA1 * pB0 + table[1, 1] * pA1 * pB1) / (pC1 + 1e-12)
    return float(pA1C1)


def gt_gaussian_EA_given_C(c0: float, wA: float, wB: float, sigma_eps: float) -> float:
    """
    Linear-Gaussian ground truth:
      E[A | C=c] = Cov(A,C) / Var(C) * c
      Cov(A,C) = wA * Var(A) = wA,  Var(C) = wA^2 + wB^2 + sigma_eps^2
    """
    varC = wA**2 + wB**2 + sigma_eps**2
    return float((wA / varC) * c0)


# ──────────────────────────────────────────────────────────────────────────────
# Inference runner
# ──────────────────────────────────────────────────────────────────────────────


def _weighted_estimate_BN(
    a_BN: torch.Tensor, pdf: dict, fn, eps: float = 1e-12
) -> float:
    # a_BN: [B,N], pdf["weights"]: [B,N] optional
    v = fn(a_BN)  # [B,N]
    if "weights" in pdf:
        w = pdf["weights"].to(v.device)
        est_B = (v * w).sum(dim=-1) / w.sum(dim=-1).clamp_min(eps)  # [B]
    else:
        est_B = v.mean(dim=-1)  # [B]
    return est_B.mean().item()


def run_inference(vbn: VBN, mode: str, inferers: list[str], batch_size: int = 512):
    results = []
    q = "A"
    do = None

    if mode == "discrete":
        # fixed evidence C=1 for GT comparability
        evidence = {"C": torch.ones(batch_size, 1, device=vbn.device, dtype=torch.long)}
        f = lambda a: (a.float() > 0.5).float()
        pbar = tqdm(inferers, desc="discrete inference...")
        for name in pbar:
            pbar.set_postfix(method=name)
            vbn.set_inference_method(name, num_samples=4096, keep_top_k=1024)
            t0 = time.time()
            pdf, samples = vbn.infer_posterior(q, evidence=evidence, do=do)
            dt = float(time.time() - t0)
            a_BN = samples[q].float()  # [B,N]
            est = _weighted_estimate_BN(a_BN, pdf, f)
            results.append((name, est, dt))
    else:
        c0 = 0.5
        evidence = {"C": torch.full((batch_size, 1), float(c0), device=vbn.device)}
        f = lambda a: a.float()
        pbar = tqdm(inferers, desc="continuous inference...")
        for name in pbar:
            pbar.set_postfix(method=name)
            vbn.set_inference_method(name, num_samples=4096, keep_top_k=1024)
            t0 = time.time()
            pdf, samples = vbn.infer_posterior(q, evidence=evidence, do=do)
            dt = float(time.time() - t0)
            a_BN = samples[q].float()  # [B,N]
            est = _weighted_estimate_BN(a_BN, pdf, f)
            results.append((name, est, dt))
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main (no argparse): runs both modes
# ──────────────────────────────────────────────────────────────────────────────


def main():
    set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inferers_d = [
        "exact.ve",
        "montecarlo.lw",
        "montecarlo.gibbs",
        "variational.mf_full",
        "lfi.abc",
        "smc.rb",
        "exact.clg",
    ]

    inferers_g = [
        "exact.gaussian",
        "montecarlo.lw",
        "variational.mf_full",
        "lfi.abc",
        "smc.rb",
    ]

    # ===== DISCRETE =====
    n_disc = 40960
    dag_d, df_d, params_d = gen_discrete_bn(n_disc)
    gt_d = gt_discrete_P_A1_given_C1(params_d["pA"], params_d["pB"], params_d["table"])
    ref_d = ref_discrete_P_A1_given_C1(df_d)
    print(f"[DISCRETE] Ground truth:   P(A=1 | C=1) = {gt_d:.6f}")
    print(f"[DISCRETE] Empirical ref.: P(A=1 | C=1) ≈ {ref_d:.6f}")

    vbn_d = VBN(dag=dag_d, device=device, seed=0)
    vbn_d.set_learning_method(
        "mle_softmax",
        num_classes=2,
        lr=0.1,
        epochs=15,
        batch_size=512,
        weight_decay=0.0,
    )
    vbn_d.fit(df_d)

    res_d = run_inference(vbn_d, "discrete", inferers=inferers_d, batch_size=512)

    # ===== GAUSSIAN =====
    n_gau = 40960
    dag_g, df_g, p_g = gen_linear_gaussian(n_gau)
    c0 = 0.5
    gt_g = gt_gaussian_EA_given_C(c0, p_g["wA"], p_g["wB"], p_g["sigma_eps"])
    ref_g = ref_gaussian_EA_given_C_close(df_g, c0)
    print(f"[GAUSSIAN] Ground truth:    E[A | C={c0}] = {gt_g:.6f}")
    print(f"[GAUSSIAN] Empirical ref.:  E[A | C={c0}] ≈ {ref_g:.6f}")

    vbn_g = VBN(dag=dag_g, device=device, seed=0)
    vbn_g.set_learning_method("linear_gaussian")
    vbn_g.fit(df_g)

    res_g = run_inference(vbn_g, "gaussian", inferers=inferers_g, batch_size=512)

    # Pretty print
    def print_results(results, mode, reference, ground_truth):
        df = pd.DataFrame(results, columns=["method", "estimate", "time (s)"])
        df["|err| vs GT"] = abs(df["estimate"] - ground_truth)
        df["|err| vs Emp"] = abs(df["estimate"] - reference)
        print(f"\n[{mode.upper()}] Results (num_samples=4096)")
        print(f"  Ground truth = {ground_truth:.6f}, Empirical ref = {reference:.6f}")
        print(df.to_string(index=False, float_format=lambda x: f"{x:0.6f}"))

    print_results(res_d, "discrete", ref_d, gt_d)
    print_results(res_g, "gaussian", ref_g, gt_g)

    # after fitting vbn_d / vbn_g
    pdf_cf, samples_cf = vbn_d.counterfactual(
        query="A",
        evidence={"C": torch.ones(512, 1, device=vbn_d.device)},  # observed C=1
        do={"B": torch.zeros(512, 1, device=vbn_d.device)},  # force B=0
        base_infer="montecarlo.lw",
        num_samples=2048,
    )
    est_cf = (samples_cf["A"] > 0.5).float().mean().item()
    print("P_cf(A=1 | C=1, do(B=0)) ≈", est_cf)


if __name__ == "__main__":
    main()
