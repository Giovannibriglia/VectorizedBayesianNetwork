#!/usr/bin/env python3
"""
01_mle_params_vs_exact_inference_cost.py

One figure with two lines vs cardinality k:
  (1) Total free CPT parameters (MLE model size)
  (2) Exact inference complexity proxy via Variable Elimination:
      peak intermediate factor entries (memory/time proxy ~ k^(w+1))

Small DAG (5 mechanisms / CPDs):
  A -> C <- B
  C -> D
  C -> E

Query: P(A | D=d, E=e)  (diagnosis-style)
Elimination order: B, C, D, E, A (we skip query/evidence vars in elimination)

Deps: numpy, matplotlib
Run: python 01_mle_params_vs_exact_inference_cost.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# -----------------------------
# DAG definition
# -----------------------------
VARS = ["A", "B", "C", "D", "E"]
PARENTS = {
    "A": [],
    "B": [],
    "C": ["A", "B"],
    "D": ["C"],
    "E": ["C"],
}


# -----------------------------
# MLE (CPT) free-parameter counts
# -----------------------------
def free_params_for_cpd(
    child: str, parents: Sequence[str], card: Dict[str, int]
) -> int:
    """
    Discrete CPT parameters with normalization constraints:
      free params = (|X|-1) * prod_{p in Pa(X)} |p|
    """
    child_states = card[child]
    parent_configs = 1
    for p in parents:
        parent_configs *= card[p]
    return (child_states - 1) * parent_configs


def total_free_params(card: Dict[str, int]) -> int:
    return sum(free_params_for_cpd(v, PARENTS[v], card) for v in VARS)


# -----------------------------
# Factor operations for Variable Elimination (Exact inference)
# -----------------------------
@dataclass
class Factor:
    scope: Tuple[str, ...]  # ordered variables
    card: Dict[str, int]  # var -> cardinality
    values: np.ndarray  # shape = [card[v] for v in scope]

    def reduce(self, evidence: Dict[str, int]) -> "Factor":
        """Condition on evidence by slicing out fixed indices."""
        if not evidence:
            return self
        slicer = []
        new_scope = []
        for v in self.scope:
            if v in evidence:
                slicer.append(evidence[v])
            else:
                slicer.append(slice(None))
                new_scope.append(v)
        new_vals = self.values[tuple(slicer)]
        return Factor(tuple(new_scope), self.card, np.asarray(new_vals))


def multiply_factors(f1: Factor, f2: Factor) -> Factor:
    """Multiply two factors via broadcasting alignment."""
    # Union scope: keep f1 order, then add f2 vars not present
    scope = list(f1.scope)
    for v in f2.scope:
        if v not in scope:
            scope.append(v)
    scope_t = tuple(scope)

    def reshape_to(f: Factor, target_scope: Tuple[str, ...]) -> np.ndarray:
        idx_map = {v: i for i, v in enumerate(f.scope)}
        present = [v for v in target_scope if v in idx_map]
        if present:
            perm = [idx_map[v] for v in present]
            vals = np.transpose(f.values, axes=perm)
        else:
            vals = f.values
        shape = []
        present_set = set(present)
        for v in target_scope:
            shape.append(f.card[v] if v in present_set else 1)
        return vals.reshape(shape)

    v1 = reshape_to(f1, scope_t)
    v2 = reshape_to(f2, scope_t)
    out = v1 * v2
    return Factor(scope_t, f1.card, out)


def sum_out(f: Factor, var: str) -> Factor:
    """Sum out one variable."""
    if var not in f.scope:
        return f
    axis = f.scope.index(var)
    out = f.values.sum(axis=axis)
    new_scope = tuple(v for v in f.scope if v != var)
    return Factor(new_scope, f.card, np.asarray(out))


def random_cpt(
    child: str, parents: Sequence[str], card: Dict[str, int], rng: np.random.Generator
) -> Factor:
    """
    Factor representing P(child | parents), stored over scope (parents..., child).
    """
    scope = tuple(list(parents) + [child])
    if not parents:
        probs = rng.dirichlet(alpha=np.ones(card[child]))
        return Factor(scope=(child,), card=card, values=probs)

    parent_shape = [card[p] for p in parents]
    c = card[child]
    vals = np.zeros(parent_shape + [c], dtype=np.float64)
    for idx in np.ndindex(*parent_shape):
        vals[idx] = rng.dirichlet(alpha=np.ones(c))
    return Factor(scope=scope, card=card, values=vals)


def build_bn_factors(card: Dict[str, int], seed: int) -> List[Factor]:
    rng = np.random.default_rng(seed)
    factors = []
    for v in VARS:
        factors.append(random_cpt(v, PARENTS[v], card, rng))
    return factors


def ve_stats(
    factors: List[Factor],
    query_vars: Sequence[str],
    evidence: Dict[str, int],
    elim_order: Sequence[str],
    measure: str = "peak_entries",  # "peak_entries" or "wall_ms"
) -> float:
    """
    Run VE and return a scalar complexity metric:
      - peak_entries: peak intermediate factor size (entries)
      - wall_ms: runtime in ms
    """
    t0 = time.perf_counter()

    work = [f.reduce(evidence) for f in factors]
    Q = set(query_vars)
    E = set(evidence.keys())

    peak = 1

    for z in elim_order:
        if z in Q or z in E:
            continue

        bucket = [f for f in work if z in f.scope]
        if not bucket:
            continue
        rest = [f for f in work if z not in f.scope]

        prod = bucket[0]
        for f in bucket[1:]:
            prod = multiply_factors(prod, f)
            if prod.values.shape:
                peak = max(peak, int(np.prod(prod.values.shape)))

        prod = sum_out(prod, z)
        if prod.values.shape:
            peak = max(peak, int(np.prod(prod.values.shape)))

        work = rest + [prod]

    # Multiply remaining factors
    prod = work[0]
    for f in work[1:]:
        prod = multiply_factors(prod, f)
        if prod.values.shape:
            peak = max(peak, int(np.prod(prod.values.shape)))

    # Sum out leftover non-query vars (if any)
    for v in list(prod.scope):
        if v not in Q:
            prod = sum_out(prod, v)

    wall_ms = (time.perf_counter() - t0) * 1000.0
    return float(peak if measure == "peak_entries" else wall_ms)


def main():
    # Sweep cardinalities
    ks = list(range(1, 1001, 100))  # edit as you like

    # Fixed query/evidence (diagnosis)
    query_vars = ["A"]
    # VE elim order (simple, fixed)
    elim_order = ["B", "C", "D", "E", "A"]

    # Choose metric for inference line:
    #   - "peak_entries" is stable and clearly exponential
    #   - "wall_ms" is more “real” but can be noisy on different machines
    inference_metric = "wall_ms"

    params_line = []
    inference_line = []

    for k in tqdm(ks, desc="Running VE"):
        card = {v: k for v in VARS}
        params_line.append(total_free_params(card))

        # Build factors (random CPTs) once per k
        factors = build_bn_factors(card, seed=123 + k)

        # Fixed evidence values per k (so plots are consistent)
        rng = np.random.default_rng(999 + k)
        evidence = {"D": int(rng.integers(0, k)), "E": int(rng.integers(0, k))}

        # Warmup then measure (median of a few runs)
        _ = ve_stats(
            factors, query_vars, evidence, elim_order, measure=inference_metric
        )
        vals = [
            ve_stats(
                factors, query_vars, evidence, elim_order, measure=inference_metric
            )
            for _ in range(5)
        ]
        inference_line.append(float(np.median(vals)))

    # Plot both lines in the same figure (different y-axes for readability)
    fig = plt.figure(dpi=500, figsize=(6, 6))
    plt.plot(ks, params_line, marker="o", linewidth=5)
    plt.xlabel("Cardinality per variable (k)", fontsize=20)
    # plt.yscale("log")
    plt.title("Total CPT parameters", fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("1_scal_cont.pdf")
    plt.show()


if __name__ == "__main__":
    main()
