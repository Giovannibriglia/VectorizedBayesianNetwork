#!/usr/bin/env python3
"""
Self-contained example: computational complexity of (1) MLE for CPTs + incremental updates
and (2) exact inference via Variable Elimination (VE), as a function of:
- variable cardinality k
- number of mechanisms (CPDs)
- structure/treewidth (small DAG)
- evidence/query

This script:
1) Defines a small DAG with 5 mechanisms (CPDs): A, B, C|A,B, D|C, E|C
2) Computes:
   - number of free parameters per CPD and total (MLE "model size")
   - empirical runtime + approximate operation counts for VE inference as k increases
   - empirical runtime for incremental MLE updates as k increases and batch size changes
3) Produces plots suitable for an intro section.

Dependencies: numpy, matplotlib
Run: python 06_mle_exact_inference_complexity.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Small DAG (5 nodes, 5 CPDs)
# A -> C <- B, and C -> D, C -> E
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
# Factor utilities (for VE)
# -----------------------------
@dataclass
class Factor:
    scope: Tuple[str, ...]  # ordered variables
    card: Dict[str, int]  # variable -> cardinality
    values: np.ndarray  # shape = [card[v] for v in scope]

    def copy(self) -> "Factor":
        return Factor(self.scope, self.card, self.values.copy())

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

    def normalize(self) -> "Factor":
        s = self.values.sum()
        if s > 0:
            self.values = self.values / s
        return self


def multiply_factors(f1: Factor, f2: Factor) -> Tuple[Factor, int]:
    """
    Multiply two factors with broadcasting/alignment.
    Returns (product_factor, approx_multiplies).
    """
    # Union scope in a stable order: f1.scope then variables from f2 not in f1
    scope = list(f1.scope)
    for v in f2.scope:
        if v not in scope:
            scope.append(v)
    scope_t = tuple(scope)

    # Build target shape
    tgt_shape = tuple(f1.card[v] for v in scope_t)

    # Reshape f1 values to target by inserting singleton axes where missing
    def reshape_to(f: Factor, target_scope: Tuple[str, ...]) -> np.ndarray:
        # For each v in target_scope, axis is f.card[v] if v in f.scope else 1
        shape = []
        # Map f axes
        idx_map = {v: i for i, v in enumerate(f.scope)}
        # Move axes of f to match the order in target_scope (only the present vars)
        present_vars = [v for v in target_scope if v in idx_map]
        if present_vars:
            perm = [idx_map[v] for v in present_vars]
            vals = np.transpose(f.values, axes=perm)
        else:
            vals = f.values  # scalar (possible after reductions)
        # Now expand dims to full target
        pv_set = set(present_vars)
        pv_iter = iter(range(len(present_vars)))
        for v in target_scope:
            if v in pv_set:
                shape.append(f.card[v])
                next(pv_iter)
            else:
                shape.append(1)
        return vals.reshape(shape)

    v1 = reshape_to(f1, scope_t)
    v2 = reshape_to(f2, scope_t)

    # Broadcasting multiplication
    out = v1 * v2

    # Approx multiply count = number of output entries
    approx_multiplies = int(np.prod(tgt_shape))
    return Factor(scope_t, f1.card, out), approx_multiplies


def sum_out(f: Factor, var: str) -> Tuple[Factor, int]:
    """Sum out one variable. Returns (new_factor, approx_additions)."""
    if var not in f.scope:
        return f, 0
    axis = f.scope.index(var)
    new_scope = tuple(v for v in f.scope if v != var)
    # summation
    out = f.values.sum(axis=axis)
    # Approx additions: for each output cell, (card[var]-1) adds
    out_entries = int(np.prod(out.shape)) if out.shape else 1
    approx_adds = out_entries * (f.card[var] - 1)
    return Factor(new_scope, f.card, np.asarray(out)), approx_adds


def variable_elimination(
    factors: List[Factor],
    query_vars: Sequence[str],
    evidence: Dict[str, int],
    elim_order: Sequence[str],
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Exact inference P(query_vars | evidence) via VE.
    Returns:
      - normalized factor values over query_vars (numpy array)
      - stats dict: mults, adds, max_factor_entries, wall_ms
    """
    t0 = time.perf_counter()

    # Reduce factors by evidence
    work = [f.reduce(evidence) for f in factors]

    mults = 0
    adds = 0
    max_entries = 0

    Q = set(query_vars)
    E = set(evidence.keys())

    for z in elim_order:
        if z in Q or z in E:
            continue

        bucket = [f for f in work if z in f.scope]
        if not bucket:
            continue
        rest = [f for f in work if z not in f.scope]

        # Multiply all in bucket
        prod = bucket[0]
        for f in bucket[1:]:
            prod, m = multiply_factors(prod, f)
            mults += m
            max_entries = max(
                max_entries, int(np.prod(prod.values.shape)) if prod.values.shape else 1
            )

        # Sum out z
        prod, a = sum_out(prod, z)
        adds += a
        max_entries = max(
            max_entries, int(np.prod(prod.values.shape)) if prod.values.shape else 1
        )

        work = rest + [prod]

    # Multiply remaining factors
    if not work:
        raise RuntimeError("No factors left (unexpected).")
    prod = work[0]
    for f in work[1:]:
        prod, m = multiply_factors(prod, f)
        mults += m
        max_entries = max(
            max_entries, int(np.prod(prod.values.shape)) if prod.values.shape else 1
        )

    # Sum out any leftover non-query vars (should be none if elim_order covered)
    for v in list(prod.scope):
        if v not in Q:
            prod, a = sum_out(prod, v)
            adds += a

    # Reorder to query_vars
    if tuple(query_vars) != prod.scope:
        # permute axes
        axis_map = {v: i for i, v in enumerate(prod.scope)}
        perm = [axis_map[v] for v in query_vars]
        prod.values = np.transpose(prod.values, axes=perm)
        prod.scope = tuple(query_vars)

    prod.normalize()

    wall_ms = (time.perf_counter() - t0) * 1000.0
    stats = {
        "mults": int(mults),
        "adds": int(adds),
        "max_factor_entries": int(max_entries),
        "wall_ms": float(wall_ms),
    }
    return prod.values, stats


# -----------------------------
# Random CPT generation (for factors)
# -----------------------------
def random_cpt(
    child: str, parents: Sequence[str], card: Dict[str, int], rng: np.random.Generator
) -> Factor:
    """
    Returns factor over (parents..., child) representing P(child | parents).
    Stored as a full table with last axis = child.
    """
    scope = tuple(list(parents) + [child])
    shape = [card[v] for v in scope]
    # For each parent configuration, sample a Dirichlet over child states.
    # We'll build with broadcasting-friendly approach.
    parent_shape = [card[v] for v in parents] if parents else []
    c = card[child]

    if not parents:
        probs = rng.dirichlet(alpha=np.ones(c))
        vals = probs.reshape((c,))
        return Factor(scope=(child,), card=card, values=vals)

    # Create table: parent_shape + [c]
    vals = np.zeros(parent_shape + [c], dtype=np.float64)
    it = np.ndindex(*parent_shape)
    for idx in it:
        vals[idx] = rng.dirichlet(alpha=np.ones(c))
    return Factor(scope=scope, card=card, values=vals)


def build_bn_factors(card: Dict[str, int], seed: int = 0) -> List[Factor]:
    rng = np.random.default_rng(seed)
    factors: List[Factor] = []
    for v in VARS:
        parents = PARENTS[v]
        factors.append(random_cpt(v, parents, card, rng))
    return factors


# -----------------------------
# MLE parameter counting + update cost model
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


def total_free_params(card: Dict[str, int]) -> Tuple[int, Dict[str, int]]:
    per = {}
    tot = 0
    for v in VARS:
        fp = free_params_for_cpd(v, PARENTS[v], card)
        per[v] = fp
        tot += fp
    return tot, per


def incremental_mle_update(
    counts: Dict[str, np.ndarray],
    samples: Dict[str, np.ndarray],
    card: Dict[str, int],
) -> float:
    """
    Simple incremental update of sufficient statistics (counts) for each CPD.
    This simulates the *mechanism update* part (not re-running inference).

    For each CPD X|Pa, we increment count[Pa, X] for each sample.
    Returns wall time in ms.
    """
    t0 = time.perf_counter()

    n = next(iter(samples.values())).shape[0]
    for x in VARS:
        pa = PARENTS[x]
        if not pa:
            # counts[x] shape: [|x|]
            np.add.at(counts[x], samples[x], 1)
        else:
            # counts[x] shape: [|pa1|, |pa2|, ..., |x|]
            idx = tuple(samples[p] for p in pa) + (samples[x],)
            np.add.at(counts[x], idx, 1)

    return (time.perf_counter() - t0) * 1000.0


def init_counts(card: Dict[str, int]) -> Dict[str, np.ndarray]:
    counts = {}
    for x in VARS:
        pa = PARENTS[x]
        shape = [card[p] for p in pa] + [card[x]]
        counts[x] = np.zeros(shape, dtype=np.int64)
    return counts


# -----------------------------
# Main experiment
# -----------------------------
def main() -> None:
    # Settings
    ks = list(range(50, 1000, 50))  # variable cardinalities
    n_repeats_inference = 5
    batch_sizes = [1024, 2048, 4096, 10240]

    # Fixed query/evidence pattern (diagnostic-ish):
    # Query: A ; Evidence: D, E
    query_vars = ["A"]
    # VE elimination order (heuristic). With this DAG, eliminating B then C then others is reasonable.
    elim_order = ["B", "C", "D", "E", "A"]

    # Collect results
    params_total = []
    params_per_cpd_A = []
    params_per_cpd_C = []

    ve_wall = []
    ve_mults = []
    ve_adds = []
    ve_peak = []

    upd_wall_by_bs = {bs: [] for bs in batch_sizes}

    for k in ks:
        card = {v: k for v in VARS}

        # --- MLE model size (free parameters) ---
        tot_fp, per_fp = total_free_params(card)
        params_total.append(tot_fp)
        params_per_cpd_A.append(per_fp["A"])
        params_per_cpd_C.append(per_fp["C"])

        # --- Build random BN factors for inference ---
        factors = build_bn_factors(card, seed=123 + k)

        # Evidence values (random but fixed per k for fairness)
        rng = np.random.default_rng(999 + k)
        evidence = {"D": int(rng.integers(0, k)), "E": int(rng.integers(0, k))}

        # --- Exact inference via VE: measure runtime + op counts ---
        wall_ms_list = []
        mults_list = []
        adds_list = []
        peak_list = []

        # Warmup
        _ = variable_elimination(factors, query_vars, evidence, elim_order)

        for _ in range(n_repeats_inference):
            _, stats = variable_elimination(factors, query_vars, evidence, elim_order)
            wall_ms_list.append(stats["wall_ms"])
            mults_list.append(stats["mults"])
            adds_list.append(stats["adds"])
            peak_list.append(stats["max_factor_entries"])

        ve_wall.append(float(np.median(wall_ms_list)))
        ve_mults.append(int(np.median(mults_list)))
        ve_adds.append(int(np.median(adds_list)))
        ve_peak.append(int(np.median(peak_list)))

        # --- Incremental MLE update time vs batch size ---
        for bs in batch_sizes:
            counts = init_counts(card)
            # Generate iid samples (uniform) purely to drive updates
            samples = {v: rng.integers(0, k, size=(bs,), dtype=np.int64) for v in VARS}
            # Warmup
            _ = incremental_mle_update(counts, samples, card)
            # Measure
            tms = incremental_mle_update(counts, samples, card)
            upd_wall_by_bs[bs].append(tms)

        print(
            f"k={k:2d} | free_params={tot_fp:8d} | VE_ms~{ve_wall[-1]:7.3f} "
            f"| VE_mults~{ve_mults[-1]:10d} | VE_peak_entries~{ve_peak[-1]:8d}"
        )

    # -----------------------------
    # Plotting
    # -----------------------------
    fig1 = plt.figure()
    plt.plot(ks, params_total, marker="o")
    plt.xlabel("Cardinality per variable (k)")
    plt.ylabel("Total free CPT parameters (sum over mechanisms)")
    plt.title("MLE model size grows polynomially with cardinalities")
    plt.grid(True)

    fig2 = plt.figure()
    plt.plot(ks, ve_wall, marker="o")
    plt.xlabel("Cardinality per variable (k)")
    plt.ylabel("Median VE wall-time (ms)")
    plt.title("Exact inference (Variable Elimination) runtime vs cardinality")
    plt.grid(True)

    fig3 = plt.figure()
    plt.plot(ks, ve_mults, marker="o", label="multiplies (approx)")
    plt.plot(ks, ve_adds, marker="o", label="adds (approx)")
    plt.xlabel("Cardinality per variable (k)")
    plt.ylabel("Operation count (approx, median)")
    plt.title("Exact inference operation growth (structure-dependent)")
    plt.grid(True)
    plt.legend()

    fig4 = plt.figure()
    for bs in batch_sizes:
        plt.plot(ks, upd_wall_by_bs[bs], marker="o", label=f"batch={bs}")
    plt.xlabel("Cardinality per variable (k)")
    plt.ylabel("Incremental MLE update time (ms)")
    plt.title("Updating mechanisms (sufficient stats) is linear in batch size")
    plt.grid(True)
    plt.legend()

    # Optional: show peak factor size (proxy for memory / treewidth effects)
    fig5 = plt.figure()
    plt.plot(ks, ve_peak, marker="o")
    plt.xlabel("Cardinality per variable (k)")
    plt.ylabel("Peak intermediate factor entries")
    plt.title("Peak factor size during VE (memory proxy)")
    plt.grid(True)

    plt.show()

    # -----------------------------
    # Quick text summary (for copy/paste into notes)
    # -----------------------------
    print("\n--- Summary (rules of thumb) ---")
    print(
        "1) MLE (discrete CPTs): free params per CPD X|Pa is (|X|-1)*Π|Pa|, total is sum over mechanisms."
    )
    print(
        "2) Exact inference (VE): time/memory dominated by largest intermediate factor ~ O(k^{w+1}) where w=treewidth."
    )
    print(
        "3) Updating mechanisms via incremental counts: O(batch * (#mechanisms)) with small constants; scales ~linearly."
    )


if __name__ == "__main__":
    main()
