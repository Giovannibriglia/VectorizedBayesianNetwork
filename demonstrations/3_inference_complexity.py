#!/usr/bin/env python3
"""
03_inference_complexity_treewidth_and_3d.py

Inference complexity:
- Exact inference via Variable Elimination is exponential in treewidth:
    time/memory ~ O(k^{w+1}) (roughly), where w is treewidth.
- We build synthetic models that induce intermediate cliques of size (w+1)
  under a natural elimination order, then measure:
    - wall time of VE
    - peak factor entries (memory proxy)
- Also produces a 3D surface over (k, w) -> runtime proxy.

Deps: numpy, matplotlib
Run: python 03_inference_complexity_treewidth_and_3d.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from tqdm import tqdm


@dataclass
class Factor:
    scope: Tuple[int, ...]  # variables as ints
    card: Dict[int, int]
    values: np.ndarray

    def reduce(self, evidence: Dict[int, int]) -> "Factor":
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


def multiply(f1: Factor, f2: Factor) -> Tuple[Factor, int]:
    scope = list(f1.scope)
    for v in f2.scope:
        if v not in scope:
            scope.append(v)
    scope_t = tuple(scope)

    def reshape_to(f: Factor, target_scope: Tuple[int, ...]) -> np.ndarray:
        idx_map = {v: i for i, v in enumerate(f.scope)}
        present = [v for v in target_scope if v in idx_map]
        if present:
            perm = [idx_map[v] for v in present]
            vals = np.transpose(f.values, axes=perm)
        else:
            vals = f.values
        shape = []
        pset = set(present)
        for v in target_scope:
            shape.append(f.card[v] if v in pset else 1)
        return vals.reshape(shape)

    v1 = reshape_to(f1, scope_t)
    v2 = reshape_to(f2, scope_t)
    out = v1 * v2
    approx_mults = int(np.prod(out.shape)) if out.shape else 1
    return Factor(scope_t, f1.card, out), approx_mults


def sum_out(f: Factor, var: int) -> Tuple[Factor, int]:
    if var not in f.scope:
        return f, 0
    axis = f.scope.index(var)
    out = f.values.sum(axis=axis)
    new_scope = tuple(v for v in f.scope if v != var)
    out_entries = int(np.prod(out.shape)) if out.shape else 1
    approx_adds = out_entries * (f.card[var] - 1)
    return Factor(new_scope, f.card, np.asarray(out)), approx_adds


def variable_elimination(
    factors: List[Factor],
    query: Sequence[int],
    evidence: Dict[int, int],
    elim_order: Sequence[int],
) -> Dict[str, float]:
    t0 = time.perf_counter()
    work = [f.reduce(evidence) for f in factors]
    mults = 0
    adds = 0
    peak = 1

    Q = set(query)
    E = set(evidence.keys())

    for z in elim_order:
        if z in Q or z in E:
            continue
        bucket = [f for f in work if z in f.scope]
        if not bucket:
            continue
        rest = [f for f in work if z not in f.scope]

        prod = bucket[0]
        for f in bucket[1:]:
            prod, m = multiply(prod, f)
            mults += m
            peak = max(
                peak, int(np.prod(prod.values.shape)) if prod.values.shape else 1
            )
        prod, a = sum_out(prod, z)
        adds += a
        peak = max(peak, int(np.prod(prod.values.shape)) if prod.values.shape else 1)
        work = rest + [prod]

    prod = work[0]
    for f in work[1:]:
        prod, m = multiply(prod, f)
        mults += m
        peak = max(peak, int(np.prod(prod.values.shape)) if prod.values.shape else 1)

    # sum out leftovers not in query
    for v in list(prod.scope):
        if v not in Q:
            prod, a = sum_out(prod, v)
            adds += a

    wall_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "wall_ms": wall_ms,
        "mults": float(mults),
        "adds": float(adds),
        "peak_entries": float(peak),
    }


def make_controlled_treewidth_model(
    n_vars: int, k: int, w: int, seed: int = 0
) -> Tuple[List[Factor], List[int]]:
    """
    Construct factors that induce intermediate cliques of size (w+1) under elim order [0,1,2,...].
    We do it by adding overlapping (w+1)-ary factors on sliding windows:
      factor on (i, i+1, ..., i+w)
    This behaves like a chain of cliques; induced width is ~w.

    Returns (factors, elim_order).
    """
    rng = np.random.default_rng(seed)
    card = {i: k for i in range(n_vars)}
    factors: List[Factor] = []
    for i in range(n_vars - w):
        scope = tuple(range(i, i + w + 1))
        shape = [k] * (w + 1)
        # random positive values
        vals = rng.random(shape) + 1e-6
        factors.append(Factor(scope=scope, card=card, values=vals))
    elim_order = list(range(n_vars))
    return factors, elim_order


def main():
    # 2D: fix k, vary w (treewidth)
    k_fixed = 4
    n_vars = 18
    ws = list(range(1, 6, 1))  # treewidth proxy
    repeats = 5

    wall = []
    peak = []

    for w in tqdm(ws, desc="VE"):
        factors, elim = make_controlled_treewidth_model(
            n_vars=n_vars, k=k_fixed, w=w, seed=123 + w
        )
        query = [n_vars - 1]
        evidence = {0: 0}  # arbitrary conditioning

        # warmup
        _ = variable_elimination(factors, query, evidence, elim)

        ms = []
        pk = []
        for _ in range(repeats):
            stats = variable_elimination(factors, query, evidence, elim)
            ms.append(stats["wall_ms"])
            pk.append(stats["peak_entries"])
        wall.append(float(np.median(ms)))
        peak.append(float(np.median(pk)))

    plt.figure(dpi=500)
    plt.plot(ws, wall, marker="o")
    plt.xlabel("Treewidth proxy (w)")
    plt.ylabel("Median VE wall-time (ms)")
    plt.title(f"Exact inference becomes intractable as treewidth grows (k={k_fixed})")
    plt.grid(True)

    plt.figure(dpi=500)
    plt.plot(ws, peak, marker="o")
    plt.xlabel("Treewidth proxy (w)")
    plt.ylabel("Peak intermediate factor entries")
    plt.title(f"Memory proxy grows ~ k^(w+1) (k={k_fixed})")
    plt.grid(True)

    # 3D surface: vary k and w
    ks = np.array([2, 3, 4, 5, 6, 8, 10, 12])
    ws2 = np.array([1, 2, 3, 4, 5, 6, 7])
    K, W = np.meshgrid(ks, ws2)

    Z_time = np.zeros_like(K, dtype=float)
    Z_peak = np.zeros_like(K, dtype=float)

    for i in tqdm(range(W.shape[0]), desc="combo"):
        for j in range(W.shape[1]):
            k = int(K[i, j])
            w = int(W[i, j])
            factors, elim = make_controlled_treewidth_model(
                n_vars=n_vars, k=k, w=w, seed=999 + 37 * w + k
            )
            stats = variable_elimination(
                factors, query=[n_vars - 1], evidence={0: 0}, elim_order=elim
            )
            Z_time[i, j] = stats["wall_ms"]
            Z_peak[i, j] = stats["peak_entries"]

    fig = plt.figure(dpi=500)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(K, W, Z_time)
    ax.set_xlabel("Cardinality (k)")
    ax.set_ylabel("Treewidth proxy (w)")
    ax.set_zlabel("VE time (ms)")
    ax.set_title("Inference complexity surface: (k, w) → time")

    fig = plt.figure(dpi=500)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(K, W, Z_peak)
    ax.set_xlabel("Cardinality (k)")
    ax.set_ylabel("Treewidth proxy (w)")
    ax.set_zlabel("Peak factor entries")
    ax.set_title("Memory surface: (k, w) → peak factor size")

    plt.show()


if __name__ == "__main__":
    main()
