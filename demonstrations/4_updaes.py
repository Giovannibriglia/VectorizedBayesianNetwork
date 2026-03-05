#!/usr/bin/env python3
"""
04_inefficient_updates_repeated_inference.py

Inefficient updates:
- Many learning/update procedures repeatedly require inference:
    e.g., EM-style updates, gradient steps, or online re-estimation where each step
    needs marginals/posteriors computed by exact inference.
- If inference is expensive (high treewidth / high k), learning cost inherits it.

We simulate:
- A controlled-treewidth factor model (same as script 03).
- An "online update loop" of T steps.
- Each step performs:
    (a) a small parameter perturbation (cheap)
    (b) EXACT inference via VE to compute a posterior (expensive)
- Plot total time vs T for different (k, w) settings.

Deps: numpy, matplotlib
Run: python 04_inefficient_updates_repeated_inference.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Factor:
    scope: Tuple[int, ...]
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


def multiply(f1: Factor, f2: Factor) -> Factor:
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
    return Factor(scope_t, f1.card, out)


def sum_out(f: Factor, var: int) -> Factor:
    if var not in f.scope:
        return f
    axis = f.scope.index(var)
    out = f.values.sum(axis=axis)
    new_scope = tuple(v for v in f.scope if v != var)
    return Factor(new_scope, f.card, np.asarray(out))


def variable_elimination(
    factors: List[Factor],
    query: Sequence[int],
    evidence: Dict[int, int],
    elim_order: Sequence[int],
) -> None:
    # We only care about runtime here, not the distribution.
    work = [f.reduce(evidence) for f in factors]
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
            prod = multiply(prod, f)
        prod = sum_out(prod, z)
        work = rest + [prod]

    prod = work[0]
    for f in work[1:]:
        prod = multiply(prod, f)

    for v in list(prod.scope):
        if v not in Q:
            prod = sum_out(prod, v)

    # normalize omitted (doesn't change asymptotic cost)


def make_controlled_treewidth_model(
    n_vars: int, k: int, w: int, seed: int = 0
) -> Tuple[List[Factor], List[int]]:
    rng = np.random.default_rng(seed)
    card = {i: k for i in range(n_vars)}
    factors: List[Factor] = []
    for i in range(n_vars - w):
        scope = tuple(range(i, i + w + 1))
        vals = rng.random([k] * (w + 1)) + 1e-6
        factors.append(Factor(scope=scope, card=card, values=vals))
    elim = list(range(n_vars))
    return factors, elim


def cheap_parameter_update(
    factors: List[Factor], rng: np.random.Generator, scale: float = 1e-3
) -> None:
    # Perturb a few random entries to mimic "learning updates" (cheap vs inference).
    for f in factors[:: max(1, len(factors) // 5)]:
        idx = tuple(rng.integers(0, s) for s in f.values.shape)
        f.values[idx] *= 1.0 + scale * rng.normal()


def run_update_loop(n_vars: int, k: int, w: int, T: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    factors, elim = make_controlled_treewidth_model(n_vars=n_vars, k=k, w=w, seed=seed)
    query = [n_vars - 1]
    evidence = {0: 0}

    # warmup
    variable_elimination(factors, query, evidence, elim)

    t0 = time.perf_counter()
    for _ in range(T):
        cheap_parameter_update(factors, rng)
        variable_elimination(
            factors, query, evidence, elim
        )  # repeated inference dominates
    return (time.perf_counter() - t0) * 1000.0


def main():
    n_vars = 18
    Ts = np.arange(1, 61, 3)

    settings = [
        {"k": 3, "w": 2},
        {"k": 4, "w": 2},
        {"k": 4, "w": 4},
        {"k": 6, "w": 4},
    ]

    plt.figure()
    for s in settings:
        times = []
        for T in Ts:
            ms = run_update_loop(
                n_vars=n_vars,
                k=s["k"],
                w=s["w"],
                T=int(T),
                seed=1000 + 17 * s["k"] + 31 * s["w"],
            )
            times.append(ms)
        plt.plot(Ts, times, marker="o", label=f'k={s["k"]}, w={s["w"]}')
    plt.xlabel("Number of update steps (T)")
    plt.ylabel("Total time (ms)")
    plt.title(
        "Inefficient updates: repeated inference makes learning inherit inference cost"
    )
    plt.grid(True)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
