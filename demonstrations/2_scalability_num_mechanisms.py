#!/usr/bin/env python3
"""
02_scalability_num_mechanisms.py

Scalability in the number of mechanisms:
- In a BN with N nodes, we have N conditional mechanisms (CPDs).
- Learning (MLE) parameter count sums over mechanisms.
- Sampling / likelihood evaluation typically touches *all* mechanisms.

We simulate a family of DAGs with bounded in-degree p:
- Each node i has up to p parents among {0..i-1}.
- All variables are discrete with cardinality k.
- Free params per node: (k-1) * k^{|Pa|}
- "Evaluate all CPDs per sample" ~ sum over nodes of table lookups,
  which is O(N) but the constant grows with parent-config complexity.

Outputs:
- Plot: total free parameters vs N for different max parents p.
- Plot: proxy evaluation cost vs N for different p.

Deps: numpy, matplotlib
Run: python 02_scalability_num_mechanisms.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def total_free_params(N: int, k: int, p: int) -> int:
    """
    Worst-case: each node has exactly p parents once possible (except first few).
    Free params per node with d parents: (k-1) * k^d
    """
    total = 0
    for i in range(N):
        d = min(p, i)  # cannot have more parents than previous nodes
        total += (k - 1) * (k**d)
    return total


def eval_cost_proxy(N: int, k: int, p: int) -> float:
    """
    Proxy for per-sample evaluation cost:
    - each node needs to index into CPT given parents + child.
    - indexing is O(1), but computing parent-config index can be thought of as O(d),
      and larger d implies larger tables and higher memory/cache pressure.
    We model: cost ~ sum_i (1 + d_i).
    """
    cost = 0.0
    for i in range(N):
        d = min(p, i)
        cost += 1.0 + d
    return cost


def main():
    Ns = np.arange(1, 1000, 50)
    k = 4
    ps = [2, 10, 50, 100]

    """plt.figure()
    for p in ps:
        y = [total_free_params(N=int(N), k=k, p=p) for N in Ns]
        plt.plot(Ns, y, marker="o", label=f"max parents p={p}")
    plt.xlabel("Number of mechanisms / nodes (N)")
    plt.ylabel("Total free CPT parameters (sum over mechanisms)")
    plt.yscale("log")
    plt.title(f"Scaling with number of mechanisms (k={k})")
    plt.grid(True)
    plt.legend()"""

    plt.figure(dpi=500, figsize=(6, 6))
    for p in ps:
        y = [eval_cost_proxy(N=int(N), k=k, p=p) for N in Ns]
        plt.plot(Ns, y, marker="o", label=f"max parents p={p}", linewidth=3)
    plt.xlabel("Number of mechanisms", fontsize=20)
    # plt.ylabel("Per-sample CPD evaluation proxy (arb. units)")
    # plt.yscale("log")
    plt.title("Total CPT parameters", fontsize=22)
    plt.grid(True)
    plt.tight_layout()
    plt.legend(fontsize=16, loc="best")
    plt.savefig("2_scal_mech.pdf")
    plt.show()


if __name__ == "__main__":
    main()
