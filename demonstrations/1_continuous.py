#!/usr/bin/env python3
"""
01_continuous_discretization_blowup.py

Scalability to continuous variables:
- Discretize a continuous child X into k bins.
- CPT free parameters for X | Pa: (k-1) * Π |Pa|
- With continuous parents discretized too, Π |Pa| explodes with k^d.
- Compare with a Linear-Gaussian CPD parameter count (rough constant wrt k):
    X = w^T Pa + b + eps, eps ~ N(0, sigma^2)
  params ~ (d weights + 1 bias + 1 variance) = d+2

Outputs:
- Plot of CPT free-params vs k for different parent counts d.
- Plot comparing CPT params vs (approx) Gaussian params.

Deps: numpy, matplotlib
Run: python 01_continuous_discretization_blowup.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def cpt_free_params(k_child: int, k_parent: int, n_parents: int) -> int:
    # (|X|-1) * Π |Pa|
    return (k_child - 1) * (k_parent**n_parents)


def gaussian_params(n_parents: int) -> int:
    # linear Gaussian: weights (n_parents) + bias + variance
    return n_parents + 2


def main():
    ks = np.arange(50, 500, 50)  # bins/cardinality
    parent_counts = [1, 10, 20, 50, 100]

    # Assume each parent is discretized into same k bins (worst-case illustrative)
    # so Π|Pa| = k^d.
    plt.figure()
    for d in parent_counts:
        y = [cpt_free_params(k_child=k, k_parent=k, n_parents=d) for k in ks]
        plt.plot(ks, y, marker="o", label=f"{d} parents")
    plt.xlabel("Discretization bins / cardinality (k)")
    plt.ylabel("Free CPT parameters for X | Pa (discrete)")
    plt.title("Discretizing continuous variables: CPT parameter blow-up")
    plt.grid(True)
    plt.legend()

    # Compare discrete CPT vs linear-Gaussian CPD parameter count
    plt.figure()
    d = 3
    y_cpt = [cpt_free_params(k_child=k, k_parent=k, n_parents=d) for k in ks]
    y_gauss = [gaussian_params(d) for _ in ks]
    plt.plot(ks, y_cpt, marker="o", label=f"CPT params (d={d} parents)")
    plt.plot(ks, y_gauss, marker="o", label=f"Linear-Gaussian params (d={d})")
    plt.xlabel("Discretization bins / cardinality (k)")
    plt.ylabel("Number of parameters")
    plt.title("Expressiveness vs scalability: CPT discretization vs Gaussian CPD")
    plt.grid(True)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
