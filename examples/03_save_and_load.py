# examples/03_save_load_and_plot.py
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch

from vbn.core import CausalBayesNet, LearnParams

# ---------------- BN & data ----------------
G = nx.DiGraph([("X", "Y"), ("Z", "Y"), ("Y", "A")])
types = {"X": "discrete", "Z": "discrete", "Y": "discrete", "A": "continuous"}
cards = {"X": 3, "Z": 2, "Y": 4}
bn = CausalBayesNet(G, types, cards)

N = 5000
rng = np.random.default_rng(7)
X = torch.tensor(rng.integers(0, 3, size=N))
Z = torch.tensor(rng.integers(0, 2, size=N))
Y = (X + 2 * Z + torch.tensor(rng.integers(0, 2, size=N))) % 4
A = 1.0 * Y.float() + torch.randn(N) * 0.5
df = pd.DataFrame({"X": X.numpy(), "Z": Z.numpy(), "Y": Y.numpy(), "A": A.numpy()})

# Fit both discrete+continuous
lp_disc = bn.fit_discrete_mle(df)
lp_lg = bn.fit_continuous_gaussian(df)

# Merge (optional): last-wins if overlapping families (here there is no conflict)
from vbn.core import merge_learnparams

lp_full = merge_learnparams(lp_disc, lp_lg)

# -------------- SAVE ----------------
out_dir = Path("artifacts")
out_dir.mkdir(parents=True, exist_ok=True)
bin_path = out_dir / "bn_learnparams.td"  # your .io uses torch.save under the hood
bn.save_params(lp_full, str(bin_path))
print(f"[saved] {bin_path}")

# -------------- LOAD ----------------
lp_loaded: LearnParams = bn.load_params(str(bin_path))
print(
    "[loaded] families:",
    "tables" if lp_loaded.discrete_tables else "-",
    "lg" if lp_loaded.lg else "-",
)


# -------------- Plot from loaded ----------------
def plot_discrete_table(table, title):
    probs = table.probs.detach().cpu().float().numpy()
    plt.figure()
    if probs.ndim == 1:
        plt.bar(range(len(probs)), probs)
        plt.xlabel("child value")
        plt.ylabel("prob")
    else:
        Pcfg, C = probs.shape
        xs = np.arange(C)
        width = 0.8 / Pcfg
        for i in range(Pcfg):
            plt.bar(xs + i * width, probs[i], width=width, label=f"pcfg={i}")
        plt.xlabel("child value")
        plt.ylabel("prob")
        plt.legend()
    plt.title(title)
    plt.tight_layout()


def quick_plot_lg_mean(bn: CausalBayesNet, node: str, df: pd.DataFrame):
    """Plots observed y vs LG mean using bn.lin_weights; device-safe."""
    # Ensure we have some online weights; touch incremental updater once
    if not (hasattr(bn, "lin_weights") and node in getattr(bn, "lin_weights", {})):
        bn.add_data(df.iloc[:1024], update_params=True)

    beta = bn.lin_weights.get(node, None)
    if beta is None:
        print(f"No online lin_weights for {node}.")
        return

    pars = bn.parents[node]
    # Build X on CPU then move to beta.device
    if pars:
        Xp = [torch.as_tensor(df[p].values, dtype=beta.dtype).view(-1, 1) for p in pars]
        X = torch.cat([torch.ones((len(df), 1), dtype=beta.dtype), *Xp], dim=1)
    else:
        X = torch.ones((len(df), 1), dtype=beta.dtype)

    X = X.to(beta.device)
    yhat = (X @ beta).detach().cpu().numpy()

    y = torch.as_tensor(df[node].values, dtype=torch.float32).numpy()

    plt.figure()
    plt.scatter(range(len(yhat)), y[: len(yhat)], s=6, alpha=0.4, label="observed")
    plt.plot(yhat, lw=2, label="LG mean")
    plt.title(f"{node}: observed vs. LG mean (from loaded params)")
    plt.xlabel("sample index")
    plt.ylabel(node)
    plt.legend()
    plt.tight_layout()


# Discrete plot (Y|X,Z)
if lp_loaded.discrete_tables and "Y" in lp_loaded.discrete_tables:
    plot_discrete_table(lp_loaded.discrete_tables["Y"], "P(Y|X,Z) – loaded")

# Continuous plot (A | Y)
quick_plot_lg_mean(bn, "A", df.iloc[:1200])

plt.show()
