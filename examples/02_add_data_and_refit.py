# examples/02_add_data_and_refit.py
from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch

from vbn.core import CausalBayesNet

# BN: X->Y<-Z, Y->A
G = nx.DiGraph([("X", "Y"), ("Z", "Y"), ("Y", "A")])
types = {"X": "discrete", "Z": "discrete", "Y": "discrete", "A": "continuous"}
cards = {"X": 3, "Z": 2, "Y": 4}
bn = CausalBayesNet(G, types, cards)

# ---------- initial data ----------
N0 = 4000
rng = np.random.default_rng(1)
X0 = torch.tensor(rng.integers(0, 3, size=N0))
Z0 = torch.tensor(rng.integers(0, 2, size=N0))
Y0 = (X0 + 2 * Z0 + torch.tensor(rng.integers(0, 2, size=N0))) % 4
A0 = 0.9 * Y0.float() + torch.randn(N0) * 0.6
df0 = pd.DataFrame({"X": X0.numpy(), "Z": Z0.numpy(), "Y": Y0.numpy(), "A": A0.numpy()})

# Fit once (discrete + LG)
lp_disc = bn.fit("discrete_mle", df0)
lp_lg = bn.fit("continuous_gaussian", df0)

# ---------- add new data in three ways ----------
# (1) pandas DataFrame
N1 = 1500
X1 = torch.tensor(rng.integers(0, 3, size=N1))
Z1 = torch.tensor(rng.integers(0, 2, size=N1))
Y1 = (X1 + Z1 + torch.tensor(rng.integers(0, 3, size=N1))) % 4
A1 = 0.9 * Y1.float() + torch.randn(N1) * 0.6
df1 = pd.DataFrame({"X": X1.numpy(), "Z": Z1.numpy(), "Y": Y1.numpy(), "A": A1.numpy()})
bn.add_data(df1, update_params=True)  # online incremental update

# (2) dict[str, Tensor]
N2 = 1200
X2 = torch.tensor(rng.integers(0, 3, size=N2))
Z2 = torch.tensor(rng.integers(0, 2, size=N2))
Y2 = (2 * X2 + Z2 + torch.tensor(rng.integers(0, 2, size=N2))) % 4
A2 = 0.9 * Y2.float() + torch.randn(N2) * 0.6
bn.add_data({"X": X2, "Z": Z2, "Y": Y2, "A": A2}, update_params=True)

# (3) TensorDict (implicitly through _to_tensordict inside add_data)
# Here we just reuse a DataFrame; add_data will convert for us.

# Also show a manual partial_fit over *all* accumulated data
bn.partial_fit(td_new=None, epochs=1, batch_size=8192)

# ---------- check effect on a discrete table ----------
tab_before = lp_disc.discrete_tables["Y"].probs.detach().cpu()
tab_after = (
    bn.tabular_probs["Y"].detach().cpu() if hasattr(bn, "tabular_probs") else tab_before
)

plt.figure()
if tab_after.ndim == 1:
    xs = np.arange(tab_after.numel())
    plt.bar(xs - 0.15, tab_before.numpy(), width=0.3, label="before")
    plt.bar(xs + 0.15, tab_after.numpy(), width=0.3, label="after")
    plt.xlabel("Y value")
    plt.ylabel("P(Y)")
else:
    # compare first parent-config row as a teaser
    xs = np.arange(tab_after.shape[1])
    plt.bar(xs - 0.15, tab_before[0].numpy(), width=0.3, label="before (pcfg0)")
    plt.bar(xs + 0.15, tab_after[0].numpy(), width=0.3, label="after (pcfg0)")
    plt.xlabel("Y value")
    plt.ylabel("P(Y | pcfg=0)")
plt.title("Discrete table drift after add_data + partial_fit")
plt.legend()
plt.tight_layout()
plt.show()
