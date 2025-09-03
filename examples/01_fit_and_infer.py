from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch

from vbn.core import CausalBayesNet, LearnParams, merge_learnparams
from vbn.utils import unpack_gaussian

# --------------------- toy BN ---------------------
# X -> Y <- Z, and Y -> A (continuous)
G = nx.DiGraph([("X", "Y"), ("Z", "Y"), ("Y", "A")])
types = {"X": "discrete", "Z": "discrete", "Y": "discrete", "A": "continuous"}
cards = {"X": 3, "Z": 2, "Y": 4}

bn = CausalBayesNet(G, types, cards)

# --------------------- data gen -------------------
N = 8000
rng = np.random.default_rng(0)
X = torch.tensor(rng.integers(0, 3, size=N))
Z = torch.tensor(rng.integers(0, 2, size=N))
# Y depends on X, Z
Y = (X + 2 * Z + torch.tensor(rng.integers(0, 2, size=N))) % 4
# A ~ 0.8*Y + noise
A = 0.8 * Y.float() + torch.randn(N) * 0.7

df = pd.DataFrame({"X": X.numpy(), "Z": Z.numpy(), "Y": Y.numpy(), "A": A.numpy()})

# ==================================================
# 1) DISCRETE: tabular MLE
lp_disc_mle: LearnParams = bn.fit("discrete_mle", df)
print("[discrete MLE] tables:", list(lp_disc_mle.discrete_tables.keys()))

# 2) DISCRETE: neural CPDs → materialize tabular
lp_disc_mlp: LearnParams = bn.fit("discrete_mlp", df, hidden=64, epochs=5)
lp_disc_mlp = bn.materialize("discrete_mlp", lp_disc_mlp)  # create tables
print("[discrete MLP→tables] tables:", list(lp_disc_mlp.discrete_tables.keys()))

# 3) CONTINUOUS: exact linear-Gaussian
lp_cont_lg: LearnParams = bn.fit("continuous_gaussian", df)
print("[continuous LG] nodes:", lp_cont_lg.lg.order if lp_cont_lg.lg else None)

# 4) CONTINUOUS: MLP heads → materialize LG
lp_cont_mlp: LearnParams = bn.fit("continuous_mlp_gaussian", df, hidden=64, epochs=5)
lp_cont_mlp = bn.materialize("continuous_mlp_gaussian", lp_cont_mlp, data=df)
print("[continuous MLP→LG] nodes:", lp_cont_mlp.lg.order if lp_cont_mlp.lg else None)

# ==================================================
# Inference examples
# (a) exact variable elimination on the discrete subgraph
discrete_exact_inference = bn.setup_inference("discrete_exact")
post_exact = bn.infer(
    discrete_exact_inference,
    lp=lp_disc_mle,
    evidence={"X": torch.tensor([1]), "Z": torch.tensor([0])},
    query=["Y"],
)
print("[inference discrete exact] P(Y | X=1,Z=0):", post_exact["Y"].softmax(-1))

# (b) approximate discrete (sampling) with do-intervention
discrete_approx_inference = bn.setup_inference("discrete_approx")
post_approx = bn.infer(
    inference_obj=discrete_approx_inference,
    lp=lp_disc_mle,
    evidence={"X": torch.tensor([1]), "Z": torch.tensor([0])},
    query=["Y"],
    num_samples=1024,
)
print("[inference discrete approx] P(Y | X=1, do(Z=1)):", post_approx["Y"].softmax(-1))

# (c) continuous exact Gaussian posterior: predict A given Y
continuous_gaussian_inference = bn.setup_inference("continuous_gaussian")
gauss_post = bn.infer(
    inference_obj=continuous_gaussian_inference,
    lp=lp_cont_lg,
    evidence={"Y": torch.tensor([2.0])},
    query=["A"],
)
mu_A, std_A = unpack_gaussian(gauss_post, "A")
print(f"[inference LG] A | Y=2 -> mean={float(mu_A):.3f}, std={float(std_A):.3f}")

# merge discrete CPDs + continuous MLP params
lp_hybrid = merge_learnparams(lp_disc_mle, lp_cont_mlp)

# (c) continuous exact Gaussian posterior (unchanged)
gauss_post = bn.infer(
    continuous_gaussian_inference,
    lp=lp_cont_lg,
    evidence={"Y": torch.tensor([2.0])},
    query=["A"],
)
mu_A, std_A = unpack_gaussian(gauss_post, "A")
print(f"[inference LG] A | Y=2 -> mean={float(mu_A):.3f}, std={float(std_A):.3f}")

# (d) continuous approximate — use the merged params that INCLUDE discrete CPDs
continuous_approx_inference = bn.setup_inference("continuous_approx")
gauss_approx = bn.infer(
    continuous_approx_inference,
    lp=lp_hybrid,
    evidence={"Y": torch.tensor([3.0])},
    query=["A"],
    num_samples=4000,
)
print("gauss_approx: ", gauss_approx)
mu_A2 = gauss_approx["A"]["mean"]
std_A2 = gauss_approx["A"]["var"].sqrt()
print(f"[inference approx] A | Y=3 -> mean={mu_A2.item():.3f}, std={std_A2.item():.3f}")


# ==================================================
# Plotting helpers
def plot_discrete_table(table, title="P(child | parents)"):
    probs = table.probs.detach().cpu().float().numpy()  # [Pcfg, C] or [C]
    plt.figure()
    if probs.ndim == 1:
        plt.bar(range(len(probs)), probs)
        plt.xlabel("child value")
        plt.ylabel("prob")
    else:
        # show all parent-config rows as grouped bars
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


def plot_lg_node(lp_lg: LearnParams, node: str, df: pd.DataFrame):
    """Scatter y vs LG mean prediction; handles CPU/CUDA device safely."""
    if lp_lg.lg is None:
        print("No LG params available.")
        return

    pars = bn.parents[node]
    # build tensors on CPU first (cheap), then move to beta.device
    y = torch.as_tensor(df[node].values, dtype=torch.float32)

    if pars:
        Xp = [
            torch.as_tensor(df[p].values, dtype=torch.float32).view(-1, 1) for p in pars
        ]
        X = torch.cat(
            [torch.ones((len(df), 1), dtype=torch.float32), *Xp], dim=1
        )  # [N, 1+|P|]
    else:
        X = torch.ones((len(df), 1), dtype=torch.float32)

    beta = bn.lin_weights.get(node, None) if hasattr(bn, "lin_weights") else None
    if beta is None:
        print(
            "No online lin_weights found; run `bn.add_data(..., update_params=True)` first."
        )
        return

    # ensure same device & dtype
    X = X.to(device=beta.device, dtype=beta.dtype)
    yhat = (X @ beta).detach().cpu().numpy()

    plt.figure()
    plt.scatter(
        range(len(yhat)), y.numpy(), s=6, alpha=0.4, label="observed", c="orange"
    )
    plt.plot(yhat, lw=2, label="LG mean", c="blue", alpha=0.6)
    plt.title(f"{node}: observed vs. LG mean (ordered by sample index)")
    plt.xlabel("sample index")
    plt.ylabel(node)
    plt.legend(loc="best")
    plt.tight_layout()


# Show a discrete CPD (Y|X,Z) from the MLE fit
plot_discrete_table(lp_disc_mle.discrete_tables["Y"], "P(Y|X,Z) – MLE")

# If you already ran add_data/partial_fit elsewhere, bn.lin_weights['A'] exists.
# Here we’ll quickly touch bn.partial_fit to populate online buffers for plotting:
bn.add_data(df.iloc[:2048], update_params=True)
plot_lg_node(lp_cont_lg, "A", df.iloc[:1000])

plt.show()
