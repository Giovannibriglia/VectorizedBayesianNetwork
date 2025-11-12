#!/usr/bin/env python3
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import torch

from vbn import VBN  # your class using LEARNING_METHODS
from vbn.io import load_vbn, save_vbn


# ----------------------------- data ---------------------------------
def make_categorical_df(N=2000, K=5, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.integers(0, K, size=N)
    b = rng.integers(0, K, size=N)
    # child depends on parents with noise (mod K) – real signal present
    c = (a + 2 * b + rng.integers(0, K // 2 + 1, size=N)) % K
    return pd.DataFrame({"feature_0": a, "feature_1": b, "feature_2": c})


# ----------------------------- evals --------------------------------
@torch.no_grad()
def eval_nll(vbn: VBN, df: pd.DataFrame, child: str, device: str) -> float:
    # build parent matrix X and target y for the child node
    parents = list(vbn.dag.predecessors(child))
    X_np = df[parents].values.astype(np.float32)
    y_np = df[child].values.astype(np.int64).reshape(-1, 1)

    X = torch.tensor(X_np, device=device)
    y = torch.tensor(y_np, device=device)
    head = vbn._nodes[child]
    nll = -head.log_prob(X, y).mean().item()
    return float(nll)


@torch.no_grad()
def show_samples(vbn: VBN, df: pd.DataFrame, child: str, n_inputs=3, n_samples=5):
    parents = list(vbn.dag.predecessors(child))
    X_np = df[parents].values[:n_inputs].astype(np.float32)
    X = torch.tensor(X_np, device=vbn.device)
    y_s = vbn._nodes[child].sample(X, n=n_samples).detach().cpu().numpy()  # [n, B]
    print(
        f"Samples for {child} given first {n_inputs} parent rows (shape {y_s.shape}):"
    )
    print(y_s)


# ----------------------------- main ---------------------------------
if __name__ == "__main__":
    K = 5
    device = "cuda"

    # DAG: feature_0, feature_1 → feature_2
    G = nx.DiGraph()
    G.add_edges_from([("feature_0", "feature_2"), ("feature_1", "feature_2")])

    # train / update splits
    df_train = make_categorical_df(N=200, K=K, seed=0)
    df_update = make_categorical_df(N=4000, K=K, seed=1)
    df_test = make_categorical_df(N=8000, K=K, seed=2)

    # -------- model A: softmax_nn (deep classifier) --------
    vbn_nn = VBN(G, seed=0, device=device)
    vbn_nn.set_learning_method(
        "softmax_nn",
        num_classes=K,  # mapped to out_dim internally
        hidden=128,
        depth=2,
        epochs=10,
        batch_size=512,
        weight_decay=0.0,
        lr=1e-3,
    )
    vbn_nn.fit(df_train)
    nll_nn_before = eval_nll(vbn_nn, df_test, child="feature_2", device=vbn_nn.device)
    print(f"[softmax_nn] test NLL before update: {nll_nn_before:.4f}")

    # knowledge update
    vbn_nn.update(df_update)
    nll_nn_after = eval_nll(vbn_nn, df_test, child="feature_2", device=vbn_nn.device)
    print(f"[softmax_nn] test NLL after  update: {nll_nn_after:.4f}")
    show_samples(vbn_nn, df_test, child="feature_2", n_inputs=3, n_samples=5)

    # -------- model B: mle_softmax (linear classifier MLE) --------
    vbn_mle = VBN(G, seed=0, device=device)
    vbn_mle.set_learning_method(
        "mle_softmax",
        num_classes=K,  # mapped to num_classes inside the CPD
        lr=0.1,
        epochs=15,
        batch_size=512,
        weight_decay=0.0,
    )
    vbn_mle.fit(df_train)
    nll_mle_before = eval_nll(
        vbn_mle, df_test, child="feature_2", device=vbn_mle.device
    )
    print(f"[mle_softmax] test NLL before update: {nll_mle_before:.4f}")

    # knowledge update
    vbn_mle.update(df_update)
    nll_mle_after = eval_nll(vbn_mle, df_test, child="feature_2", device=vbn_mle.device)
    print(f"[mle_softmax] test NLL after  update: {nll_mle_after:.4f}")
    show_samples(vbn_mle, df_test, child="feature_2", n_inputs=3, n_samples=5)

    # -------- compact comparison --------
    print("\n=== Comparison (lower NLL is better) ===")
    print(f"softmax_nn  : before {nll_nn_before:.4f}  | after {nll_nn_after:.4f}")
    print(f"mle_softmax : before {nll_mle_before:.4f} | after {nll_mle_after:.4f}")

    # save
    save_vbn(vbn_mle, "checkpoints/my_bn.pt")

    # load
    vbn2 = load_vbn("checkpoints/my_bn.pt", map_location="cuda")
    nll_mle_after = eval_nll(vbn2, df_test, child="feature_2", device=vbn_mle.device)
    print(f"[mle_softmax] test NLL after  update: {nll_mle_after:.4f}")
    show_samples(vbn2, df_test, child="feature_2", n_inputs=3, n_samples=5)
