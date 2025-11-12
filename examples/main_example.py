#!/usr/bin/env python3
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from vbn import VBN


# ----------------------------- data ---------------------------------
def make_categorical_df(N=2000, K=5, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.integers(0, K, size=N)
    b = rng.integers(0, K, size=N)
    # child depends on parents with noise (mod K) – real signal present
    c = (a + 2 * b + rng.integers(0, K // 2 + 1, size=N)) % K
    return pd.DataFrame({"feature_0": a, "feature_1": b, "feature_2": c})


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

    # -------- model: mle_softmax (linear classifier MLE) --------
    vbn_mle = VBN(G, seed=0, device=device)
    vbn_mle.set_learning_method(
        "mle_softmax",
        num_classes=K,  # mapped to num_classes inside the CPD
        epochs=50,
        batch_size=512,
    )
    vbn_mle.fit(df_train)
    vbn_mle.set_inference_method("exact.ve")
    pdf, samples = vbn_mle.infer_posterior("feature_2")
    print(pdf["weights"].shape, samples)

    # knowledge update
    vbn_mle.update(df_update)

    vbn_mle.set_inference_method("exact.ve")
    pdf, samples = vbn_mle.infer_posterior("feature_2")
    print(pdf["weights"].shape, samples)
