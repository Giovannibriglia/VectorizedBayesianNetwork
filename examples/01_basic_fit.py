import os

import networkx as nx
import pandas as pd
import torch
from vbn import VBN
from vbn.display import plot_cpd_fit


def make_df(n=200, seed=0):
    gen = torch.Generator().manual_seed(seed)
    x0 = torch.randn(n, generator=gen)
    x1 = torch.randn(n, generator=gen)
    x2 = 0.5 * x0 - 0.2 * x1 + 0.1 * torch.randn(n, generator=gen)
    return pd.DataFrame(
        {"feature_0": x0.numpy(), "feature_1": x1.numpy(), "feature_2": x2.numpy()}
    )


def main():
    os.makedirs("out", exist_ok=True)
    df = make_df()
    g = nx.DiGraph()
    g.add_edges_from([("feature_0", "feature_2"), ("feature_1", "feature_2")])

    vbn = VBN(g, seed=0, device="cpu")
    vbn.set_learning_method(
        method=vbn.config.learning.node_wise,
        nodes_cpds={
            "feature_0": {"cpd": "softmax_nn"},
            "feature_1": {"cpd": "softmax_nn"},
            "feature_2": {"cpd": "mdn", "n_components": 3},
        },
    )
    vbn.fit(df)

    parents_grid = torch.tensor([[0.0, 0.0], [1.0, -1.0]])
    handle = vbn.cpd("feature_2")
    plot_cpd_fit(
        handle,
        parents_grid=parents_grid,
        n_samples=512,
        save_path="out/cpd_feature_2.png",
    )
    print("Fit complete. CPD plot saved to out/cpd_feature_2.png")


if __name__ == "__main__":
    main()
