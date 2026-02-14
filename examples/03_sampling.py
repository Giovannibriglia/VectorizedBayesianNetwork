import os

import networkx as nx
import pandas as pd
import torch
from vbn import defaults, VBN
from vbn.display import plot_sampling_outcome


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
        method=defaults.learning("node_wise"),
        nodes_cpds={
            "feature_0": defaults.cpd("softmax_nn"),
            "feature_1": defaults.cpd("softmax_nn"),
            "feature_2": {**defaults.cpd("mdn"), "n_components": 3},
        },
    )
    vbn.fit(df)

    vbn.set_sampling_method(defaults.sampling("gibbs"), n_samples=50)
    query = {
        "target": "feature_2",
        "evidence": {
            "feature_0": torch.tensor([[0.4], [0.8]]),
            "feature_1": torch.tensor([[0.1], [0.2]]),
        },
    }
    samples = vbn.sample(query, n_samples=50)
    assert not samples.requires_grad
    plot_sampling_outcome(
        samples,
        batch_index=0,
        save_path="out/03_sampling_outcome.png",
    )
    print("Sampling plot saved to out/sampling_outcome.png")


if __name__ == "__main__":
    main()
