import networkx as nx
import pandas as pd
import torch
from vbn import VBN


def make_df(n=200):
    x0 = torch.randn(n)
    x1 = torch.randn(n)
    x2 = 0.5 * x0 - 0.2 * x1 + 0.1 * torch.randn(n)
    return pd.DataFrame(
        {"feature_0": x0.numpy(), "feature_1": x1.numpy(), "feature_2": x2.numpy()}
    )


def main():
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

    vbn.set_sampling_method(vbn.config.sampling.gibbs, n_samples=20)
    query = {
        "target": "feature_2",
        "evidence": {
            "feature_0": torch.tensor([[0.4]]),
            "feature_1": torch.tensor([[0.1]]),
        },
    }
    samples = vbn.sample(query, n_samples=20)
    print("samples shape:", samples.shape)


if __name__ == "__main__":
    main()
