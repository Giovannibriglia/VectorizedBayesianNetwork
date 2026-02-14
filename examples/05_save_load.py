import os

import networkx as nx
import pandas as pd
import torch
from vbn import VBN
from vbn.display import plot_inference_posterior, plot_sampling_outcome


def make_df(n=200, seed=0):
    gen = torch.Generator().manual_seed(seed)
    x0 = torch.randn(n, generator=gen)
    x1 = torch.randn(n, generator=gen)
    x2 = 0.5 * x0 - 0.2 * x1 + 0.1 * torch.randn(n, generator=gen)
    return pd.DataFrame(
        {"feature_0": x0.numpy(), "feature_1": x1.numpy(), "feature_2": x2.numpy()}
    )


def main():
    os.makedirs("examples/out", exist_ok=True)
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

    vbn.set_inference_method(
        vbn.config.inference.monte_carlo_marginalization, n_samples=50
    )
    vbn.set_sampling_method(vbn.config.sampling.gibbs, n_samples=50)

    model_path = "examples/out/model.pt"
    vbn.save(model_path)
    print(f"Saved model to {model_path}")

    loaded = VBN.load(model_path, map_location="cpu")
    query = {
        "target": "feature_2",
        "evidence": {
            "feature_0": torch.tensor([[0.2]]),
            "feature_1": torch.tensor([[-0.1]]),
        },
    }
    pdf, samples = loaded.infer_posterior(query)
    print("loaded pdf shape:", pdf.shape)
    print("loaded samples shape:", samples.shape)
    plot_inference_posterior(
        pdf,
        samples,
        save_path="examples/out/loaded_inference_posterior.png",
    )

    samp = loaded.sample(query, n_samples=50)
    print("loaded sampling shape:", samp.shape)
    plot_sampling_outcome(
        samp,
        save_path="examples/out/loaded_sampling_outcome.png",
    )
    print("Loaded model plots saved under examples/out/")


if __name__ == "__main__":
    main()
