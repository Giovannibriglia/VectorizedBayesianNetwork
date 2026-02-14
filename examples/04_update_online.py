import networkx as nx
import pandas as pd
import torch
from vbn import defaults, VBN
from vbn.display import plot_inference_posterior


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
        method=defaults.learning("node_wise"),
        nodes_cpds={
            "feature_0": defaults.cpd("softmax_nn"),
            "feature_1": defaults.cpd("softmax_nn"),
            "feature_2": {**defaults.cpd("mdn"), "n_components": 3},
        },
    )
    vbn.fit(df)

    vbn.set_inference_method(
        defaults.inference("monte_carlo_marginalization"), n_samples=16
    )
    query = {
        "target": "feature_2",
        "evidence": {
            "feature_0": torch.tensor([[0.2]]),
            "feature_1": torch.tensor([[-0.1]]),
        },
    }
    pdf, samples = vbn.infer_posterior(query)
    plot_inference_posterior(
        pdf,
        samples,
        save_path="out/04_inference_posterior.png",
    )

    new_df = make_df(200)
    vbn.update(new_df, update_method="online_sgd", lr=1e-3, n_steps=2)
    print("Update complete")

    pdf, samples = vbn.infer_posterior(query)
    plot_inference_posterior(
        pdf,
        samples,
        save_path="out/04_inference_posterior_after_refit.png",
    )


if __name__ == "__main__":
    main()
