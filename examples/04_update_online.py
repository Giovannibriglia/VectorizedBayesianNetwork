import os
from pathlib import Path

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
    if os.getenv("CI") and "VBN_SKIP_PLOTS" not in os.environ:
        os.environ["VBN_SKIP_PLOTS"] = "1"
    os.environ.setdefault("MPLBACKEND", "Agg")
    # Directory of the current script
    SCRIPT_DIR = Path(__file__).resolve().parent

    # Create "out" inside the script directory
    OUT_DIR = SCRIPT_DIR / "out"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = make_df(n=2000)
    g = nx.DiGraph()
    g.add_edges_from([("feature_0", "feature_2"), ("feature_1", "feature_2")])

    vbn = VBN(g, seed=0, device="cpu")
    learning_conf = {**defaults.learning("node_wise"), "epochs": 5, "batch_size": 64}
    vbn.set_learning_method(
        method=learning_conf,
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
        save_path=os.path.join(OUT_DIR, "04_inference_posterior.png"),
    )

    new_df = make_df(500)
    vbn.update(
        new_df, defaults.update("ema")
    )  # vbn.update(new_df, update_method="ema", lr=1e-3, n_steps=2)
    print("Update complete")

    pdf, samples = vbn.infer_posterior(query)
    plot_inference_posterior(
        pdf,
        samples,
        save_path=os.path.join(OUT_DIR, "04_inference_posterior_after_refit.png"),
    )


if __name__ == "__main__":
    main()
