import os
from pathlib import Path

import networkx as nx
import pandas as pd
import torch
from vbn import defaults, VBN
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
    if os.getenv("CI") and "VBN_SKIP_PLOTS" not in os.environ:
        os.environ["VBN_SKIP_PLOTS"] = "1"
    os.environ.setdefault("MPLBACKEND", "Agg")
    # Directory of the current script
    SCRIPT_DIR = Path(__file__).resolve().parent

    # Create "out" inside the script directory
    OUT_DIR = SCRIPT_DIR / "out"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = make_df()
    g = nx.DiGraph()
    g.add_edges_from([("feature_0", "feature_2"), ("feature_1", "feature_2")])

    vbn = VBN(g, seed=0, device="cpu")
    vbn.set_learning_method(
        method=defaults.learning("node_wise"),
        nodes_cpds={
            "feature_0": defaults.cpd("gaussian_nn"),
            "feature_1": defaults.cpd("gaussian_nn"),
            "feature_2": {**defaults.cpd("mdn"), "n_components": 3},
        },
    )
    vbn.fit(df)

    parents_grid = torch.tensor([[0.0, 0.0], [0.5, -0.5], [1.0, -1.0]])
    handle = vbn.cpd("feature_2")
    plot_cpd_fit(
        handle,
        parents_grid=parents_grid,
        n_samples=256,
        save_path=os.path.join(OUT_DIR, "01_basic_fit_cpd_features2.png"),
    )

    vbn_cat = VBN(g, seed=0, device="cpu")
    vbn_cat.set_learning_method(
        method=defaults.learning("node_wise"),
        nodes_cpds={
            "feature_0": defaults.cpd("softmax_nn"),
            "feature_1": defaults.cpd("softmax_nn"),
            "feature_2": defaults.cpd("softmax_nn"),
        },
    )
    vbn_cat.fit(df)
    cat_samples = vbn_cat.cpd("feature_1").sample(None, 5)
    print("Categorical samples shape:", cat_samples.shape)
    print("Fit complete.")


if __name__ == "__main__":
    main()
