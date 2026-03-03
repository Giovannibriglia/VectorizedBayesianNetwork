import argparse
import os
from pathlib import Path

import networkx as nx
import pandas as pd
import torch
from _common import auto_device, print_env_header, require_optional, seed_all
from vbn import defaults, VBN


def make_df(n=200, seed=0):
    gen = torch.Generator().manual_seed(seed)
    x0 = torch.randn(n, generator=gen)
    x1 = torch.randn(n, generator=gen)
    x2 = 0.5 * x0 - 0.2 * x1 + 0.1 * torch.randn(n, generator=gen)
    return pd.DataFrame(
        {"feature_0": x0.numpy(), "feature_1": x1.numpy(), "feature_2": x2.numpy()}
    )


def _parse_args():
    parser = argparse.ArgumentParser(description="Posterior inference example.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--plot", action="store_true", help="Enable plotting output.")
    return parser.parse_args()


def main():
    args = _parse_args()
    seed_all(args.seed)
    device = auto_device()
    print_env_header("02_infer_posterior", device)

    df = make_df(seed=args.seed)
    g = nx.DiGraph()
    g.add_edges_from([("feature_0", "feature_2"), ("feature_1", "feature_2")])

    fit_conf = {"epochs": 8, "batch_size": 256, "lr": 1e-3, "weight_decay": 0.0}
    vbn = VBN(g, seed=args.seed, device=device)
    vbn.set_learning_method(
        method=defaults.learning("node_wise"),
        nodes_cpds={
            "feature_0": {**defaults.cpd("gaussian_nn"), "fit": dict(fit_conf)},
            "feature_1": {**defaults.cpd("gaussian_nn"), "fit": dict(fit_conf)},
            "feature_2": {
                **defaults.cpd("mdn"),
                "n_components": 3,
                "fit": dict(fit_conf),
            },
        },
    )
    vbn.fit(df)

    vbn.set_inference_method(
        defaults.inference("monte_carlo_marginalization"), n_samples=128
    )
    query = {
        "target": "feature_2",
        "evidence": {
            "feature_0": torch.tensor([[0.2]], device=device),
            "feature_1": torch.tensor([[-0.1]], device=device),
        },
    }
    pdf, samples = vbn.infer_posterior(query)
    assert not pdf.requires_grad and not samples.requires_grad
    if args.plot:
        os.environ.setdefault("MPLBACKEND", "Agg")
        require_optional("matplotlib.pyplot", "plotting")
        from vbn.display import plot_inference_posterior

        out_dir = Path(__file__).resolve().parent / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_inference_posterior(
            pdf,
            samples,
            save_path=os.path.join(out_dir, "02_inference_posterior.png"),
        )
    print("Posterior computed.")


if __name__ == "__main__":
    main()
