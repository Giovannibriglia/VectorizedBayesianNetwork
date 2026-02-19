import os
from pathlib import Path

import networkx as nx
import pandas as pd
import torch
from tqdm import tqdm
from vbn import CPD_REGISTRY, defaults, VBN


def make_df(n=1000, seed=0):
    gen = torch.Generator().manual_seed(seed)
    x1 = torch.randn(n, generator=gen)
    x2 = torch.randn(n, generator=gen)
    noise = 0.3 * torch.randn(n, generator=gen)
    y = 1.5 * x1 - 0.7 * x2 + noise
    return pd.DataFrame({"x1": x1.numpy(), "x2": x2.numpy(), "y": y.numpy()})


def _skip_plots() -> bool:
    flag = os.getenv("VBN_SKIP_PLOTS", "0").strip().lower()
    return flag in {"1", "true", "yes"}


def _maybe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except Exception:
        if not os.getenv("CI"):
            print(
                "matplotlib is not installed; skipping plots. "
                "Install it with 'pip install matplotlib'."
            )
        return None
    return plt


def main():
    if os.getenv("CI") and "VBN_SKIP_PLOTS" not in os.environ:
        os.environ["VBN_SKIP_PLOTS"] = "1"
    os.environ.setdefault("MPLBACKEND", "Agg")

    skip_plots = _skip_plots()
    plt = None
    if not skip_plots:
        plt = _maybe_import_matplotlib()
        if plt is None:
            skip_plots = True

    out_dir = None
    if not skip_plots:
        out_dir = Path(__file__).resolve().parent / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

    df = make_df(n=5000, seed=0)

    g = nx.DiGraph()
    g.add_edges_from([("x1", "y"), ("x2", "y")])

    cpd_keys = sorted(CPD_REGISTRY.keys())
    if not cpd_keys:
        print("No CPDs found in registry.")
        return

    x1_grid = torch.linspace(-2.5, 2.5, steps=8)

    # Store results for all CPDs
    results = {}

    pbar = tqdm(cpd_keys, desc="training...")
    for cpd_key in pbar:
        pbar.set_postfix(cpd_key=cpd_key)

        vbn = VBN(g, seed=0, device="cpu")
        nodes_cpds = {node: defaults.cpd(cpd_key) for node in ("x1", "x2", "y")}
        vbn.set_learning_method(
            method=defaults.learning("node_wise"), nodes_cpds=nodes_cpds
        )
        vbn.fit(df, verbosity=0)

        handle = vbn.cpd("y")

        parents_grid = torch.zeros(x1_grid.shape[0], len(handle.parents))
        for idx, parent in enumerate(handle.parents):
            if parent == "x1":
                parents_grid[:, idx] = x1_grid
            else:
                parents_grid[:, idx] = 0.0

        with torch.no_grad():
            samples = handle.sample(parents_grid, n_samples=200)

        if samples.dim() == 2:
            samples = samples.unsqueeze(0)

        mean = samples.mean(dim=1).squeeze(-1)
        std = samples.std(dim=1).squeeze(-1)

        results[cpd_key] = {
            "mean": mean.detach().cpu().numpy(),
            "std": std.detach().cpu().numpy(),
        }

        # print(f"{cpd_key}: fit ok, sample shape {tuple(samples.shape)}")

    if skip_plots:
        return

    # === SINGLE FIGURE ===
    fig = plt.figure(dpi=500)

    x_vals = x1_grid.detach().cpu().numpy()

    for idx, (cpd_key, stats) in enumerate(results.items()):
        color = f"C{idx % 10}"

        mean = stats["mean"]
        std = stats["std"]

        plt.plot(x_vals, mean, color=color, label=cpd_key)
        plt.fill_between(
            x_vals,
            mean - std,
            mean + std,
            color=color,
            alpha=0.15,
        )

    plt.title("CPD comparison for node y (x2 = 0 slice)")
    plt.xlabel("x1")
    plt.ylabel("y")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "01_continuous_cpds_benchmark.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
