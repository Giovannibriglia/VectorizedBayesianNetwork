import os
from pathlib import Path

import networkx as nx
import pandas as pd
import torch
from tqdm import tqdm
from vbn import CPD_REGISTRY, defaults, VBN

K = 5


def make_df(n=1000, seed=0):
    gen = torch.Generator().manual_seed(seed)
    x1 = torch.randn(n, generator=gen)
    x2 = torch.randn(n, generator=gen)

    slopes = torch.linspace(0.0, 4.0, K)
    x1_boundaries = torch.tensor([-2.0, -0.5, 0.8, 2.0])
    intercepts = torch.zeros(K)
    for i in range(K - 1):
        intercepts[i + 1] = intercepts[i] - x1_boundaries[i]
    x2_weights = torch.tensor([0.3, -0.2, 0.0, 0.2, -0.3])

    logits = (
        x1[:, None] * slopes[None, :]
        + x2[:, None] * x2_weights[None, :]
        + intercepts[None, :]
    )
    y = torch.distributions.Categorical(logits=logits).sample()

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


def _cpd_config(cpd_key: str) -> dict:
    try:
        if cpd_key != "softmax_nn":
            return defaults.cpd(cpd_key)
        else:
            return {**defaults.cpd("softmax_nn"), "n_classes": K}
    except Exception:
        return {"cpd": CPD_REGISTRY[cpd_key]}


def _samples_to_labels(samples: torch.Tensor) -> torch.Tensor:
    if samples.dim() == 3 and samples.shape[-1] == 1:
        samples = samples.squeeze(-1)
    elif samples.dim() > 3:
        raise ValueError(f"Unexpected sample shape {tuple(samples.shape)}")
    labels = torch.round(samples.float()).long()
    return labels.clamp(0, K - 1)


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

    out_path = None
    if not skip_plots:
        out_dir = Path(__file__).resolve().parent / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "01_categorical_cpds_benchmark.png"

    df = make_df(n=5000, seed=0)
    g = nx.DiGraph()
    g.add_edges_from([("x1", "y"), ("x2", "y")])

    x1_grid = torch.linspace(-3.0, 3.0, steps=80)
    cpd_keys = sorted(CPD_REGISTRY.keys())
    results = {}

    pbar = tqdm(cpd_keys, desc="training...")
    for cpd_key in pbar:
        pbar.set_postfix(cpd_key=cpd_key)

        vbn = VBN(g, seed=0, device="cpu")

        nodes_cpds = {node: _cpd_config(cpd_key) for node in ("x1", "x2", "y")}
        vbn.set_learning_method(
            method=defaults.learning("node_wise"), nodes_cpds=nodes_cpds
        )
        try:
            vbn.fit(df, verbosity=0)
            handle = vbn.cpd("y")
            parents_grid = torch.zeros(x1_grid.shape[0], len(handle.parents))
            for idx, parent in enumerate(handle.parents):
                if parent == "x1":
                    parents_grid[:, idx] = x1_grid
                elif parent == "x2":
                    parents_grid[:, idx] = 0.0
                else:
                    raise ValueError(f"Unexpected parent '{parent}' for node 'y'")
            with torch.no_grad():
                samples = handle.sample(parents_grid, n_samples=500)
            labels = _samples_to_labels(samples.detach().cpu())
            probs = torch.zeros(x1_grid.shape[0], K)
            for k in range(K):
                probs[:, k] = (labels == k).float().mean(dim=1)
            results[cpd_key] = probs
            # print(f"{cpd_key}: fit ok, sample shape {tuple(samples.shape)}")
        except Exception as exc:
            print(f"{cpd_key}: skipped ({exc})")
            continue

    if skip_plots:
        return
    if not results:
        print("No CPDs succeeded; nothing to plot.")
        return

    fig, axes = plt.subplots(nrows=K, ncols=1, figsize=(7, 2.2 * K), sharex=True)
    x_vals = x1_grid.detach().cpu().numpy()
    for k in range(K):
        ax = axes[k]
        for cpd_key, probs in results.items():
            ax.plot(x_vals, probs[:, k].numpy(), label=cpd_key)
        ax.set_ylabel(f"P(y={k})")
        if k == 0:
            ax.legend(fontsize=7, ncol=2)
        if k == K - 1:
            ax.set_xlabel("x1")
    fig.suptitle("Categorical CPD comparison for node y (x2=0 slice)")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, bbox_inches="tight", dpi=500)
    plt.close(fig)


if __name__ == "__main__":
    main()
