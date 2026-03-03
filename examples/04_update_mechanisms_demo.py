from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
import torch
from _common import auto_device, print_env_header, require_optional, seed_all
from vbn import defaults, UPDATE_REGISTRY, VBN
from vbn.core.utils import ensure_2d

# =========================
# Experiment configuration
# =========================

# Initial (offline) fit dataset size
N0 = 2048

# Stream settings
T = 50  # number of online steps
NB = 1024  # stream batch size per step
EVAL_EVERY = 10  # evaluate every N online steps

# Evaluation settings
DELTA = 1.0  # perturbation for slope estimate
NS = 128  # MC samples for slope estimate
N_VAL = 1_000  # validation size per evaluation

# External "mechanisms" (simulated outside learner) params
ALPHA_EMA = 0.8  # smoothing factor for "ema" metric (NOT model EMA)

# Data-generating process (concept drift)
A0, B0 = 1.5, -0.7  # pre-drift coefficients (used only for initial fit df0)
A1, B1 = -1.5, -0.7  # post-drift coefficients (used for stream + validation)
NOISE_STD = 0.3

# Per-CPD training hyperparams
FIT_CONF = {"epochs": 50, "batch_size": 1024, "lr": 1e-3, "weight_decay": 0.0}
UPDATE_CONF = {"n_steps": 2, "batch_size": NB, "lr": 1e-3, "weight_decay": 0.0}


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Compare update mechanisms under concept drift."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--plot", action="store_true", help="Enable plotting output.")
    parser.add_argument(
        "--verbose", action="store_true", help="Enable progress output."
    )
    return parser.parse_args()


# =========================
# Data generation
# =========================


def make_linear_stream_df(
    n: int, *, a: float, b: float, noise_std: float, gen: torch.Generator
) -> pd.DataFrame:
    x1 = torch.randn(n, generator=gen)
    x2 = torch.randn(n, generator=gen)
    noise = noise_std * torch.randn(n, generator=gen)
    y = a * x1 + b * x2 + noise
    return pd.DataFrame({"x1": x1.numpy(), "x2": x2.numpy(), "y": y.numpy()})


# =========================
# VBN construction
# =========================


def build_nodes_cpds() -> dict:
    base = copy.deepcopy(defaults.cpd("softmax_nn"))
    base.setdefault("fit", {})
    base.setdefault("update", {})
    base["fit"] = {**base["fit"], **FIT_CONF}
    base["update"] = {**base["update"], **UPDATE_CONF}
    return {
        "x1": copy.deepcopy(base),
        "x2": copy.deepcopy(base),
        "y": copy.deepcopy(base),
    }


def build_vbn(*, device: str, seed: int) -> VBN:
    g = nx.DiGraph()
    g.add_edges_from([("x1", "y"), ("x2", "y")])
    vbn = VBN(g, seed=seed, device=device)
    vbn.set_learning_method(
        method=defaults.learning("node_wise"),
        nodes_cpds=build_nodes_cpds(),
    )
    return vbn


# =========================
# Metrics
# =========================


def estimate_slope(vbn: VBN, *, delta: float, n_samples: int) -> float:
    """
    Slope estimate via two interventions on x1:
      a_hat = (E[y|x1=+d,x2=0] - E[y|x1=-d,x2=0]) / (2d)
    """
    cpd = vbn.cpd("y")
    parents_pos = {
        "x1": torch.tensor([[delta]], device=vbn.device),
        "x2": torch.tensor([[0.0]], device=vbn.device),
    }
    parents_neg = {
        "x1": torch.tensor([[-delta]], device=vbn.device),
        "x2": torch.tensor([[0.0]], device=vbn.device),
    }

    with torch.no_grad():
        y_pos = cpd.sample(parents_pos, n_samples)
        y_neg = cpd.sample(parents_neg, n_samples)
        slope = (y_pos.mean() - y_neg.mean()) / (2.0 * delta)
    return float(slope)


def estimate_nll(vbn: VBN, df_val: pd.DataFrame) -> float:
    cpd = vbn.cpd("y")
    parents = {
        "x1": torch.tensor(df_val["x1"].values, device=vbn.device),
        "x2": torch.tensor(df_val["x2"].values, device=vbn.device),
    }
    y = torch.tensor(df_val["y"].values, device=vbn.device)
    y = ensure_2d(y).unsqueeze(1)
    with torch.no_grad():
        log_prob = cpd.log_prob(y, parents)
        return float((-log_prob).mean())


def estimate_nll_gt(
    df_val: pd.DataFrame, *, a: float, b: float, noise_std: float
) -> float:
    x1 = df_val["x1"].values
    x2 = df_val["x2"].values
    y = df_val["y"].values
    mu = a * x1 + b * x2
    var = noise_std**2
    log_prob = -0.5 * (
        ((y - mu) ** 2) / var + 2.0 * np.log(noise_std) + np.log(2.0 * np.pi)
    )
    return float(-log_prob.mean())


# =========================
# Main demo
# =========================


def main() -> None:
    args = _parse_args()
    seed_all(args.seed)
    device = auto_device()
    print_env_header("04_update_mechanisms_demo", device)

    if args.verbose:
        print("Using CPD 'gaussian_nn' on device", device)

    mechanisms = list(UPDATE_REGISTRY.keys())

    data_gen = torch.Generator().manual_seed(args.seed)
    val_gen = torch.Generator().manual_seed(args.seed + 123)
    df0 = make_linear_stream_df(N0, a=A0, b=B0, noise_std=NOISE_STD, gen=data_gen)
    stream_batches = [
        make_linear_stream_df(NB, a=A1, b=B1, noise_std=NOISE_STD, gen=data_gen)
        for _ in range(T)
    ]

    eval_steps = [0] + [t for t in range(1, T + 1) if (t % EVAL_EVERY) == 0]
    val_batches = [
        make_linear_stream_df(N_VAL, a=A1, b=B1, noise_std=NOISE_STD, gen=val_gen)
        for _ in eval_steps
    ]

    a_gt = [A0 if step == 0 else A1 for step in eval_steps]
    nll_gt = [
        estimate_nll_gt(dfv, a=A1, b=B1, noise_std=NOISE_STD) for dfv in val_batches
    ]

    base_vbn = build_vbn(device=device, seed=args.seed)
    base_vbn.fit(df0, verbosity=int(args.verbose))

    learner_update_method = defaults.update("online_sgd")

    results = {m: {"steps": [], "a_hat": [], "nll": []} for m in mechanisms}

    tqdm = None
    if args.verbose:
        try:
            from tqdm import tqdm as _tqdm

            tqdm = _tqdm
        except Exception:
            tqdm = None

    for mech in mechanisms:
        vbn = copy.deepcopy(base_vbn)
        ema_metric: Optional[float] = None

        eval_i = 0
        a_raw = estimate_slope(vbn, delta=DELTA, n_samples=NS)
        nll = estimate_nll(vbn, val_batches[eval_i])
        eval_i += 1

        if mech == "ema":
            ema_metric = a_raw
            a_store = ema_metric
        else:
            a_store = a_raw

        results[mech]["steps"].append(0)
        results[mech]["a_hat"].append(a_store)
        results[mech]["nll"].append(nll)

        step_iter = range(1, T + 1)
        if tqdm is not None:
            step_iter = tqdm(step_iter, desc=f"{mech} updates")

        for t in step_iter:
            df_t = stream_batches[t - 1]
            vbn.update(df_t, update_method=learner_update_method, verbosity=0)

            if (t % EVAL_EVERY) == 0:
                a_raw = estimate_slope(vbn, delta=DELTA, n_samples=NS)
                nll = estimate_nll(vbn, val_batches[eval_i])
                eval_i += 1

                if mech == "ema":
                    ema_metric = (
                        a_raw
                        if ema_metric is None
                        else (ALPHA_EMA * ema_metric + (1.0 - ALPHA_EMA) * a_raw)
                    )
                    a_store = ema_metric
                else:
                    a_store = a_raw

                results[mech]["steps"].append(t)
                results[mech]["a_hat"].append(a_store)
                results[mech]["nll"].append(nll)

                if tqdm is not None:
                    step_iter.set_postfix(a_hat=f"{a_store:.3f}", nll=f"{nll:.3f}")

    if args.plot:
        os.environ.setdefault("MPLBACKEND", "Agg")
        require_optional("matplotlib.pyplot", "plotting")
        import matplotlib.pyplot as plt

        out_dir = Path(__file__).resolve().parent / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "04_update_mechanisms_demo.png"

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for mech in mechanisms:
            axes[0].plot(results[mech]["steps"], results[mech]["a_hat"], label=mech)
            axes[1].plot(results[mech]["steps"], results[mech]["nll"], label=mech)

        axes[0].plot(eval_steps, a_gt, "k--", label="ground truth")
        axes[1].plot(
            [s for s in eval_steps if s >= 1],
            [v for s, v in zip(eval_steps, nll_gt) if s >= 1],
            "k--",
            label="ground truth",
        )

        drift_x = 0.5
        for ax in axes:
            ax.axvline(drift_x, color="black", linestyle=":", linewidth=1.0, alpha=0.5)

        axes[0].set_title("Estimated slope a_hat(t)")
        axes[0].set_xlabel("Update step")
        axes[0].set_ylabel("a_hat")

        axes[1].set_title("Current-regime NLL")
        axes[1].set_xlabel("Update step")
        axes[1].set_ylabel("NLL")

        for ax in axes:
            ax.legend()

        fig.tight_layout()
        fig.savefig(out_path, dpi=500)
        plt.close(fig)
        if args.verbose:
            print(f"Saved plot: {out_path}")

    final_step = results[mechanisms[0]]["steps"][-1]
    print(
        f"\nGround truth: a0={A0:+.3f} | a1={A1:+.3f} | b={B1:+.3f} | noise_std={NOISE_STD:.3f}"
    )
    print(f"\nFinal metrics at step {final_step}:")
    for mech in mechanisms:
        a_last = results[mech]["a_hat"][-1]
        nll_last = results[mech]["nll"][-1]
        print(f"{mech:>14} | a_hat={a_last:+.3f} | a1={A1:+.3f} | nll={nll_last:.3f}")


if __name__ == "__main__":
    main()
