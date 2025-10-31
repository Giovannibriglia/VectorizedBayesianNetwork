# vbn/plotting.py
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

Tensor = torch.Tensor

# --------------------------- helpers ---------------------------


def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _with_ci(
    mean: np.ndarray,
    std: Optional[np.ndarray] = None,
    n: Optional[int] = None,
    z: float = 1.96,
):
    if std is None:  # assume bernoulli variance for proportions
        p = np.clip(mean, 1e-9, 1 - 1e-9)
        var = p * (1 - p)
        if n is None:
            n = 1
        std = np.sqrt(var / max(n, 1))
    return mean, z * std


# --------------------------- LEARNING ---------------------------


def plot_learning_curves(
    log: Dict[str, Sequence[float]], title: str = "Learning Curves"
):
    """
    log: {"train_nll":[...], "val_nll":[...], "train_mae":[...], ...}
    """
    plt.figure()
    for k, v in log.items():
        plt.plot(v, label=k)
    plt.xlabel("epoch")
    plt.ylabel("metric")
    plt.title(title)
    plt.legend()
    plt.tight_layout()


def plot_cpd_discrete_heatmap(
    cpt: Tensor, parent_cards: Sequence[int], node_card: int, title="CPD heatmap"
):
    """
    Displays a CPT as a heatmap. Rows are parent assignments (flattened), columns are child categories.
    cpt: [prod(parent_cards), K] or canonical viewable to that shape
    """
    K = int(node_card)
    prodC = int(np.prod(parent_cards)) if len(parent_cards) else 1
    M = _to_np(cpt).reshape(prodC, K)
    plt.figure()
    plt.imshow(M, aspect="auto")
    plt.colorbar()
    plt.xlabel("child category")
    plt.ylabel("parent assignment (flattened)")
    plt.title(title)
    plt.tight_layout()


def plot_calibration(
    prob: np.ndarray, y_true: np.ndarray, n_bins: int = 10, title="Calibration"
):
    """
    Reliability diagram for binary prob. predictions.
    prob: predicted P(Y=1)
    y_true: {0,1}
    """
    prob = _to_np(prob).ravel()
    y = _to_np(y_true).ravel()
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(prob, bins) - 1
    bin_centers, acc, conf = [], [], []
    for b in range(n_bins):
        m = idx == b
        if not np.any(m):
            continue
        p_hat = prob[m].mean()
        a_hat = y[m].mean()
        bin_centers.append(p_hat)
        acc.append(a_hat)
        conf.append(p_hat)
    plt.figure()
    plt.plot([0, 1], [0, 1], "--")
    plt.scatter(conf, acc)
    plt.xlabel("predicted probability")
    plt.ylabel("empirical accuracy")
    plt.title(title)
    plt.tight_layout()


# --------------------------- INFERENCE ---------------------------


def plot_posterior_bars(
    exact: np.ndarray,
    approx: Dict[str, np.ndarray],
    labels: Optional[Sequence[str]] = None,
    title: str = "Posterior comparison",
):
    """
    Bar chart comparing categorical posterior across methods vs exact baseline.
    exact: [K]
    approx: {"lw":[K], "smc":[K], "lbp":[K], ...}
    """
    exact = _to_np(exact).ravel()
    K = exact.shape[0]
    if labels is None:
        labels = [f"k={i}" for i in range(K)]

    methods = ["exact"] + list(approx.keys())
    X = np.arange(K)
    w = 0.8 / len(methods)

    plt.figure()
    plt.bar(X - 0.4 + w / 2, exact, width=w, label="exact")
    for i, (name, p) in enumerate(approx.items(), start=1):
        plt.bar(X - 0.4 + w / 2 + i * w, _to_np(p).ravel(), width=w, label=name)
    plt.xticks(X, labels)
    plt.ylabel("probability")
    plt.title(title)
    plt.legend()
    plt.tight_layout()


def plot_speed_accuracy_frontier(
    rows: List[Tuple[str, float, float]], title="Speed–Accuracy Frontier", xlog=True
):
    """
    rows: list of (method, time_seconds, kl_to_exact)
    """
    methods, times, kls = zip(*rows)
    times = np.array(times)
    kls = np.array(kls)
    plt.figure()
    plt.scatter(times, kls)
    for m, t, k in rows:
        plt.annotate(m, (t, k))
    if xlog:
        plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("time [s]")
    plt.ylabel("KL(exact || method)")
    plt.title(title)
    plt.tight_layout()


def plot_lbp_convergence(residuals: Sequence[float], title="LBP convergence"):
    plt.figure()
    plt.semilogy(residuals)
    plt.xlabel("iteration")
    plt.ylabel("message residual (log)")
    plt.title(title)
    plt.tight_layout()


def plot_smc_ess(ess_traj: Sequence[float], N: int, title="SMC ESS trajectory"):
    plt.figure()
    plt.plot(ess_traj)
    plt.axhline(0.5 * N, linestyle="--")  # threshold
    plt.xlabel("topological step")
    plt.ylabel("ESS")
    plt.title(title)
    plt.tight_layout()


# --------------------------- SAMPLING ---------------------------


def plot_discrete_freq_vs_truth(
    freqs: np.ndarray, truth: np.ndarray, title="Discrete: sample freq vs truth"
):
    """
    freqs, truth: [K]
    """
    freqs = _to_np(freqs).ravel()
    truth = _to_np(truth).ravel()
    X = np.arange(len(freqs))
    w = 0.35
    plt.figure()
    plt.bar(X - w / 2, truth, width=w, label="truth")
    plt.bar(X + w / 2, freqs, width=w, label="samples")
    plt.ylabel("probability")
    plt.xticks(X, [f"k={i}" for i in X])
    plt.title(title)
    plt.legend()
    plt.tight_layout()


def plot_continuous_pdf_vs_samples(
    x_grid: np.ndarray,
    pdf: np.ndarray,
    samples: np.ndarray,
    title="Continuous: pdf vs samples",
):
    """
    Plots analytic 1D pdf and histogram of samples.
    """
    x_grid = _to_np(x_grid)
    pdf = _to_np(pdf)
    samples = _to_np(samples).ravel()
    plt.figure()
    plt.plot(x_grid, pdf, label="analytic")
    plt.hist(samples, bins=40, density=True, alpha=0.4, label="samples")
    plt.xlabel("x")
    plt.ylabel("density")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()


def plot_rmse_vs_n(
    ns: Sequence[int],
    rmses_mc: Sequence[float],
    rmses_qmc: Optional[Sequence[float]] = None,
    title="RMSE vs N",
):
    plt.figure()
    plt.loglog(ns, rmses_mc, marker="o", label="MC")
    if rmses_qmc is not None:
        plt.loglog(ns, rmses_qmc, marker="o", label="QMC")
    plt.xlabel("N samples")
    plt.ylabel("RMSE")
    plt.title(title)
    plt.legend()
    plt.tight_layout()


def plot_autocorr(x: np.ndarray, max_lag: int = 100, title="Autocorrelation"):
    """
    x: [N] chain from Gibbs / MCMC
    """
    x = _to_np(x).ravel()
    x = (x - x.mean()) / (x.std() + 1e-12)
    ac = np.correlate(x, x, mode="full")
    ac = ac[ac.size // 2 :]
    ac = ac / ac[0]
    plt.figure()
    plt.plot(np.arange(min(max_lag, len(ac))), ac[:max_lag])
    plt.xlabel("lag")
    plt.ylabel("autocorrelation")
    plt.title(title)
    plt.tight_layout()


def plot_ate_bar(ey_do_a0, ey_do_a1, title="Interventional effect on Y"):

    if isinstance(ey_do_a0, torch.Tensor):
        ey_do_a0 = ey_do_a0.detach().cpu().item()
    if isinstance(ey_do_a1, torch.Tensor):
        ey_do_a1 = ey_do_a1.detach().cpu().item()

    means = np.array([ey_do_a0, ey_do_a1])
    plt.figure()
    plt.bar([0, 1], means, color=["#4682B4", "#CD5C5C"])
    plt.xticks([0, 1], ["do(A=0)", "do(A=1)"])
    plt.ylabel("E[Y]")
    plt.title(title)
    plt.tight_layout()
