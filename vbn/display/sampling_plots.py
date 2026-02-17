from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from vbn.display.plots import _finalize_figure, _import_plt, _to_numpy


def _select_batch(x: torch.Tensor, batch_index: int) -> torch.Tensor:
    if x.dim() <= 1:
        return x
    return x[batch_index]


def plot_sampling_outcome(
    samples: torch.Tensor,
    *,
    batch_index: int = 0,
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    plt = _import_plt()
    if plt is None:
        return
    if not isinstance(samples, torch.Tensor):
        samples = torch.tensor(samples)

    samples_b = _select_batch(samples, batch_index)
    if samples_b.dim() == 1:
        samples_b = samples_b.unsqueeze(-1)
    if samples_b.dim() != 2:
        raise ValueError("samples must have shape [N] or [N,D] after batch selection")

    n, d = samples_b.shape
    samples_np = _to_numpy(samples_b)

    if d == 1:
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 5), squeeze=False)
        ax_trace = axes[0, 0]
        ax_hist = axes[1, 0]
        ax_trace.plot(np.arange(n), samples_np[:, 0], linewidth=1.0)
        ax_trace.set_title("trace")
        ax_trace.set_xlabel("step")
        ax_trace.set_ylabel("value")
        ax_hist.hist(samples_np[:, 0], bins=30, density=True, alpha=0.7)
        ax_hist.set_title("marginal")
        ax_hist.set_xlabel("value")
        ax_hist.set_ylabel("density")
    else:
        fig, axes = plt.subplots(
            nrows=d,
            ncols=1,
            figsize=(6, max(2.5, 2.5 * d)),
            squeeze=False,
        )
        for dim in range(d):
            ax = axes[dim, 0]
            ax.hist(samples_np[:, dim], bins=30, density=True, alpha=0.7)
            ax.set_title(f"marginal (dim {dim})")
            ax.set_xlabel("value")
            ax.set_ylabel("density")

    _finalize_figure(fig, save_path, show)
