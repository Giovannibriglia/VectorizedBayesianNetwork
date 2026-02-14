from __future__ import annotations

from typing import Optional

import torch

from vbn.display.plots import (
    _finalize_figure,
    _import_plt,
    _normalize_weights,
    _to_numpy,
)


def _select_batch(x: torch.Tensor, batch_index: int) -> torch.Tensor:
    if x.dim() <= 1:
        return x
    return x[batch_index]


def plot_inference_posterior(
    pdf: torch.Tensor,
    samples: torch.Tensor,
    *,
    batch_index: int = 0,
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    if not isinstance(samples, torch.Tensor):
        samples = torch.tensor(samples)
    if not isinstance(pdf, torch.Tensor):
        pdf = torch.tensor(pdf)

    samples_b = _select_batch(samples, batch_index)
    pdf_b = _select_batch(pdf, batch_index)

    if samples_b.dim() == 1:
        samples_b = samples_b.unsqueeze(-1)
    if samples_b.dim() != 2:
        raise ValueError("samples must have shape [S] or [S,D] after batch selection")

    s, d = samples_b.shape
    samples_np = _to_numpy(samples_b)
    pdf_np = _to_numpy(pdf_b)

    plt = _import_plt()
    fig, axes = plt.subplots(
        nrows=d,
        ncols=1,
        figsize=(6, max(2.5, 2.5 * d)),
        squeeze=False,
    )

    for dim in range(d):
        ax = axes[dim, 0]
        if pdf_np.ndim == 1:
            weights = pdf_np
        elif pdf_np.ndim == 2 and pdf_np.shape[0] == s and pdf_np.shape[1] == d:
            weights = pdf_np[:, dim]
        else:
            weights = pdf_np.reshape(s, -1).mean(axis=-1)
        weights = _normalize_weights(weights)
        ax.hist(samples_np[:, dim], bins=30, density=True, weights=weights, alpha=0.7)
        title = f"posterior (dim {dim})" if d > 1 else "posterior"
        ax.set_title(title)
        ax.set_xlabel("value")
        ax.set_ylabel("density")

    _finalize_figure(fig, save_path, show)
