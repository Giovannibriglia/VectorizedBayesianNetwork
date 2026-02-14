from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from vbn.core.utils import ensure_2d, ensure_tensor
from vbn.display.plots import _finalize_figure, _import_plt, _to_numpy


def _format_parent_row(row: np.ndarray) -> str:
    values = ", ".join(f"{v:.2f}" for v in row.tolist())
    return f"[{values}]"


def plot_cpd_fit(
    vbn,
    node: str,
    *,
    parents_grid: Optional[torch.Tensor] = None,
    n_samples: int = 512,
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    if node not in vbn.nodes:
        raise ValueError(f"Unknown node '{node}'.")

    cpd = vbn.nodes[node]
    if parents_grid is not None and cpd.input_dim == 0:
        raise ValueError("parents_grid provided for node with no parents")

    if parents_grid is None:
        if cpd.input_dim == 0:
            parents_tensor = None
            labels = ["unconditional"]
        else:
            parents_tensor = torch.zeros(1, cpd.input_dim, device=vbn.device)
            labels = ["parents=0"]
    else:
        parents_tensor = ensure_2d(ensure_tensor(parents_grid, device=vbn.device))
        if parents_tensor.shape[-1] != cpd.input_dim:
            raise ValueError(
                f"parents_grid last dim {parents_tensor.shape[-1]} does not match input_dim {cpd.input_dim}"
            )
        labels = [
            f"parents={_format_parent_row(row)}" for row in _to_numpy(parents_tensor)
        ]

    with torch.no_grad():
        samples = cpd.sample(parents_tensor, int(n_samples))
    if samples.dim() == 2:
        samples = samples.unsqueeze(0)
    samples_np = _to_numpy(samples)

    b, _, d = samples_np.shape
    plt = _import_plt()
    fig, axes = plt.subplots(
        nrows=d,
        ncols=1,
        figsize=(6, max(2.5, 2.5 * d)),
        squeeze=False,
    )

    for dim in range(d):
        ax = axes[dim, 0]
        for i in range(b):
            data = samples_np[i, :, dim]
            hist, edges = np.histogram(data, bins=30, density=True)
            centers = 0.5 * (edges[1:] + edges[:-1])
            label = labels[i] if b > 1 else None
            ax.plot(centers, hist, label=label)
        title = f"{node} (dim {dim})" if d > 1 else f"{node}"
        ax.set_title(title)
        ax.set_xlabel("value")
        ax.set_ylabel("density")
        if b > 1:
            ax.legend(fontsize=8)

    _finalize_figure(fig, save_path, show)
