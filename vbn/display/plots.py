from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import torch


def plots_enabled() -> bool:
    return os.getenv("VBN_SKIP_PLOTS", "0") not in ("1", "true", "True")


def _import_plt():
    if not plots_enabled():
        return None
    try:
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - import guard
        if not os.getenv("CI"):
            print(
                "matplotlib is not installed; skipping plots. Install it with 'pip install matplotlib'.",
                file=sys.stderr,
            )
        return None
    return plt


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _normalize_weights(weights: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if weights is None:
        return None
    weights = np.clip(weights, a_min=0.0, a_max=None)
    total = float(weights.sum())
    if total > 0:
        weights = weights / total
    return weights


def _finalize_figure(fig, save_path: Optional[str], show: bool) -> None:
    plt = _import_plt()
    if plt is None:
        return
    fig.tight_layout()
    if save_path:
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=1000)
    if show:
        plt.show()
    plt.close(fig)
