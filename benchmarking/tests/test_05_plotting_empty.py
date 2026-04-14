from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd


def test_plot_error_vs_size_skips_empty(tmp_path: Path) -> None:
    module = importlib.import_module("benchmarking.scripts.05_report_results")
    summary_style = module.SUMMARY_STYLES["mean"]

    df = pd.DataFrame(
        {
            "method_id": ["m1", "m2"],
            "n_nodes": [1, 2],
            "kl_mean": [np.nan, np.nan],
            "kl_std": [np.nan, np.nan],
        }
    )
    out = module._plot_error_vs_size(
        df,
        size_col="n_nodes",
        metric="kl",
        summary_style=summary_style,
        out_dir=tmp_path,
        title_prefix="Test",
        filename_prefix="skip_plot",
    )

    assert out is None
    assert not (tmp_path / "skip_plot.png").exists()
