from __future__ import annotations

from typing import Dict

import numpy as np


def _safe_normalize(arr: np.ndarray, axis: int = -1, eps: float = 1e-9) -> np.ndarray:
    arr = np.maximum(arr, 0)
    arr = arr + eps
    denom = arr.sum(axis=axis, keepdims=True)
    return arr / denom


def categorical_kl(
    p: np.ndarray, q: np.ndarray, axis: int = -1, eps: float = 1e-9
) -> np.ndarray:
    p_n = _safe_normalize(p, axis=axis, eps=eps)
    q_n = _safe_normalize(q, axis=axis, eps=eps)
    return np.sum(p_n * (np.log(p_n + eps) - np.log(q_n + eps)), axis=axis)


def categorical_tv(
    p: np.ndarray, q: np.ndarray, axis: int = -1, eps: float = 1e-9
) -> np.ndarray:
    p_n = _safe_normalize(p, axis=axis, eps=eps)
    q_n = _safe_normalize(q, axis=axis, eps=eps)
    return 0.5 * np.sum(np.abs(p_n - q_n), axis=axis)


def categorical_wasserstein(
    p: np.ndarray, q: np.ndarray, axis: int = -1, eps: float = 1e-9
) -> np.ndarray:
    p_n = _safe_normalize(p, axis=axis, eps=eps)
    q_n = _safe_normalize(q, axis=axis, eps=eps)
    cdf_p = np.cumsum(p_n, axis=axis)
    cdf_q = np.cumsum(q_n, axis=axis)
    return np.sum(np.abs(cdf_p - cdf_q), axis=axis)


def cpd_distance(true_cpd: np.ndarray, pred_cpd: np.ndarray) -> Dict[str, float]:
    # true/pred shape: parents..., states
    kl = categorical_kl(true_cpd, pred_cpd)
    w1 = categorical_wasserstein(true_cpd, pred_cpd)
    tv = categorical_tv(true_cpd, pred_cpd)
    return {
        "kl_mean": float(np.mean(kl)),
        "kl_max": float(np.max(kl)),
        "w1_mean": float(np.mean(w1)),
        "w1_max": float(np.max(w1)),
        "tv_mean": float(np.mean(tv)),
        "tv_max": float(np.max(tv)),
    }


def aggregate_cpd_metrics(
    true_cpds: Dict[str, np.ndarray], pred_cpds: Dict[str, np.ndarray]
) -> Dict[str, float]:
    metrics = {}
    totals = {"kl": [], "w1": [], "tv": []}
    for node, true_cpd in true_cpds.items():
        if node not in pred_cpds:
            continue
        pred = pred_cpds[node]
        if pred.shape != true_cpd.shape and pred.size == true_cpd.size:
            pred = pred.reshape(true_cpd.shape)
        m = cpd_distance(true_cpd, pred)
        for key, val in m.items():
            metrics[f"{node}_{key}"] = val
        totals["kl"].append(m["kl_mean"])
        totals["w1"].append(m["w1_mean"])
        totals["tv"].append(m["tv_mean"])
    if totals["kl"]:
        metrics["kl_mean_overall"] = float(np.mean(totals["kl"]))
        metrics["w1_mean_overall"] = float(np.mean(totals["w1"]))
        metrics["tv_mean_overall"] = float(np.mean(totals["tv"]))
    return metrics
