from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy.stats import wasserstein_distance as _scipy_wasserstein

_NEG_TOL = 1e-12
_SUM_TOL = 1e-6


def _as_prob_array(values: Iterable[float], eps: float = 0.0) -> np.ndarray | None:
    arr = np.asarray(list(values), dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    if eps > 0.0:
        arr = np.clip(arr, eps, None)
    if np.any(arr < 0.0):
        if np.any(arr < -_NEG_TOL):
            return None
        arr = arr.copy()
        arr[arr < 0.0] = 0.0
    total = float(arr.sum())
    if not math.isfinite(total) or total <= 0.0:
        return None
    if not math.isclose(total, 1.0, abs_tol=_SUM_TOL):
        arr = arr / total
    return arr


def jensen_shannon_divergence(
    p: np.ndarray | Iterable[float],
    q: np.ndarray | Iterable[float],
    eps: float = 0.0,
) -> float:
    """
    Compute JSD(P,Q) = 0.5*KL(P||M) + 0.5*KL(Q||M), where M=(P+Q)/2.
    Assumes p,q are 1D arrays, non-negative, and sum to 1.
    Uses natural log, returns value in [0, log(2)].
    """
    p_arr = _as_prob_array(p, eps=eps)
    q_arr = _as_prob_array(q, eps=eps)
    if p_arr is None or q_arr is None:
        return float("nan")
    if p_arr.shape != q_arr.shape:
        return float("nan")
    m = 0.5 * (p_arr + q_arr)
    mask_p = (p_arr > 0.0) & (m > 0.0)
    mask_q = (q_arr > 0.0) & (m > 0.0)
    kl_pm = np.sum(p_arr[mask_p] * np.log(p_arr[mask_p] / m[mask_p]))
    kl_qm = np.sum(q_arr[mask_q] * np.log(q_arr[mask_q] / m[mask_q]))
    jsd = 0.5 * (kl_pm + kl_qm)
    return float(jsd)


def jensen_shannon_divergence_normalized(
    p: np.ndarray | Iterable[float], q: np.ndarray | Iterable[float]
) -> float:
    jsd = jensen_shannon_divergence(p, q)
    return float(jsd / np.log(2.0))


def _normalize_probs(probs: Iterable[float], eps: float) -> np.ndarray:
    arr = np.asarray(list(probs), dtype=float)
    arr = np.clip(arr, eps, 1.0)
    total = float(arr.sum())
    if not math.isfinite(total) or total <= 0:
        return np.full_like(arr, 1.0 / len(arr))
    return arr / total


def kl_divergence(p: Iterable[float], q: Iterable[float], eps: float) -> float:
    p_arr = _normalize_probs(p, eps)
    q_arr = _normalize_probs(q, eps)
    return float(np.sum(p_arr * np.log(p_arr / q_arr)))


def _wasserstein_from_probs(p_arr: np.ndarray, q_arr: np.ndarray) -> float:
    k = len(p_arr)
    xs = np.arange(k, dtype=float)
    if _scipy_wasserstein is not None:
        return float(_scipy_wasserstein(xs, xs, p_arr, q_arr))
    cdf_p = np.cumsum(p_arr)
    cdf_q = np.cumsum(q_arr)
    return float(np.sum(np.abs(cdf_p - cdf_q)))


def wasserstein_distance(p: Iterable[float], q: Iterable[float], eps: float) -> float:
    p_arr = _normalize_probs(p, eps)
    q_arr = _normalize_probs(q, eps)
    return _wasserstein_from_probs(p_arr, q_arr)


def _compute_discrete_metrics(
    gt_probs: Iterable[float],
    pred_probs: Iterable[float],
    eps: float,
    *,
    compute_jsd: bool,
) -> tuple[float, float, float, float]:
    p_arr = _normalize_probs(gt_probs, eps)
    q_arr = _normalize_probs(pred_probs, eps)
    kl = float(np.sum(p_arr * np.log(p_arr / q_arr)))
    wass = _wasserstein_from_probs(p_arr, q_arr)
    if not compute_jsd:
        return kl, wass, float("nan"), float("nan")
    jsd = jensen_shannon_divergence(p_arr, q_arr)
    jsd_norm = jensen_shannon_divergence_normalized(p_arr, q_arr)
    return kl, wass, jsd, jsd_norm
