import argparse
import gc
import json
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

# Ensure repo root is on sys.path when executed as a script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ============================================================
# Configuration
# ============================================================


@dataclass
class DataConfig:
    card: int
    n_samples_df: int
    n_states: int
    n_actions: int
    mode: str = "linear"  # "linear" or "nonlinear"
    seed: int = 42
    include_next_state: bool = False


@dataclass
class ExperimentConfig:
    n_samples_df: int = 32000
    n_states_list: Sequence[int] = (1,)
    n_actions_list: Sequence[int] = (1,)
    cards: Sequence[int] = (10, 20, 50, 70, 100, 200, 500, 700, 1000, 2000)
    mode: str = "linear"
    seed: int = 42
    n_mc_ground_truth: int = 20000
    n_inference_queries: int = 128
    inference_seed: int = 123
    vbn_inference_n_samples: int = 512
    vbn_inference_batch_size: int = 128
    vbn_device: str = "auto"
    out_dir: str = "stress_test/out"
    include_next_state: bool = True
    softmax_max_classes: Optional[int] = None
    softmax_label_smoothing: float = 0.0
    softmax_class_weighting: str = "none"
    embedded_softmax_embedding_dim: int = 8
    embedded_softmax_hidden_dims: Sequence[int] = (64, 64)
    categorical_table_alpha: float = 1.0
    categorical_table_alpha_mode: str = "total_mass"
    categorical_table_prior: str = "global"
    metrics: Sequence[str] = ("kl", "js", "ws", "fit_time")
    inference_metrics: Sequence[str] = (
        "kl",
        "js",
        "ws",
        "inference_time",
        "mse",
        "mae",
        "r2",
    )
    aggregation_mode: str = "iqm"  # "mean" or "iqm"
    spread_mode: Optional[str] = "iqr_std"  # None -> std for mean, iqr_std for iqm


# ============================================================
# Device helpers
# ============================================================


def resolve_torch_device(device_str: str) -> torch.device:
    if device_str is None:
        device_str = "auto"
    if isinstance(device_str, torch.device):
        return device_str

    key = str(device_str).strip().lower()
    if key == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if key.startswith("cuda") and not torch.cuda.is_available():
        warnings.warn(
            "CUDA requested but unavailable; falling back to CPU.",
            RuntimeWarning,
        )
        return torch.device("cpu")

    try:
        return torch.device(device_str)
    except Exception as exc:
        raise ValueError(f"Invalid torch device '{device_str}'.") from exc


def to_device_tensor(
    values: Any,
    device: torch.device,
    dtype: torch.dtype,
    *,
    ensure_2d: bool = False,
) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        tensor = values.to(device=device, dtype=dtype)
    else:
        tensor = torch.tensor(values, device=device, dtype=dtype)

    if ensure_2d:
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(-1)
        elif tensor.dim() != 2:
            raise ValueError(f"Expected 1D or 2D tensor, got shape {tensor.shape}")

    return tensor


def build_vbn_evidence_tensors(
    evidence: pd.DataFrame | Dict[str, Any],
    device: torch.device,
    *,
    dtype: torch.dtype = torch.long,
) -> Dict[str, torch.Tensor]:
    if isinstance(evidence, pd.DataFrame):
        items = {col: evidence[col].to_numpy() for col in evidence.columns}
    elif isinstance(evidence, dict):
        items = evidence
    else:
        raise TypeError("evidence must be a pandas DataFrame or a dict")

    return {
        key: to_device_tensor(values, device=device, dtype=dtype, ensure_2d=True)
        for key, values in items.items()
    }


# ============================================================
# Metrics
# ============================================================


def normalize_probs(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, None)
    return p / p.sum(axis=-1, keepdims=True)


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = normalize_probs(p, eps=eps)
    q = normalize_probs(q, eps=eps)
    return np.sum(p * (np.log(p) - np.log(q)), axis=-1)


def js_divergence_normalized(
    p: np.ndarray, q: np.ndarray, eps: float = 1e-12
) -> np.ndarray:
    p = normalize_probs(p, eps=eps)
    q = normalize_probs(q, eps=eps)
    m = 0.5 * (p + q)
    js = 0.5 * kl_divergence(p, m, eps=eps) + 0.5 * kl_divergence(q, m, eps=eps)
    return js / np.log(2.0)


def wasserstein_1d(p: np.ndarray, q: np.ndarray, support: np.ndarray) -> np.ndarray:
    p = normalize_probs(p)
    q = normalize_probs(q)
    order = np.argsort(support)
    support = support[order]
    p = p[:, order]
    q = q[:, order]

    cdf_p = np.cumsum(p, axis=1)
    cdf_q = np.cumsum(q, axis=1)
    dx = np.diff(support)

    if dx.size == 0:
        return np.zeros(p.shape[0], dtype=np.float64)

    return np.sum(np.abs(cdf_p[:, :-1] - cdf_q[:, :-1]) * dx, axis=1)


def interquartile_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    x = np.sort(x)
    lower = int(np.floor(0.25 * x.size))
    upper = int(np.ceil(0.75 * x.size))
    if upper <= lower:
        return float(np.mean(x))
    return float(np.mean(x[lower:upper]))


def iqr_std(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size <= 1:
        return 0.0 if x.size == 1 else float("nan")
    q25, q75 = np.percentile(x, [25.0, 75.0])
    return float((q75 - q25) / 1.349)


def summarize_metric_values(
    values: np.ndarray,
    *,
    aggregation_mode: str = "iqm",
    spread_mode: Optional[str] = None,
) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "value": float("nan"),
            "spread": float("nan"),
            "lower": float("nan"),
            "upper": float("nan"),
            "n": 0,
        }

    aggregation_key = str(aggregation_mode).strip().lower()
    if aggregation_key not in {"mean", "iqm"}:
        raise ValueError(
            f"Unsupported aggregation_mode '{aggregation_mode}'. Use 'mean' or 'iqm'."
        )

    if spread_mode is None:
        spread_key = "std" if aggregation_key == "mean" else "iqr_std"
    else:
        spread_key = str(spread_mode).strip().lower()

    if spread_key not in {"std", "iqr_std", "none"}:
        raise ValueError(
            f"Unsupported spread_mode '{spread_mode}'. Use 'std', 'iqr_std', or 'none'."
        )

    center = (
        float(np.mean(values))
        if aggregation_key == "mean"
        else interquartile_mean(values)
    )

    if spread_key == "std":
        spread = float(np.std(values))
    elif spread_key == "iqr_std":
        spread = iqr_std(values)
    else:
        spread = 0.0

    return {
        "value": center,
        "spread": float(spread),
        "lower": float(center - spread),
        "upper": float(center + spread),
        "n": int(values.size),
    }


def compare_reward_cpds(
    reference_pmf: np.ndarray,
    candidate_pmf: np.ndarray,
    support: np.ndarray,
    *,
    aggregation_mode: str = "iqm",
    spread_mode: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    if reference_pmf.shape != candidate_pmf.shape:
        raise ValueError(
            f"PMF shape mismatch: reference {reference_pmf.shape}, candidate {candidate_pmf.shape}"
        )

    if reference_pmf.size == 0:
        empty = summarize_metric_values(
            np.array([], dtype=np.float64),
            aggregation_mode=aggregation_mode,
            spread_mode=spread_mode,
        )
        return {"kl": dict(empty), "js": dict(empty), "ws": dict(empty)}

    metric_arrays = {
        "kl": kl_divergence(reference_pmf, candidate_pmf),
        "js": js_divergence_normalized(reference_pmf, candidate_pmf),
        "ws": wasserstein_1d(reference_pmf, candidate_pmf, support),
    }
    return {
        metric_name: summarize_metric_values(
            metric_values,
            aggregation_mode=aggregation_mode,
            spread_mode=spread_mode,
        )
        for metric_name, metric_values in metric_arrays.items()
    }


def posterior_mode_from_pmf(pmf: np.ndarray, support: np.ndarray) -> np.ndarray:
    pmf = np.asarray(pmf)
    support = np.asarray(support)
    if pmf.ndim != 2:
        raise ValueError(f"PMF must be 2D, got shape {pmf.shape}")
    if support.ndim != 1:
        raise ValueError(f"Support must be 1D, got shape {support.shape}")
    indices = np.argmax(pmf, axis=1)
    return support[indices]


def compute_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true/y_pred shape mismatch: {y_true.shape} vs {y_pred.shape}"
        )

    diff = y_true - y_pred
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))

    ss_res = float(np.sum(diff**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        r2 = 1.0 if ss_res == 0.0 else 0.0
    else:
        r2 = 1.0 - ss_res / ss_tot

    return {"mse": mse, "mae": mae, "r2": float(r2)}


def compare_posterior_point_predictions(
    reference_pmf: np.ndarray,
    candidate_pmf: np.ndarray,
    support: np.ndarray,
) -> Dict[str, float]:
    if reference_pmf.shape != candidate_pmf.shape:
        raise ValueError(
            f"PMF shape mismatch: reference {reference_pmf.shape}, candidate {candidate_pmf.shape}"
        )
    if reference_pmf.size == 0:
        return {"mse": float("nan"), "mae": float("nan"), "r2": float("nan")}

    y_true = posterior_mode_from_pmf(reference_pmf, support)
    y_pred = posterior_mode_from_pmf(candidate_pmf, support)
    return compute_regression_metrics(y_true, y_pred)


def validate_pmf(pmf: np.ndarray, name: str, atol: float = 1e-6) -> None:
    if pmf.ndim != 2:
        raise ValueError(f"{name} PMF must be 2D, got shape {pmf.shape}")
    if not np.all(np.isfinite(pmf)):
        raise ValueError(f"{name} PMF contains non-finite values")
    if (pmf < -atol).any():
        raise ValueError(f"{name} PMF contains negative entries below {-atol}")
    row_sums = pmf.sum(axis=1)
    if row_sums.size == 0:
        return
    if not np.allclose(row_sums, 1.0, atol=atol):
        raise ValueError(
            f"{name} PMF rows do not sum to 1. "
            f"Min={row_sums.min()}, Max={row_sums.max()}"
        )


def validate_backend_result(
    result: "BackendResult", reward_support: np.ndarray, atol: float = 1e-6
) -> None:
    support = np.asarray(result.support, dtype=np.int64)
    reward_support = np.asarray(reward_support, dtype=np.int64)

    if support.shape != reward_support.shape or not np.array_equal(
        support, reward_support
    ):
        raise ValueError(
            f"Support mismatch for {result.name}: {support} vs {reward_support}"
        )

    if result.pmf.shape[1] != reward_support.shape[0]:
        raise ValueError(
            f"PMF/support mismatch for {result.name}: "
            f"pmf.shape={result.pmf.shape}, support={reward_support.shape}"
        )

    if result.pmf.shape[0] != len(result.parents_df):
        raise ValueError(
            f"PMF/parents mismatch for {result.name}: "
            f"pmf.shape={result.pmf.shape}, parents={len(result.parents_df)}"
        )

    validate_pmf(result.pmf, result.name, atol=atol)


# ============================================================
# Data generation
# ============================================================


def extract_state_action_arrays(
    df: pd.DataFrame, n_states: int, n_actions: int
) -> Tuple[np.ndarray, np.ndarray]:
    if n_states > 0:
        states = np.stack(
            [df[f"state_{i}"].values for i in range(n_states)], axis=1
        ).astype(np.float64)
    else:
        states = np.zeros((len(df), 0), dtype=np.float64)

    if n_actions > 0:
        actions = np.stack(
            [df[f"action_{j}"].values for j in range(n_actions)], axis=1
        ).astype(np.float64)
    else:
        actions = np.zeros((len(df), 0), dtype=np.float64)

    return states, actions


def compute_reward_base_from_arrays(
    states: np.ndarray, actions: np.ndarray, mode: str, card: int
) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    n_rows = states.shape[0]

    base = np.zeros(n_rows, dtype=np.float64)

    if mode == "linear":
        if states.shape[1] > 0:
            weights_s = np.arange(1, states.shape[1] + 1, dtype=np.float64)
            base += states @ weights_s
        if actions.shape[1] > 0:
            weights_a = np.arange(1, actions.shape[1] + 1, dtype=np.float64)
            base += actions @ weights_a
        return base

    if mode != "nonlinear":
        raise ValueError(f"Unsupported mode: {mode}")

    if states.shape[1] > 0:
        base += np.sum(np.sin(states) + 0.1 * (states**2), axis=1)

    if actions.shape[1] > 0:
        base += np.sum(np.cos(actions) + 0.05 * (actions**2), axis=1)

    if states.shape[1] > 0 and actions.shape[1] > 0:
        base += 0.2 * (states.sum(axis=1) * actions.sum(axis=1))
        base += 0.1 * np.sin(states.sum(axis=1) + actions.sum(axis=1))

    if actions.shape[1] > 1:
        sum_a = actions.sum(axis=1)
        sum_a2 = np.sum(actions**2, axis=1)
        base += 0.05 * (0.5 * (sum_a**2 - sum_a2))

    return base


def finalize_reward_from_base(
    base: np.ndarray, reward_noise: np.ndarray, card: int
) -> np.ndarray:
    reward = (base + reward_noise).astype(np.int64) % card
    return reward


def compute_reward_from_arrays(
    states: np.ndarray,
    actions: np.ndarray,
    reward_noise: np.ndarray,
    mode: str,
    card: int,
) -> np.ndarray:
    base = compute_reward_base_from_arrays(states, actions, mode=mode, card=card)
    return finalize_reward_from_base(base, reward_noise, card=card)


def define_df(cfg: DataConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    data: Dict[str, np.ndarray] = {}

    for i in range(cfg.n_states):
        data[f"state_{i}"] = rng.integers(0, cfg.card, size=cfg.n_samples_df)

    for j in range(cfg.n_actions):
        noise = rng.integers(0, cfg.card, size=cfg.n_samples_df)
        base_state = data[f"state_{j % cfg.n_states}"]

        if cfg.mode == "linear":
            action = (2 * base_state + noise) % cfg.card
        elif cfg.mode == "nonlinear":
            action = (
                np.sin(base_state) * (cfg.card / 2)
                + (base_state * noise) % cfg.card
                + np.cos(noise)
            )
        else:
            raise ValueError(f"Unsupported mode: {cfg.mode}")

        data[f"action_{j}"] = np.asarray(action, dtype=np.int32) % cfg.card

    if cfg.n_states > 0:
        states = np.stack(
            [data[f"state_{i}"] for i in range(cfg.n_states)], axis=1
        ).astype(np.float64)
    else:
        states = np.zeros((cfg.n_samples_df, 0), dtype=np.float64)

    if cfg.n_actions > 0:
        actions = np.stack(
            [data[f"action_{j}"] for j in range(cfg.n_actions)], axis=1
        ).astype(np.float64)
    else:
        actions = np.zeros((cfg.n_samples_df, 0), dtype=np.float64)

    reward_noise = rng.normal(size=cfg.n_samples_df)
    data["reward"] = compute_reward_from_arrays(
        states, actions, reward_noise, mode=cfg.mode, card=cfg.card
    ).astype(np.int32)

    if cfg.include_next_state:
        for i in range(cfg.n_states):
            s = data[f"state_{i}"]
            a = data[f"action_{i % cfg.n_actions}"]
            noise = rng.integers(1, max(2, cfg.card), size=cfg.n_samples_df)

            if cfg.mode == "linear":
                next_s = s + 2 * a + noise
            elif cfg.mode == "nonlinear":
                next_s = (
                    s + a + np.sin(s + a) * (cfg.card / 3) + (s * a) % cfg.card + noise
                )
            else:
                raise ValueError(f"Unsupported mode: {cfg.mode}")

            data[f"next_state_{i}"] = np.asarray(next_s, dtype=np.int32) % cfg.card

    return pd.DataFrame(data)


def get_rl_dag(
    n_states: int, n_actions: int, *, include_next_state: bool = True
) -> nx.DiGraph:
    dag = nx.DiGraph()

    for s in range(n_states):
        for a in range(n_actions):
            dag.add_edge(f"state_{s}", f"action_{a}")

    for s in range(n_states):
        if include_next_state:
            dag.add_edge(f"state_{s}", f"next_state_{s}")
        dag.add_edge(f"state_{s}", "reward")

    for a in range(n_actions):
        if include_next_state and n_states > 0:
            dag.add_edge(f"action_{a}", f"next_state_{a % n_states}")
        dag.add_edge(f"action_{a}", "reward")

    return dag


def get_state_action_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("state_") or c.startswith("action_")]


def get_parent_combinations(df: pd.DataFrame) -> pd.DataFrame:
    return df[get_state_action_columns(df)].drop_duplicates().reset_index(drop=True)


def get_reward_support(df: pd.DataFrame, card: Optional[int] = None) -> np.ndarray:
    if card is not None:
        return np.arange(card, dtype=np.int64)
    return np.sort(df["reward"].unique()).astype(np.int64)


# ============================================================
# Inference utilities
# ============================================================


def get_inference_evidence_columns(
    n_states: int, n_actions: int, include_actions: bool = False
) -> List[str]:
    cols = [f"state_{i}" for i in range(n_states)]
    if include_actions:
        cols += [f"action_{j}" for j in range(n_actions)]
    return cols


def sample_inference_queries(
    df: pd.DataFrame,
    n_states: int,
    n_actions: int,
    n_queries: int,
    seed: int,
    *,
    include_actions: bool = False,
) -> pd.DataFrame:
    evidence_cols = get_inference_evidence_columns(
        n_states, n_actions, include_actions=include_actions
    )
    candidates = df[evidence_cols].drop_duplicates().reset_index(drop=True)
    if len(candidates) == 0 or n_queries <= 0:
        return candidates.iloc[:0].copy()

    n_queries = min(int(n_queries), len(candidates))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(candidates), size=n_queries, replace=False)
    return candidates.iloc[idx].reset_index(drop=True)


def sample_actions_from_states(
    states: np.ndarray, n_actions: int, card: int, mode: str, rng: np.random.Generator
) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64)
    n_samples = states.shape[0]
    n_states = states.shape[1]

    if n_actions <= 0:
        return np.zeros((n_samples, 0), dtype=np.float64)

    actions = np.zeros((n_samples, n_actions), dtype=np.float64)
    for j in range(n_actions):
        if n_states == 0:
            base_state = np.zeros(n_samples, dtype=np.float64)
        else:
            base_state = states[:, j % n_states]
        noise = rng.integers(0, card, size=n_samples)
        if mode == "linear":
            action = (2 * base_state + noise) % card
        elif mode == "nonlinear":
            action = (
                np.sin(base_state) * (card / 2)
                + (base_state * noise) % card
                + np.cos(noise)
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        actions[:, j] = np.asarray(action, dtype=np.int32) % card

    return actions


def estimate_ground_truth_inference_pmf(
    evidence_df: pd.DataFrame,
    n_states: int,
    n_actions: int,
    card: int,
    mode: str,
    n_mc_samples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    reward_support = np.arange(card, dtype=np.int64)

    state_cols = [f"state_{i}" for i in range(n_states)]
    action_cols = [
        f"action_{j}" for j in range(n_actions) if f"action_{j}" in evidence_df.columns
    ]

    if n_states > 0:
        states = evidence_df[state_cols].to_numpy(dtype=np.float64)
    else:
        states = np.zeros((len(evidence_df), 0), dtype=np.float64)
    observed_actions = None
    if action_cols:
        observed_actions = evidence_df[action_cols].to_numpy(dtype=np.float64)

    pmf = np.zeros((len(evidence_df), card), dtype=np.float64)
    for i in range(len(evidence_df)):
        state_row = np.repeat(states[i : i + 1], n_mc_samples, axis=0)
        actions = sample_actions_from_states(
            state_row, n_actions=n_actions, card=card, mode=mode, rng=rng
        )
        if observed_actions is not None:
            for idx, col in enumerate(action_cols):
                action_idx = int(col.split("_")[1])
                actions[:, action_idx] = observed_actions[i, idx]
        reward_noise = rng.normal(size=n_mc_samples)
        rewards = compute_reward_from_arrays(
            state_row, actions, reward_noise, mode=mode, card=card
        ).astype(np.int64)
        counts = np.bincount(rewards, minlength=card).astype(np.float64)
        pmf[i] = counts / counts.sum()

    return reward_support, pmf


def map_samples_to_support_indices(
    samples: np.ndarray, support: np.ndarray
) -> np.ndarray:
    support = np.asarray(support, dtype=np.float64)
    if support.ndim != 1:
        raise ValueError("support must be 1D")
    if support.size == 0:
        raise ValueError("support must be non-empty")

    support_sorted = np.sort(support)
    contiguous = np.allclose(np.diff(support_sorted), 1.0)
    if contiguous:
        min_val = support_sorted[0]
        max_val = support_sorted[-1]
        idx = np.rint(samples).astype(np.int64)
        idx = np.clip(idx, int(min_val), int(max_val)) - int(min_val)
        return idx

    flat = samples.reshape(-1)
    idx = np.searchsorted(support_sorted, flat, side="left")
    idx = np.clip(idx, 0, len(support_sorted) - 1)
    left_idx = np.clip(idx - 1, 0, len(support_sorted) - 1)
    right = support_sorted[idx]
    left = support_sorted[left_idx]
    choose_left = np.abs(flat - left) <= np.abs(flat - right)
    idx = np.where(choose_left, left_idx, idx)
    return idx.reshape(samples.shape)


def weighted_samples_to_pmf(
    samples: np.ndarray, weights: np.ndarray, support: np.ndarray
) -> np.ndarray:
    samples = np.asarray(samples)
    weights = np.asarray(weights)

    if samples.ndim == 3:
        if samples.shape[-1] != 1:
            raise ValueError(f"Expected 1D target samples, got {samples.shape}")
        samples = samples[..., 0]
    if weights.ndim == 3:
        weights = weights[..., 0]

    if samples.shape != weights.shape:
        raise ValueError(
            f"Sample/weight shape mismatch: {samples.shape} vs {weights.shape}"
        )

    indices = map_samples_to_support_indices(samples, support)
    pmf = np.zeros((samples.shape[0], len(support)), dtype=np.float64)
    weights = np.clip(weights, 0.0, None)

    for i in range(samples.shape[0]):
        np.add.at(pmf[i], indices[i], weights[i])

    return normalize_probs(pmf)


# ============================================================
# PMF extraction utilities
# ============================================================


def pgmpy_reward_pmf(
    cpd,
    parents_df: pd.DataFrame,
    reward_support: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    evidence = list(getattr(cpd, "evidence", None) or cpd.variables[1:])
    state_names = getattr(cpd, "state_names", None) or {}
    cardinality = dict(zip(cpd.variables, cpd.cardinality))

    if reward_support is None:
        reward_support = state_names.get("reward")
    if reward_support is None:
        reward_support = list(
            range(cardinality.get("reward", cpd.get_values().shape[0]))
        )

    reward_support = np.asarray(reward_support, dtype=np.int64)
    reward_support = np.sort(reward_support)

    reward_states = state_names.get("reward", reward_support)
    reward_index = {v: i for i, v in enumerate(reward_states)}

    values = np.asarray(cpd.get_values())
    pmf = np.zeros((len(parents_df), len(reward_support)), dtype=np.float64)

    evidence_index = {}
    for var in evidence:
        names = state_names.get(var)
        if names is None:
            names = list(range(cardinality[var]))
        evidence_index[var] = {v: i for i, v in enumerate(names)}

    strides = []
    stride = 1
    for var in reversed(evidence):
        strides.append((var, stride))
        stride *= cardinality[var]
    strides = list(reversed(strides))

    for i, row in parents_df.iterrows():
        col = 0
        for var, s in strides:
            col += evidence_index[var][row[var]] * s
        reward_probs = values[:, col]

        if list(reward_support) != list(reward_states):
            aligned = np.zeros(len(reward_support), dtype=np.float64)
            for idx, v in enumerate(reward_support):
                if v in reward_index:
                    aligned[idx] = reward_probs[reward_index[v]]
            reward_probs = aligned

        pmf[i] = reward_probs

    return reward_support, pmf


def vbn_reward_pmf(
    handle,
    parents_df: pd.DataFrame,
    reward_support: Sequence[int],
    device: torch.device,
) -> np.ndarray:
    reward_support = np.asarray(reward_support)
    reward_support = np.sort(reward_support)

    parents_tensor = torch.cat(
        [
            to_device_tensor(
                parents_df[p].values,
                device=device,
                dtype=torch.long,
                ensure_2d=True,
            )
            for p in handle.parents
        ],
        dim=-1,
    )  # [B, Dp]

    reward_vals = to_device_tensor(
        reward_support, device=device, dtype=torch.long, ensure_2d=False
    ).view(1, -1, 1)
    b, k = parents_tensor.shape[0], reward_vals.shape[1]

    parents_grid = parents_tensor.unsqueeze(1).expand(b, k, -1)  # [B, K, Dp]
    reward_grid = reward_vals.expand(b, k, 1)  # [B, K, 1]

    logp = handle.conditional_log_prob(reward_grid, parents_grid)
    pmf = torch.softmax(logp, dim=1)

    return pmf.detach().cpu().numpy()


def estimate_ground_truth_reward_pmf(
    parents_df: pd.DataFrame,
    n_states: int,
    n_actions: int,
    card: int,
    mode: str,
    n_mc_samples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    reward_support = np.arange(card, dtype=np.int64)

    states, actions = extract_state_action_arrays(parents_df, n_states, n_actions)
    base = compute_reward_base_from_arrays(states, actions, mode=mode, card=card)

    pmf = np.zeros((len(parents_df), card), dtype=np.float64)
    for i in range(len(parents_df)):
        reward_noise = rng.normal(size=n_mc_samples)
        rewards = finalize_reward_from_base(base[i], reward_noise, card=card)
        counts = np.bincount(rewards, minlength=card).astype(np.float64)
        pmf[i] = counts / counts.sum()

    return reward_support, pmf


# ============================================================
# Backend interface
# ============================================================


@dataclass
class BackendResult:
    name: str
    fit_time: float
    support: np.ndarray
    parents_df: pd.DataFrame
    pmf: np.ndarray
    artifact: Optional[object] = None


class RewardBackend:
    name: str

    def fit_reward_pmf(
        self,
        dag: nx.DiGraph,
        df: pd.DataFrame,
        reward_support: Optional[Sequence[int]] = None,
    ) -> BackendResult:
        raise NotImplementedError

    def infer_reward_posterior(
        self,
        result: BackendResult,
        evidence_df: pd.DataFrame,
        reward_support: Sequence[int],
        **kwargs,
    ) -> Tuple[np.ndarray, float]:
        raise NotImplementedError


@dataclass
class GroundTruthBackend(RewardBackend):
    n_states: int
    n_actions: int
    card: int
    mode: str
    n_mc_samples: int = 20000
    seed: int = 42

    name = "ground_truth"

    def fit_reward_pmf(
        self,
        dag: nx.DiGraph,
        df: pd.DataFrame,
        reward_support: Optional[Sequence[int]] = None,
    ) -> BackendResult:
        del dag
        t0 = time.time()

        parents_df = get_parent_combinations(df)
        support, pmf = estimate_ground_truth_reward_pmf(
            parents_df=parents_df,
            n_states=self.n_states,
            n_actions=self.n_actions,
            card=self.card,
            mode=self.mode,
            n_mc_samples=self.n_mc_samples,
            seed=self.seed,
        )

        if reward_support is not None:
            reward_support = np.asarray(reward_support, dtype=np.int64)
            if not np.array_equal(reward_support, support):
                raise ValueError(
                    "Ground truth support mismatch: " f"{support} vs {reward_support}"
                )

        return BackendResult(
            name=self.name,
            fit_time=time.time() - t0,
            support=np.asarray(support),
            parents_df=parents_df,
            pmf=pmf,
            artifact=None,
        )


class PgmpyBackend(RewardBackend):
    name = "pgmpy"

    def fit_reward_pmf(
        self,
        dag: nx.DiGraph,
        df: pd.DataFrame,
        reward_support: Optional[Sequence[int]] = None,
    ) -> BackendResult:
        from pgmpy.estimators import MaximumLikelihoodEstimator
        from pgmpy.models import BayesianNetwork

        bn = BayesianNetwork(dag.edges)
        t0 = time.time()

        mle_est = MaximumLikelihoodEstimator(model=bn, data=df)
        cpds = [mle_est.estimate_cpd(node=node) for node in bn.nodes()]
        bn.add_cpds(*cpds)
        bn.check_model()
        cpd = bn.get_cpds("reward")

        parents_df = get_parent_combinations(df)
        support, pmf = pgmpy_reward_pmf(cpd, parents_df, reward_support=reward_support)

        return BackendResult(
            name=self.name,
            fit_time=time.time() - t0,
            support=np.asarray(support),
            parents_df=parents_df,
            pmf=pmf,
            artifact=bn,
        )

    def infer_reward_posterior(
        self,
        result: BackendResult,
        evidence_df: pd.DataFrame,
        reward_support: Sequence[int],
        **kwargs,
    ) -> Tuple[np.ndarray, float]:
        from pgmpy.inference import VariableElimination

        model = result.artifact
        if model is None:
            raise ValueError("pgmpy backend missing fitted model for inference")

        reward_support = np.asarray(reward_support, dtype=np.int64)
        reward_cpd = model.get_cpds("reward")
        state_names = getattr(reward_cpd, "state_names", None) or {}
        reward_states = state_names.get("reward") or list(reward_support)
        reward_index = {v: i for i, v in enumerate(reward_states)}

        inference = VariableElimination(model)
        pmf = np.zeros((len(evidence_df), len(reward_support)), dtype=np.float64)
        evidence_cols = list(evidence_df.columns)

        t = 0.0
        for i, row in enumerate(evidence_df.itertuples(index=False, name=None)):
            evidence = {col: int(val) for col, val in zip(evidence_cols, row)}
            ttt = time.time()
            factor = inference.query(
                variables=["reward"], evidence=evidence, show_progress=False
            )
            t += time.time() - ttt
            values = np.asarray(factor.values).reshape(-1)
            if list(reward_support) != list(reward_states):
                aligned = np.zeros(len(reward_support), dtype=np.float64)
                for idx, v in enumerate(reward_support):
                    if v in reward_index:
                        aligned[idx] = values[reward_index[v]]
                pmf[i] = aligned
            else:
                pmf[i] = values

        return pmf, t


@dataclass
class VBNBackend(RewardBackend):
    cpd_name: str
    inf_method: str = "monte_carlo_marginalization"
    inf_n_samples: int = 512
    seed: int = 42
    device: torch.device = torch.device("cpu")
    softmax_max_classes: Optional[int] = None
    softmax_label_smoothing: float = 0.0
    softmax_class_weighting: str = "none"
    embedded_softmax_embedding_dim: int = 8
    embedded_softmax_hidden_dims: Sequence[int] = (64, 64)
    categorical_table_alpha: float = 1.0
    categorical_table_alpha_mode: str = "total_mass"
    categorical_table_prior: str = "global"

    @property
    def name(self) -> str:
        return self.cpd_result_name

    @property
    def cpd_result_name(self) -> str:
        return f"vbn_{self.cpd_name}"

    @property
    def inference_result_name(self) -> str:
        return f"vbn_{self.cpd_name}_{self.inf_method}"

    @property
    def training_cache_key(self) -> Tuple[str, str, int, str]:
        return ("vbn", self.cpd_name, int(self.seed), str(self.device))

    def fit_reward_pmf(
        self,
        dag: nx.DiGraph,
        df: pd.DataFrame,
        reward_support: Optional[Sequence[int]] = None,
    ) -> BackendResult:
        from vbn import defaults, VBN

        vbn = VBN(dag, seed=self.seed, device=self.device)
        fit_conf = {"batch_size": max(1, int(len(df) / 4))}

        nodes_cpds: Dict[str, Dict] = {}
        for feat in dag.nodes:
            base = defaults.cpd(self.cpd_name)
            if self.cpd_name == "softmax_nn":
                n_classes = int(df[feat].nunique())
                max_classes = self.softmax_max_classes
                if max_classes is not None and max_classes > 0:
                    n_classes = min(n_classes, int(max_classes))
                base = {
                    **base,
                    "n_classes": n_classes,
                    "label_smoothing": float(self.softmax_label_smoothing),
                    "class_weighting": str(self.softmax_class_weighting),
                }
            elif self.cpd_name == "categorical_embedded_softmax":
                n_classes = int(df[feat].nunique())
                max_classes = self.softmax_max_classes
                if max_classes is not None and max_classes > 0:
                    n_classes = min(n_classes, int(max_classes))
                base = {
                    **base,
                    "n_classes": n_classes,
                    "embedding_dim": int(self.embedded_softmax_embedding_dim),
                    "hidden_dims": tuple(self.embedded_softmax_hidden_dims),
                    "label_smoothing": float(self.softmax_label_smoothing),
                    "class_weighting": str(self.softmax_class_weighting),
                }
            elif self.cpd_name == "categorical_table":
                base = {
                    **base,
                    "alpha": float(self.categorical_table_alpha),
                    "alpha_mode": str(self.categorical_table_alpha_mode),
                    "prior": str(self.categorical_table_prior),
                }
            nodes_cpds[feat] = {**base, "fit": dict(fit_conf)}

        vbn.set_learning_method(
            method=defaults.learning("node_wise"),
            nodes_cpds=nodes_cpds,
        )

        t0 = time.time()
        vbn.fit(df, verbosity=0)

        parents_df = get_parent_combinations(df)
        handle = vbn.get_cpd("reward")

        if reward_support is None:
            reward_support = np.sort(df["reward"].unique())

        pmf = vbn_reward_pmf(
            handle,
            parents_df,
            reward_support=reward_support,
            device=vbn.device,
        )

        return BackendResult(
            name=self.cpd_result_name,
            fit_time=time.time() - t0,
            support=np.asarray(reward_support),
            parents_df=parents_df,
            pmf=pmf,
            artifact=vbn,
        )

    def infer_reward_posterior(
        self,
        result: BackendResult,
        evidence_df: pd.DataFrame,
        reward_support: Sequence[int],
        **kwargs,
    ) -> Tuple[np.ndarray, float]:
        vbn = result.artifact
        if vbn is None:
            raise ValueError("VBN backend missing fitted model for inference")

        requested_n_samples = int(kwargs.get("n_samples", self.inf_n_samples))
        batch_size = int(kwargs.get("batch_size", 256))
        batch_size = max(1, batch_size)
        reward_support = np.asarray(reward_support, dtype=np.int64)
        effective_n_samples = resolve_effective_vbn_n_samples(
            requested_n_samples, reward_support
        )

        pmf = np.zeros((len(evidence_df), len(reward_support)), dtype=np.float64)
        if len(evidence_df) == 0:
            return pmf, 0.0

        vbn.set_inference_method(self.inf_method, n_samples=effective_n_samples)

        evidence_cols = list(evidence_df.columns)
        infer_weights = self.inf_method.lower() != "monte_carlo_marginalization"

        t = 0.0
        for start in range(0, len(evidence_df), batch_size):
            end = min(start + batch_size, len(evidence_df))
            batch = evidence_df.iloc[start:end]
            evidence = build_vbn_evidence_tensors(
                batch[evidence_cols], device=vbn.device, dtype=torch.long
            )
            query = {"target": "reward", "evidence": evidence}
            with torch.no_grad():
                ttt = time.time()
                weights, samples = vbn.infer_posterior(
                    query, n_samples=effective_n_samples
                )
                t += time.time() - ttt
            weights_np = weights.detach().cpu().numpy()
            samples_np = samples.detach().cpu().numpy()
            if not infer_weights:
                weights_np = np.ones_like(weights_np, dtype=np.float64)
            pmf[start:end] = weighted_samples_to_pmf(
                samples_np, weights_np, reward_support
            )

        return pmf, t


# ============================================================
# Logging and persistence
# ============================================================


def resolve_effective_vbn_n_samples(
    requested_n_samples: int, reward_support: Sequence[int]
) -> int:
    support_size = int(len(reward_support))
    requested = int(requested_n_samples)
    return int(max(requested, 4 * support_size))


def unique_preserve_order(items: Sequence[str]) -> List[str]:
    seen = set()
    ordered = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def get_backend_cpd_name(backend: RewardBackend) -> str:
    if isinstance(backend, VBNBackend):
        return backend.cpd_result_name
    return backend.name


def get_backend_inference_name(backend: RewardBackend) -> str:
    if isinstance(backend, VBNBackend):
        return backend.inference_result_name
    return backend.name


def get_vbn_training_key(backend: VBNBackend) -> Tuple[str, str, int, str]:
    return backend.training_cache_key


def group_vbn_backends_by_training_key(
    backends: Sequence[VBNBackend],
) -> Dict[Tuple[str, str, int, str], List[VBNBackend]]:
    groups: Dict[Tuple[str, str, int, str], List[VBNBackend]] = {}
    for backend in backends:
        key = get_vbn_training_key(backend)
        groups.setdefault(key, []).append(backend)
    return groups


def to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, torch.dtype):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if dataclass_isinstance(obj):
        return to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return str(obj)


def dataclass_isinstance(obj: Any) -> bool:
    return hasattr(obj, "__dataclass_fields__")


def make_benchmark_dir(out_dir: str, timestamp: Optional[str] = None) -> Path:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bench_dir = root / f"benchmark_{timestamp}"
    bench_dir.mkdir(parents=True, exist_ok=False)
    return bench_dir


def save_json(path: Path | str, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2)


def load_json(path: Path | str) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def initialize_logs(
    evaluated_backends: Sequence[Any],
    metrics: Sequence[str] = ("kl", "js", "ws", "fit_time"),
    track_ground_truth_time: bool = False,
) -> Dict[str, Any]:
    evaluated_names: List[str] = []
    for item in evaluated_backends:
        if isinstance(item, str):
            evaluated_names.append(item)
        else:
            evaluated_names.append(item.name)
    evaluated_names = unique_preserve_order(evaluated_names)
    logs: Dict[str, Any] = {
        "metrics": {
            metric: {name: [] for name in evaluated_names} for metric in metrics
        },
        "backends": evaluated_names,
    }
    if track_ground_truth_time:
        logs["ground_truth_time"] = []
    return logs


def append_metric(
    logs: Dict[str, Any],
    metric_name: str,
    algo_name: str,
    card: int,
    value: float,
    *,
    spread: Optional[float] = None,
    lower: Optional[float] = None,
    upper: Optional[float] = None,
    n: Optional[int] = None,
) -> None:
    metrics = logs.setdefault("metrics", {})
    if metric_name not in metrics:
        metrics[metric_name] = {name: [] for name in logs.get("backends", [])}
    if algo_name not in metrics[metric_name]:
        metrics[metric_name][algo_name] = []

    entries: List[Dict[str, float]] = metrics[metric_name][algo_name]
    card_val = int(card)
    entry_payload: Dict[str, Any] = {"card": card_val, "value": float(value)}
    if spread is not None:
        entry_payload["spread"] = float(spread)
    if lower is not None:
        entry_payload["lower"] = float(lower)
    if upper is not None:
        entry_payload["upper"] = float(upper)
    if n is not None:
        entry_payload["n"] = int(n)

    for entry in entries:
        if entry.get("card") == card_val:
            entry.update(entry_payload)
            return
    entries.append(entry_payload)


def append_fit_time(logs: Dict[str, Any], result: BackendResult, card: int) -> None:
    append_metric(logs, "fit_time", result.name, card, result.fit_time)


def append_comparison_metrics(
    logs: Dict[str, Any],
    algo_name: str,
    metrics_dict: Dict[str, Any],
    card: int,
) -> None:
    for metric_name, value in metrics_dict.items():
        if isinstance(value, dict):
            append_metric(
                logs,
                metric_name,
                algo_name,
                card,
                value.get("value", float("nan")),
                spread=value.get("spread"),
                lower=value.get("lower"),
                upper=value.get("upper"),
                n=value.get("n"),
            )
        else:
            append_metric(logs, metric_name, algo_name, card, value)


def save_logs(logs: Dict[str, Any], path: Path | str) -> None:
    save_json(path, logs)


def load_logs(path: Path | str) -> Dict[str, Any]:
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid log format in {path}")
    return data


def flatten_logs_to_dataframe(logs: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    metrics = logs.get("metrics", {})
    for metric, by_backend in metrics.items():
        for backend, entries in by_backend.items():
            for entry in entries:
                rows.append(
                    {
                        "card": entry.get("card"),
                        "backend": backend,
                        "metric": metric,
                        "value": entry.get("value"),
                        "spread": entry.get("spread"),
                        "lower": entry.get("lower"),
                        "upper": entry.get("upper"),
                        "n": entry.get("n"),
                    }
                )
    df = pd.DataFrame(
        rows,
        columns=["card", "backend", "metric", "value", "spread", "lower", "upper", "n"],
    )
    if not df.empty:
        df = df.sort_values(["metric", "backend", "card"]).reset_index(drop=True)
    return df


def save_dataframe_csv(
    df: pd.DataFrame, path: Path | str, columns: Optional[List[str]] = None
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty and columns is not None:
        df = pd.DataFrame(columns=columns)
    df.to_csv(path, index=False)


def persist_benchmark(
    run_dir: Path, cpd_logs: Dict[str, Any], inference_logs: Dict[str, Any]
) -> None:
    save_logs(cpd_logs, run_dir / "cpd_logs.json")
    save_logs(inference_logs, run_dir / "inference_logs.json")
    save_dataframe_csv(
        flatten_logs_to_dataframe(cpd_logs),
        run_dir / "cpd_metrics.csv",
        columns=["card", "backend", "metric", "value", "spread", "lower", "upper", "n"],
    )
    save_dataframe_csv(
        flatten_logs_to_dataframe(inference_logs),
        run_dir / "inference_metrics.csv",
        columns=["card", "backend", "metric", "value", "spread", "lower", "upper", "n"],
    )


# ============================================================
# Plotting
# ============================================================


def plot_metrics_grid(
    logs: Dict[str, Any],
    metrics: Sequence[str],
    title: str = "",
    out_path: Optional[Path | str] = None,
    *,
    show_fill_between: bool = True,
) -> None:
    n_metrics = len(metrics)
    n_cols = 2
    n_rows = int(np.ceil(n_metrics / n_cols)) if n_metrics > 0 else 1

    fig, axes = plt.subplots(n_rows, n_cols, dpi=500, figsize=(10, 5 * n_rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.reshape(-1)

    metrics_map = logs.get("metrics", logs)

    for ax, metric in zip(axes, metrics):
        for algo, entries in metrics_map.get(metric, {}).items():
            if not entries:
                continue
            lowers = None
            uppers = None
            if isinstance(entries[0], dict):
                cards = np.array([e["card"] for e in entries], dtype=np.float64)
                values = np.array([e["value"] for e in entries], dtype=np.float64)
                if show_fill_between:
                    lower_vals = [e.get("lower") for e in entries]
                    upper_vals = [e.get("upper") for e in entries]
                    if any(v is not None for v in lower_vals) and any(
                        v is not None for v in upper_vals
                    ):
                        lowers = np.array(
                            [np.nan if v is None else float(v) for v in lower_vals],
                            dtype=np.float64,
                        )
                        uppers = np.array(
                            [np.nan if v is None else float(v) for v in upper_vals],
                            dtype=np.float64,
                        )
            else:
                cards = np.arange(len(entries), dtype=np.float64)
                values = np.array(entries, dtype=np.float64)
            if cards.size == 0:
                continue
            order = np.argsort(cards)
            cards = cards[order]
            values = values[order]
            line = ax.plot(cards, values, marker="o", label=algo)[0]
            if show_fill_between and lowers is not None and uppers is not None:
                lowers = lowers[order]
                uppers = uppers[order]
                valid = np.isfinite(lowers) & np.isfinite(uppers)
                if np.any(valid):
                    ax.fill_between(
                        cards[valid],
                        lowers[valid],
                        uppers[valid],
                        alpha=0.2,
                        color=line.get_color(),
                    )
        ax.set_xlabel("cardinality")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.grid(True)
        if ax == axes[0]:
            ax.legend(loc="best")

    for ax in axes[n_metrics:]:
        ax.axis("off")

    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
    plt.close(fig)


def split_inference_metrics(
    metrics: Sequence[str],
) -> Tuple[List[str], List[str]]:
    distribution_set = {"kl", "js", "ws", "inference_time"}
    point_set = {"mse", "mae", "r2"}
    dist = [m for m in metrics if m in distribution_set]
    point = [m for m in metrics if m in point_set]
    return dist, point


# ============================================================
# Experiment
# ============================================================


def run_single_experiment(
    n_states: int,
    n_actions: int,
    exp_cfg: ExperimentConfig,
    evaluated_backends: Sequence[RewardBackend],
    run_dir: Path,
    log_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cpd_names = unique_preserve_order(
        [get_backend_cpd_name(b) for b in evaluated_backends]
    )
    inference_names = unique_preserve_order(
        [get_backend_inference_name(b) for b in evaluated_backends]
    )
    cpd_logs = initialize_logs(cpd_names, metrics=exp_cfg.metrics)
    inference_logs = initialize_logs(inference_names, metrics=exp_cfg.inference_metrics)
    if log_meta:
        cpd_logs["meta"] = dict(log_meta)
        inference_logs["meta"] = dict(log_meta)
    persist_benchmark(run_dir, cpd_logs, inference_logs)

    non_vbn_backends = [
        backend for backend in evaluated_backends if not isinstance(backend, VBNBackend)
    ]
    vbn_backends = [
        backend for backend in evaluated_backends if isinstance(backend, VBNBackend)
    ]
    vbn_groups = group_vbn_backends_by_training_key(vbn_backends)

    pbar = tqdm(exp_cfg.cards, desc=f"#states={n_states}, #actions={n_actions}")
    for card in pbar:
        training_cache: Dict[Tuple[str, str, int, str], BackendResult] = {}
        dag = get_rl_dag(
            n_states, n_actions, include_next_state=exp_cfg.include_next_state
        )

        data_cfg = DataConfig(
            card=card,
            n_samples_df=exp_cfg.n_samples_df,
            n_states=n_states,
            n_actions=n_actions,
            mode=exp_cfg.mode,
            seed=exp_cfg.seed,
            include_next_state=exp_cfg.include_next_state,
        )
        df = define_df(data_cfg)

        reward_support = get_reward_support(df, card=card)

        pbar.set_postfix_str("ground truth cpd")
        ground_truth_backend = GroundTruthBackend(
            n_states=n_states,
            n_actions=n_actions,
            card=card,
            mode=exp_cfg.mode,
            n_mc_samples=exp_cfg.n_mc_ground_truth,
            seed=exp_cfg.seed,
        )
        ground_truth = ground_truth_backend.fit_reward_pmf(
            dag, df, reward_support=reward_support
        )
        validate_backend_result(ground_truth, reward_support)

        pbar.set_postfix_str("ground truth inference")
        inference_queries = sample_inference_queries(
            df,
            n_states=n_states,
            n_actions=n_actions,
            n_queries=exp_cfg.n_inference_queries,
            seed=exp_cfg.inference_seed,
            include_actions=True,
        )

        gt_inference_support, gt_inference_pmf = estimate_ground_truth_inference_pmf(
            inference_queries,
            n_states=n_states,
            n_actions=n_actions,
            card=card,
            mode=exp_cfg.mode,
            n_mc_samples=exp_cfg.n_mc_ground_truth,
            seed=exp_cfg.inference_seed,
        )
        if not np.array_equal(gt_inference_support, reward_support):
            raise ValueError(
                "Ground truth inference support mismatch: "
                f"{gt_inference_support} vs {reward_support}"
            )
        validate_pmf(gt_inference_pmf, "ground_truth_inference")

        for backend in non_vbn_backends:
            cpd_name = get_backend_cpd_name(backend)
            inference_name = get_backend_inference_name(backend)
            pbar.set_postfix_str(f"{cpd_name} learning")
            result = backend.fit_reward_pmf(dag, df, reward_support=reward_support)
            validate_backend_result(result, reward_support)
            append_fit_time(cpd_logs, result, card)

            if not ground_truth.parents_df.equals(result.parents_df):
                raise ValueError(
                    f"Parents mismatch between {ground_truth.name} and {result.name}"
                )

            metrics_dict = compare_reward_cpds(
                reference_pmf=ground_truth.pmf,
                candidate_pmf=result.pmf,
                support=reward_support,
                aggregation_mode=exp_cfg.aggregation_mode,
                spread_mode=exp_cfg.spread_mode,
            )
            append_comparison_metrics(cpd_logs, cpd_name, metrics_dict, card)
            persist_benchmark(run_dir, cpd_logs, inference_logs)

            pbar.set_postfix_str(f"{inference_name} inference")
            inferred_pmf, inference_time = backend.infer_reward_posterior(
                result,
                inference_queries,
                reward_support,
                n_samples=exp_cfg.vbn_inference_n_samples,
                batch_size=exp_cfg.vbn_inference_batch_size,
            )
            validate_pmf(inferred_pmf, f"{inference_name}_inference")
            inference_metrics = compare_reward_cpds(
                reference_pmf=gt_inference_pmf,
                candidate_pmf=inferred_pmf,
                support=reward_support,
                aggregation_mode=exp_cfg.aggregation_mode,
                spread_mode=exp_cfg.spread_mode,
            )
            append_comparison_metrics(
                inference_logs, inference_name, inference_metrics, card
            )
            append_metric(
                inference_logs, "inference_time", inference_name, card, inference_time
            )
            point_metrics = compare_posterior_point_predictions(
                reference_pmf=gt_inference_pmf,
                candidate_pmf=inferred_pmf,
                support=reward_support,
            )
            append_comparison_metrics(
                inference_logs, inference_name, point_metrics, card
            )
            persist_benchmark(run_dir, cpd_logs, inference_logs)
            if result.artifact is not None and str(
                getattr(result.artifact, "device", "")
            ).startswith("cuda"):
                torch.cuda.empty_cache()
            del inferred_pmf, inference_metrics, point_metrics, metrics_dict, result

        for key, group in vbn_groups.items():
            train_backend = group[0]
            cpd_name = train_backend.cpd_result_name
            pbar.set_postfix_str(f"{cpd_name} learning")
            if key not in training_cache:
                result = train_backend.fit_reward_pmf(
                    dag, df, reward_support=reward_support
                )
                validate_backend_result(result, reward_support)
                training_cache[key] = result
            else:
                result = training_cache[key]

            if not ground_truth.parents_df.equals(result.parents_df):
                raise ValueError(
                    f"Parents mismatch between {ground_truth.name} and {result.name}"
                )

            metrics_dict = compare_reward_cpds(
                reference_pmf=ground_truth.pmf,
                candidate_pmf=result.pmf,
                support=reward_support,
                aggregation_mode=exp_cfg.aggregation_mode,
                spread_mode=exp_cfg.spread_mode,
            )
            append_fit_time(cpd_logs, result, card)
            append_comparison_metrics(cpd_logs, cpd_name, metrics_dict, card)
            persist_benchmark(run_dir, cpd_logs, inference_logs)
            del metrics_dict

            for inf_backend in group:
                inference_name = inf_backend.inference_result_name
                pbar.set_postfix_str(f"{inference_name} inference")
                inferred_pmf, inference_time = inf_backend.infer_reward_posterior(
                    result,
                    inference_queries,
                    reward_support,
                    n_samples=exp_cfg.vbn_inference_n_samples,
                    batch_size=exp_cfg.vbn_inference_batch_size,
                )
                validate_pmf(inferred_pmf, f"{inference_name}_inference")
                inference_metrics = compare_reward_cpds(
                    reference_pmf=gt_inference_pmf,
                    candidate_pmf=inferred_pmf,
                    support=reward_support,
                )
                append_comparison_metrics(
                    inference_logs, inference_name, inference_metrics, card
                )
                append_metric(
                    inference_logs,
                    "inference_time",
                    inference_name,
                    card,
                    inference_time,
                )
                point_metrics = compare_posterior_point_predictions(
                    reference_pmf=gt_inference_pmf,
                    candidate_pmf=inferred_pmf,
                    support=reward_support,
                )
                append_comparison_metrics(
                    inference_logs, inference_name, point_metrics, card
                )
                persist_benchmark(run_dir, cpd_logs, inference_logs)
                del inferred_pmf, inference_metrics, point_metrics

            if key in training_cache:
                del training_cache[key]
            if result.artifact is not None and str(
                getattr(result.artifact, "device", "")
            ).startswith("cuda"):
                torch.cuda.empty_cache()
            del result

        del df
        # Release per-card intermediates to avoid carrying state across cards.
        del inference_queries
        del ground_truth
        del ground_truth_backend
        del gt_inference_pmf
        del gt_inference_support
        del reward_support
        del dag
        del data_cfg
        training_cache.clear()
        del training_cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    del non_vbn_backends, vbn_groups

    return cpd_logs, inference_logs


def run_experiments(exp_cfg: ExperimentConfig) -> None:
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark_dir = make_benchmark_dir(exp_cfg.out_dir, timestamp=run_timestamp)

    resolved_device = resolve_torch_device(exp_cfg.vbn_device)
    print("VBN Device: ", resolved_device)
    device_warning = None
    if (
        isinstance(exp_cfg.vbn_device, str)
        and exp_cfg.vbn_device.lower().startswith("cuda")
        and not torch.cuda.is_available()
    ):
        device_warning = "CUDA requested but unavailable; fell back to CPU."
    """VBNBackend(
        cpd_name="kde",
        inf_method="monte_carlo_marginalization",
        inf_n_samples=exp_cfg.vbn_inference_n_samples,
        seed=exp_cfg.seed,
        device=resolved_device,
    ),
    VBNBackend(
        cpd_name="kde",
        inf_method="likelihood_weighting",
        inf_n_samples=exp_cfg.vbn_inference_n_samples,
        seed=exp_cfg.seed,
        device=resolved_device,
    ),
    VBNBackend(
        cpd_name="kde",
        inf_method="lbp",
        inf_n_samples=exp_cfg.vbn_inference_n_samples,
        seed=exp_cfg.seed,
        device=resolved_device,
    ),
    VBNBackend(
        cpd_name="kde",
        inf_method="importance_sampling",
        inf_n_samples=exp_cfg.vbn_inference_n_samples,
        seed=exp_cfg.seed,
        device=resolved_device,
    ),
    VBNBackend(
        cpd_name="gaussian_nn",
        inf_method="monte_carlo_marginalization",
        inf_n_samples=exp_cfg.vbn_inference_n_samples,
        seed=exp_cfg.seed,
        device=resolved_device,
    ),
    VBNBackend(
        cpd_name="gaussian_nn",
        inf_method="likelihood_weighting",
        inf_n_samples=exp_cfg.vbn_inference_n_samples,
        seed=exp_cfg.seed,
        device=resolved_device,
    ),
    VBNBackend(
        cpd_name="gaussian_nn",
        inf_method="lbp",
        inf_n_samples=exp_cfg.vbn_inference_n_samples,
        seed=exp_cfg.seed,
        device=resolved_device,
    ),
    VBNBackend(
        cpd_name="gaussian_nn",
        inf_method="importance_sampling",
        inf_n_samples=exp_cfg.vbn_inference_n_samples,
        seed=exp_cfg.seed,
        device=resolved_device,
    )""",

    """
    VBNBackend(
            cpd_name="softmax_nn",
            inf_method="categorical_exact",
            inf_n_samples=exp_cfg.vbn_inference_n_samples,
            seed=exp_cfg.seed,
            device=resolved_device,
            softmax_max_classes=exp_cfg.softmax_max_classes,
            softmax_label_smoothing=exp_cfg.softmax_label_smoothing,
            softmax_class_weighting=exp_cfg.softmax_class_weighting,
            embedded_softmax_embedding_dim=exp_cfg.embedded_softmax_embedding_dim,
            embedded_softmax_hidden_dims=exp_cfg.embedded_softmax_hidden_dims,
            categorical_table_alpha=exp_cfg.categorical_table_alpha,
        ),
        VBNBackend(
            cpd_name="categorical_embedded_softmax",
            inf_method="categorical_exact",
            inf_n_samples=exp_cfg.vbn_inference_n_samples,
            seed=exp_cfg.seed,
            device=resolved_device,
            softmax_max_classes=exp_cfg.softmax_max_classes,
            softmax_label_smoothing=exp_cfg.softmax_label_smoothing,
            softmax_class_weighting=exp_cfg.softmax_class_weighting,
            embedded_softmax_embedding_dim=exp_cfg.embedded_softmax_embedding_dim,
            embedded_softmax_hidden_dims=exp_cfg.embedded_softmax_hidden_dims,
            categorical_table_alpha=exp_cfg.categorical_table_alpha,
        ),
    """

    evaluated_backends = [
        PgmpyBackend(),
        VBNBackend(
            cpd_name="categorical_table",
            inf_method="categorical_exact",
            inf_n_samples=exp_cfg.vbn_inference_n_samples,
            seed=exp_cfg.seed,
            device=resolved_device,
            softmax_max_classes=exp_cfg.softmax_max_classes,
            softmax_label_smoothing=exp_cfg.softmax_label_smoothing,
            softmax_class_weighting=exp_cfg.softmax_class_weighting,
            embedded_softmax_embedding_dim=exp_cfg.embedded_softmax_embedding_dim,
            embedded_softmax_hidden_dims=exp_cfg.embedded_softmax_hidden_dims,
            categorical_table_alpha=exp_cfg.categorical_table_alpha,
            categorical_table_alpha_mode=exp_cfg.categorical_table_alpha_mode,
            categorical_table_prior=exp_cfg.categorical_table_prior,
        ),
        VBNBackend(
            cpd_name="categorical_table",
            inf_method="resampled_importance_sampling",
            inf_n_samples=exp_cfg.vbn_inference_n_samples,
            seed=exp_cfg.seed,
            device=resolved_device,
            softmax_max_classes=exp_cfg.softmax_max_classes,
            softmax_label_smoothing=exp_cfg.softmax_label_smoothing,
            softmax_class_weighting=exp_cfg.softmax_class_weighting,
            embedded_softmax_embedding_dim=exp_cfg.embedded_softmax_embedding_dim,
            embedded_softmax_hidden_dims=exp_cfg.embedded_softmax_hidden_dims,
            categorical_table_alpha=exp_cfg.categorical_table_alpha,
            categorical_table_alpha_mode=exp_cfg.categorical_table_alpha_mode,
            categorical_table_prior=exp_cfg.categorical_table_prior,
        ),
        VBNBackend(
            cpd_name="categorical_table",
            inf_method="resampled_importance_sampling",
            inf_n_samples=exp_cfg.vbn_inference_n_samples,
            seed=exp_cfg.seed,
            device=resolved_device,
            softmax_max_classes=exp_cfg.softmax_max_classes,
            softmax_label_smoothing=exp_cfg.softmax_label_smoothing,
            softmax_class_weighting=exp_cfg.softmax_class_weighting,
            embedded_softmax_embedding_dim=exp_cfg.embedded_softmax_embedding_dim,
            embedded_softmax_hidden_dims=exp_cfg.embedded_softmax_hidden_dims,
            categorical_table_alpha=exp_cfg.categorical_table_alpha,
            categorical_table_alpha_mode=exp_cfg.categorical_table_alpha_mode,
            categorical_table_prior=exp_cfg.categorical_table_prior,
        ),
    ]
    cpd_backend_names = unique_preserve_order(
        [get_backend_cpd_name(b) for b in evaluated_backends]
    )
    inference_backend_names = unique_preserve_order(
        [get_backend_inference_name(b) for b in evaluated_backends]
    )

    n_runs = len(exp_cfg.n_states_list) * len(exp_cfg.n_actions_list)
    use_subdirs = n_runs > 1

    top_level_meta = {
        "timestamp": run_timestamp,
        "resolved_device": str(resolved_device),
        "vbn_device_requested": exp_cfg.vbn_device,
        "vbn_device_warning": device_warning,
        "n_runs": n_runs,
        "backends": [b.name for b in evaluated_backends],
        "cpd_backends": cpd_backend_names,
        "inference_backends": inference_backend_names,
    }
    save_json(benchmark_dir / "run_metadata.json", top_level_meta)

    for n_states in exp_cfg.n_states_list:
        for n_actions in exp_cfg.n_actions_list:
            run_dir = (
                benchmark_dir
                if not use_subdirs
                else benchmark_dir / f"states_{n_states}_actions_{n_actions}"
            )
            run_dir.mkdir(parents=True, exist_ok=True)

            config_payload = {
                "timestamp": run_timestamp,
                "resolved_device": str(resolved_device),
                "vbn_device_requested": exp_cfg.vbn_device,
                "vbn_device_warning": device_warning,
                "n_states": n_states,
                "n_actions": n_actions,
                "backends": [b.name for b in evaluated_backends],
                "cpd_backends": cpd_backend_names,
                "inference_backends": inference_backend_names,
                "config": asdict(exp_cfg),
            }
            save_json(run_dir / "config.json", config_payload)

            run_single_experiment(
                n_states=n_states,
                n_actions=n_actions,
                exp_cfg=exp_cfg,
                evaluated_backends=evaluated_backends,
                run_dir=run_dir,
                log_meta={
                    "timestamp": run_timestamp,
                    "resolved_device": str(resolved_device),
                    "vbn_device_requested": exp_cfg.vbn_device,
                    "vbn_device_warning": device_warning,
                    "n_states": n_states,
                    "n_actions": n_actions,
                },
            )

            cpd_logs_disk = load_logs(run_dir / "cpd_logs.json")
            inference_logs_disk = load_logs(run_dir / "inference_logs.json")

            dist_metrics, point_metrics = split_inference_metrics(
                exp_cfg.inference_metrics
            )
            spread_label = exp_cfg.spread_mode or (
                "std" if exp_cfg.aggregation_mode == "mean" else "iqr_std"
            )
            prefix = (
                f"#states: {n_states}, #actions: {n_actions}, "
                f"aggregation: {exp_cfg.aggregation_mode}, spread: {spread_label}"
            )

            plot_metrics_grid(
                logs=cpd_logs_disk,
                metrics=exp_cfg.metrics,
                title=f"{prefix} - CPD metrics",
                out_path=run_dir / "cpd_metrics.png",
            )
            plot_metrics_grid(
                logs=inference_logs_disk,
                metrics=dist_metrics,
                title=f"{prefix} - Inference distribution metrics",
                out_path=run_dir / "inference_distribution_metrics.png",
            )
            if point_metrics:
                plot_metrics_grid(
                    logs=inference_logs_disk,
                    metrics=point_metrics,
                    title=f"{prefix} - Inference point metrics",
                    out_path=run_dir / "inference_point_metrics.png",
                )


# ============================================================
# CLI helpers
# ============================================================


def parse_int_list(values: Any) -> Tuple[int, ...]:
    if isinstance(values, (list, tuple)):
        return tuple(int(v) for v in values)
    if values is None:
        return tuple()
    text = str(values).replace(",", " ").strip()
    if not text:
        return tuple()
    return tuple(int(part) for part in text.split())


def parse_str_list(values: Any) -> Tuple[str, ...]:
    if isinstance(values, (list, tuple)):
        return tuple(str(v) for v in values)
    if values is None:
        return tuple()
    text = str(values).replace(",", " ").strip()
    if not text:
        return tuple()
    return tuple(part for part in text.split() if part)


def parse_optional_int(values: Any) -> Optional[int]:
    if values is None:
        return None
    text = str(values).strip().lower()
    if text in {"", "none", "null"}:
        return None
    return int(values)


def build_arg_parser() -> argparse.ArgumentParser:
    default_cfg = ExperimentConfig()
    parser = argparse.ArgumentParser(
        description="Run VBN stress-test experiments with configurable settings."
    )
    parser.add_argument("--n-samples-df", type=int, default=default_cfg.n_samples_df)
    parser.add_argument("--mode", type=str, default=default_cfg.mode)
    parser.add_argument("--seed", type=int, default=default_cfg.seed)
    parser.add_argument(
        "--cards",
        type=parse_int_list,
        default=default_cfg.cards,
        help="Comma or space-separated list of cardinalities.",
    )
    parser.add_argument(
        "--n-states-list",
        type=parse_int_list,
        default=default_cfg.n_states_list,
        help="Comma or space-separated list of state counts.",
    )
    parser.add_argument(
        "--n-actions-list",
        type=parse_int_list,
        default=default_cfg.n_actions_list,
        help="Comma or space-separated list of action counts.",
    )
    parser.add_argument(
        "--n-mc-ground-truth", type=int, default=default_cfg.n_mc_ground_truth
    )
    parser.add_argument(
        "--n-inference-queries",
        type=int,
        default=default_cfg.n_inference_queries,
    )
    parser.add_argument(
        "--inference-seed", type=int, default=default_cfg.inference_seed
    )
    parser.add_argument(
        "--vbn-inference-n-samples",
        type=int,
        default=default_cfg.vbn_inference_n_samples,
    )
    parser.add_argument(
        "--vbn-inference-batch-size",
        type=int,
        default=default_cfg.vbn_inference_batch_size,
    )
    parser.add_argument(
        "--vbn-device",
        type=str,
        default=default_cfg.vbn_device,
        help="Torch device: auto, cpu, cuda, or cuda:0.",
    )
    parser.add_argument("--out-dir", type=str, default=default_cfg.out_dir)
    parser.add_argument(
        "--include-next-state",
        action=argparse.BooleanOptionalAction,
        default=default_cfg.include_next_state,
        help="Include next_state nodes and data generation.",
    )
    parser.add_argument(
        "--softmax-max-classes",
        type=int,
        default=(
            0
            if default_cfg.softmax_max_classes is None
            else default_cfg.softmax_max_classes
        ),
        help="Cap softmax_nn n_classes; 0 disables the cap.",
    )
    parser.add_argument(
        "--softmax-label-smoothing",
        type=float,
        default=default_cfg.softmax_label_smoothing,
        help="Label smoothing for softmax-based CPDs.",
    )
    parser.add_argument(
        "--softmax-class-weighting",
        type=str,
        default=default_cfg.softmax_class_weighting,
        choices=("none", "inverse_freq"),
        help="Class weighting for softmax-based CPDs.",
    )
    parser.add_argument(
        "--embedded-softmax-embedding-dim",
        type=int,
        default=default_cfg.embedded_softmax_embedding_dim,
        help="Embedding dimension for categorical_embedded_softmax.",
    )
    parser.add_argument(
        "--embedded-softmax-hidden-dims",
        type=parse_int_list,
        default=default_cfg.embedded_softmax_hidden_dims,
        help="Hidden dims for categorical_embedded_softmax.",
    )
    parser.add_argument(
        "--categorical-table-alpha",
        type=float,
        default=default_cfg.categorical_table_alpha,
        help=(
            "Dirichlet/Laplace smoothing mass for categorical_table. Interpreted per "
            "class when alpha_mode=per_class, or as total mass when alpha_mode=total_mass."
        ),
    )
    parser.add_argument(
        "--categorical-table-alpha-mode",
        type=str,
        default=default_cfg.categorical_table_alpha_mode,
        choices=("per_class", "total_mass"),
        help="Interpretation of categorical_table alpha.",
    )
    parser.add_argument(
        "--categorical-table-prior",
        type=str,
        default=default_cfg.categorical_table_prior,
        choices=("uniform", "global"),
        help="Base prior for categorical_table smoothing.",
    )
    parser.add_argument(
        "--aggregation-mode",
        type=str,
        default=default_cfg.aggregation_mode,
        choices=("mean", "iqm"),
        help="How to aggregate per-query metrics into a single value.",
    )
    parser.add_argument(
        "--spread-mode",
        type=str,
        default=default_cfg.spread_mode,
        choices=("std", "iqr_std", "none"),
        help="Spread used for fill_between. Default: std for mean, iqr_std for iqm.",
    )
    parser.add_argument(
        "--metrics",
        type=parse_str_list,
        default=default_cfg.metrics,
        help="Comma or space-separated CPD metrics.",
    )
    parser.add_argument(
        "--inference-metrics",
        type=parse_str_list,
        default=default_cfg.inference_metrics,
        help="Comma or space-separated inference metrics.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    if args.softmax_max_classes is None:
        max_classes = None
    else:
        max_classes = int(args.softmax_max_classes)
        if max_classes <= 0:
            max_classes = None
    return ExperimentConfig(
        n_samples_df=args.n_samples_df,
        n_states_list=tuple(args.n_states_list),
        n_actions_list=tuple(args.n_actions_list),
        cards=tuple(args.cards),
        mode=args.mode,
        seed=args.seed,
        n_mc_ground_truth=args.n_mc_ground_truth,
        n_inference_queries=args.n_inference_queries,
        inference_seed=args.inference_seed,
        vbn_inference_n_samples=args.vbn_inference_n_samples,
        vbn_inference_batch_size=args.vbn_inference_batch_size,
        vbn_device=args.vbn_device,
        out_dir=args.out_dir,
        include_next_state=args.include_next_state,
        softmax_max_classes=max_classes,
        softmax_label_smoothing=args.softmax_label_smoothing,
        softmax_class_weighting=args.softmax_class_weighting,
        embedded_softmax_embedding_dim=args.embedded_softmax_embedding_dim,
        embedded_softmax_hidden_dims=tuple(args.embedded_softmax_hidden_dims),
        categorical_table_alpha=args.categorical_table_alpha,
        categorical_table_alpha_mode=args.categorical_table_alpha_mode,
        categorical_table_prior=args.categorical_table_prior,
        metrics=tuple(args.metrics),
        inference_metrics=tuple(args.inference_metrics),
        aggregation_mode=args.aggregation_mode,
        spread_mode=None if args.spread_mode == "none" else args.spread_mode,
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    exp_cfg = config_from_args(args)
    run_experiments(exp_cfg)
