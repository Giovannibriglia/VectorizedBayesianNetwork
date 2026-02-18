from __future__ import annotations

import ast
from typing import Any, Callable, Dict

import numpy as np
import torch


def coerce_scalar(value: Any) -> Any:
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, torch.Tensor) and value.ndim == 0:
        return value.item()
    return value


def _is_numeric_string(value: str) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


def _coerce_number(value: Any, target_type: type, key: str) -> Any:
    value = coerce_scalar(value)
    if isinstance(value, str):
        raw = value.strip()
        if not _is_numeric_string(raw):
            raise ValueError(
                f"Invalid hyperparameter {key}='{value}' (expected {target_type.__name__})."
            )
        value = float(raw) if target_type is float else int(float(raw))
    try:
        return target_type(value)
    except Exception as exc:
        raise ValueError(
            f"Invalid hyperparameter {key}='{value}' (expected {target_type.__name__})."
        ) from exc


def _coerce_bool(value: Any, key: str) -> bool:
    value = coerce_scalar(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"true", "1", "yes"}:
            return True
        if raw in {"false", "0", "no"}:
            return False
    raise ValueError(f"Invalid hyperparameter {key}='{value}' (expected bool).")


def list_of(element_type: type) -> Callable[[Any, str], list]:
    def _coerce(value: Any, key: str) -> list:
        value = coerce_scalar(value)
        if isinstance(value, str):
            raw = value.strip()
            try:
                parsed = ast.literal_eval(raw)
            except Exception:
                parsed = [v.strip() for v in raw.split(",") if v.strip()]
            value = parsed
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"Invalid hyperparameter {key}='{value}' (expected list).")
        out = []
        for item in value:
            out.append(_coerce_number(item, element_type, key))
        return out

    return _coerce


def coerce_numbers(values: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    coerced = dict(values)
    for key, caster in schema.items():
        if key not in coerced:
            continue
        val = coerced[key]
        if caster is int:
            coerced[key] = _coerce_number(val, int, key)
        elif caster is float:
            coerced[key] = _coerce_number(val, float, key)
        elif caster is bool:
            coerced[key] = _coerce_bool(val, key)
        elif callable(caster):
            coerced[key] = caster(val, key)
        else:
            coerced[key] = coerce_scalar(val)
    return coerced


FIT_SCHEMA = {
    "epochs": int,
    "batch_size": int,
    "lr": float,
    "weight_decay": float,
    "n_steps": int,
    "show_progress": bool,
    "verbosity": int,
    "max_grad_norm": float,
}

UPDATE_SCHEMA = {
    "lr": float,
    "n_steps": int,
    "batch_size": int,
    "weight_decay": float,
}

CPD_SCHEMAS = {
    "gaussian_nn": {
        "hidden_dims": list_of(int),
        "min_scale": float,
    },
    "softmax_nn": {
        "n_classes": int,
        "hidden_dims": list_of(int),
        "label_smoothing": float,
        "min_bin_width": float,
        "within_bin_scale": float,
        "within_bin_clip": bool,
    },
    "mdn": {
        "n_components": int,
        "hidden_dims": list_of(int),
        "min_scale": float,
    },
    "kde": {
        "bandwidth": float,
        "parent_bandwidth": float,
        "max_points": int,
        "min_scale": float,
    },
}
