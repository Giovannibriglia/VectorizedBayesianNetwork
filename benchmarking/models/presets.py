from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal

import yaml

from .config import ComponentSpec, make_component, ModelBenchmarkConfig

Backend = Literal["vbn", "pgmpy"]
Mode = Literal["cpds", "inference"]

_VBN_PRESETS_PATH = Path(__file__).parent / "presets" / "vbn.yaml"
_PGMPY_PRESETS_PATH = Path(__file__).parent / "presets" / "pgmpy.yaml"

_DEFAULT_VBN_INFERENCE = "importance_sampling"
_DEFAULT_PGMPY_INFERENCE = "exact_variable_elimination"

_PRESET_CACHE: dict[tuple[str, str], dict[str, dict]] = {}


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Preset file not found: {path}")
    payload = yaml.safe_load(path.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Preset file must contain a mapping: {path}")
    return payload


def load_presets(
    *,
    vbn_path: Path | None = None,
    pgmpy_path: Path | None = None,
) -> dict[str, dict]:
    vbn_path = Path(vbn_path) if vbn_path is not None else _VBN_PRESETS_PATH
    pgmpy_path = Path(pgmpy_path) if pgmpy_path is not None else _PGMPY_PRESETS_PATH
    return {
        "vbn": _read_yaml(vbn_path),
        "pgmpy": _read_yaml(pgmpy_path),
    }


def _get_cached_presets() -> dict[str, dict]:
    cache_key = (str(_VBN_PRESETS_PATH), str(_PGMPY_PRESETS_PATH))
    if cache_key in _PRESET_CACHE:
        return _PRESET_CACHE[cache_key]
    presets = load_presets()
    _PRESET_CACHE[cache_key] = presets
    return presets


def _require_mapping(value: Any, *, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _require_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _normalize_kwargs(value: Any) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("kwargs must be a mapping")
    return dict(value)


def _normalize_vbn_preset(mode: Mode, preset: dict) -> dict:
    learning = _require_mapping(preset.get("learning"), label="learning")
    learning_method = _require_string(learning.get("method"), label="learning.method")
    learning_kwargs = _normalize_kwargs(learning.get("kwargs"))

    cpds = _require_mapping(preset.get("cpds"), label="cpds")
    default = _require_mapping(cpds.get("default"), label="cpds.default")
    default_method = _require_string(default.get("method"), label="cpds.default.method")
    default_kwargs = _normalize_kwargs(default.get("kwargs"))

    per_node_raw = cpds.get("per_node") or {}
    if not isinstance(per_node_raw, dict):
        raise ValueError("cpds.per_node must be a mapping when provided")
    per_node: dict[str, dict] = {}
    for node, conf in per_node_raw.items():
        conf_map = _require_mapping(conf, label=f"cpds.per_node[{node}]")
        method = _require_string(
            conf_map.get("method"), label=f"cpds.per_node[{node}].method"
        )
        kwargs = _normalize_kwargs(conf_map.get("kwargs"))
        per_node[str(node)] = {"method": method, "kwargs": kwargs}

    inference = None
    sampling = None
    if mode == "inference":
        inference_map = _require_mapping(preset.get("inference"), label="inference")
        inference_method = _require_string(
            inference_map.get("method"), label="inference.method"
        )
        inference_kwargs = _normalize_kwargs(inference_map.get("kwargs"))
        inference = {"method": inference_method, "kwargs": inference_kwargs}

        sampling_raw = preset.get("sampling")
        if sampling_raw is not None:
            sampling_map = _require_mapping(sampling_raw, label="sampling")
            sampling_method = _require_string(
                sampling_map.get("method"), label="sampling.method"
            )
            sampling_kwargs = _normalize_kwargs(sampling_map.get("kwargs"))
            sampling = {"method": sampling_method, "kwargs": sampling_kwargs}

    normalized: dict[str, Any] = {
        "backend": "vbn",
        "mode": mode,
        "learning": {"method": learning_method, "kwargs": learning_kwargs},
        "cpds": {
            "default": {"method": default_method, "kwargs": default_kwargs},
            "per_node": per_node,
        },
    }
    if inference is not None:
        normalized["inference"] = inference
    if sampling is not None:
        normalized["sampling"] = sampling
    return normalized


def _normalize_pgmpy_preset(mode: Mode, preset: dict) -> dict:
    cpds = _require_mapping(preset.get("cpds"), label="cpds")
    estimator = _require_string(cpds.get("estimator"), label="cpds.estimator")
    cpd_kwargs = _normalize_kwargs(cpds.get("kwargs"))

    inference = None
    if mode == "inference":
        inference_map = _require_mapping(preset.get("inference"), label="inference")
        inference_method = _require_string(
            inference_map.get("method"), label="inference.method"
        )
        inference_kwargs = _normalize_kwargs(inference_map.get("kwargs"))
        inference = {"method": inference_method, "kwargs": inference_kwargs}

    normalized: dict[str, Any] = {
        "backend": "pgmpy",
        "mode": mode,
        "cpds": {"estimator": estimator, "kwargs": cpd_kwargs},
    }
    if inference is not None:
        normalized["inference"] = inference
    return normalized


def validate_preset(backend: Backend, mode: Mode, preset: dict) -> dict:
    if backend == "vbn":
        return _normalize_vbn_preset(mode, preset)
    if backend == "pgmpy":
        return _normalize_pgmpy_preset(mode, preset)
    raise ValueError(f"Unsupported backend '{backend}'")


def list_presets(backend: Backend, mode: Mode) -> list[str]:
    presets = _get_cached_presets()
    if backend not in presets:
        available = ", ".join(sorted(presets)) or "<none>"
        raise KeyError(f"Unknown backend '{backend}'. Available backends: {available}")
    mode_map = presets[backend].get(mode, {})
    if not isinstance(mode_map, dict):
        return []
    return sorted(mode_map)


def get_preset(backend: Backend, mode: Mode, preset_name: str) -> dict:
    presets = _get_cached_presets()
    if backend not in presets:
        available = ", ".join(sorted(presets)) or "<none>"
        raise KeyError(f"Unknown backend '{backend}'. Available backends: {available}")
    mode_map = presets[backend].get(mode, {})
    if not isinstance(mode_map, dict):
        raise KeyError(
            f"No presets available for backend '{backend}' and mode '{mode}'."
        )
    if preset_name not in mode_map:
        available = ", ".join(sorted(mode_map)) or "<none>"
        raise KeyError(
            f"Unknown preset '{preset_name}' for backend '{backend}' (mode={mode}). "
            f"Available: {available}"
        )
    return validate_preset(backend, mode, mode_map[preset_name])


def _component_from_method(
    component: str,
    method: str,
    kwargs: dict,
) -> ComponentSpec:
    return make_component(component, method, kwargs=kwargs)


def _vbn_config_from_preset(
    mode: Mode, preset: dict, config_id: str
) -> ModelBenchmarkConfig:
    learning = preset.get("learning") or {}
    learning_spec = _component_from_method(
        "learning",
        learning.get("method", "node_wise"),
        learning.get("kwargs", {}),
    )

    cpds = preset.get("cpds") or {}
    default = cpds.get("default") or {}
    cpd_kwargs = dict(default.get("kwargs", {}))
    per_node = cpds.get("per_node") or {}
    if per_node:
        cpd_kwargs = dict(cpd_kwargs)
        cpd_kwargs["per_node"] = per_node
    cpd_spec = _component_from_method(
        "cpd",
        default.get("method", "softmax_nn"),
        cpd_kwargs,
    )

    inference = preset.get("inference") or {}
    inference_method = inference.get("method") or _DEFAULT_VBN_INFERENCE
    inference_spec = _component_from_method(
        "inference",
        inference_method,
        inference.get("kwargs", {}),
    )

    return ModelBenchmarkConfig(
        model="vbn",
        config_id=config_id,
        learning=learning_spec,
        cpd=cpd_spec,
        inference=inference_spec,
    )


def _pgmpy_config_from_preset(
    mode: Mode, preset: dict, config_id: str
) -> ModelBenchmarkConfig:
    cpds = preset.get("cpds") or {}
    estimator = cpds.get("estimator", "mle")
    kwargs = cpds.get("kwargs", {})
    learning_spec = _component_from_method("learning", estimator, kwargs)

    cpd_name = f"tabular_{estimator}"
    cpd_spec = _component_from_method("cpd", cpd_name, kwargs)

    inference = preset.get("inference") or {}
    inference_method = inference.get("method") or _DEFAULT_PGMPY_INFERENCE
    inference_spec = _component_from_method(
        "inference",
        inference_method,
        inference.get("kwargs", {}),
    )

    return ModelBenchmarkConfig(
        model="pgmpy",
        config_id=config_id,
        learning=learning_spec,
        cpd=cpd_spec,
        inference=inference_spec,
    )


def get_preset_config(
    backend: Backend, mode: Mode, config_id: str
) -> ModelBenchmarkConfig:
    preset = get_preset(backend, mode, config_id)
    if backend == "vbn":
        return _vbn_config_from_preset(mode, preset, config_id)
    if backend == "pgmpy":
        return _pgmpy_config_from_preset(mode, preset, config_id)
    raise KeyError(f"Unknown backend '{backend}'")


def get_model_presets(
    model: Backend,
    mode: Mode | None = None,
) -> Dict[str, ModelBenchmarkConfig]:
    resolved_mode: Mode = mode or "inference"
    return {
        name: get_preset_config(model, resolved_mode, name)
        for name in list_presets(model, resolved_mode)
    }


def get_default_config(
    model: Backend, mode: Mode | None = None
) -> ModelBenchmarkConfig:
    resolved_mode: Mode = mode or "inference"
    return get_preset_config(model, resolved_mode, "default")
