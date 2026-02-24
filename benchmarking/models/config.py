from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    key: str
    kwargs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "key": self.key,
            "kwargs": dict(self.kwargs),
        }


@dataclass(frozen=True)
class ModelBenchmarkConfig:
    model: str
    config_id: str
    learning: ComponentSpec
    cpd: ComponentSpec
    inference: ComponentSpec

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "config_id": self.config_id,
            "learning": self.learning.to_dict(),
            "cpd": self.cpd.to_dict(),
            "inference": self.inference.to_dict(),
        }

    def run_key(self) -> str:
        return (
            f"{self.model}/{self.config_id}"
            f"|learn={self.learning.name}"
            f"|cpd={self.cpd.name}"
            f"|inf={self.inference.name}"
        )


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(config: ModelBenchmarkConfig) -> str:
    payload = _canonical_json(config.to_dict())
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def make_component(
    component: str,
    name: str,
    *,
    key: str | None = None,
    kwargs: Mapping[str, Any] | None = None,
) -> ComponentSpec:
    resolved_key = key if key is not None else f"{component}:{name}"
    return ComponentSpec(name=name, key=resolved_key, kwargs=dict(kwargs or {}))


def apply_overrides(
    config: ModelBenchmarkConfig, overrides: Mapping[str, Any] | None
) -> ModelBenchmarkConfig:
    if not overrides:
        return config
    allowed = {"learning", "cpd", "inference"}
    unknown = set(overrides) - allowed
    if unknown:
        raise ValueError(
            "Override keys must be one of learning/cpd/inference. "
            f"Unknown: {sorted(unknown)}"
        )

    learning = _apply_component_override(
        "learning", config.learning, overrides.get("learning")
    )
    cpd = _apply_component_override("cpd", config.cpd, overrides.get("cpd"))
    inference = _apply_component_override(
        "inference", config.inference, overrides.get("inference")
    )
    return ModelBenchmarkConfig(
        model=config.model,
        config_id=config.config_id,
        learning=learning,
        cpd=cpd,
        inference=inference,
    )


def _apply_component_override(
    component: str,
    base: ComponentSpec,
    override: Mapping[str, Any] | None,
) -> ComponentSpec:
    if not override:
        return base
    if not isinstance(override, Mapping):
        raise ValueError(
            f"Override for '{component}' must be an object (mapping), "
            f"got {type(override).__name__}."
        )
    allowed = {"name", "key", "kwargs"}
    unknown = set(override) - allowed
    if unknown:
        raise ValueError(
            f"Override for '{component}' must only include name/key/kwargs. "
            f"Unknown: {sorted(unknown)}"
        )

    name = override.get("name", base.name)
    if not isinstance(name, str):
        raise ValueError(f"Override for '{component}.name' must be a string")

    kwargs_override = override.get("kwargs")
    if kwargs_override is None:
        merged_kwargs = dict(base.kwargs)
    elif isinstance(kwargs_override, Mapping):
        merged_kwargs = {**base.kwargs, **dict(kwargs_override)}
    else:
        raise ValueError(
            f"Override for '{component}.kwargs' must be an object (mapping)"
        )

    if "key" in override and override["key"] is not None:
        key = override["key"]
    elif "name" in override:
        key = f"{component}:{name}"
    else:
        key = base.key
    if not isinstance(key, str):
        raise ValueError(f"Override for '{component}.key' must be a string")

    return ComponentSpec(name=name, key=key, kwargs=merged_kwargs)
