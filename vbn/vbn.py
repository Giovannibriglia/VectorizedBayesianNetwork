from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from types import SimpleNamespace
from typing import Dict, Optional

import networkx as nx
import pandas as pd
import torch
import yaml

from vbn.core.base import Query
from vbn.core.dags import StaticDAG
from vbn.core.registry import (
    INFERENCE_REGISTRY,
    LEARNING_REGISTRY,
    SAMPLING_REGISTRY,
    UPDATE_REGISTRY,
)
from vbn.core.utils import ensure_2d, ensure_tensor, resolve_device, set_seed
from vbn.utils import df_to_tensor_dict, dict_to_device


@dataclass
class ConfigItem:
    name: str
    params: Dict


class ConfigNamespace(SimpleNamespace):
    def __getitem__(self, item):
        return getattr(self, item)


def _load_configs() -> ConfigNamespace:
    categories = {}
    base = resources.files("vbn.configs")
    for category in ["cpds", "learning", "inference", "sampling", "update"]:
        cat_dir = base / category
        items = {}
        if cat_dir.is_dir():
            for path in sorted(cat_dir.iterdir(), key=lambda p: p.name):
                if path.name.endswith(".yaml"):
                    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                    name = data.pop("name", path.stem)
                    items[path.stem] = ConfigItem(name=name, params=data)
        categories[category] = ConfigNamespace(**items)
    return ConfigNamespace(**categories)


def _resolve_method(
    method,
    registry: Dict[str, type],
    config_namespace: Optional[ConfigNamespace] = None,
):
    if isinstance(method, ConfigItem):
        return method.name, method.params
    if isinstance(method, str):
        key = method.lower().strip()
        if key not in registry:
            raise ValueError(
                f"Unknown method '{method}'. Available: {list(registry.keys())}"
            )
        return key, {}
    if callable(method):
        return method, {}
    raise TypeError("method must be a string, ConfigItem, or callable")


class VBN:
    """Vectorized Bayesian Network main interface."""

    def __init__(
        self,
        dag: nx.DiGraph,
        seed: Optional[int] = None,
        device: Optional[str | torch.device] = None,
    ) -> None:
        self.device = resolve_device(device)
        set_seed(seed)
        self.seed = seed
        self.dag = StaticDAG(dag)
        self.nodes: Dict[str, torch.nn.Module] = {}
        self.config = _load_configs()

        self._learning = None
        self._inference = None
        self._sampling = None
        self._update_policy = None

    # ----------------- configuration -----------------
    def set_learning_method(
        self, method, nodes_cpds: Optional[Dict[str, Dict]] = None, **kwargs
    ):
        if isinstance(method, ConfigItem):
            name, base_params = method.name, method.params
        elif isinstance(method, str):
            key = method.lower().strip()
            if key not in LEARNING_REGISTRY:
                raise ValueError(
                    f"Unknown learning method '{method}'. Available: {list(LEARNING_REGISTRY.keys())}"
                )
            name, base_params = key, {}
        elif callable(method):
            self._learning = method
            return
        else:
            raise TypeError("method must be a string, ConfigItem, or callable")
        params = {**base_params, **kwargs}
        learning_cls = LEARNING_REGISTRY[name]
        self._learning = learning_cls(nodes_cpds=nodes_cpds, **params)

    def set_inference_method(self, method, **kwargs):
        if isinstance(method, ConfigItem):
            name, base_params = method.name, method.params
        elif isinstance(method, str):
            key = method.lower().strip()
            if key not in INFERENCE_REGISTRY:
                raise ValueError(
                    f"Unknown inference method '{method}'. Available: {list(INFERENCE_REGISTRY.keys())}"
                )
            name, base_params = key, {}
        elif callable(method):
            self._inference = method
            return
        else:
            raise TypeError("method must be a string, ConfigItem, or callable")
        params = {**base_params, **kwargs}
        inference_cls = INFERENCE_REGISTRY[name]
        self._inference = inference_cls(**params)

    def set_sampling_method(self, method, **kwargs):
        if isinstance(method, ConfigItem):
            name, base_params = method.name, method.params
        elif isinstance(method, str):
            key = method.lower().strip()
            if key not in SAMPLING_REGISTRY:
                raise ValueError(
                    f"Unknown sampling method '{method}'. Available: {list(SAMPLING_REGISTRY.keys())}"
                )
            name, base_params = key, {}
        elif callable(method):
            self._sampling = method
            return
        else:
            raise TypeError("method must be a string, ConfigItem, or callable")
        params = {**base_params, **kwargs}
        sampling_cls = SAMPLING_REGISTRY[name]
        self._sampling = sampling_cls(**params)

    # ----------------- data prep -----------------
    def _prepare_data(
        self, data: Dict[str, torch.Tensor] | pd.DataFrame
    ) -> Dict[str, torch.Tensor]:
        if isinstance(data, pd.DataFrame):
            tensor_data = df_to_tensor_dict(data, device=self.device)
        elif isinstance(data, dict):
            tensor_data = dict_to_device(data, device=self.device)
        else:
            raise TypeError("data must be pandas DataFrame or dict[str, Tensor]")

        missing = [n for n in self.dag.nodes() if n not in tensor_data]
        if missing:
            raise ValueError(f"Missing data for DAG nodes: {missing}")
        return tensor_data

    # ----------------- fit/update -----------------
    def fit(self, data: Dict[str, torch.Tensor] | pd.DataFrame, **kwargs) -> None:
        if self._learning is None:
            raise RuntimeError("Call set_learning_method(...) before fit().")
        tensor_data = self._prepare_data(data)
        self.nodes = self._learning.fit(self, tensor_data, **kwargs)

    def update(
        self,
        data: Dict[str, torch.Tensor] | pd.DataFrame,
        update_method: Optional[str] = None,
        **kwargs,
    ):
        if not self.nodes:
            raise RuntimeError("Call fit(...) before update(...).")
        tensor_data = self._prepare_data(data)
        if update_method is not None:
            if isinstance(update_method, ConfigItem):
                name, base_params = update_method.name, update_method.params
                params = {**base_params, **kwargs}
            else:
                key = update_method.lower().strip()
                if key not in UPDATE_REGISTRY:
                    raise ValueError(
                        f"Unknown update method '{update_method}'. Available: {list(UPDATE_REGISTRY.keys())}"
                    )
                params = kwargs
                name = key
            update_cls = UPDATE_REGISTRY[name]
            init_kwargs = {
                k: v for k, v in params.items() if k in {"max_size", "replay_ratio"}
            }
            if self._update_policy is None or not isinstance(
                self._update_policy, update_cls
            ):
                self._update_policy = update_cls(**init_kwargs)
            else:
                for k, v in init_kwargs.items():
                    setattr(self._update_policy, k, v)
            policy_kwargs = {
                k: v for k, v in params.items() if k not in {"max_size", "replay_ratio"}
            }
        else:
            if self._update_policy is None:
                raise RuntimeError(
                    "update_method must be provided for the first update call"
                )
            policy_kwargs = kwargs
        self.nodes = self._update_policy.update(self, tensor_data, **policy_kwargs)

    # ----------------- inference/sampling -----------------
    def infer_posterior(self, query: Dict | Query, **kwargs):
        if self._inference is None:
            raise RuntimeError(
                "Call set_inference_method(...) before infer_posterior()."
            )
        q = self._normalize_query(query)
        return self._inference.infer_posterior(self, q, **kwargs)

    def sample(self, query: Dict | Query, n_samples: int = 200, **kwargs):
        if self._sampling is None:
            raise RuntimeError("Call set_sampling_method(...) before sample().")
        q = self._normalize_query(query)
        return self._sampling.sample(self, q, n_samples=n_samples, **kwargs)

    def _normalize_query(self, query: Dict | Query) -> Query:
        if isinstance(query, Query):
            evidence = {
                k: ensure_2d(ensure_tensor(v, device=self.device))
                for k, v in query.evidence.items()
            }
            return Query(target=query.target, evidence=evidence)
        if not isinstance(query, dict):
            raise TypeError("query must be a dict or Query")
        target = query.get("target") or query.get("target_feature")
        if target is None:
            raise ValueError("query must contain 'target'")
        evidence = {
            k: ensure_2d(ensure_tensor(v, device=self.device))
            for k, v in query.get("evidence", {}).items()
        }
        return Query(target=target, evidence=evidence)

    # ----------------- device management -----------------
    def to_device(self, device: str | torch.device) -> None:
        self.device = resolve_device(device)
        for cpd in self.nodes.values():
            cpd.to(self.device)
            if hasattr(cpd, "device"):
                cpd.device = self.device
        if hasattr(self._update_policy, "_buffer"):
            new_buffer = {}
            for k, (p, x) in self._update_policy._buffer.items():
                new_buffer[k] = (p.to(self.device), x.to(self.device))
            self._update_policy._buffer = new_buffer
