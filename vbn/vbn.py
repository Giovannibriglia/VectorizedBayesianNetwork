from __future__ import annotations

import json
import os
from dataclasses import dataclass
from importlib import metadata, resources
from types import SimpleNamespace
from typing import Any, Dict, Optional

import networkx as nx
import pandas as pd
import torch
import yaml

from vbn.core.base import Query
from vbn.core.cpd_handle import CPDHandle
from vbn.core.dags import StaticDAG
from vbn.core.registry import (
    CPD_REGISTRY,
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


def _get_vbn_version() -> Optional[str]:
    try:
        return metadata.version("vbn")
    except Exception:
        return None


def _resolve_checkpoint_paths(path: str) -> tuple[str, Optional[str]]:
    _, ext = os.path.splitext(path)
    if ext in {".pt", ".pth", ".ckpt"}:
        return path, None
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, "checkpoint.pt"), os.path.join(path, "meta.json")


def _cpd_registry_key(cpd) -> str:
    for key, cls in CPD_REGISTRY.items():
        if isinstance(cpd, cls):
            return key
    raise ValueError(f"CPD class '{cpd.__class__.__name__}' is not registered")


def _cpd_key_from_class(cpd_cls) -> Optional[str]:
    for key, cls in CPD_REGISTRY.items():
        if cls is cpd_cls:
            return key
    return None


def _serialize_nodes_cpds(nodes_cpds: Optional[Dict[str, Dict]]) -> Optional[Dict]:
    if nodes_cpds is None:
        return None
    serialized: Dict[str, Dict] = {}
    for node, conf in nodes_cpds.items():
        if conf is None:
            serialized[node] = {}
            continue
        conf_copy = dict(conf)
        cpd_value = conf_copy.get("cpd")
        if cpd_value is None or isinstance(cpd_value, str):
            serialized[node] = conf_copy
            continue
        key = _cpd_key_from_class(cpd_value)
        if key is None:
            raise ValueError(
                f"CPD class '{cpd_value}' for node '{node}' is not registered"
            )
        conf_copy["cpd"] = key
        serialized[node] = conf_copy
    return serialized


def _infer_dtype(nodes: Dict[str, torch.nn.Module]) -> Optional[str]:
    for cpd in nodes.values():
        for param in cpd.parameters():
            return str(param.dtype)
        for buffer in cpd.buffers():
            return str(buffer.dtype)
        extra = None
        if hasattr(cpd, "get_extra_state"):
            extra = cpd.get_extra_state()
        if isinstance(extra, dict):
            for value in extra.values():
                if isinstance(value, torch.Tensor):
                    return str(value.dtype)
    return None


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
        self._learning_config: Optional[Dict[str, Any]] = None
        self._inference_config: Optional[Dict[str, Any]] = None
        self._sampling_config: Optional[Dict[str, Any]] = None
        self._update_config: Optional[Dict[str, Any]] = None

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
            self._learning_config = {
                "callable": True,
                "name": getattr(method, "__qualname__", str(method)),
            }
            return
        else:
            raise TypeError("method must be a string, ConfigItem, or callable")
        params = {**base_params, **kwargs}
        serialized_nodes_cpds = _serialize_nodes_cpds(nodes_cpds)
        learning_cls = LEARNING_REGISTRY[name]
        self._learning = learning_cls(nodes_cpds=serialized_nodes_cpds, **params)
        self._learning_config = {
            "name": name,
            "params": params,
            "nodes_cpds": serialized_nodes_cpds,
        }

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
            self._inference_config = {
                "callable": True,
                "name": getattr(method, "__qualname__", str(method)),
            }
            return
        else:
            raise TypeError("method must be a string, ConfigItem, or callable")
        params = {**base_params, **kwargs}
        inference_cls = INFERENCE_REGISTRY[name]
        self._inference = inference_cls(**params)
        self._inference_config = {"name": name, "params": params}

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
            self._sampling_config = {
                "callable": True,
                "name": getattr(method, "__qualname__", str(method)),
            }
            return
        else:
            raise TypeError("method must be a string, ConfigItem, or callable")
        params = {**base_params, **kwargs}
        sampling_cls = SAMPLING_REGISTRY[name]
        self._sampling = sampling_cls(**params)
        self._sampling_config = {"name": name, "params": params}

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
            self._update_config = {
                "name": name,
                "params": params,
                "init_kwargs": init_kwargs,
                "policy_kwargs": policy_kwargs,
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
        pdf, samples = self._inference.infer_posterior(self, q, **kwargs)
        return pdf.detach(), samples.detach()

    def sample(self, query: Dict | Query, n_samples: int = 200, **kwargs):
        if self._sampling is None:
            raise RuntimeError("Call set_sampling_method(...) before sample().")
        q = self._normalize_query(query)
        samples = self._sampling.sample(self, q, n_samples=n_samples, **kwargs)
        if isinstance(samples, dict):
            return {k: v.detach() for k, v in samples.items()}
        return samples.detach()

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

    # ----------------- CPD access -----------------
    def cpd(self, node: str) -> CPDHandle:
        return CPDHandle(self, node)

    # ----------------- persistence -----------------
    def save(
        self,
        path: str,
        *,
        include_configs: bool = True,
        extra: Optional[dict] = None,
    ) -> None:
        missing = [n for n in self.dag.nodes() if n not in self.nodes]
        if missing:
            raise RuntimeError(
                f"Cannot save model with missing CPDs for nodes: {missing}"
            )
        if include_configs:
            for label, cfg in [
                ("learning", self._learning_config),
                ("inference", self._inference_config),
                ("sampling", self._sampling_config),
                ("update", self._update_config),
            ]:
                if cfg and cfg.get("callable"):
                    raise ValueError(
                        f"Cannot serialize callable {label} method: {cfg.get('name')}"
                    )

        checkpoint_path, meta_path = _resolve_checkpoint_paths(path)

        dag_info = {
            "nodes": list(self.dag.nodes()),
            "edges": list(self.dag.edges()),
            "topological_order": list(self.dag.topological_order()),
            "parents": {n: list(self.dag.parents(n)) for n in self.dag.nodes()},
        }

        nodes_state: Dict[str, Dict[str, object]] = {}
        for node in self.dag.topological_order():
            cpd = self.nodes[node]
            key = _cpd_registry_key(cpd)
            init_kwargs = {}
            if hasattr(cpd, "get_init_kwargs"):
                init_kwargs = cpd.get_init_kwargs() or {}
            extra_state = None
            if hasattr(cpd, "get_extra_state"):
                extra_state = cpd.get_extra_state()
            nodes_state[node] = {
                "cpd_key": key,
                "class_name": cpd.__class__.__name__,
                "input_dim": cpd.input_dim,
                "output_dim": cpd.output_dim,
                "seed": self.seed,
                "init_kwargs": init_kwargs,
                "state_dict": cpd.state_dict(),
                "extra_state": extra_state,
            }

        meta = {
            "vbn_version": _get_vbn_version(),
            "torch_version": str(torch.__version__),
            "dtype": _infer_dtype(self.nodes),
            "device": str(self.device),
            "seed": self.seed,
        }

        checkpoint = {
            "dag": dag_info,
            "nodes": nodes_state,
            "meta": meta,
            "extra": extra,
        }
        if include_configs:
            checkpoint["config"] = {
                "learning": self._learning_config,
                "inference": self._inference_config,
                "sampling": self._sampling_config,
                "update": self._update_config,
            }
            if self._update_policy is not None and hasattr(
                self._update_policy, "get_state"
            ):
                checkpoint["update_state"] = self._update_policy.get_state()

        torch.save(checkpoint, checkpoint_path)

        if meta_path is not None:
            summary = {
                "meta": meta,
                "dag": dag_info,
                "nodes": {k: {"cpd_key": v["cpd_key"]} for k, v in nodes_state.items()},
                "config": checkpoint.get("config"),
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

    @classmethod
    def load(
        cls,
        path: str,
        *,
        map_location: str | torch.device = "auto",
    ) -> "VBN":
        _, ext = os.path.splitext(path)
        if ext in {".pt", ".pth", ".ckpt"}:
            checkpoint_path = path
        else:
            checkpoint_path = os.path.join(path, "checkpoint.pt")
        device = resolve_device(map_location)
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )

        dag_info = checkpoint.get("dag", {})
        g = nx.DiGraph()
        g.add_nodes_from(dag_info.get("nodes", []))
        g.add_edges_from(dag_info.get("edges", []))

        meta = checkpoint.get("meta", {})
        seed = meta.get("seed")
        vbn = cls(g, seed=seed, device=device)

        config = checkpoint.get("config") or {}
        learning_cfg = config.get("learning")
        if learning_cfg and learning_cfg.get("name"):
            vbn.set_learning_method(
                learning_cfg.get("name"),
                nodes_cpds=learning_cfg.get("nodes_cpds"),
                **(learning_cfg.get("params") or {}),
            )
        inference_cfg = config.get("inference")
        if inference_cfg and inference_cfg.get("name"):
            vbn.set_inference_method(
                inference_cfg.get("name"),
                **(inference_cfg.get("params") or {}),
            )
        sampling_cfg = config.get("sampling")
        if sampling_cfg and sampling_cfg.get("name"):
            vbn.set_sampling_method(
                sampling_cfg.get("name"),
                **(sampling_cfg.get("params") or {}),
            )
        update_cfg = config.get("update")
        if update_cfg:
            update_name = update_cfg.get("name")
            if update_name:
                update_cls = UPDATE_REGISTRY.get(update_name)
                if update_cls is None:
                    raise ValueError(
                        f"Unknown update method '{update_name}' in checkpoint"
                    )
                init_kwargs = update_cfg.get("init_kwargs") or {}
                vbn._update_policy = update_cls(**init_kwargs)
                vbn._update_config = update_cfg

        nodes_state = checkpoint.get("nodes", {})
        vbn.nodes = {}
        for node, info in nodes_state.items():
            cpd_key = info.get("cpd_key")
            if cpd_key not in CPD_REGISTRY:
                raise ValueError(f"Unknown CPD key '{cpd_key}' for node '{node}'")
            cpd_cls = CPD_REGISTRY[cpd_key]
            init_kwargs = info.get("init_kwargs") or {}
            cpd = cpd_cls(
                input_dim=int(info.get("input_dim", 0)),
                output_dim=int(info.get("output_dim", 1)),
                device=device,
                seed=info.get("seed", seed),
                **init_kwargs,
            )
            cpd.load_state_dict(info.get("state_dict") or {})
            extra_state = info.get("extra_state")
            if extra_state is not None and hasattr(cpd, "set_extra_state"):
                cpd.set_extra_state(extra_state)
            cpd.to(device)
            if hasattr(cpd, "device"):
                cpd.device = device
            vbn.nodes[node] = cpd

        update_state = checkpoint.get("update_state")
        if vbn._update_policy is not None and update_state is not None:
            if hasattr(vbn._update_policy, "set_state"):
                vbn._update_policy.set_state(update_state)

        return vbn
