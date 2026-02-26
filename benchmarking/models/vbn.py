from __future__ import annotations

import math
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from vbn.vbn import VBN

from .base import BaseBenchmarkModel
from .registry import register_benchmark_model

try:
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover
    import importlib_metadata  # type: ignore


def _package_version() -> str | None:
    try:
        return importlib_metadata.version("vbn")
    except Exception:
        return None


def _extract_evidence(query: dict) -> dict:
    values = query.get("evidence_values")
    if values is None:
        evidence = query.get("evidence")
        if isinstance(evidence, dict):
            values = evidence.get("values")
    if values is None:
        return {}
    if not isinstance(values, dict):
        return {}
    return {k: v for k, v in values.items() if v is not None}


def _get_n_mc(query: dict, default: int = 200) -> int:
    kwargs = query.get("generator_kwargs")
    if isinstance(kwargs, dict) and "n_mc" in kwargs:
        try:
            return int(kwargs["n_mc"])
        except Exception:
            pass
    if "n_mc" in query:
        try:
            return int(query["n_mc"])
        except Exception:
            pass
    return int(default)


def _domain_node(domain: dict, node: str) -> dict:
    if not isinstance(domain, dict):
        return {}
    nodes = domain.get("nodes", {})
    if isinstance(nodes, dict):
        return nodes.get(node, {})
    return {}


def _to_float(value: Any) -> float:
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    try:
        return float(value)
    except Exception:
        return float("nan")


def _normalize_probs(probs: Iterable[float]) -> list[float]:
    arr = np.asarray(list(probs), dtype=float)
    total = float(arr.sum())
    if not math.isfinite(total) or total <= 0:
        return (np.ones_like(arr) / len(arr)).tolist()
    return (arr / total).tolist()


def _build_nodes_cpds(
    domain: dict,
    cpd_name: str,
    cpd_kwargs: dict,
    *,
    per_node: dict | None = None,
) -> dict:
    nodes_cpds: dict[str, dict] = {}
    nodes = domain.get("nodes", {}) if isinstance(domain, dict) else {}
    per_node = per_node or {}
    for node, meta in nodes.items():
        node_type = meta.get("type")
        if node_type not in {"discrete", "continuous"}:
            continue
        node_override = per_node.get(node) if isinstance(per_node, dict) else None
        if isinstance(node_override, dict):
            method = node_override.get("method", cpd_name)
            override_kwargs = dict(node_override.get("kwargs") or {})
            kwargs = {**dict(cpd_kwargs or {}), **override_kwargs}
        else:
            method = cpd_name
            kwargs = dict(cpd_kwargs or {})
        conf = {"cpd": method}
        if kwargs:
            conf.update(kwargs)
        if node_type == "discrete" and method == "softmax_nn":
            k = len(meta.get("states") or [])
            if k > 0:
                conf["n_classes"] = k
        nodes_cpds[node] = conf
    return nodes_cpds


def _estimate_discrete_posterior(
    samples: torch.Tensor,
    weights: torch.Tensor,
    k: int,
) -> list[float]:
    if samples.dim() == 3:
        samples = samples[:, :, 0]
    if samples.dim() == 2:
        samples = samples[0]
    if weights.dim() == 2:
        weights = weights[0]
    vals = samples.detach().cpu().numpy().reshape(-1)
    wts = weights.detach().cpu().numpy().reshape(-1)
    hist = np.zeros(int(k), dtype=float)
    for value, weight in zip(vals, wts):
        if not math.isfinite(weight):
            continue
        idx = int(round(float(value)))
        if idx < 0 or idx >= k:
            continue
        hist[idx] += float(weight)
    return _normalize_probs(hist)


@register_benchmark_model
class VBNBenchmarkModel(BaseBenchmarkModel):
    name = "vbn"
    family = "neural_bn"
    version = _package_version()

    def __init__(
        self,
        *,
        dag,
        seed: int,
        domain: dict,
        benchmark_config=None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            dag=dag,
            seed=seed,
            domain=domain,
            benchmark_config=benchmark_config,
            **kwargs,
        )
        self._vbn = VBN(dag, seed=seed)
        if benchmark_config is not None:
            learning_method = benchmark_config.learning.name
            learning_kwargs = dict(benchmark_config.learning.kwargs)
            inference_method = benchmark_config.inference.name
            inference_kwargs = dict(benchmark_config.inference.kwargs)
            nodes_cpds = kwargs.get("nodes_cpds")
            if nodes_cpds is None:
                cpd_kwargs = dict(benchmark_config.cpd.kwargs)
                per_node = cpd_kwargs.pop("per_node", None)
                nodes_cpds = _build_nodes_cpds(
                    domain,
                    benchmark_config.cpd.name,
                    cpd_kwargs,
                    per_node=per_node,
                )
            self._vbn.set_learning_method(
                learning_method, nodes_cpds=nodes_cpds, **learning_kwargs
            )
            self._vbn.set_inference_method(inference_method, **inference_kwargs)
        else:
            learning_method = kwargs.get("learning_method", "node_wise")
            inference_method = kwargs.get("inference_method", "importance_sampling")
            nodes_cpds = kwargs.get("nodes_cpds")

            if nodes_cpds is None:
                nodes_cpds = {}
                nodes = domain.get("nodes", {}) if isinstance(domain, dict) else {}
                for node, meta in nodes.items():
                    if meta.get("type") == "discrete":
                        k = len(meta.get("states") or [])
                        if k > 0:
                            nodes_cpds[node] = {"cpd": "softmax_nn", "n_classes": k}
                    elif meta.get("type") == "continuous":
                        nodes_cpds[node] = {"cpd": "gaussian_nn"}

            self._vbn.set_learning_method(learning_method, nodes_cpds=nodes_cpds)
            self._vbn.set_inference_method(inference_method)

    def supports(self) -> dict:
        return {
            "can_fit": True,
            "can_answer_cpd": True,
            "can_answer_inference": True,
            "uses_inference": True,
        }

    def fit(
        self, data_df: pd.DataFrame, *, progress: bool = True, **kwargs: Any
    ) -> None:
        verbosity = kwargs.pop("verbosity", None)
        show_progress = kwargs.pop("show_progress", None)
        if verbosity is None:
            verbosity = 1 if progress else 0
        if show_progress is None:
            show_progress = bool(progress)
        self._vbn.fit(
            data_df, verbosity=verbosity, show_progress=show_progress, **kwargs
        )

    def _cpd_distribution(self, target: str, evidence: dict) -> dict:
        meta = _domain_node(self.domain, target)
        states = meta.get("states") or []
        k = len(states)
        if k == 0:
            return {
                "format": "categorical_probs",
                "probs": None,
                "k": 0,
                "support": [],
            }
        parents = list(self._vbn.dag.parents(target))
        if parents and not all(p in evidence for p in parents):
            return {
                "format": "categorical_probs",
                "probs": None,
                "k": k,
                "support": list(range(k)),
            }
        parents_values = {p: [_to_float(evidence[p])] for p in parents}
        probs = []
        cpd = self._vbn.cpd(target)
        for idx in range(k):
            x = torch.tensor([[float(idx)]], device=self._vbn.device)
            logp = cpd.log_prob(x, parents_values).detach().cpu().numpy().item()
            probs.append(math.exp(logp))
        return {
            "format": "categorical_probs",
            "probs": _normalize_probs(probs),
            "k": k,
            "support": list(range(k)),
        }

    def _infer_distribution(self, target: str, evidence: dict, n_samples: int) -> dict:
        meta = _domain_node(self.domain, target)
        states = meta.get("states") or []
        k = len(states)
        if k == 0:
            return {
                "format": "categorical_probs",
                "probs": None,
                "k": 0,
                "support": [],
            }
        evidence_values = {k: [_to_float(v)] for k, v in evidence.items()}
        pdf, samples = self._vbn.infer_posterior(
            {"target": target, "evidence": evidence_values}, n_samples=int(n_samples)
        )
        probs = _estimate_discrete_posterior(samples, pdf, k)
        return {
            "format": "categorical_probs",
            "probs": probs,
            "k": k,
            "support": list(range(k)),
        }

    def answer_cpd_query(self, query: dict) -> dict:
        start = time.perf_counter()
        try:
            target = query.get("target")
            if not target:
                raise ValueError("Missing target in query")
            evidence = _extract_evidence(query)
            parents = list(self._vbn.dag.parents(target))
            if parents and all(p in evidence for p in parents):
                result = self._cpd_distribution(target, evidence)
            else:
                n_samples = _get_n_mc(query, default=200)
                result = self._infer_distribution(target, evidence, n_samples)
            ok = result.get("probs") is not None
            error = None
        except Exception as exc:
            ok = False
            error = f"{type(exc).__name__}: {exc}"
            result = None
        timing_ms = (time.perf_counter() - start) * 1000.0
        return {"ok": ok, "error": error, "timing_ms": timing_ms, "result": result}

    def answer_inference_query(self, query: dict) -> dict:
        start = time.perf_counter()
        try:
            target = query.get("target")
            if not target:
                raise ValueError("Missing target in query")
            evidence = _extract_evidence(query)
            n_samples = _get_n_mc(query, default=200)
            result = self._infer_distribution(target, evidence, n_samples)
            ok = result.get("probs") is not None
            error = None
        except Exception as exc:
            ok = False
            error = f"{type(exc).__name__}: {exc}"
            result = None
        timing_ms = (time.perf_counter() - start) * 1000.0
        return {"ok": ok, "error": error, "timing_ms": timing_ms, "result": result}
