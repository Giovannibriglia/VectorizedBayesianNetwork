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
            if values is None and "vars" not in evidence:
                values = evidence
    if values is None:
        return {}
    if not isinstance(values, dict):
        return {}
    return {k: v for k, v in values.items() if v is not None}


def _extract_do(query: dict) -> dict:
    values = query.get("do_values")
    if values is None:
        do = query.get("do")
        if isinstance(do, dict):
            values = do.get("values")
            if values is None and "vars" not in do:
                values = do
    if values is None:
        return {}
    if not isinstance(values, dict):
        return {}
    return {k: v for k, v in values.items() if v is not None}


def get_device_info() -> str:
    """
    Returns a human-readable description of the current torch device.
    """
    if not torch.cuda.is_available():
        return "cpu"

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(device)
    total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)

    return f"cuda:0 ({gpu_name}, {total_mem:.1f}GB)"


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


def _is_continuous(domain: dict, node: str) -> bool:
    meta = _domain_node(domain, node)
    node_type = meta.get("type")
    if node_type == "continuous":
        return True
    if node_type == "discrete":
        return False
    states = meta.get("states") or []
    return len(states) == 0


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


def _result_ok(result: dict | None) -> bool:
    if not isinstance(result, dict):
        return False
    fmt = result.get("format")
    if fmt == "categorical_probs":
        return result.get("probs") is not None
    if fmt == "normal_params":
        mean = result.get("mean")
        std = result.get("std")
        if mean is not None and std is not None:
            return math.isfinite(float(mean)) and math.isfinite(float(std))
        samples = result.get("samples") or []
        return bool(samples)
    if fmt == "samples_1d":
        samples = result.get("samples") or []
        return bool(samples)
    return False


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


def _estimate_discrete_posterior_batch(
    samples: torch.Tensor,
    weights: torch.Tensor,
    k: int,
) -> list[list[float]]:
    if samples.dim() == 3:
        samples = samples[:, :, 0]
    if samples.dim() != 2:
        raise ValueError(f"Expected samples with 2D shape, got {tuple(samples.shape)}")
    if weights.dim() != 2:
        raise ValueError(f"Expected weights with 2D shape, got {tuple(weights.shape)}")
    if samples.shape[0] != weights.shape[0]:
        raise ValueError("Samples/weights batch size mismatch")
    probs: list[list[float]] = []
    for idx in range(samples.shape[0]):
        probs.append(_estimate_discrete_posterior(samples[idx], weights[idx], k))
    return probs


@register_benchmark_model
class VBNBenchmarkModel(BaseBenchmarkModel):
    name = "vbn"
    family = "neural_bn"
    version = _package_version()
    supports_batched_inference_queries = True

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

    def _continuous_params(
        self, cpd_handle: Any, parents_values: dict
    ) -> tuple[float | None, float | None]:
        candidates = [cpd_handle]
        cpd_obj = getattr(cpd_handle, "cpd", None)
        if cpd_obj is not None and cpd_obj is not cpd_handle:
            candidates.append(cpd_obj)
        parents_tensor = None
        if hasattr(cpd_handle, "_parents_tensor"):
            try:
                parents_tensor = cpd_handle._parents_tensor(parents_values)
            except Exception:
                parents_tensor = None
        method_names = (
            "conditional_mean_std",
            "mean_std",
            "get_mean_and_std",
            "mean_and_std",
            "mean_stddev",
        )
        for obj in candidates:
            for name in method_names:
                if not hasattr(obj, name):
                    continue
                try:
                    fn = getattr(obj, name)
                    out = fn(
                        parents_tensor if parents_tensor is not None else parents_values
                    )
                    if isinstance(out, tuple) and len(out) >= 2:
                        mean = float(np.asarray(out[0]).reshape(-1)[0])
                        std = float(np.asarray(out[1]).reshape(-1)[0])
                        if math.isfinite(mean) and math.isfinite(std):
                            return mean, std
                except Exception:
                    continue
        for obj in candidates:
            if not hasattr(obj, "_params"):
                continue
            try:
                loc, scale = obj._params(parents_tensor)
                mean = float(np.asarray(loc).reshape(-1)[0])
                std = float(np.asarray(scale).reshape(-1)[0])
                if math.isfinite(mean) and math.isfinite(std):
                    return mean, std
            except Exception:
                continue
        return None, None

    def _extract_samples_1d(self, samples: Any) -> np.ndarray:
        if isinstance(samples, torch.Tensor):
            samples = samples.detach().cpu().numpy()
        arr = np.asarray(samples)
        if arr.ndim == 0:
            return arr.reshape(1)
        if arr.ndim == 3:
            arr = arr.reshape(-1, arr.shape[-1])
        if arr.ndim == 2:
            if arr.shape[1] == 1:
                return arr[:, 0]
            raise ValueError("Multivariate continuous targets are unsupported")
        if arr.ndim == 1:
            return arr
        raise ValueError(f"Unsupported sample shape {arr.shape}")

    def _continuous_from_samples(
        self, samples: Any, weights: Any | None = None
    ) -> dict:
        vals = self._extract_samples_1d(samples)
        if vals.size == 0:
            raise ValueError("No samples returned for continuous target")
        vals = vals.astype(float)
        wts = None
        if weights is not None:
            if isinstance(weights, torch.Tensor):
                weights = weights.detach().cpu().numpy()
            wts_arr = np.asarray(weights).reshape(-1)
            if wts_arr.size == vals.size and np.isfinite(wts_arr).any():
                wts = wts_arr.astype(float)
        if wts is not None:
            wts = np.clip(wts, 0.0, np.inf)
            total = float(wts.sum())
            if total > 0:
                wts = wts / total
                mean = float(np.sum(wts * vals))
                var = float(np.sum(wts * (vals - mean) ** 2))
                std = math.sqrt(var)
            else:
                mean = float(np.mean(vals))
                std = float(np.std(vals, ddof=0))
        else:
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=0))
        max_keep = min(int(vals.size), 2048)
        sample_list = [float(x) for x in vals[:max_keep]]
        if math.isfinite(mean) and math.isfinite(std):
            return {
                "format": "normal_params",
                "mean": mean,
                "std": std,
                "n_samples": int(vals.size),
                "samples": sample_list,
            }
        return {
            "format": "samples_1d",
            "samples": sample_list,
            "n_samples": int(vals.size),
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

    def _cpd_distribution(
        self, target: str, evidence: dict, do: dict | None = None
    ) -> dict:
        do = do or {}
        meta = _domain_node(self.domain, target)
        if target in do:
            do_value = _to_float(do[target])
            if _is_continuous(self.domain, target):
                return {
                    "format": "normal_params",
                    "mean": float(do_value),
                    "std": 0.0,
                    "n_samples": None,
                    "samples": None,
                }
            states = meta.get("states") or []
            k = len(states)
            if k == 0:
                return {
                    "format": "categorical_probs",
                    "probs": None,
                    "k": 0,
                    "support": [],
                }
            idx = int(round(float(do_value)))
            if idx < 0 or idx >= k:
                raise ValueError("do value out of bounds for discrete target")
            probs = [0.0] * k
            probs[idx] = 1.0
            return {
                "format": "categorical_probs",
                "probs": probs,
                "k": k,
                "support": list(range(k)),
            }
        if _is_continuous(self.domain, target):
            parents = list(self._vbn.dag.parents(target))
            if parents and not all(p in evidence for p in parents):
                raise ValueError("Missing parents for continuous CPD")
            parents_values = {p: [_to_float(evidence[p])] for p in parents}
            cpd_handle = self._vbn.cpd(target)
            mean, std = self._continuous_params(cpd_handle, parents_values)
            if mean is not None and std is not None:
                return {
                    "format": "normal_params",
                    "mean": float(mean),
                    "std": float(std),
                    "n_samples": None,
                    "samples": None,
                }
            samples = cpd_handle.sample(parents_values, n_samples=1024)
            return self._continuous_from_samples(samples)

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

    def _infer_distribution(
        self, target: str, evidence: dict, do: dict, n_samples: int
    ) -> dict:
        meta = _domain_node(self.domain, target)
        if _is_continuous(self.domain, target):
            evidence_values = {k: [_to_float(v)] for k, v in evidence.items()}
            do_values = {k: [_to_float(v)] for k, v in do.items()}
            pdf, samples = self._vbn.infer_posterior(
                {"target": target, "evidence": evidence_values, "do": do_values},
                n_samples=int(n_samples),
            )
            return self._continuous_from_samples(samples, weights=pdf)

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
        do_values = {k: [_to_float(v)] for k, v in do.items()}
        pdf, samples = self._vbn.infer_posterior(
            {"target": target, "evidence": evidence_values, "do": do_values},
            n_samples=int(n_samples),
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
            do = _extract_do(query)
            overlap = set(evidence) & set(do)
            if overlap:
                raise ValueError("Nodes cannot be in both evidence and do")
            assignment = {**evidence, **do}
            parents = list(self._vbn.dag.parents(target))
            if target in do:
                result = self._cpd_distribution(target, assignment, do=do)
            elif all(p in assignment for p in parents):
                result = self._cpd_distribution(target, assignment, do=do)
            else:
                n_samples = _get_n_mc(query, default=200)
                result = self._infer_distribution(target, evidence, do, n_samples)
            ok = _result_ok(result)
            error = None if ok else "Unsupported or empty CPD result"
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
            do = _extract_do(query)
            overlap = set(evidence) & set(do)
            if overlap:
                raise ValueError("Nodes cannot be in both evidence and do")
            n_samples = _get_n_mc(query, default=200)
            result = self._infer_distribution(target, evidence, do, n_samples)
            ok = _result_ok(result)
            error = None if ok else "Unsupported or empty inference result"
        except Exception as exc:
            ok = False
            error = f"{type(exc).__name__}: {exc}"
            result = None
        timing_ms = (time.perf_counter() - start) * 1000.0
        return {"ok": ok, "error": error, "timing_ms": timing_ms, "result": result}

    def answer_inference_queries(self, queries: list[dict]) -> list[dict]:
        if not queries:
            return []
        target = queries[0].get("target")
        if not target:
            raise ValueError("Missing target in query")
        if any(_extract_do(q) for q in queries):
            return [self.answer_inference_query(q) for q in queries]
        if _is_continuous(self.domain, target):
            return [self.answer_inference_query(q) for q in queries]
        evidence_vars = None
        for query in queries:
            evidence = (
                query.get("evidence") if isinstance(query.get("evidence"), dict) else {}
            )
            vars_list = evidence.get("vars") or query.get("evidence_vars") or []
            if not isinstance(vars_list, list):
                vars_list = list(vars_list) if vars_list is not None else []
            if evidence_vars is None:
                evidence_vars = list(vars_list)
            elif list(vars_list) != list(evidence_vars):
                raise ValueError("Inconsistent evidence vars in batch")
        evidence_vars = evidence_vars or []

        if not evidence_vars:
            n_samples = _get_n_mc(queries[0], default=200)
            pdf, samples = self._vbn.infer_posterior(
                {"target": target, "evidence": {}}, n_samples=int(n_samples)
            )
            meta = _domain_node(self.domain, target)
            states = meta.get("states") or []
            k = len(states)
            if k == 0:
                result = {
                    "format": "categorical_probs",
                    "probs": None,
                    "k": 0,
                    "support": [],
                }
                return [
                    {"ok": False, "error": "Empty target state space", "result": result}
                    for _ in queries
                ]
            probs = _estimate_discrete_posterior(samples, pdf, k)
            output = {
                "format": "categorical_probs",
                "probs": probs,
                "k": k,
                "support": list(range(k)),
            }
            return [{"ok": True, "error": None, "result": output} for _ in queries]

        evidence_values: dict[str, list[float]] = {v: [] for v in evidence_vars}
        for query in queries:
            values = query.get("evidence_values")
            if values is None:
                evidence = query.get("evidence")
                if isinstance(evidence, dict):
                    values = evidence.get("values")
            if values is None or not isinstance(values, dict):
                raise ValueError("Missing evidence values in batch")
            for var in evidence_vars:
                if var not in values or values[var] is None:
                    raise ValueError("Missing evidence value in batch")
                evidence_values[var].append(values[var])

        def _is_int_like(value: Any) -> bool:
            return isinstance(value, (int, bool, np.integer))

        batched_evidence: dict[str, torch.Tensor] = {}
        for var in evidence_vars:
            values_list = evidence_values[var]
            dtype = (
                torch.long
                if values_list and all(_is_int_like(v) for v in values_list)
                else torch.float32
            )
            batched_evidence[var] = torch.tensor(
                values_list, dtype=dtype, device=self._vbn.device
            )

        n_samples = _get_n_mc(queries[0], default=200)
        pdf, samples = self._vbn.infer_posterior(
            {"target": target, "evidence": batched_evidence}, n_samples=int(n_samples)
        )
        meta = _domain_node(self.domain, target)
        states = meta.get("states") or []
        k = len(states)
        if k == 0:
            result = {
                "format": "categorical_probs",
                "probs": None,
                "k": 0,
                "support": [],
            }
            return [
                {"ok": False, "error": "Empty target state space", "result": result}
                for _ in queries
            ]
        probs_list = _estimate_discrete_posterior_batch(samples, pdf, k)
        if len(probs_list) != len(queries):
            return [self.answer_inference_query(query) for query in queries]
        results = []
        for probs in probs_list:
            output = {
                "format": "categorical_probs",
                "probs": probs,
                "k": k,
                "support": list(range(k)),
            }
            results.append({"ok": True, "error": None, "result": output})
        return results
