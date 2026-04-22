from __future__ import annotations

import math
import time
from collections import defaultdict
from copy import deepcopy
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .base import BaseBenchmarkModel
from .registry import register_benchmark_model

try:
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover
    import importlib_metadata  # type: ignore


def _package_version() -> str | None:
    for package_name in ("pyro-ppl", "pyro"):
        try:
            return importlib_metadata.version(package_name)
        except Exception:
            continue
    return None


def _require_pyro():
    try:
        import pyro
        import pyro.distributions as dist
        import torch
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "pyro is required for the pyro benchmark model. "
            "Install it with 'pip install pyro-ppl torch'."
        ) from exc
    return torch, pyro, dist


def _resolve_torch_device(torch_mod):
    return torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")


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


def _to_int(value: Any) -> int:
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    return int(round(float(value)))


def _to_float(value: Any) -> float:
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    return float(value)


def _to_cache_scalar(value: Any) -> float | str:
    val = _to_float(value)
    if math.isfinite(val):
        return float(round(val, 12))
    return str(val)


def _inference_cache_key(
    query: dict, *, default_n_mc: int, inference_name: str
) -> tuple:
    target = query.get("target")
    if not target:
        raise ValueError("Missing target in query")
    evidence = _extract_evidence(query)
    do = _extract_do(query)
    n_samples = _get_n_mc(query, default=default_n_mc)
    evidence_key = tuple(
        sorted((str(k), _to_cache_scalar(v)) for k, v in evidence.items())
    )
    do_key = tuple(sorted((str(k), _to_cache_scalar(v)) for k, v in do.items()))
    return (str(target), int(n_samples), str(inference_name), evidence_key, do_key)


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
        return mean is not None and std is not None
    return False


def _domain_nodes(domain: dict) -> dict:
    if not isinstance(domain, dict):
        return {}
    nodes = domain.get("nodes", {})
    return nodes if isinstance(nodes, dict) else {}


def _sorted_nodes(dag) -> list[Any]:
    try:
        import networkx as nx

        return list(nx.topological_sort(dag))
    except Exception:
        pass

    nodes = list(dag.nodes())
    node_set = set(nodes)
    indegree = {node: 0 for node in nodes}
    children: dict[Any, list[Any]] = {node: [] for node in nodes}
    if hasattr(dag, "predecessors"):
        for node in nodes:
            try:
                parents = list(dag.predecessors(node))
            except Exception:
                parents = []
            for parent in parents:
                if parent not in node_set or parent == node:
                    continue
                indegree[node] += 1
                children[parent].append(node)
    elif hasattr(dag, "edges"):
        for src, dst in list(dag.edges()):
            if src not in node_set or dst not in node_set or src == dst:
                continue
            indegree[dst] += 1
            children[src].append(dst)

    ready = [node for node in nodes if indegree[node] == 0]
    try:
        ready.sort()
    except TypeError:
        ready.sort(key=lambda n: str(n))
    order: list[Any] = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        next_nodes = children.get(node, [])
        try:
            next_nodes = sorted(next_nodes)
        except TypeError:
            next_nodes = sorted(next_nodes, key=lambda n: str(n))
        for child in next_nodes:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
        try:
            ready.sort()
        except TypeError:
            ready.sort(key=lambda n: str(n))
    if len(order) == len(nodes):
        return order

    try:
        return sorted(nodes)
    except TypeError:
        return sorted(nodes, key=lambda n: str(n))


def _fit_linear_gaussian(
    y: np.ndarray,
    x: np.ndarray,
    *,
    min_std: float,
) -> tuple[np.ndarray, float]:
    y = np.asarray(y, dtype=float).reshape(-1)
    if y.size == 0:
        raise ValueError("Cannot fit linear Gaussian with empty target")
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.size == 0:
        design = np.ones((y.size, 1), dtype=float)
    else:
        design = np.column_stack([np.ones(y.size, dtype=float), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ beta
    dof = max(1, int(y.size - design.shape[1]))
    var = float(np.dot(residuals, residuals) / dof)
    if not math.isfinite(var) or var < 0:
        var = 0.0
    std = max(math.sqrt(var), float(min_std))
    return np.asarray(beta, dtype=float), float(std)


@register_benchmark_model
class PyroBenchmarkModel(BaseBenchmarkModel):
    name = "pyro"
    family = "pyro"
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
        self._torch, self._pyro, self._dist = _require_pyro()
        self.device = _resolve_torch_device(self._torch)
        self._pyro.set_rng_seed(int(self.seed))
        self._topo = _sorted_nodes(dag)
        self._parents = {
            node: list(dag.predecessors(node)) if hasattr(dag, "predecessors") else []
            for node in self._topo
        }
        self._node_type: dict[str, str] = {}
        self._k_by_node: dict[str, int] = {}
        self._discrete_parents: dict[str, list[str]] = {}
        self._continuous_parents: dict[str, list[str]] = {}
        self._cpds: dict[str, dict[tuple[int, ...], np.ndarray]] = {}
        self._default_probs: dict[str, np.ndarray] = {}
        self._gaussian_cpds: dict[str, dict[tuple[int, ...], dict[str, Any]]] = {}
        self._gaussian_fallback: dict[str, dict[str, Any]] = {}
        self._fitted = False

    def supports(self) -> dict:
        return {
            "can_fit": True,
            "can_answer_cpd": True,
            "can_answer_inference": True,
            "uses_inference": True,
        }

    def _is_discrete_node(self, node: str) -> bool:
        return self._node_type.get(node) == "discrete"

    def _coerce_discrete_value(self, node: str, value: Any) -> int:
        if node not in self._k_by_node:
            raise ValueError(f"Unknown discrete node '{node}'")
        idx = _to_int(value)
        k = int(self._k_by_node[node])
        if idx < 0 or idx >= k:
            raise ValueError(f"Value out of range for '{node}': {idx}")
        return int(idx)

    def _coerce_continuous_value(self, node: str, value: Any) -> float:
        val = _to_float(value)
        if not math.isfinite(val):
            raise ValueError(f"Non-finite value for continuous node '{node}': {value}")
        return float(val)

    def _coerce_assignment(self, values: dict) -> dict[str, int | float]:
        coerced: dict[str, int | float] = {}
        for node, value in values.items():
            if node not in self._node_type:
                raise ValueError(f"Unknown node '{node}'")
            if self._is_discrete_node(node):
                coerced[node] = self._coerce_discrete_value(node, value)
            else:
                coerced[node] = self._coerce_continuous_value(node, value)
        return coerced

    def fit(
        self, data_df: pd.DataFrame, *, progress: bool = True, **kwargs: Any
    ) -> None:
        del progress, kwargs
        nodes_meta = _domain_nodes(self.domain)
        if not nodes_meta:
            raise ValueError("Domain metadata missing 'nodes'")

        learning_kwargs = (
            dict(self.benchmark_config.learning.kwargs)
            if self.benchmark_config is not None
            else {}
        )
        alpha = float(learning_kwargs.get("alpha", 1.0))
        if alpha <= 0:
            raise ValueError("pyro learning alpha must be > 0")
        min_std = float(learning_kwargs.get("min_std", 1e-3))
        if min_std <= 0:
            raise ValueError("pyro learning min_std must be > 0")

        missing_cols = [node for node in self._topo if node not in data_df.columns]
        if missing_cols:
            raise ValueError(f"Data missing columns for nodes: {sorted(missing_cols)}")
        if int(len(data_df.index)) <= 0:
            raise ValueError("Cannot fit pyro backend with empty data")

        numeric_df = pd.DataFrame(index=data_df.index)
        discrete_encoded: dict[str, np.ndarray] = {}
        self._node_type = {}
        self._k_by_node = {}

        for node in self._topo:
            meta = nodes_meta.get(node) or {}
            states = list(meta.get("states") or [])
            node_type = str(meta.get("type") or "").strip().lower()
            if node_type not in {"discrete", "continuous"}:
                node_type = "discrete" if states else "continuous"
            self._node_type[node] = node_type

            col = pd.to_numeric(data_df[node], errors="coerce")
            if col.isna().any():
                raise ValueError(f"Non-numeric samples for '{node}'")
            arr = col.to_numpy(dtype=float)

            if node_type == "discrete":
                if not states:
                    raise ValueError(
                        "pyro discrete nodes require explicit states. "
                        f"Missing states for '{node}'"
                    )
                rounded = np.rint(arr).astype(int)
                if not np.allclose(arr, rounded, atol=1e-6):
                    bad = arr[np.abs(arr - rounded) > 1e-6][:5].tolist()
                    raise ValueError(f"Non-integer discrete values for '{node}': {bad}")
                k = int(len(states))
                if (rounded < 0).any() or (rounded >= k).any():
                    bad = rounded[(rounded < 0) | (rounded >= k)][:5].tolist()
                    raise ValueError(f"Out-of-range values for '{node}': {bad}")
                self._k_by_node[node] = k
                discrete_encoded[node] = rounded
                numeric_df[node] = rounded.astype(float)
            else:
                numeric_df[node] = arr

        self._discrete_parents = {}
        self._continuous_parents = {}
        for node in self._topo:
            parents = list(self._parents.get(node, []))
            d_par = [p for p in parents if self._is_discrete_node(p)]
            c_par = [p for p in parents if not self._is_discrete_node(p)]
            self._discrete_parents[node] = d_par
            self._continuous_parents[node] = c_par
            if self._is_discrete_node(node) and c_par:
                raise NotImplementedError(
                    "pyro backend supports CLG only when discrete nodes have "
                    "discrete parents. Continuous parent(s) for "
                    f"'{node}': {sorted(c_par)}"
                )

        n_rows = int(len(numeric_df.index))
        self._cpds = {}
        self._default_probs = {}
        self._gaussian_cpds = {}
        self._gaussian_fallback = {}

        for node in self._topo:
            if not self._is_discrete_node(node):
                continue
            parents = self._discrete_parents.get(node, [])
            k = int(self._k_by_node[node])
            counts: dict[tuple[int, ...], np.ndarray] = defaultdict(
                lambda: np.zeros(k, dtype=float)
            )
            node_vals = discrete_encoded[node]
            for row_idx in range(n_rows):
                key = tuple(int(discrete_encoded[p][row_idx]) for p in parents)
                counts[key][int(node_vals[row_idx])] += 1.0

            cpd_table: dict[tuple[int, ...], np.ndarray] = {}
            for key, count_vec in counts.items():
                probs = (count_vec + alpha) / (float(count_vec.sum()) + alpha * k)
                cpd_table[key] = probs.astype(float)
            self._cpds[node] = cpd_table
            self._default_probs[node] = np.full(k, 1.0 / float(k), dtype=float)

        for node in self._topo:
            if self._is_discrete_node(node):
                continue
            d_par = self._discrete_parents.get(node, [])
            c_par = self._continuous_parents.get(node, [])
            y_all = numeric_df[node].to_numpy(dtype=float).reshape(-1)
            if c_par:
                x_all = numeric_df[c_par].to_numpy(dtype=float)
            else:
                x_all = np.empty((n_rows, 0), dtype=float)

            fallback_beta, fallback_std = _fit_linear_gaussian(
                y_all, x_all, min_std=min_std
            )
            self._gaussian_fallback[node] = {
                "beta": fallback_beta,
                "std": float(fallback_std),
            }

            groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
            for row_idx in range(n_rows):
                key = tuple(int(discrete_encoded[p][row_idx]) for p in d_par)
                groups[key].append(row_idx)

            node_table: dict[tuple[int, ...], dict[str, Any]] = {}
            for key, idxs in groups.items():
                idx_arr = np.asarray(idxs, dtype=int)
                y_sub = y_all[idx_arr]
                x_sub = x_all[idx_arr] if c_par else np.empty((idx_arr.size, 0))
                beta, std = _fit_linear_gaussian(y_sub, x_sub, min_std=min_std)
                node_table[key] = {"beta": beta, "std": float(std)}
            self._gaussian_cpds[node] = node_table

        self._fitted = True

    def _cpd_probs_for(self, node: str, parent_values: dict[str, int]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model is not fitted")
        if node not in self._cpds:
            raise ValueError(f"Unknown discrete node '{node}'")
        parents = self._discrete_parents.get(node, [])
        key = tuple(self._coerce_discrete_value(p, parent_values[p]) for p in parents)
        table = self._cpds[node]
        probs = table.get(key)
        if probs is None:
            return self._default_probs[node].copy()
        return probs.copy()

    def _gaussian_params_for(
        self, node: str, parent_values: dict[str, int | float]
    ) -> tuple[float, float]:
        if not self._fitted:
            raise RuntimeError("Model is not fitted")
        if node not in self._gaussian_fallback:
            raise ValueError(f"Unknown continuous node '{node}'")
        d_par = self._discrete_parents.get(node, [])
        c_par = self._continuous_parents.get(node, [])
        key = tuple(self._coerce_discrete_value(p, parent_values[p]) for p in d_par)
        params = self._gaussian_cpds.get(node, {}).get(key)
        if params is None:
            params = self._gaussian_fallback[node]
        beta = np.asarray(params["beta"], dtype=float).reshape(-1)
        std = float(params["std"])
        if c_par:
            x = np.asarray(
                [self._coerce_continuous_value(p, parent_values[p]) for p in c_par],
                dtype=float,
            )
            mean = float(beta[0] + np.dot(beta[1:], x))
        else:
            mean = float(beta[0])
        std = max(float(std), 1e-8)
        return mean, std

    def _direct_cpd_distribution(
        self, target: str, evidence: dict[str, int | float], do: dict | None = None
    ) -> dict:
        do = do or {}
        if target not in self._node_type:
            raise ValueError(f"Unknown target '{target}'")

        if target in do:
            if self._is_discrete_node(target):
                k = int(self._k_by_node[target])
                idx = self._coerce_discrete_value(target, do[target])
                probs = np.zeros(k, dtype=float)
                probs[idx] = 1.0
                return {
                    "format": "categorical_probs",
                    "k": int(k),
                    "probs": probs.tolist(),
                    "support": list(range(int(k))),
                }
            val = self._coerce_continuous_value(target, do[target])
            return {
                "format": "normal_params",
                "mean": float(val),
                "std": 0.0,
                "n_samples": None,
                "samples": None,
            }

        parents = self._parents.get(target, [])
        if parents and not all(parent in evidence for parent in parents):
            raise ValueError("Missing parent assignments for direct CPD query")

        if self._is_discrete_node(target):
            k = int(self._k_by_node[target])
            parent_values = {
                p: self._coerce_discrete_value(p, evidence[p])
                for p in self._discrete_parents.get(target, [])
            }
            probs = self._cpd_probs_for(target, parent_values)
            return {
                "format": "categorical_probs",
                "k": int(k),
                "probs": _normalize_probs(probs),
                "support": list(range(int(k))),
            }

        mean, std = self._gaussian_params_for(target, evidence)
        return {
            "format": "normal_params",
            "mean": float(mean),
            "std": float(std),
            "n_samples": None,
            "samples": None,
        }

    def _infer_distribution(
        self,
        target: str,
        evidence: dict[str, int | float],
        do: dict[str, int | float],
        n_samples: int,
    ) -> dict:
        if not self._fitted:
            raise RuntimeError("Model is not fitted")
        if target not in self._node_type:
            raise ValueError(f"Unknown target '{target}'")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")

        inference_name = (
            str(self.benchmark_config.inference.name)
            if self.benchmark_config is not None
            else "likelihood_weighting"
        )
        supported = {
            "likelihood_weighting",
            "ancestral_importance_sampling",
            "importance_sampling",
        }
        if inference_name not in supported:
            raise ValueError(
                f"Unsupported pyro inference method '{inference_name}'. "
                f"Supported: {sorted(supported)}"
            )

        overlap = set(evidence) & set(do)
        if overlap:
            raise ValueError("Nodes cannot be in both evidence and do")

        local_evidence = self._coerce_assignment(evidence)
        local_do = self._coerce_assignment(do)

        self._torch.manual_seed(int(self.seed))
        if self.device.type == "cuda":
            self._torch.cuda.manual_seed_all(int(self.seed))

        target_is_discrete = self._is_discrete_node(target)
        if target_is_discrete:
            target_vals = np.zeros(int(n_samples), dtype=int)
        else:
            target_vals = np.zeros(int(n_samples), dtype=float)
        logw = np.zeros(int(n_samples), dtype=float)

        for sample_idx in range(int(n_samples)):
            assignment: dict[str, int | float] = {}
            for node in self._topo:
                if node in local_do:
                    assignment[node] = local_do[node]
                    continue

                if self._is_discrete_node(node):
                    node_k = int(self._k_by_node[node])
                    parent_vals = {
                        p: self._coerce_discrete_value(p, assignment[p])
                        for p in self._discrete_parents.get(node, [])
                    }
                    probs = self._cpd_probs_for(node, parent_vals)
                    probs_t = self._torch.tensor(
                        probs, dtype=self._torch.float32, device=self.device
                    )
                    dist = self._dist.Categorical(probs=probs_t)
                    if node in local_evidence:
                        val = self._coerce_discrete_value(node, local_evidence[node])
                        if val < 0 or val >= node_k:
                            raise ValueError(
                                f"evidence value out of range for '{node}': {val}"
                            )
                        log_prob = dist.log_prob(
                            self._torch.tensor(
                                val, dtype=self._torch.int64, device=self.device
                            )
                        )
                        lp = float(log_prob.item())
                        if math.isfinite(lp):
                            logw[sample_idx] += lp
                        else:
                            logw[sample_idx] = float("-inf")
                        assignment[node] = int(val)
                    else:
                        sampled = int(dist.sample().item())
                        assignment[node] = sampled
                    continue

                parent_vals = {p: assignment[p] for p in self._parents.get(node, [])}
                mean, std = self._gaussian_params_for(node, parent_vals)
                mean_t = self._torch.tensor(
                    float(mean), dtype=self._torch.float32, device=self.device
                )
                std_t = self._torch.tensor(
                    max(float(std), 1e-8), dtype=self._torch.float32, device=self.device
                )
                dist = self._dist.Normal(loc=mean_t, scale=std_t)
                if node in local_evidence:
                    val = self._coerce_continuous_value(node, local_evidence[node])
                    log_prob = dist.log_prob(
                        self._torch.tensor(
                            float(val), dtype=self._torch.float32, device=self.device
                        )
                    )
                    lp = float(log_prob.item())
                    if math.isfinite(lp):
                        logw[sample_idx] += lp
                    else:
                        logw[sample_idx] = float("-inf")
                    assignment[node] = float(val)
                else:
                    sampled = float(dist.sample().item())
                    assignment[node] = sampled

            if target_is_discrete:
                target_vals[sample_idx] = self._coerce_discrete_value(
                    target, assignment[target]
                )
            else:
                target_vals[sample_idx] = self._coerce_continuous_value(
                    target, assignment[target]
                )

        if not np.isfinite(logw).any():
            weights = np.ones(int(n_samples), dtype=float) / float(n_samples)
        else:
            finite_max = float(np.nanmax(logw))
            weights = np.exp(logw - finite_max)
            total = float(weights.sum())
            if total <= 0 or not math.isfinite(total):
                weights = np.ones(int(n_samples), dtype=float) / float(n_samples)
            else:
                weights /= total

        if target_is_discrete:
            target_k = int(self._k_by_node[target])
            hist = np.bincount(
                target_vals.astype(int), weights=weights, minlength=target_k
            ).astype(float)
            probs = _normalize_probs(hist)
            return {
                "format": "categorical_probs",
                "k": int(target_k),
                "probs": probs,
                "support": list(range(int(target_k))),
            }

        target_vals_f = target_vals.astype(float)
        mean = float(np.sum(weights * target_vals_f))
        var = float(np.sum(weights * (target_vals_f - mean) ** 2))
        std = math.sqrt(var) if var >= 0 else float("nan")
        kept = target_vals_f[: min(len(target_vals_f), 2048)]
        return {
            "format": "normal_params",
            "mean": float(mean),
            "std": float(std),
            "n_samples": int(n_samples),
            "samples": [float(x) for x in kept],
        }

    def answer_cpd_query(self, query: dict) -> dict:
        start = time.perf_counter()
        try:
            target = query.get("target")
            if not target:
                raise ValueError("Missing target in query")
            evidence = self._coerce_assignment(_extract_evidence(query))
            do = self._coerce_assignment(_extract_do(query))
            assignment = {**evidence, **do}
            parents = self._parents.get(target, [])
            if target in do or all(parent in assignment for parent in parents):
                result = self._direct_cpd_distribution(target, assignment, do=do)
            else:
                n_samples = _get_n_mc(query, default=512)
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
            evidence = self._coerce_assignment(_extract_evidence(query))
            do = self._coerce_assignment(_extract_do(query))
            n_samples = _get_n_mc(query, default=512)
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
        inference_name = (
            str(self.benchmark_config.inference.name)
            if self.benchmark_config is not None
            else "likelihood_weighting"
        )
        cached: dict[tuple, dict] = {}
        responses: list[dict] = []
        for query in queries:
            try:
                key = _inference_cache_key(
                    query, default_n_mc=512, inference_name=inference_name
                )
            except Exception:
                responses.append(self.answer_inference_query(query))
                continue
            if key not in cached:
                cached[key] = self.answer_inference_query(query)
                responses.append(cached[key])
                continue
            reused = deepcopy(cached[key])
            reused["timing_ms"] = 0.0
            responses.append(reused)
        return responses
