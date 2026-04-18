from __future__ import annotations

import math
import time
from collections import defaultdict
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


def _normalize_probs(probs: Iterable[float]) -> list[float]:
    arr = np.asarray(list(probs), dtype=float)
    total = float(arr.sum())
    if not math.isfinite(total) or total <= 0:
        return (np.ones_like(arr) / len(arr)).tolist()
    return (arr / total).tolist()


def _result_ok(result: dict | None) -> bool:
    if not isinstance(result, dict):
        return False
    if result.get("format") != "categorical_probs":
        return False
    return result.get("probs") is not None


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


@register_benchmark_model
class PyroBenchmarkModel(BaseBenchmarkModel):
    name = "pyro"
    family = "pyro"
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
        self._torch, self._pyro, self._dist = _require_pyro()
        self.device = _resolve_torch_device(self._torch)
        self._pyro.set_rng_seed(int(self.seed))
        self._topo = _sorted_nodes(dag)
        self._parents = {
            node: list(dag.predecessors(node)) if hasattr(dag, "predecessors") else []
            for node in self._topo
        }
        self._k_by_node: dict[str, int] = {}
        self._cpds: dict[str, dict[tuple[int, ...], np.ndarray]] = {}
        self._default_probs: dict[str, np.ndarray] = {}
        self._fitted = False

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

        self._k_by_node = {}
        for node in self._topo:
            meta = nodes_meta.get(node) or {}
            if meta.get("type") == "continuous":
                raise NotImplementedError(
                    "pyro backend currently supports discrete nodes only"
                )
            states = list(meta.get("states") or [])
            if not states:
                raise ValueError(
                    "pyro backend requires explicit discrete states for each node. "
                    f"Missing states for '{node}'"
                )
            self._k_by_node[node] = len(states)

        missing_cols = [node for node in self._topo if node not in data_df.columns]
        if missing_cols:
            raise ValueError(f"Data missing columns for nodes: {sorted(missing_cols)}")

        encoded_df = pd.DataFrame(index=data_df.index)
        for node in self._topo:
            k = self._k_by_node[node]
            col = pd.to_numeric(data_df[node], errors="coerce")
            if col.isna().any():
                raise ValueError(f"Non-numeric samples for '{node}'")
            vals = col.astype(int)
            if (vals < 0).any() or (vals >= k).any():
                bad = vals[(vals < 0) | (vals >= k)].head(5).tolist()
                raise ValueError(f"Out-of-range values for '{node}': {bad}")
            encoded_df[node] = vals

        self._cpds = {}
        self._default_probs = {}
        data = encoded_df[self._topo].to_numpy(dtype=int)
        index_by_node = {node: idx for idx, node in enumerate(self._topo)}
        for node in self._topo:
            parents = self._parents.get(node, [])
            k = int(self._k_by_node[node])
            node_idx = index_by_node[node]
            parent_indices = [index_by_node[p] for p in parents]
            counts: dict[tuple[int, ...], np.ndarray] = defaultdict(
                lambda: np.zeros(k, dtype=float)
            )
            for row in data:
                key = tuple(int(row[p_idx]) for p_idx in parent_indices)
                counts[key][int(row[node_idx])] += 1.0

            cpd_table: dict[tuple[int, ...], np.ndarray] = {}
            for key, count_vec in counts.items():
                probs = (count_vec + alpha) / (float(count_vec.sum()) + alpha * k)
                cpd_table[key] = probs.astype(float)
            self._cpds[node] = cpd_table
            self._default_probs[node] = np.full(k, 1.0 / float(k), dtype=float)

        self._fitted = True

    def _cpd_probs_for(self, node: str, parent_values: dict[str, int]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model is not fitted")
        if node not in self._cpds:
            raise ValueError(f"Unknown node '{node}'")
        parents = self._parents.get(node, [])
        key = tuple(int(parent_values[p]) for p in parents)
        table = self._cpds[node]
        probs = table.get(key)
        if probs is None:
            return self._default_probs[node].copy()
        return probs.copy()

    def _direct_cpd_distribution(
        self, target: str, evidence: dict, do: dict | None = None
    ) -> dict:
        do = do or {}
        k = self._k_by_node.get(target)
        if k is None:
            raise ValueError(f"Unknown target '{target}'")

        if target in do:
            idx = _to_int(do[target])
            if idx < 0 or idx >= k:
                raise ValueError("do value out of bounds for target")
            probs = np.zeros(k, dtype=float)
            probs[idx] = 1.0
            return {
                "format": "categorical_probs",
                "k": int(k),
                "probs": probs.tolist(),
                "support": list(range(int(k))),
            }

        parents = self._parents.get(target, [])
        if parents and not all(parent in evidence for parent in parents):
            raise ValueError("Missing parent assignments for direct CPD query")
        parent_values = {p: _to_int(evidence[p]) for p in parents}
        probs = self._cpd_probs_for(target, parent_values)
        return {
            "format": "categorical_probs",
            "k": int(k),
            "probs": _normalize_probs(probs),
            "support": list(range(int(k))),
        }

    def _infer_distribution(
        self, target: str, evidence: dict, do: dict, n_samples: int
    ) -> dict:
        if not self._fitted:
            raise RuntimeError("Model is not fitted")
        if target not in self._k_by_node:
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

        local_evidence = {k: _to_int(v) for k, v in evidence.items()}
        local_do = {k: _to_int(v) for k, v in do.items()}
        overlap = set(local_evidence) & set(local_do)
        if overlap:
            raise ValueError("Nodes cannot be in both evidence and do")

        self._torch.manual_seed(int(self.seed))
        if self.device.type == "cuda":
            self._torch.cuda.manual_seed_all(int(self.seed))
        target_k = int(self._k_by_node[target])
        target_vals = np.zeros(int(n_samples), dtype=int)
        logw = np.zeros(int(n_samples), dtype=float)

        for sample_idx in range(int(n_samples)):
            assignment: dict[str, int] = {}
            for node in self._topo:
                node_k = int(self._k_by_node[node])
                if node in local_do:
                    val = int(local_do[node])
                    if val < 0 or val >= node_k:
                        raise ValueError(f"do value out of range for '{node}': {val}")
                    assignment[node] = val
                    continue

                parent_vals = {p: assignment[p] for p in self._parents.get(node, [])}
                probs = self._cpd_probs_for(node, parent_vals)
                probs_t = self._torch.tensor(
                    probs, dtype=self._torch.float32, device=self.device
                )
                dist = self._dist.Categorical(probs=probs_t)

                if node in local_evidence:
                    val = int(local_evidence[node])
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
                    assignment[node] = val
                else:
                    sampled = int(dist.sample().item())
                    assignment[node] = sampled

            target_vals[sample_idx] = int(assignment[target])

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

        hist = np.bincount(target_vals, weights=weights, minlength=target_k).astype(
            float
        )
        probs = _normalize_probs(hist)
        return {
            "format": "categorical_probs",
            "k": int(target_k),
            "probs": probs,
            "support": list(range(int(target_k))),
        }

    def answer_cpd_query(self, query: dict) -> dict:
        start = time.perf_counter()
        try:
            target = query.get("target")
            if not target:
                raise ValueError("Missing target in query")
            evidence = _extract_evidence(query)
            do = _extract_do(query)
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
            evidence = _extract_evidence(query)
            do = _extract_do(query)
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
