from __future__ import annotations

import math
import random
import time
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
    try:
        return importlib_metadata.version("pgmpy")
    except Exception:
        return None


def _require_pgmpy():
    try:
        from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator
        from pgmpy.inference import VariableElimination
        from pgmpy.models import BayesianNetwork
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "pgmpy is required for the pgmpy benchmark model. "
            "Install it with 'pip install pgmpy'."
        ) from exc
    return (
        BayesianNetwork,
        MaximumLikelihoodEstimator,
        BayesianEstimator,
        VariableElimination,
    )


def _sorted_edges(dag) -> list[tuple[Any, Any]]:
    edges = list(dag.edges())
    try:
        return sorted(edges)
    except TypeError:
        return sorted(edges, key=lambda e: (str(e[0]), str(e[1])))


def _sorted_nodes(dag) -> list[Any]:
    nodes = list(dag.nodes())
    try:
        return sorted(nodes)
    except TypeError:
        return sorted(nodes, key=lambda n: str(n))


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


def _to_int(value: Any) -> int:
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    return int(value)


def _normalize_probs(probs: Iterable[float]) -> list[float]:
    arr = np.asarray(list(probs), dtype=float)
    total = float(arr.sum())
    if not math.isfinite(total) or total <= 0:
        return (np.ones_like(arr) / len(arr)).tolist()
    return (arr / total).tolist()


def _map_codes_to_states(
    series: pd.Series, states: list[str], var: str
) -> pd.Categorical:
    codes = pd.to_numeric(series, errors="coerce")
    if codes.isna().any():
        bad = series[codes.isna()].head(5).tolist()
        raise ValueError(f"Non-numeric codes for '{var}': {bad}")
    codes_int = codes.astype(int)
    if (codes_int < 0).any() or (codes_int >= len(states)).any():
        bad_values = codes_int[(codes_int < 0) | (codes_int >= len(states))].unique()
        sample = [int(v) for v in bad_values[:5]]
        raise ValueError(f"Out-of-range codes for '{var}': {sample}")
    mapping = {idx: state for idx, state in enumerate(states)}
    mapped = codes_int.map(mapping)
    return pd.Categorical(mapped, categories=list(states), ordered=False)


def _domain_nodes(domain: dict) -> dict:
    if not isinstance(domain, dict):
        return {}
    nodes = domain.get("nodes", {})
    return nodes if isinstance(nodes, dict) else {}


@register_benchmark_model
class PgmpyBenchmarkModel(BaseBenchmarkModel):
    name = "pgmpy"
    family = "pgmpy"
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
        random.seed(self.seed)
        np.random.seed(self.seed)

        BayesianNetwork, _, _, _ = _require_pgmpy()
        edges = _sorted_edges(dag)
        self._model = BayesianNetwork(edges)
        self._model.add_nodes_from(_sorted_nodes(dag))
        self._inference_engine = None
        self.state_map: dict[str, list[str]] = {}

    def supports(self) -> dict:
        return {
            "can_fit": True,
            "can_answer_cpd": True,
            "can_answer_inference": True,
            "uses_inference": True,
        }

    def _validate_domain(self) -> dict:
        if not isinstance(self.domain, dict):
            raise ValueError("Domain metadata missing or invalid")
        if self.domain.get("unsupported"):
            reason = self.domain.get("reason") or "Unsupported domain"
            raise ValueError(reason)
        nodes = _domain_nodes(self.domain)
        if not nodes:
            raise ValueError("Domain metadata missing 'nodes'")
        for node, meta in nodes.items():
            node_type = meta.get("type")
            if node_type == "continuous":
                raise ValueError(
                    "Continuous variables are not supported by pgmpy baselines"
                )
            if node_type != "discrete":
                raise ValueError(
                    f"Unsupported node type '{node_type}' for '{node}' in domain"
                )
            states = meta.get("states") or []
            if not states:
                raise ValueError(f"Missing states for discrete node '{node}'")
        return nodes

    def fit(
        self, data_df: pd.DataFrame, *, progress: bool = True, **kwargs: Any
    ) -> None:
        del progress
        nodes = self._validate_domain()

        missing_cols = [node for node in nodes if node not in data_df.columns]
        if missing_cols:
            raise ValueError(f"Data missing columns for nodes: {sorted(missing_cols)}")

        df_states = data_df.copy()
        self.state_map = {}
        for node, meta in nodes.items():
            states = list(meta.get("states") or [])
            self.state_map[node] = states
            df_states[node] = _map_codes_to_states(df_states[node], states, node)

        model_nodes = list(self._model.nodes())
        df_states = df_states[model_nodes]

        (
            BayesianNetwork,
            MaximumLikelihoodEstimator,
            BayesianEstimator,
            VariableElimination,
        ) = _require_pgmpy()
        del BayesianNetwork

        learning_name = (
            self.benchmark_config.learning.name
            if self.benchmark_config is not None
            else "mle"
        )
        learning_kwargs = (
            dict(self.benchmark_config.learning.kwargs)
            if self.benchmark_config is not None
            else {}
        )

        if learning_name == "mle":
            estimator = MaximumLikelihoodEstimator(self._model, data=df_states)
            cpds = estimator.get_parameters()
        elif learning_name == "bdeu":
            ess = learning_kwargs.get("equivalent_sample_size", 10)
            estimator = BayesianEstimator(self._model, data=df_states)
            cpds = estimator.get_parameters(
                prior_type="BDeu",
                equivalent_sample_size=float(ess),
            )
        else:
            raise ValueError(f"Unsupported learning method '{learning_name}'")

        self._model.add_cpds(*cpds)
        if not self._model.check_model():
            raise ValueError("pgmpy model validation failed after fitting")

        self._inference_engine = VariableElimination(self._model)

    def _format_evidence(self, evidence: dict) -> dict:
        mapped = {}
        for var, value in evidence.items():
            if var not in self.state_map:
                raise ValueError(f"Unknown evidence variable '{var}'")
            states = self.state_map[var]
            idx = _to_int(value)
            if idx < 0 or idx >= len(states):
                raise ValueError(f"Evidence code out of range for '{var}': {idx}")
            mapped[var] = states[idx]
        return mapped

    def _infer_distribution(self, target: str, evidence: dict) -> dict:
        if self._inference_engine is None:
            raise RuntimeError("Model is not fitted")
        if target not in self.state_map:
            raise ValueError(f"Unknown target variable '{target}'")

        evidence_states = self._format_evidence(evidence)
        phi = self._inference_engine.query(
            variables=[target],
            evidence=evidence_states,
            show_progress=False,
        )

        factor = phi
        if isinstance(phi, dict):
            factor = phi[target]

        if not hasattr(factor, "state_names"):
            raise ValueError("pgmpy query result missing state names")

        state_names = factor.state_names.get(target)
        if state_names is None:
            raise ValueError("pgmpy query result missing target state names")

        values = np.asarray(factor.values, dtype=float).reshape(-1)
        name_to_index = {name: idx for idx, name in enumerate(state_names)}

        states = self.state_map[target]
        probs = []
        for state in states:
            if state not in name_to_index:
                raise ValueError(
                    f"State '{state}' not found in pgmpy output for '{target}'"
                )
            probs.append(float(values[name_to_index[state]]))

        return {
            "format": "categorical_probs",
            "k": len(states),
            "probs": _normalize_probs(probs),
            "support": list(range(len(states))),
        }

    def answer_cpd_query(self, query: dict) -> dict:
        start = time.perf_counter()
        try:
            target = query.get("target")
            if not target:
                raise ValueError("Missing target in query")
            evidence = _extract_evidence(query)
            result = self._infer_distribution(target, evidence)
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
            result = self._infer_distribution(target, evidence)
            ok = result.get("probs") is not None
            error = None
        except Exception as exc:
            ok = False
            error = f"{type(exc).__name__}: {exc}"
            result = None
        timing_ms = (time.perf_counter() - start) * 1000.0
        return {"ok": ok, "error": error, "timing_ms": timing_ms, "result": result}
