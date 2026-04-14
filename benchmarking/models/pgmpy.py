from __future__ import annotations

import logging
import math
import random
import time
from types import SimpleNamespace
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

        try:
            from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
        except Exception:
            from pgmpy.models import BayesianNetwork

        try:
            from pgmpy.factors.continuous import LinearGaussianCPD
            from pgmpy.models import LinearGaussianBayesianNetwork
        except Exception:
            LinearGaussianBayesianNetwork = None
            LinearGaussianCPD = None
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
        LinearGaussianBayesianNetwork,
        LinearGaussianCPD,
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
    return int(value)


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


def _target_discrete_cardinality(domain: dict, target: str) -> int:
    nodes = _domain_nodes(domain)
    meta = nodes.get(target, {}) if isinstance(nodes, dict) else {}
    states = meta.get("states")
    if isinstance(states, list) and states:
        return int(len(states))
    codes = meta.get("codes")
    if isinstance(codes, dict) and codes:
        values: list[int] = []
        for code in codes.values():
            try:
                values.append(int(code))
            except Exception:
                return 0
        unique_vals = sorted(set(values))
        if unique_vals:
            return int(len(unique_vals))
    return 0


def _normal_to_categorical_probs(
    mean: float, std: float, *, k: int
) -> list[float] | None:
    if k <= 0:
        return None
    if not math.isfinite(mean) or not math.isfinite(std) or std < 0:
        return None
    if std <= 1e-12:
        idx = int(round(mean))
        idx = max(0, min(int(k) - 1, idx))
        probs = np.zeros(int(k), dtype=float)
        probs[idx] = 1.0
        return probs.tolist()

    sigma = float(std)
    mu = float(mean)
    sqrt2 = math.sqrt(2.0)

    def _cdf(x: float) -> float:
        if x == float("-inf"):
            return 0.0
        if x == float("inf"):
            return 1.0
        z = (x - mu) / (sigma * sqrt2)
        return 0.5 * (1.0 + math.erf(z))

    probs = np.zeros(int(k), dtype=float)
    for idx in range(int(k)):
        left = float("-inf") if idx == 0 else float(idx) - 0.5
        right = float("inf") if idx == int(k) - 1 else float(idx) + 0.5
        prob = _cdf(right) - _cdf(left)
        if prob < 0 and abs(prob) <= 1e-12:
            prob = 0.0
        probs[idx] = max(float(prob), 0.0)
    return _normalize_probs(probs)


def _samples_to_categorical_probs(
    samples: Iterable[float], *, k: int
) -> list[float] | None:
    if k <= 0:
        return None
    arr = np.asarray(list(samples), dtype=float).reshape(-1)
    if arr.size == 0:
        return None
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    edges = np.arange(int(k) + 1, dtype=float) - 0.5
    edges[0] = float("-inf")
    edges[-1] = float("inf")
    hist, _ = np.histogram(arr, bins=edges)
    return _normalize_probs(hist.astype(float))


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


def _normalize_joint_gaussian(
    joint: Any, *, fallback_order: list[Any]
) -> tuple[np.ndarray, np.ndarray, list[Any]]:
    mean = None
    cov = None
    variables: list[Any] | None = None

    if isinstance(joint, dict):
        mean = joint.get("mean")
        cov = joint.get("covariance", joint.get("cov"))
        vars_raw = joint.get("variables")
        if vars_raw is not None:
            variables = list(vars_raw)
    elif isinstance(joint, (tuple, list)):
        if len(joint) < 2:
            raise ValueError(
                "Invalid joint Gaussian tuple/list: expected at least (mean, covariance)"
            )
        mean = joint[0]
        cov = joint[1]
        if len(joint) >= 3 and joint[2] is not None:
            variables = list(joint[2])
    else:
        mean = getattr(joint, "mean", None)
        cov = getattr(joint, "covariance", getattr(joint, "cov", None))
        vars_raw = getattr(joint, "variables", None)
        if vars_raw is not None:
            variables = list(vars_raw)

    if mean is None or cov is None:
        raise ValueError("Failed to decode joint Gaussian mean/covariance")

    mean_arr = np.asarray(mean, dtype=float).reshape(-1)
    cov_arr = np.asarray(cov, dtype=float)
    if cov_arr.ndim != 2:
        raise ValueError("Joint Gaussian covariance must be a 2D matrix")

    if not variables:
        variables = list(fallback_order)
    return mean_arr, cov_arr, list(variables)


def _extract_lg_cpd_coefficients(
    cpd: Any, *, node: Any, parents: list[Any]
) -> tuple[float, np.ndarray]:
    beta_candidates = (
        getattr(cpd, "beta", None),
        getattr(cpd, "evidence_mean", None),
        getattr(cpd, "mean", None),
    )
    beta_arr = np.asarray([], dtype=float)
    for candidate in beta_candidates:
        if candidate is None:
            continue
        try:
            arr = np.asarray(candidate, dtype=float).reshape(-1)
        except Exception:
            continue
        if arr.size > 0:
            beta_arr = arr
            break
    if beta_arr.size == 0:
        raise ValueError(f"Missing CPD coefficients for '{node}'")

    if beta_arr.size == len(parents) + 1:
        intercept = float(beta_arr[0])
        coeffs = np.asarray(beta_arr[1:], dtype=float)
    elif beta_arr.size == len(parents):
        intercept = 0.0
        coeffs = np.asarray(beta_arr, dtype=float)
    else:
        raise ValueError(
            f"CPD coefficient size mismatch for '{node}': "
            f"got {beta_arr.size}, expected {len(parents)} or {len(parents) + 1}"
        )
    return intercept, coeffs


def _extract_lg_cpd_variance(cpd: Any, *, node: Any) -> float:
    variance_candidates = (
        getattr(cpd, "variance", None),
        getattr(cpd, "var", None),
    )
    for candidate in variance_candidates:
        if candidate is None:
            continue
        try:
            var = float(np.asarray(candidate, dtype=float).reshape(-1)[0])
        except Exception:
            continue
        if math.isfinite(var) and var >= 0:
            return var

    std = getattr(cpd, "std", None)
    if std is not None:
        try:
            std_val = float(np.asarray(std, dtype=float).reshape(-1)[0])
            var = std_val * std_val
            if math.isfinite(var):
                return max(var, 0.0)
        except Exception:
            pass

    raise ValueError(f"Missing CPD variance/std for '{node}'")


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


@register_benchmark_model
class PgmpyBenchmarkModel(BaseBenchmarkModel):
    name = "pgmpy"
    family = "pgmpy"
    version = _package_version()
    _logger = logging.getLogger("benchmarking.pgmpy")

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

        # Create pgmpy models lazily in fit(): gaussian presets should not require
        # discrete BayesianNetwork construction at model-init time.
        self._model = None
        self._model_edges = _sorted_edges(dag)
        self._model_nodes = _sorted_nodes(dag)
        self._lg_model = None
        self._inference_engine = None
        self.state_map: dict[str, list[str]] = {}
        self._continuous = False
        self._lg_joint = None
        self._lg_order: list[str] = []
        self._lg_index: dict[str, int] = {}
        self._rng = np.random.default_rng(self.seed)

    def supports(self) -> dict:
        return {
            "can_fit": True,
            "can_answer_cpd": True,
            "can_answer_inference": True,
            "uses_inference": True,
        }

    def _nx_graph(self):
        g = self.dag
        if not hasattr(g, "edges") or not hasattr(g, "predecessors"):
            raise TypeError(
                f"Expected dag to be a networkx DiGraph-like object, got {type(g)}"
            )
        return g

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
            if node_type not in {"discrete", "continuous"}:
                raise ValueError(
                    f"Unsupported node type '{node_type}' for '{node}' in domain"
                )
        return nodes

    def fit(
        self, data_df: pd.DataFrame, *, progress: bool = True, **kwargs: Any
    ) -> None:
        del progress
        nodes = self._validate_domain()

        missing_cols = [node for node in nodes if node not in data_df.columns]
        if missing_cols:
            raise ValueError(f"Data missing columns for nodes: {sorted(missing_cols)}")

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
        learning_name_norm = str(learning_name).strip().lower()

        if learning_name_norm.startswith("gaussian"):
            self._fit_gaussian_pgmpy(data_df, nodes)
            return

        self._continuous = False
        self._lg_model = None
        self._lg_joint = None
        self._lg_order = []
        self._lg_index = {}

        df_states = data_df.copy()
        self.state_map = {}
        for node, meta in nodes.items():
            states = list(meta.get("states") or [])
            if not states:
                raise ValueError(
                    "Tabular pgmpy fitting requires explicit states for each node. "
                    f"Missing states for '{node}'"
                )
            self.state_map[node] = states
            df_states[node] = _map_codes_to_states(df_states[node], states, node)

        (
            BayesianNetwork,
            MaximumLikelihoodEstimator,
            BayesianEstimator,
            VariableElimination,
            _,
            _,
        ) = _require_pgmpy()
        self._model = BayesianNetwork(self._model_edges)
        self._model.add_nodes_from(self._model_nodes)
        model_nodes = list(self._model.nodes())
        df_states = df_states[model_nodes]

        if learning_name_norm == "mle":
            estimator = MaximumLikelihoodEstimator(self._model, data=df_states)
            cpds = estimator.get_parameters()
        elif learning_name_norm == "bdeu":
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

    def _fit_gaussian_pgmpy(self, data_df: pd.DataFrame, nodes: dict) -> None:
        (
            _,
            _,
            _,
            _,
            LinearGaussianBayesianNetwork,
            _,
        ) = _require_pgmpy()
        if LinearGaussianBayesianNetwork is None:
            raise NotImplementedError(
                "pgmpy gaussian backend requires LinearGaussianBayesianNetwork"
            )
        self._continuous = True
        nodes_list = list(nodes.keys())
        graph = self._nx_graph()
        edges = list(graph.edges())
        lg = LinearGaussianBayesianNetwork(edges)
        lg.add_nodes_from(nodes_list)
        fit_fn = getattr(lg, "fit", None)
        if fit_fn is None:
            raise NotImplementedError(
                "pgmpy gaussian backend requires LinearGaussianBayesianNetwork.fit"
            )
        lg.fit(data_df[nodes_list])
        self._lg_model = lg
        if hasattr(lg, "check_model") and not lg.check_model():
            raise ValueError("pgmpy LinearGaussianBayesianNetwork validation failed")
        try:
            joint_raw = lg.to_joint_gaussian()
        except Exception as exc:
            raise RuntimeError(
                "Failed to derive joint Gaussian from LinearGaussianBayesianNetwork"
            ) from exc
        mu, cov, variables = _normalize_joint_gaussian(
            joint_raw, fallback_order=list(lg.nodes())
        )
        self._lg_joint = SimpleNamespace(
            mean=mu, covariance=cov, variables=list(variables)
        )
        self._lg_order = list(variables)
        self._lg_index = {v: i for i, v in enumerate(self._lg_order)}
        if set(self._lg_order) != set(nodes_list):
            raise ValueError(
                "Joint Gaussian variables do not match dataset nodes: "
                f"{sorted(set(self._lg_order) ^ set(nodes_list))}"
            )
        n_vars = len(self._lg_order)
        if mu.shape[0] != n_vars or cov.shape != (n_vars, n_vars):
            raise ValueError("Joint Gaussian mean/cov dimensions do not match nodes")

    def _coerce_continuous_output_for_target(
        self, *, target: str, result: dict
    ) -> dict:
        if not isinstance(result, dict):
            return result
        k = _target_discrete_cardinality(self.domain, target)
        if k <= 0:
            return result
        if result.get("format") == "categorical_probs":
            return result

        probs: list[float] | None = None
        if result.get("format") == "normal_params":
            mean = result.get("mean")
            std = result.get("std")
            try:
                probs = _normal_to_categorical_probs(float(mean), float(std), k=k)
            except Exception:
                probs = None
            if probs is None:
                probs = _samples_to_categorical_probs(result.get("samples") or [], k=k)
        elif result.get("format") == "samples_1d":
            probs = _samples_to_categorical_probs(result.get("samples") or [], k=k)

        if probs is None:
            return result
        return {
            "format": "categorical_probs",
            "k": int(k),
            "probs": probs,
            "support": list(range(int(k))),
        }

    def _gaussian_exact_condition(
        self,
        *,
        target: str,
        evidence: dict,
        n_samples: int = 0,
        max_return_samples: int = 2048,
        seed: int | None = None,
    ) -> dict:
        if self._lg_joint is None or self._lg_model is None:
            raise RuntimeError("Gaussian model is not fitted")
        if target not in self._lg_index:
            raise ValueError(f"Unknown target variable '{target}'")

        mu = np.asarray(self._lg_joint.mean, dtype=float).reshape(-1)
        cov = np.asarray(self._lg_joint.covariance, dtype=float)
        if mu.shape[0] != cov.shape[0]:
            raise RuntimeError("Joint Gaussian mean/cov dimension mismatch")
        idx_target = self._lg_index[target]

        if not evidence:
            mean = float(mu[idx_target])
            var = float(cov[idx_target, idx_target])
            std = math.sqrt(var) if var >= 0 else float("nan")
            sample_list: list[float] | None = None
            if n_samples > 0:
                rng = np.random.default_rng(seed)
                draws = rng.normal(mean, std, size=min(n_samples, max_return_samples))
                sample_list = [float(x) for x in draws]
            return {
                "format": "normal_params",
                "mean": mean,
                "std": std,
                "n_samples": len(sample_list) if sample_list is not None else None,
                "samples": sample_list,
            }

        evidence_vars = [var for var in evidence if var in self._lg_index]
        if len(evidence_vars) != len(evidence):
            missing = sorted(set(evidence) - set(evidence_vars))
            raise ValueError(f"Evidence variables not in model: {missing}")
        for var in evidence_vars:
            if not math.isfinite(float(evidence[var])):
                raise ValueError(
                    f"Non-finite evidence value for '{var}': {evidence[var]}"
                )
        idx_e = [self._lg_index[var] for var in evidence_vars]
        mu_x = mu[idx_target]
        mu_e = mu[idx_e]
        cov_xx = cov[idx_target, idx_target]
        cov_xe = cov[np.ix_([idx_target], idx_e)]
        cov_ee = cov[np.ix_(idx_e, idx_e)]
        cov_ee = cov_ee + 1e-8 * np.eye(len(idx_e))
        diff = np.asarray([evidence[v] for v in evidence_vars]) - mu_e
        try:
            cov_ee_inv = np.linalg.inv(cov_ee)
        except np.linalg.LinAlgError:
            cov_ee_inv = np.linalg.pinv(cov_ee)
        mean = float(mu_x + (cov_xe @ cov_ee_inv @ diff).reshape(-1)[0])
        var = float(cov_xx - (cov_xe @ cov_ee_inv @ cov_xe.T).reshape(-1)[0])
        if var < 0 and var > -1e-12:
            var = 0.0
        std = math.sqrt(var) if var >= 0 else float("nan")

        sample_list: list[float] | None = None
        if n_samples > 0 and math.isfinite(std):
            rng = np.random.default_rng(seed)
            draws = rng.normal(mean, std, size=min(n_samples, max_return_samples))
            sample_list = [float(x) for x in draws]
        return {
            "format": "normal_params",
            "mean": mean,
            "std": std,
            "n_samples": len(sample_list) if sample_list is not None else None,
            "samples": sample_list,
        }

    def _gaussian_forward_sample(
        self,
        *,
        target: str,
        evidence: dict,
        n_samples: int,
        n_resample: int,
        evidence_tau: float | None,
        seed: int | None = None,
        max_return_samples: int = 2048,
    ) -> dict:
        if self._lg_joint is None or self._lg_model is None:
            raise RuntimeError("Gaussian model is not fitted")
        if target not in self._lg_index:
            raise ValueError(f"Unknown target variable '{target}'")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive for gaussian_forward_sample")
        rng = np.random.default_rng(seed)
        graph = self._nx_graph()
        topo = list(graph.nodes())
        try:
            import networkx as nx

            topo = list(nx.topological_sort(graph))
        except Exception:
            pass

        samples: dict[str, np.ndarray] = {}
        for node in topo:
            cpd = self._lg_model.get_cpds(node)
            if cpd is None:
                raise ValueError(f"Missing CPD for '{node}'")
            parents = list(getattr(cpd, "evidence", []) or [])
            intercept, coeffs = _extract_lg_cpd_coefficients(
                cpd, node=node, parents=parents
            )
            if len(coeffs) != len(parents):
                raise ValueError(
                    f"CPD parent count mismatch for '{node}': "
                    f"{len(coeffs)} coeffs vs {len(parents)} parents"
                )
            mean = np.full(int(n_samples), intercept, dtype=float)
            for parent, coeff in zip(parents, coeffs):
                mean += float(coeff) * samples[parent]
            variance = _extract_lg_cpd_variance(cpd, node=node)
            std = math.sqrt(variance) if variance >= 0 else float("nan")
            draws = rng.normal(mean, std, size=int(n_samples))
            samples[node] = draws.astype(float)

        logw = np.zeros(int(n_samples), dtype=float)
        if evidence:
            for var, value in evidence.items():
                if var not in samples:
                    raise ValueError(f"Evidence variables not in model: {var}")
                if not math.isfinite(float(value)):
                    raise ValueError(f"Non-finite evidence value for '{var}': {value}")
                if evidence_tau is None:
                    idx = self._lg_index[var]
                    cov = np.asarray(self._lg_joint.covariance, dtype=float)
                    std_var = (
                        math.sqrt(float(cov[idx, idx]))
                        if cov[idx, idx] >= 0
                        else float("nan")
                    )
                    tau = (
                        1e-3 * std_var
                        if math.isfinite(std_var) and std_var > 0
                        else 1e-2
                    )
                else:
                    tau = float(evidence_tau)
                tau = max(tau, 1e-8)
                diff = samples[var] - float(value)
                logw += -0.5 * ((diff / tau) ** 2 + math.log(2.0 * math.pi * tau * tau))

        if not np.isfinite(logw).any():
            weights = np.ones(int(n_samples), dtype=float) / float(n_samples)
        else:
            logw = logw - float(np.nanmax(logw))
            weights = np.exp(logw)
            total = float(weights.sum())
            if total <= 0 or not math.isfinite(total):
                weights = np.ones(int(n_samples), dtype=float) / float(n_samples)
            else:
                weights /= total

        target_vals = samples[target]
        mean = float(np.sum(weights * target_vals))
        var = float(np.sum(weights * (target_vals - mean) ** 2))
        std = math.sqrt(var) if var >= 0 else float("nan")

        resample_n = min(int(n_resample), int(n_samples)) if n_resample > 0 else 0
        sample_list: list[float] | None = None
        if resample_n > 0:
            idx = rng.choice(len(target_vals), size=resample_n, replace=True, p=weights)
            sample_vals = target_vals[idx]
            sample_list = [float(x) for x in sample_vals[:max_return_samples]]
        return {
            "format": "normal_params",
            "mean": mean,
            "std": std,
            "n_samples": len(sample_list) if sample_list is not None else None,
            "samples": sample_list,
        }

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

    def _infer_distribution(
        self,
        target: str,
        evidence: dict,
        *,
        n_samples: int | None = None,
        n_resample: int | None = None,
    ) -> dict:
        if self._continuous:
            if self._lg_joint is None or self._lg_model is None:
                raise RuntimeError("Gaussian model is not fitted")
            inference_kwargs = {}
            inference_name = "gaussian_exact"
            if self.benchmark_config is not None:
                inference_kwargs = dict(self.benchmark_config.inference.kwargs)
                inference_name = str(
                    self.benchmark_config.inference.name or inference_name
                )
            if n_samples is None:
                n_samples = int(inference_kwargs.get("n_samples_infer", 0))
            if n_resample is None:
                n_resample = int(inference_kwargs.get("n_resample", 1024))
            seed = inference_kwargs.get("seed")
            evidence_vals = {k: _to_float(v) for k, v in evidence.items()}
            if inference_name == "gaussian_forward_sample":
                n_samples = int(n_samples) if int(n_samples) > 0 else 4096
                evidence_tau = inference_kwargs.get("evidence_tau")
                return self._gaussian_forward_sample(
                    target=target,
                    evidence=evidence_vals,
                    n_samples=int(n_samples),
                    n_resample=int(n_resample),
                    evidence_tau=evidence_tau,
                    seed=seed,
                )
            return self._gaussian_exact_condition(
                target=target,
                evidence=evidence_vals,
                n_samples=int(n_samples),
                seed=seed,
            )

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
            if self._continuous:
                result = self._gaussian_exact_condition(
                    target=target,
                    evidence={k: _to_float(v) for k, v in evidence.items()},
                    n_samples=0,
                )
                result = self._coerce_continuous_output_for_target(
                    target=target, result=result
                )
            else:
                result = self._infer_distribution(target, evidence)
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
            if self._continuous:
                n_samples = _get_n_mc(
                    query,
                    default=(
                        self.benchmark_config.inference.kwargs.get(
                            "n_samples_infer", 4096
                        )
                        if self.benchmark_config is not None
                        else 4096
                    ),
                )
                result = self._infer_distribution(target, evidence, n_samples=n_samples)
                result = self._coerce_continuous_output_for_target(
                    target=target, result=result
                )
            else:
                result = self._infer_distribution(target, evidence)
            ok = _result_ok(result)
            error = None if ok else "Unsupported or empty inference result"
        except Exception as exc:
            ok = False
            error = f"{type(exc).__name__}: {exc}"
            result = None
        timing_ms = (time.perf_counter() - start) * 1000.0
        return {"ok": ok, "error": error, "timing_ms": timing_ms, "result": result}
