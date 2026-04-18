from __future__ import annotations

import math
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
        return importlib_metadata.version("gpytorch")
    except Exception:
        return None


def _require_gpytorch():
    try:
        import gpytorch
        import torch
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "gpytorch is required for the gpytorch benchmark model. "
            "Install it with 'pip install gpytorch torch'."
        ) from exc
    return torch, gpytorch


def _resolve_torch_device(torch_mod):
    return torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")


def _maybe_to_device(obj, device):
    if hasattr(obj, "to"):
        return obj.to(device)
    return obj


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
    samples: Iterable[float],
    *,
    k: int,
    weights: Iterable[float] | None = None,
) -> list[float] | None:
    if k <= 0:
        return None
    arr = np.asarray(list(samples), dtype=float).reshape(-1)
    if arr.size == 0:
        return None
    mask = np.isfinite(arr)
    arr = arr[mask]
    if arr.size == 0:
        return None
    hist_weights = None
    if weights is not None:
        w = np.asarray(list(weights), dtype=float).reshape(-1)
        if w.size == mask.size:
            w = w[mask]
            if np.isfinite(w).any():
                w = np.clip(w, 0.0, np.inf)
                total = float(w.sum())
                if total > 0:
                    hist_weights = w / total
    edges = np.arange(int(k) + 1, dtype=float) - 0.5
    edges[0] = float("-inf")
    edges[-1] = float("inf")
    hist, _ = np.histogram(arr, bins=edges, weights=hist_weights)
    return _normalize_probs(hist.astype(float))


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


def _make_exact_gp_model(gpytorch, train_x, train_y, likelihood, kernel_name: str):
    class _ExactGP(gpytorch.models.ExactGP):
        def __init__(self, x, y, lik) -> None:
            super().__init__(x, y, lik)
            self.mean_module = gpytorch.means.ConstantMean()
            base_kernel = None
            if kernel_name in {"rbf", "se"}:
                base_kernel = gpytorch.kernels.RBFKernel()
            elif kernel_name in {"matern32", "matern_32"}:
                base_kernel = gpytorch.kernels.MaternKernel(nu=1.5)
            elif kernel_name in {"matern52", "matern_52"}:
                base_kernel = gpytorch.kernels.MaternKernel(nu=2.5)
            else:
                raise ValueError(
                    f"Unsupported gpytorch kernel '{kernel_name}'. "
                    "Supported: rbf, se, matern32, matern52"
                )
            self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    return _ExactGP(train_x, train_y, likelihood)


@register_benchmark_model
class GpytorchBenchmarkModel(BaseBenchmarkModel):
    name = "gpytorch"
    family = "gpytorch"
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
        self._topo = _sorted_nodes(dag)
        self._parents = {
            node: list(dag.predecessors(node)) if hasattr(dag, "predecessors") else []
            for node in self._topo
        }
        self._torch = None
        self._gpytorch = None
        self.device = None
        self._nodes: dict[str, dict[str, Any]] = {}
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
        torch, gpytorch = _require_gpytorch()
        self._torch = torch
        self._gpytorch = gpytorch
        self.device = _resolve_torch_device(torch)
        torch.manual_seed(int(self.seed))
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(int(self.seed))

        nodes_meta = _domain_nodes(self.domain)
        if not nodes_meta:
            raise ValueError("Domain metadata missing 'nodes'")
        missing_cols = [node for node in self._topo if node not in data_df.columns]
        if missing_cols:
            raise ValueError(f"Data missing columns for nodes: {sorted(missing_cols)}")

        learning_kwargs = (
            dict(self.benchmark_config.learning.kwargs)
            if self.benchmark_config is not None
            else {}
        )
        kernel_name = str(learning_kwargs.get("kernel", "rbf")).strip().lower()
        training_steps = int(learning_kwargs.get("training_steps", 35))
        learning_rate = float(learning_kwargs.get("lr", 0.1))
        max_train_size = int(learning_kwargs.get("max_train_size", 1024))
        min_noise = float(learning_kwargs.get("min_noise", 1e-4))
        min_std = float(learning_kwargs.get("min_std", 1e-3))
        if training_steps < 1:
            raise ValueError("gpytorch training_steps must be >= 1")
        if learning_rate <= 0:
            raise ValueError("gpytorch lr must be > 0")
        if max_train_size < 8:
            raise ValueError("gpytorch max_train_size must be >= 8")
        if min_std <= 0:
            raise ValueError("gpytorch min_std must be > 0")

        rng = np.random.default_rng(self.seed)
        self._nodes = {}
        for node in self._topo:
            y_raw = pd.to_numeric(data_df[node], errors="coerce")
            if y_raw.isna().any():
                raise ValueError(f"Non-numeric samples for '{node}'")
            y_all = y_raw.to_numpy(dtype=float)
            parents = self._parents.get(node, [])
            if not parents:
                mean = float(np.mean(y_all))
                std = float(np.std(y_all, ddof=0))
                std = max(std, min_std)
                self._nodes[node] = {
                    "kind": "constant",
                    "parents": [],
                    "mean": mean,
                    "std": std,
                }
                continue

            x_all = (
                data_df[parents]
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy(dtype=float)
            )
            if not np.isfinite(x_all).all():
                raise ValueError(f"Non-finite parent features for '{node}'")
            if not np.isfinite(y_all).all():
                raise ValueError(f"Non-finite target values for '{node}'")

            n_total = x_all.shape[0]
            if n_total > max_train_size:
                idx = rng.choice(n_total, size=max_train_size, replace=False)
                idx = np.sort(idx)
                x_use = x_all[idx]
                y_use = y_all[idx]
            else:
                x_use = x_all
                y_use = y_all

            x_mean = x_use.mean(axis=0)
            x_scale = x_use.std(axis=0)
            x_scale = np.where(x_scale < 1e-8, 1.0, x_scale)
            y_mean = float(np.mean(y_use))
            y_scale = float(np.std(y_use))
            if not math.isfinite(y_scale) or y_scale < 1e-8:
                y_scale = 1.0

            x_train_np = (x_use - x_mean) / x_scale
            y_train_np = (y_use - y_mean) / y_scale
            x_train = torch.tensor(x_train_np, dtype=torch.float32, device=self.device)
            y_train = torch.tensor(y_train_np, dtype=torch.float32, device=self.device)

            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            likelihood = _maybe_to_device(likelihood, self.device)
            try:
                likelihood.noise_covar.noise = max(min_noise, 1e-8)
            except Exception:
                pass

            model = _make_exact_gp_model(
                gpytorch,
                x_train,
                y_train,
                likelihood,
                kernel_name=kernel_name,
            )
            model = _maybe_to_device(model, self.device)
            model.train()
            likelihood.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            failed = False
            for _ in range(training_steps):
                optimizer.zero_grad()
                output = model(x_train)
                loss = -mll(output, y_train)
                if not bool(torch.isfinite(loss).all()):
                    failed = True
                    break
                loss.backward()
                optimizer.step()
            if failed:
                mean = float(np.mean(y_all))
                std = float(np.std(y_all, ddof=0))
                std = max(std, min_std)
                self._nodes[node] = {
                    "kind": "constant",
                    "parents": [],
                    "mean": mean,
                    "std": std,
                }
                continue

            model.eval()
            likelihood.eval()
            self._nodes[node] = {
                "kind": "gp",
                "parents": list(parents),
                "model": model,
                "likelihood": likelihood,
                "x_mean": x_mean.astype(float),
                "x_scale": x_scale.astype(float),
                "y_mean": float(y_mean),
                "y_scale": float(y_scale),
                "min_std": float(min_std),
            }

        self._fitted = True

    def _predict_node_normal(
        self, node: str, parent_values: dict[str, float]
    ) -> tuple[float, float]:
        if not self._fitted:
            raise RuntimeError("Model is not fitted")
        if node not in self._nodes:
            raise ValueError(f"Unknown node '{node}'")
        info = self._nodes[node]
        if info["kind"] == "constant":
            return float(info["mean"]), float(info["std"])

        if self._torch is None or self._gpytorch is None:
            raise RuntimeError("gpytorch backend modules are not initialized")

        parents = info["parents"]
        x_raw = np.asarray([float(parent_values[p]) for p in parents], dtype=float)
        x_std = (x_raw - info["x_mean"]) / info["x_scale"]
        x_tensor = self._torch.tensor(
            x_std.reshape(1, -1), dtype=self._torch.float32, device=self.device
        )

        model = info["model"]
        likelihood = info["likelihood"]
        with self._torch.no_grad(), self._gpytorch.settings.fast_pred_var():
            pred = likelihood(model(x_tensor))
            mean_std = float(pred.mean.reshape(-1)[0].item())
            var_std = float(pred.variance.reshape(-1)[0].item())
        if not math.isfinite(var_std) or var_std < 0:
            var_std = 0.0

        y_scale = float(info["y_scale"])
        mean = float(mean_std * y_scale + info["y_mean"])
        std = float(math.sqrt(var_std) * abs(y_scale))
        std = max(std, float(info["min_std"]))
        return mean, std

    def _format_target_distribution(
        self,
        *,
        target: str,
        mean: float,
        std: float,
        samples: list[float] | None = None,
        n_samples: int | None = None,
    ) -> dict:
        k = _target_discrete_cardinality(self.domain, target)
        if k > 0:
            probs = _normal_to_categorical_probs(mean, std, k=k)
            if probs is None and samples is not None:
                probs = _samples_to_categorical_probs(samples, k=k)
            return {
                "format": "categorical_probs",
                "k": int(k),
                "probs": probs,
                "support": list(range(int(k))),
            }
        return {
            "format": "normal_params",
            "mean": float(mean),
            "std": float(std),
            "n_samples": n_samples,
            "samples": samples,
        }

    def _direct_distribution(
        self, target: str, evidence: dict, do: dict | None = None
    ) -> dict:
        do = do or {}
        if target in do:
            target_val = _to_float(do[target])
            k = _target_discrete_cardinality(self.domain, target)
            if k > 0:
                idx = int(round(target_val))
                idx = max(0, min(k - 1, idx))
                probs = np.zeros(k, dtype=float)
                probs[idx] = 1.0
                return {
                    "format": "categorical_probs",
                    "k": int(k),
                    "probs": probs.tolist(),
                    "support": list(range(int(k))),
                }
            return {
                "format": "normal_params",
                "mean": float(target_val),
                "std": 0.0,
                "n_samples": None,
                "samples": None,
            }

        parents = self._parents.get(target, [])
        if parents and not all(parent in evidence for parent in parents):
            raise ValueError("Missing parent assignments for direct GP prediction")
        parent_values = {p: _to_float(evidence[p]) for p in parents}
        mean, std = self._predict_node_normal(target, parent_values)
        return self._format_target_distribution(
            target=target, mean=mean, std=std, samples=None, n_samples=None
        )

    def _infer_distribution(
        self, target: str, evidence: dict, do: dict, n_samples: int
    ) -> dict:
        if not self._fitted:
            raise RuntimeError("Model is not fitted")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")

        inference_name = (
            str(self.benchmark_config.inference.name)
            if self.benchmark_config is not None
            else "gp_forward_sample"
        )
        supported = {"gp_posterior", "gp_forward_sample"}
        if inference_name not in supported:
            raise ValueError(
                f"Unsupported gpytorch inference method '{inference_name}'. "
                f"Supported: {sorted(supported)}"
            )

        overlap = set(evidence) & set(do)
        if overlap:
            raise ValueError("Nodes cannot be in both evidence and do")
        evidence_vals = {k: _to_float(v) for k, v in evidence.items()}
        do_vals = {k: _to_float(v) for k, v in do.items()}

        if inference_name == "gp_posterior":
            parents = self._parents.get(target, [])
            assignment = {**evidence_vals, **do_vals}
            if target in do_vals or all(parent in assignment for parent in parents):
                return self._direct_distribution(target, assignment, do=do_vals)

        rng = np.random.default_rng(self.seed)
        target_samples = np.zeros(int(n_samples), dtype=float)
        logw = np.zeros(int(n_samples), dtype=float)
        two_pi = 2.0 * math.pi

        for idx in range(int(n_samples)):
            assignment: dict[str, float] = {}
            for node in self._topo:
                if node in do_vals:
                    assignment[node] = float(do_vals[node])
                    continue

                parents = self._parents.get(node, [])
                parent_values = {p: float(assignment[p]) for p in parents}
                mean, std = self._predict_node_normal(node, parent_values)
                std = max(float(std), 1e-8)
                if node in evidence_vals:
                    val = float(evidence_vals[node])
                    assignment[node] = val
                    if (
                        math.isfinite(val)
                        and math.isfinite(mean)
                        and math.isfinite(std)
                    ):
                        z = (val - mean) / std
                        logw[idx] += -0.5 * (z * z + math.log(two_pi * std * std))
                    else:
                        logw[idx] = float("-inf")
                else:
                    assignment[node] = float(rng.normal(mean, std))

            target_samples[idx] = float(assignment[target])

        if not np.isfinite(logw).any():
            weights = np.ones(int(n_samples), dtype=float) / float(n_samples)
        else:
            centered = logw - float(np.nanmax(logw))
            weights = np.exp(centered)
            total = float(weights.sum())
            if total <= 0 or not math.isfinite(total):
                weights = np.ones(int(n_samples), dtype=float) / float(n_samples)
            else:
                weights /= total

        k = _target_discrete_cardinality(self.domain, target)
        if k > 0:
            probs = _samples_to_categorical_probs(target_samples, k=k, weights=weights)
            return {
                "format": "categorical_probs",
                "k": int(k),
                "probs": probs,
                "support": list(range(int(k))),
            }

        mean = float(np.sum(weights * target_samples))
        var = float(np.sum(weights * (target_samples - mean) ** 2))
        std = math.sqrt(var) if var >= 0 else float("nan")
        kept = target_samples[: min(len(target_samples), 2048)]
        return {
            "format": "normal_params",
            "mean": mean,
            "std": std,
            "n_samples": int(n_samples),
            "samples": [float(x) for x in kept],
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
                result = self._direct_distribution(target, assignment, do=do)
            else:
                n_samples = _get_n_mc(query, default=1024)
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
            n_samples = _get_n_mc(query, default=1024)
            result = self._infer_distribution(target, evidence, do, n_samples)
            ok = _result_ok(result)
            error = None if ok else "Unsupported or empty inference result"
        except Exception as exc:
            ok = False
            error = f"{type(exc).__name__}: {exc}"
            result = None
        timing_ms = (time.perf_counter() - start) * 1000.0
        return {"ok": ok, "error": error, "timing_ms": timing_ms, "result": result}
