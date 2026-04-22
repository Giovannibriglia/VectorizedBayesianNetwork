from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


class BNFitParseError(RuntimeError):
    pass


@dataclass(frozen=True)
class GaussianParams:
    intercept: float
    coeffs: Dict[str, float]
    sigma: float


@dataclass(frozen=True)
class DiscreteCPD:
    node: str
    parents: List[str]
    states: List[str]
    parent_state_sizes: List[int]
    probs: Dict[Tuple[int, ...], np.ndarray]


@dataclass(frozen=True)
class ContinuousCPD:
    node: str
    parents: List[str]
    discrete_parents: List[str]
    continuous_parents: List[str]
    discrete_parent_state_sizes: List[int]
    params: Dict[Tuple[int, ...], GaussianParams]


@dataclass(frozen=True)
class BNFitModel:
    nodes: List[str]
    parents: Dict[str, List[str]]
    node_types: Dict[str, str]
    states: Dict[str, List[str]]
    discrete_cpds: Dict[str, DiscreteCPD]
    continuous_cpds: Dict[str, ContinuousCPD]
    topo: List[str]


def load_bnfit(path: Path) -> BNFitModel:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing bn.fit file: {path}")

    errors: list[str] = []

    try:
        return _load_bnfit_rdata(path)
    except Exception as exc:
        errors.append(f"rdata: {type(exc).__name__}: {exc}")

    try:
        return _load_bnfit_pyreadr(path)
    except Exception as exc:
        errors.append(f"pyreadr: {type(exc).__name__}: {exc}")

    if path.suffix.lower() == ".rds":
        try:
            return _load_bnfit_rds2py(path)
        except Exception as exc:
            errors.append(f"rds2py: {type(exc).__name__}: {exc}")

    msg = "Failed to parse bn.fit"
    if errors:
        msg = f"{msg} ({'; '.join(errors)})"
    raise BNFitParseError(msg)


def find_bnfit_path(
    dataset_dir: Path, download_meta: dict | None = None
) -> Path | None:
    dataset_dir = Path(dataset_dir)
    candidates: list[Path] = []
    if isinstance(download_meta, dict):
        downloaded = download_meta.get("downloaded_files") or {}
        if isinstance(downloaded, dict):
            rds_name = downloaded.get("rds")
            rda_name = downloaded.get("rda")
            if rds_name:
                candidates.append(dataset_dir / rds_name)
            if rda_name:
                candidates.append(dataset_dir / rda_name)
    candidates.extend([dataset_dir / "model.rds", dataset_dir / "model.rda"])
    for path in candidates:
        if path.exists():
            return path
    return None


def build_bnfit_from_spec(spec: dict) -> BNFitModel:
    if not isinstance(spec, dict):
        raise BNFitParseError("Spec must be a dict")
    nodes_spec = spec.get("nodes")
    if not isinstance(nodes_spec, dict) or not nodes_spec:
        raise BNFitParseError("Spec missing nodes")

    nodes: list[str] = []
    parents: dict[str, list[str]] = {}
    node_types: dict[str, str] = {}
    states: dict[str, list[str]] = {}
    discrete_cpds: dict[str, DiscreteCPD] = {}
    continuous_cpds: dict[str, ContinuousCPD] = {}

    for node, meta in nodes_spec.items():
        if not isinstance(meta, dict):
            raise BNFitParseError(f"Invalid node spec for {node}")
        nodes.append(node)
        parents[node] = list(meta.get("parents") or [])
        ntype = str(meta.get("type") or "").lower()
        if ntype not in {"discrete", "gaussian", "clgaussian"}:
            raise BNFitParseError(f"Unsupported node type '{ntype}' for {node}")
        if ntype == "discrete":
            node_types[node] = "discrete"
            node_states = list(meta.get("states") or [])
            if not node_states:
                raise BNFitParseError(f"Missing states for {node}")
            states[node] = node_states
            probs_map = meta.get("cpt") or {}
            if not isinstance(probs_map, dict):
                raise BNFitParseError(f"Invalid CPT for {node}")
            probs: Dict[Tuple[int, ...], np.ndarray] = {}
            parent_sizes = []
            for parent in parents[node]:
                p_states = nodes_spec.get(parent, {}).get("states") or []
                parent_sizes.append(len(p_states))
            for key, values in probs_map.items():
                if key is None:
                    key = ()
                if not isinstance(key, tuple):
                    key = tuple(key)
                arr = np.asarray(list(values), dtype=float)
                probs[key] = arr
            discrete_cpds[node] = DiscreteCPD(
                node=node,
                parents=parents[node],
                states=node_states,
                parent_state_sizes=parent_sizes,
                probs=probs,
            )
        else:
            node_types[node] = "continuous"
            disc_parents = list(meta.get("discrete_parents") or [])
            cont_parents = list(meta.get("continuous_parents") or [])
            if not cont_parents:
                cont_parents = [p for p in parents[node] if p not in disc_parents]
            if ntype == "gaussian":
                disc_parents = []
            params_spec = meta.get("params")
            if params_spec is None:
                intercept = float(meta.get("intercept", 0.0))
                coeffs = dict(meta.get("coeffs") or {})
                sigma = float(meta.get("sd", meta.get("sigma", 1.0)))
                params_spec = {
                    (): {"intercept": intercept, "coeffs": coeffs, "sd": sigma}
                }
            params: Dict[Tuple[int, ...], GaussianParams] = {}
            state_sizes: list[int] = []
            for parent in disc_parents:
                p_states = nodes_spec.get(parent, {}).get("states") or []
                state_sizes.append(len(p_states))
            for key, values in params_spec.items():
                if key is None:
                    key = ()
                if not isinstance(key, tuple):
                    key = tuple(key)
                if isinstance(values, GaussianParams):
                    params[key] = values
                    continue
                if not isinstance(values, dict):
                    raise BNFitParseError(f"Invalid params for {node}")
                intercept = float(values.get("intercept", 0.0))
                coeffs = dict(values.get("coeffs") or {})
                sigma = float(values.get("sd", values.get("sigma", 1.0)))
                params[key] = GaussianParams(
                    intercept=intercept, coeffs=coeffs, sigma=sigma
                )
            continuous_cpds[node] = ContinuousCPD(
                node=node,
                parents=parents[node],
                discrete_parents=disc_parents,
                continuous_parents=cont_parents,
                discrete_parent_state_sizes=state_sizes,
                params=params,
            )

    topo = validate_dag(nodes, parents)
    return BNFitModel(
        nodes=nodes,
        parents=parents,
        node_types=node_types,
        states=states,
        discrete_cpds=discrete_cpds,
        continuous_cpds=continuous_cpds,
        topo=topo,
    )


def can_parse_bnfit(path: Path) -> bool:
    try:
        load_bnfit(path)
    except Exception:
        return False
    return True


def validate_dag(nodes: Iterable[str], parents: Dict[str, List[str]]) -> List[str]:
    nodes = list(nodes)
    node_set = set(nodes)
    cleaned_parents: Dict[str, List[str]] = {}
    for node in nodes:
        raw = parents.get(node, [])
        cleaned = _clean_parent_names(raw, node_set)
        cleaned_parents[node] = cleaned

    indegree: Dict[str, int] = {node: 0 for node in nodes}
    children: Dict[str, List[str]] = {node: [] for node in nodes}
    for node in nodes:
        for parent in cleaned_parents.get(node, []):
            if parent == node:
                continue
            indegree[node] += 1
            children.setdefault(parent, []).append(node)
    import heapq

    heap = [n for n in nodes if indegree.get(n, 0) == 0]
    heapq.heapify(heap)
    order: List[str] = []
    while heap:
        node = heapq.heappop(heap)
        order.append(node)
        for child in sorted(children.get(node, [])):
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(heap, child)
    if len(order) != len(nodes):
        edges = [
            (p, n) for n, ps in cleaned_parents.items() for p in ps if p is not None
        ]
        self_loops = [(n, p) for n, ps in cleaned_parents.items() for p in ps if p == n]
        bad_parents = [
            (n, p) for n, ps in cleaned_parents.items() for p in ps if p not in node_set
        ]
        avg_parents = float(len(edges)) / float(len(nodes)) if nodes else 0.0
        raise BNFitParseError(
            "Cycle detected in BN; cannot topologically sort. "
            f"n_nodes={len(nodes)} n_edges={len(edges)} "
            f"avg_parents={avg_parents:.2f} "
            f"self_loops={self_loops[:10]} "
            f"bad_parents={bad_parents[:10]} "
            f"sample_edges={edges[:30]}"
        )
    return order


def topological_order(nodes: Iterable[str], parents: Dict[str, List[str]]) -> List[str]:
    return validate_dag(nodes, parents)


def extract_arcs_bnlearn(
    obj: Any, *, nodes: Iterable[str] | None = None
) -> List[Tuple[str, str]]:
    node_set = set(nodes) if nodes is not None else None
    candidates = list(_iter_arcs_objects(obj))
    best_edges: list[tuple[str, str]] = []
    for candidate in candidates:
        edges = _edges_from_arcs_obj(candidate)
        if not edges:
            continue
        cleaned: list[tuple[str, str]] = []
        for src, dst in edges:
            if src is None or dst is None:
                continue
            s = _clean_name(src)
            t = _clean_name(dst)
            if not s or not t:
                continue
            if s == t:
                continue
            if node_set is not None and (s not in node_set or t not in node_set):
                continue
            cleaned.append((s, t))
        if len(cleaned) > len(best_edges):
            best_edges = cleaned
    return best_edges


def _iter_arcs_objects(obj: Any, *, _seen: set[int] | None = None) -> Iterable[Any]:
    if _seen is None:
        _seen = set()
    oid = id(obj)
    if oid in _seen:
        return []
    _seen.add(oid)

    value = _r_value(obj)
    if value is not None and value is not obj:
        yield from _iter_arcs_objects(value, _seen=_seen)

    if isinstance(obj, dict):
        for key, val in obj.items():
            if str(key).lower() == "arcs":
                yield val
            yield from _iter_arcs_objects(val, _seen=_seen)
        return []

    if hasattr(obj, "arcs"):
        try:
            arcs_val = getattr(obj, "arcs")
            yield arcs_val
            yield from _iter_arcs_objects(arcs_val, _seen=_seen)
        except Exception:
            pass

    if isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _iter_arcs_objects(item, _seen=_seen)

    return []


def _edges_from_arcs_obj(arcs_obj: Any) -> List[Tuple[Any, Any]]:
    if arcs_obj is None:
        return []
    value = _r_value(arcs_obj)
    if value is not None and value is not arcs_obj:
        arcs_obj = value

    # List of pairs
    if isinstance(arcs_obj, (list, tuple)):
        if not arcs_obj:
            return []
        if all(isinstance(item, (list, tuple)) and len(item) >= 2 for item in arcs_obj):
            return [(item[0], item[1]) for item in arcs_obj]
        if (
            len(arcs_obj) == 2
            and isinstance(arcs_obj[0], (list, tuple))
            and isinstance(arcs_obj[1], (list, tuple))
        ):
            return list(zip(arcs_obj[0], arcs_obj[1]))

    # pandas DataFrame or similar
    if hasattr(arcs_obj, "to_numpy") and hasattr(arcs_obj, "shape"):
        try:
            arr = arcs_obj.to_numpy()
            return _edges_from_array(arr)
        except Exception:
            pass

    # numpy array
    if isinstance(arcs_obj, np.ndarray):
        return _edges_from_array(arcs_obj)

    # rdata dict with data + dim
    if isinstance(arcs_obj, dict) and "data" in arcs_obj:
        data = arcs_obj.get("data")
        dim = _as_int_list(_r_attr(arcs_obj, "dim"))
        try:
            arr = np.asarray(data, dtype=object)
            if dim:
                try:
                    arr = arr.reshape(tuple(dim), order="F")
                except Exception:
                    arr = arr.reshape(tuple(dim))
            return _edges_from_array(arr)
        except Exception:
            return []

    return []


def _edges_from_array(arr: np.ndarray) -> List[Tuple[Any, Any]]:
    if arr.ndim == 1:
        if len(arr) % 2 != 0:
            return []
        arr = arr.reshape((-1, 2))
    if arr.ndim != 2:
        return []
    if arr.shape[1] == 2:
        return [(row[0], row[1]) for row in arr]
    if arr.shape[0] == 2:
        return list(zip(arr[0], arr[1]))
    return []


def _clean_name(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8", errors="ignore")
        except Exception:
            value = str(value)
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    try:
        name = str(value)
    except Exception:
        return None
    name = name.strip()
    if not name or name.lower() in {"na", "nan"}:
        return None
    return name


def _clean_parent_names(values: Iterable[Any], node_set: set[str]) -> List[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        name = _clean_name(value)
        if not name or name not in node_set:
            continue
        if name in seen:
            continue
        seen.add(name)
        cleaned.append(name)
    return cleaned


def sample_bnfit(
    model: BNFitModel, *, n_samples: int, seed: int | None = None
) -> Dict[str, np.ndarray]:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    rng = np.random.default_rng(seed)
    topo = list(model.topo)
    discrete_tables = _build_discrete_tables(model)

    samples: Dict[str, np.ndarray] = {}
    for node in topo:
        ntype = model.node_types.get(node, "discrete")
        if ntype == "discrete":
            table = discrete_tables.get(node)
            if table is None:
                raise BNFitParseError(f"Missing discrete CPD for {node}")
            if not table.parents:
                draws = rng.choice(table.k, size=n_samples, p=table.table[0])
                samples[node] = draws.astype(np.int32)
                continue
            parent_arrays = [samples[parent] for parent in table.parents]
            parent_index = np.zeros(n_samples, dtype=np.int64)
            for arr, mult in zip(parent_arrays, table.multipliers):
                parent_index += arr.astype(np.int64) * int(mult)
            out = np.empty(n_samples, dtype=np.int32)
            for idx in np.unique(parent_index):
                mask = parent_index == idx
                out[mask] = rng.choice(
                    table.k, size=int(mask.sum()), p=table.table[int(idx)]
                )
            samples[node] = out
        else:
            cpd = model.continuous_cpds.get(node)
            if cpd is None:
                raise BNFitParseError(f"Missing continuous CPD for {node}")
            if not cpd.discrete_parents:
                mean = _linear_gaussian_mean(cpd, samples, None, n_samples=n_samples)
                draws = rng.normal(mean, cpd.params[()].sigma, size=n_samples)
                samples[node] = draws.astype(float)
                continue
            disc_assign = _discrete_assignments(samples, cpd.discrete_parents)
            uniq, inverse = np.unique(disc_assign, axis=0, return_inverse=True)
            out = np.empty(n_samples, dtype=float)
            for idx, assign in enumerate(uniq):
                key = tuple(int(v) for v in assign.tolist())
                params = cpd.params.get(key)
                if params is None:
                    raise BNFitParseError(f"Missing CLG params for {node} at {key}")
                mask = inverse == idx
                mean = _linear_gaussian_mean(
                    cpd, samples, params, n_samples=n_samples, mask=mask
                )
                out[mask] = rng.normal(mean, params.sigma, size=int(mask.sum()))
            samples[node] = out
    return samples


def likelihood_weighted_samples(
    model: BNFitModel,
    *,
    evidence: Dict[str, Any],
    n_samples: int,
    seed: int | None = None,
) -> tuple[Dict[str, np.ndarray], np.ndarray]:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    rng = np.random.default_rng(seed)
    topo = list(model.topo)
    discrete_tables = _build_discrete_tables(model)

    samples: Dict[str, np.ndarray] = {}
    log_weights = np.zeros(n_samples, dtype=float)

    for node in topo:
        ntype = model.node_types.get(node, "discrete")
        if ntype == "discrete":
            table = discrete_tables.get(node)
            if table is None:
                raise BNFitParseError(f"Missing discrete CPD for {node}")
            if node in evidence:
                value = evidence[node]
                try:
                    ev_code = int(value)
                except Exception:
                    ev_code = None
                if ev_code is None or ev_code < 0 or ev_code >= table.k:
                    log_weights[:] = -np.inf
                    samples[node] = np.full(n_samples, -1, dtype=np.int32)
                    continue
                if not table.parents:
                    probs = table.table[0]
                    p = probs[ev_code]
                    if p <= 0:
                        log_weights[:] = -np.inf
                    else:
                        log_weights += math.log(p)
                    samples[node] = np.full(n_samples, ev_code, dtype=np.int32)
                    continue
                parent_arrays = [samples[parent] for parent in table.parents]
                parent_index = np.zeros(n_samples, dtype=np.int64)
                for arr, mult in zip(parent_arrays, table.multipliers):
                    parent_index += arr.astype(np.int64) * int(mult)
                probs = table.table[parent_index, ev_code]
                with np.errstate(divide="ignore"):
                    log_weights += np.log(probs)
                samples[node] = np.full(n_samples, ev_code, dtype=np.int32)
            else:
                if not table.parents:
                    draws = rng.choice(table.k, size=n_samples, p=table.table[0])
                    samples[node] = draws.astype(np.int32)
                else:
                    parent_arrays = [samples[parent] for parent in table.parents]
                    parent_index = np.zeros(n_samples, dtype=np.int64)
                    for arr, mult in zip(parent_arrays, table.multipliers):
                        parent_index += arr.astype(np.int64) * int(mult)
                    out = np.empty(n_samples, dtype=np.int32)
                    for idx in np.unique(parent_index):
                        mask = parent_index == idx
                        out[mask] = rng.choice(
                            table.k, size=int(mask.sum()), p=table.table[int(idx)]
                        )
                    samples[node] = out
        else:
            cpd = model.continuous_cpds.get(node)
            if cpd is None:
                raise BNFitParseError(f"Missing continuous CPD for {node}")
            if node in evidence:
                try:
                    ev_value = float(evidence[node])
                except Exception:
                    ev_value = float("nan")
                if not math.isfinite(ev_value):
                    log_weights[:] = -np.inf
                    samples[node] = np.full(n_samples, np.nan, dtype=float)
                    continue
                if not cpd.discrete_parents:
                    mean = _linear_gaussian_mean(
                        cpd, samples, None, n_samples=n_samples
                    )
                    log_weights += _normal_logpdf(ev_value, mean, cpd.params[()].sigma)
                    samples[node] = np.full(n_samples, ev_value, dtype=float)
                    continue
                disc_assign = _discrete_assignments(samples, cpd.discrete_parents)
                uniq, inverse = np.unique(disc_assign, axis=0, return_inverse=True)
                logw = np.empty(n_samples, dtype=float)
                for idx, assign in enumerate(uniq):
                    key = tuple(int(v) for v in assign.tolist())
                    params = cpd.params.get(key)
                    if params is None:
                        logw[inverse == idx] = -np.inf
                        continue
                    mask = inverse == idx
                    mean = _linear_gaussian_mean(
                        cpd, samples, params, n_samples=n_samples, mask=mask
                    )
                    logw[mask] = _normal_logpdf(ev_value, mean, params.sigma)
                log_weights += logw
                samples[node] = np.full(n_samples, ev_value, dtype=float)
            else:
                if not cpd.discrete_parents:
                    mean = _linear_gaussian_mean(
                        cpd, samples, None, n_samples=n_samples
                    )
                    draws = rng.normal(mean, cpd.params[()].sigma, size=n_samples)
                    samples[node] = draws.astype(float)
                else:
                    disc_assign = _discrete_assignments(samples, cpd.discrete_parents)
                    uniq, inverse = np.unique(disc_assign, axis=0, return_inverse=True)
                    out = np.empty(n_samples, dtype=float)
                    for idx, assign in enumerate(uniq):
                        key = tuple(int(v) for v in assign.tolist())
                        params = cpd.params.get(key)
                        if params is None:
                            raise BNFitParseError(
                                f"Missing CLG params for {node} at {key}"
                            )
                        mask = inverse == idx
                        mean = _linear_gaussian_mean(
                            cpd, samples, params, n_samples=n_samples, mask=mask
                        )
                        out[mask] = rng.normal(mean, params.sigma, size=int(mask.sum()))
                    samples[node] = out

    if not np.isfinite(log_weights).any():
        weights = np.zeros(n_samples, dtype=float)
    else:
        max_log = np.nanmax(log_weights)
        weights = np.exp(log_weights - max_log)
    return samples, weights


def estimate_continuous_ranges(
    model: BNFitModel,
    *,
    n_samples: int = 5000,
    seed: int | None = None,
    q_low: float = 0.01,
    q_high: float = 0.99,
) -> Dict[str, dict]:
    samples = sample_bnfit(model, n_samples=n_samples, seed=seed)
    ranges: Dict[str, dict] = {}
    for node, values in samples.items():
        if model.node_types.get(node) != "continuous":
            continue
        arr = np.asarray(values, dtype=float)
        if arr.size == 0 or not np.isfinite(arr).any():
            continue
        low = float(np.quantile(arr, q_low))
        high = float(np.quantile(arr, q_high))
        if not math.isfinite(low) or not math.isfinite(high):
            continue
        if low == high:
            low -= 1.0
            high += 1.0
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=0))
        ranges[node] = {
            "low": low,
            "high": high,
            "mean": mean,
            "std": std,
        }
    return ranges


@dataclass(frozen=True)
class _DiscreteTable:
    node: str
    parents: List[str]
    table: np.ndarray
    multipliers: np.ndarray
    k: int


def _build_discrete_tables(model: BNFitModel) -> Dict[str, _DiscreteTable]:
    tables: Dict[str, _DiscreteTable] = {}
    for node, cpd in model.discrete_cpds.items():
        parents = cpd.parents
        k = len(cpd.states)
        if not parents:
            probs = cpd.probs.get(tuple())
            if probs is None:
                raise BNFitParseError(f"Missing CPD row for {node}")
            table = _normalize_probs(np.asarray(probs, dtype=float), k, node).reshape(
                1, -1
            )
            tables[node] = _DiscreteTable(
                node=node,
                parents=[],
                table=table,
                multipliers=np.array([], dtype=int),
                k=k,
            )
            continue
        parent_sizes = list(cpd.parent_state_sizes)
        combos = list(_product_indices(parent_sizes))
        table = np.zeros((len(combos), k), dtype=float)
        for idx, combo in enumerate(combos):
            probs = cpd.probs.get(tuple(combo))
            if probs is None:
                raise BNFitParseError(
                    f"Missing CPD entry for {node} with parents {combo}"
                )
            table[idx] = _normalize_probs(np.asarray(probs, dtype=float), k, node)
        multipliers = []
        prod = 1
        for size in reversed(parent_sizes):
            multipliers.append(prod)
            prod *= size
        multipliers = list(reversed(multipliers))
        tables[node] = _DiscreteTable(
            node=node,
            parents=parents,
            table=table,
            multipliers=np.asarray(multipliers, dtype=int),
            k=k,
        )
    return tables


def _normalize_probs(probs: np.ndarray, expected: int, node: str) -> np.ndarray:
    if probs.shape[-1] != expected:
        raise BNFitParseError(
            f"CPD for {node} has {len(probs)} entries; expected {expected}"
        )
    arr = np.asarray(probs, dtype=float)
    total = float(arr.sum())
    if total <= 0 or not math.isfinite(total):
        raise BNFitParseError(f"CPD for {node} has non-positive total probability")
    return arr / total


def _product_indices(sizes: List[int]) -> Iterable[Tuple[int, ...]]:
    if not sizes:
        yield tuple()
        return
    ranges = [range(size) for size in sizes]
    from itertools import product

    yield from product(*ranges)


def _linear_gaussian_mean(
    cpd: ContinuousCPD,
    samples: Dict[str, np.ndarray],
    params: GaussianParams | None,
    *,
    n_samples: int,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    if params is None:
        params = cpd.params[tuple()]
    if not cpd.continuous_parents:
        if mask is None:
            return np.full(int(n_samples), params.intercept, dtype=float)
        return np.full(int(mask.sum()), params.intercept, dtype=float)
    vals = None
    for parent in cpd.continuous_parents:
        coeff = params.coeffs.get(parent, 0.0)
        if coeff == 0.0:
            continue
        arr = samples[parent]
        if mask is not None:
            arr = arr[mask]
        if vals is None:
            vals = coeff * arr
        else:
            vals = vals + coeff * arr
    if vals is None:
        base = np.zeros(int(mask.sum()) if mask is not None else int(n_samples))
    else:
        base = vals
    return base + params.intercept


def _discrete_assignments(
    samples: Dict[str, np.ndarray], parents: List[str]
) -> np.ndarray:
    if not parents:
        return np.zeros((len(next(iter(samples.values()))), 0), dtype=int)
    return np.stack([samples[parent] for parent in parents], axis=1)


def _normal_logpdf(x: float, mean: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0 or not math.isfinite(sigma):
        return np.full_like(mean, -np.inf, dtype=float)
    var = sigma * sigma
    return -0.5 * (math.log(2.0 * math.pi * var) + ((mean - x) ** 2) / var)


def _load_bnfit_rdata(path: Path) -> BNFitModel:
    try:
        import rdata
        from rdata import conversion
    except Exception as exc:
        raise BNFitParseError(
            "rdata is required to read bn.fit RDS/RDA files."
        ) from exc

    parsed = None
    if path.suffix.lower() == ".rda":
        parsed = rdata.read_rda(str(path), expand_altrep=True)
    else:
        parsed = rdata.read_rds(str(path), expand_altrep=True)

    candidates: list[Any] = []
    try:
        converted = conversion.convert(parsed)
        candidates.append(converted)
    except Exception:
        pass
    candidates.append(parsed)

    for obj in candidates:
        if obj is None:
            continue
        if path.suffix.lower() == ".rda" and isinstance(obj, dict):
            if "bn" in obj:
                try:
                    return _parse_bnfit_object_with_raw(obj["bn"], raw_obj=parsed)
                except Exception:
                    pass
            for candidate in obj.values():
                try:
                    return _parse_bnfit_object_with_raw(candidate, raw_obj=parsed)
                except Exception:
                    continue
        try:
            return _parse_bnfit_object_with_raw(obj, raw_obj=parsed)
        except Exception:
            continue

    raise BNFitParseError("rdata parsed object but bn.fit could not be decoded")


def _load_bnfit_pyreadr(path: Path) -> BNFitModel:
    try:
        import pyreadr
    except Exception as exc:
        raise BNFitParseError(
            "pyreadr is required to read bn.fit RDS/RDA files."
        ) from exc

    rdata = pyreadr.read_r(str(path))
    if not isinstance(rdata, dict) or not rdata:
        raise BNFitParseError("pyreadr returned empty object")

    if len(rdata) == 1:
        obj = next(iter(rdata.values()))
    else:
        obj = None
        for candidate in rdata.values():
            if _is_bnfit_object(candidate):
                obj = candidate
                break
        if obj is None:
            obj = next(iter(rdata.values()))
    return _parse_bnfit_object_with_raw(obj, raw_obj=obj)


def _load_bnfit_rds2py(path: Path) -> BNFitModel:
    try:
        import rds2py
    except Exception as exc:
        raise BNFitParseError("rds2py is required to read bn.fit RDS files.") from exc
    obj = rds2py.read_rds(str(path))
    return _parse_bnfit_object_with_raw(obj, raw_obj=obj)


def _parse_bnfit_object(obj: Any) -> BNFitModel:
    return _parse_bnfit_object_with_raw(obj, raw_obj=obj)


def _parse_bnfit_object_with_raw(obj: Any, raw_obj: Any) -> BNFitModel:
    if isinstance(obj, BNFitModel):
        return obj
    if isinstance(obj, dict) and "nodes" in obj and isinstance(obj.get("nodes"), dict):
        return build_bnfit_from_spec(obj)

    nodes_map = _as_named_list(obj)
    if not nodes_map:
        raise BNFitParseError("bn.fit object missing nodes")

    nodes: list[str] = []
    parents: dict[str, list[str]] = {}
    node_types: dict[str, str] = {}
    states: dict[str, list[str]] = {}
    discrete_cpds: dict[str, DiscreteCPD] = {}
    continuous_cpds: dict[str, ContinuousCPD] = {}

    node_objs: dict[str, Any] = {}
    node_classes: dict[str, List[str]] = {}

    raw_parents: dict[str, list[str]] = {}

    for node_name, node_obj in nodes_map.items():
        node = _as_string(_r_list_get(node_obj, "node")) or str(node_name)
        node = str(node)
        nodes.append(node)
        node_objs[node] = node_obj
        parent_list = _as_str_list(_r_list_get(node_obj, "parents"))
        raw_parents[node] = parent_list
        classes = _r_class(node_obj)
        node_classes[node] = classes
        ntype = _infer_node_type(classes, node_obj)
        node_types[node] = "discrete" if ntype == "discrete" else "continuous"

    arcs = extract_arcs_bnlearn(raw_obj, nodes=nodes)
    if not arcs and obj is not raw_obj:
        arcs = extract_arcs_bnlearn(obj, nodes=nodes)
    if arcs:
        parents = {n: [] for n in nodes}
        for src, dst in arcs:
            parents.setdefault(dst, []).append(src)
    else:
        parents = {n: list(raw_parents.get(n, [])) for n in nodes}

    parents = {n: _clean_parent_names(parents.get(n, []), set(nodes)) for n in nodes}

    for node in nodes:
        if node_types.get(node) != "discrete":
            continue
        node_obj = node_objs[node]
        prob = _r_list_get(node_obj, "prob")
        if prob is None:
            raise BNFitParseError(f"Missing CPT for node {node}")
        parent_list = raw_parents.get(node) or parents.get(node, [])
        node_states, parent_states, probs_map = _parse_discrete_prob(
            prob, node, parent_list
        )
        states[node] = node_states
        parent_sizes = [len(s) for s in parent_states]
        discrete_cpds[node] = DiscreteCPD(
            node=node,
            parents=parent_list,
            states=node_states,
            parent_state_sizes=parent_sizes,
            probs=probs_map,
        )

    for node in nodes:
        if node_types.get(node) != "continuous":
            continue
        node_obj = node_objs[node]
        classes = node_classes.get(node, [])
        parent_list = raw_parents.get(node) or parents.get(node, [])
        if _has_class(classes, "cgnode"):
            cpd = _parse_cgnode(node_obj, node, parent_list, states)
        else:
            cpd = _parse_gnode(node_obj, node, parent_list)
        continuous_cpds[node] = cpd

    topo = validate_dag(nodes, parents)
    return BNFitModel(
        nodes=nodes,
        parents=parents,
        node_types=node_types,
        states=states,
        discrete_cpds=discrete_cpds,
        continuous_cpds=continuous_cpds,
        topo=topo,
    )


def _parse_discrete_prob(
    prob: Any, node: str, parents: List[str]
) -> tuple[List[str], List[List[str]], Dict[Tuple[int, ...], np.ndarray]]:
    arr, dimnames = _as_array_with_dimnames(prob)
    if arr is None:
        raise BNFitParseError(f"Missing CPT array for {node}")
    if dimnames is None or not dimnames:
        raise BNFitParseError(f"Missing CPT dimnames for {node}")

    node_states = [str(s) for s in dimnames[0]]
    parent_states = [list(map(str, levels)) for levels in dimnames[1:]]
    if parents and len(parent_states) != len(parents):
        # Best effort: ignore parents mismatch when dimnames missing.
        parent_states = parent_states[: len(parents)]
    probs_map: Dict[Tuple[int, ...], np.ndarray] = {}

    if not parents:
        probs_map[tuple()] = np.asarray(arr).reshape(-1)
        return node_states, parent_states, probs_map

    """state_to_idx = [
        {state: idx for idx, state in enumerate(states)} for states in parent_states
    ]"""
    for combo in _product_indices([len(s) for s in parent_states]):
        idxs = tuple(int(v) for v in combo)
        slicer = (slice(None),) + idxs
        probs = np.asarray(arr[slicer], dtype=float).reshape(-1)
        probs_map[idxs] = probs
    return node_states, parent_states, probs_map


def _parse_gnode(node_obj: Any, node: str, parents: List[str]) -> ContinuousCPD:
    coeff_obj = _r_list_get(node_obj, "coefficients")
    if coeff_obj is None:
        coeff_obj = _r_list_get(node_obj, "coef")
    values, names = _as_named_vector(coeff_obj)
    if values is None or len(values) == 0:
        raise BNFitParseError(f"Missing coefficients for {node}")
    intercept = None
    coeffs: Dict[str, float] = {}
    if names:
        for name, val in zip(names, values):
            if name in {"(Intercept)", "Intercept"}:
                intercept = float(val)
            else:
                coeffs[str(name)] = float(val)
        if intercept is None:
            intercept = float(values[0])
            for parent, val in zip(parents, values[1:]):
                coeffs.setdefault(parent, float(val))
    else:
        intercept = float(values[0])
        for parent, val in zip(parents, values[1:]):
            coeffs[parent] = float(val)

    sd_obj = _r_list_get(node_obj, "sd")
    sigma = float(_as_float(sd_obj, default=1.0))

    params = {tuple(): GaussianParams(intercept=intercept, coeffs=coeffs, sigma=sigma)}
    return ContinuousCPD(
        node=node,
        parents=parents,
        discrete_parents=[],
        continuous_parents=list(parents),
        discrete_parent_state_sizes=[],
        params=params,
    )


def _parse_cgnode(
    node_obj: Any, node: str, parents: List[str], states: Dict[str, List[str]]
) -> ContinuousCPD:
    dparents_idx = [i - 1 for i in _as_int_list(_r_list_get(node_obj, "dparents"))]
    gparents_idx = [i - 1 for i in _as_int_list(_r_list_get(node_obj, "gparents"))]
    discrete_parents = [parents[i] for i in dparents_idx if 0 <= i < len(parents)]
    continuous_parents = [parents[i] for i in gparents_idx if 0 <= i < len(parents)]
    dlevels = _r_list_get(node_obj, "dlevels")
    dlevels_list = _as_list_of_str_lists(dlevels)
    if discrete_parents and not dlevels_list:
        dlevels_list = [states.get(p, []) for p in discrete_parents]

    coeff_obj = _r_list_get(node_obj, "coefficients")
    if coeff_obj is None:
        coeff_obj = _r_list_get(node_obj, "coef")
    coef_mat, row_names, _ = _as_matrix(coeff_obj)
    if coef_mat is None:
        raise BNFitParseError(f"Missing coefficients matrix for {node}")
    sd_obj = _r_list_get(node_obj, "sd")
    sd_vals = _as_float_list(sd_obj)
    if not sd_vals:
        sd_vals = [1.0]

    configs = _as_int_list(_r_list_get(node_obj, "configs"))
    if not configs:
        configs = list(range(1, coef_mat.shape[1] + 1))

    if not row_names:
        row_names = ["(Intercept)"] + list(continuous_parents)

    intercept_idx = None
    for idx, name in enumerate(row_names):
        if name in {"(Intercept)", "Intercept"}:
            intercept_idx = idx
            break
    if intercept_idx is None:
        intercept_idx = 0

    row_lookup = {name: idx for idx, name in enumerate(row_names)}

    if continuous_parents:
        cont_indices = []
        for parent in continuous_parents:
            if parent in row_lookup:
                cont_indices.append(row_lookup[parent])
            else:
                cont_indices.append(None)
    else:
        cont_indices = []

    combos = _expand_grid(dlevels_list) if dlevels_list else [tuple()]
    params: Dict[Tuple[int, ...], GaussianParams] = {}
    for col_idx, cfg in enumerate(configs):
        cfg_index = int(cfg) - 1
        if cfg_index < 0 or cfg_index >= len(combos):
            continue
        assignment_states = combos[cfg_index]
        assignment_codes: Tuple[int, ...] = tuple()
        if discrete_parents:
            assignment_codes = tuple(
                _state_index(states, parent, state)
                for parent, state in zip(discrete_parents, assignment_states)
            )
        coeffs: Dict[str, float] = {}
        for parent, row_idx in zip(continuous_parents, cont_indices):
            if row_idx is None:
                continue
            coeffs[parent] = float(coef_mat[row_idx, col_idx])
        intercept = float(coef_mat[intercept_idx, col_idx])
        sigma = float(sd_vals[col_idx] if col_idx < len(sd_vals) else sd_vals[-1])
        params[assignment_codes] = GaussianParams(
            intercept=intercept, coeffs=coeffs, sigma=sigma
        )

    state_sizes = [len(states.get(p, [])) for p in discrete_parents]
    return ContinuousCPD(
        node=node,
        parents=parents,
        discrete_parents=discrete_parents,
        continuous_parents=continuous_parents,
        discrete_parent_state_sizes=state_sizes,
        params=params,
    )


def _state_index(states: Dict[str, List[str]], parent: str, state: str) -> int:
    values = states.get(parent, [])
    if state in values:
        return values.index(state)
    return 0


def _expand_grid(levels: List[List[str]]) -> List[Tuple[str, ...]]:
    if not levels:
        return [tuple()]
    from itertools import product

    combos = []
    for combo in product(*reversed(levels)):
        combos.append(tuple(reversed(combo)))
    return combos


def _infer_node_type(classes: List[str], node_obj: Any) -> str:
    if _has_class(classes, "dnode") or _has_class(classes, "onode"):
        return "discrete"
    if _has_class(classes, "cgnode") or _has_class(classes, "cnode"):
        return "continuous"
    if _has_class(classes, "gnode"):
        return "continuous"
    if _r_list_get(node_obj, "prob") is not None:
        return "discrete"
    if _r_list_get(node_obj, "coefficients") is not None:
        return "continuous"
    return "discrete"


def _has_class(classes: List[str], name: str) -> bool:
    return any(name in cls for cls in classes)


def _is_bnfit_object(obj: Any) -> bool:
    classes = _r_class(obj)
    return any("bn.fit" in cls for cls in classes)


def _r_value(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, bytes, bytearray, memoryview, np.ndarray, np.generic)):
        return None
    if type(obj).__module__.startswith("xarray"):
        return None
    for attr in ("value", "data"):
        if hasattr(obj, attr):
            try:
                return getattr(obj, attr)
            except Exception:
                continue
    return None


def _r_class(obj: Any) -> List[str]:
    if obj is None:
        return []
    if hasattr(obj, "rclass"):
        try:
            rclass = obj.rclass
            return [str(c) for c in rclass]
        except Exception:
            pass
    if hasattr(obj, "attributes"):
        try:
            attrs = getattr(obj, "attributes")
            if isinstance(attrs, dict):
                cls_attr = attrs.get("class")
                if cls_attr is not None:
                    return _as_str_list(cls_attr)
        except Exception:
            pass
    if isinstance(obj, dict):
        class_name = obj.get("class_name")
        if class_name:
            if isinstance(class_name, list):
                return [str(c) for c in class_name]
            return [str(class_name)]
        attrs = obj.get("attributes") or {}
        cls_attr = attrs.get("class")
        if cls_attr is not None:
            return _as_str_list(cls_attr)
    return []


def _as_named_list(obj: Any) -> Dict[str, Any]:
    value = _r_value(obj)
    if value is not None and value is not obj:
        if isinstance(value, dict):
            return {str(k): v for k, v in value.items()}
        if isinstance(value, list):
            names = _as_str_list(_r_attr(obj, "names"))
            if names and len(names) == len(value):
                return dict(zip(names, value))
            return {str(i): v for i, v in enumerate(value)}
    if isinstance(obj, dict):
        if "type" in obj and obj.get("type") in {"list", "pairlist", "vector"}:
            data = obj.get("data") or []
            names = _as_str_list(_r_attr(obj, "names"))
            if isinstance(data, dict):
                return {str(k): v for k, v in data.items()}
            if names and isinstance(data, (list, tuple)) and len(names) == len(data):
                return dict(zip(names, data))
            return {str(i): v for i, v in enumerate(data)}
        return {str(k): v for k, v in obj.items()}
    if hasattr(obj, "items"):
        try:
            return {str(k): v for k, v in obj.items()}
        except Exception:
            return {}
    return {}


def _r_list_get(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    value = _r_value(obj)
    if value is not None and value is not obj:
        if isinstance(value, dict):
            return value.get(key)
        if isinstance(value, list):
            names = _as_str_list(_r_attr(obj, "names"))
            if names and len(names) == len(value):
                try:
                    idx = names.index(key)
                except ValueError:
                    return None
                return value[idx]
    if isinstance(obj, dict):
        if "type" in obj and obj.get("type") in {"list", "pairlist", "vector"}:
            data = obj.get("data") or []
            names = _as_str_list(_r_attr(obj, "names"))
            if names and isinstance(data, (list, tuple)) and len(names) == len(data):
                try:
                    idx = names.index(key)
                except ValueError:
                    return None
                return data[idx]
            if isinstance(data, dict):
                return data.get(key)
        return obj.get(key)
    if hasattr(obj, "keys"):
        try:
            return obj.get(key)  # type: ignore[attr-defined]
        except Exception:
            pass
    return None


def _as_string(obj: Any) -> str | None:
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj
    if isinstance(obj, np.ndarray):
        if obj.size == 0:
            return None
        return _as_string(obj.reshape(-1)[0])
    if isinstance(obj, np.generic):
        try:
            return str(obj.item())
        except Exception:
            return str(obj)
    if isinstance(obj, (list, tuple)) and obj:
        return str(obj[0])
    try:
        return str(obj)
    except Exception:
        return None


def _as_int_list(obj: Any) -> List[int]:
    values = _as_list(obj)
    out: list[int] = []
    for value in values:
        try:
            out.append(int(value))
        except Exception:
            continue
    return out


def _as_float_list(obj: Any) -> List[float]:
    values = _as_list(obj)
    out: list[float] = []
    for value in values:
        try:
            out.append(float(value))
        except Exception:
            continue
    return out


def _as_str_list(obj: Any) -> List[str]:
    values = _as_list(obj)
    return [str(v) for v in values if v is not None]


def _as_list(obj: Any) -> List[Any]:
    if obj is None:
        return []
    value = _r_value(obj)
    if value is not None and value is not obj:
        return _as_list(value)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, np.ndarray):
        return obj.reshape(-1).tolist()
    if isinstance(obj, dict):
        rtype = str(obj.get("type") or "").lower()
        if rtype in {"character", "integer", "double", "string", "logical"}:
            data = obj.get("data")
            if isinstance(data, list):
                return data
            if isinstance(data, np.ndarray):
                return data.reshape(-1).tolist()
            if data is not None:
                return [data]
        if "data" in obj and obj.get("type") == "factor":
            data = obj.get("data")
            levels = _as_list(_r_attr(obj, "levels"))
            if isinstance(data, list):
                out = []
                for idx in data:
                    try:
                        out.append(levels[int(idx) - 1])
                    except Exception:
                        continue
                return out
        if "data" in obj and obj.get("type") in {"list", "pairlist", "vector"}:
            data = obj.get("data")
            if isinstance(data, list):
                return data
        if "data" in obj and isinstance(obj.get("data"), list):
            return obj.get("data")
    if hasattr(obj, "tolist"):
        try:
            return list(obj.tolist())
        except Exception:
            pass
    return [obj]


def _r_attr(obj: Any, name: str) -> Any:
    if hasattr(obj, "attributes"):
        try:
            attrs = getattr(obj, "attributes")
            if isinstance(attrs, dict):
                return attrs.get(name)
        except Exception:
            pass
    if not isinstance(obj, dict):
        return None
    attrs = obj.get("attributes") or {}
    return attrs.get(name)


def _as_float(obj: Any, default: float) -> float:
    if obj is None:
        return float(default)
    try:
        if isinstance(obj, (list, tuple, np.ndarray)) and len(obj) > 0:
            return float(obj[0])
        return float(obj)
    except Exception:
        return float(default)


def _as_array_with_dimnames(
    obj: Any,
) -> tuple[np.ndarray | None, List[List[str]] | None]:
    if obj is None:
        return None, None
    if _is_xarray_dataarray(obj):
        try:
            arr = np.asarray(obj.to_numpy(), dtype=float)
        except Exception:
            arr = np.asarray(getattr(obj, "values"), dtype=float)
        return arr, _xarray_dimnames(obj)
    value = _r_value(obj)
    if value is not None and value is not obj:
        if isinstance(value, np.ndarray):
            arr = np.asarray(value, dtype=float)
            dim = _as_int_list(_r_attr(obj, "dim"))
            dimnames = _normalize_dimnames(_r_attr(obj, "dimnames"))
            if dim:
                try:
                    arr = arr.reshape(tuple(dim), order="F")
                except Exception:
                    arr = arr.reshape(tuple(dim))
            return arr, dimnames
    if hasattr(obj, "matrix"):
        try:
            arr = np.asarray(obj.matrix, dtype=float)
            dimnames = obj.dimnames if hasattr(obj, "dimnames") else None
            return arr, _normalize_dimnames(dimnames)
        except Exception:
            pass
    if isinstance(obj, np.ndarray):
        arr = np.asarray(obj, dtype=float)
        dimnames = getattr(obj, "dimnames", None)
        return arr, _normalize_dimnames(dimnames)
    if isinstance(obj, dict):
        rtype = obj.get("type")
        data = obj.get("data")
        attrs = obj.get("attributes") or {}
        dim = _as_int_list(attrs.get("dim"))
        dimnames = _normalize_dimnames(attrs.get("dimnames"))
        if data is not None:
            arr = np.asarray(data, dtype=float)
            if dim:
                try:
                    arr = arr.reshape(tuple(dim), order="F")
                except Exception:
                    arr = arr.reshape(tuple(dim))
            return arr, dimnames
        if rtype in {"matrix", "array"}:
            arr = np.asarray(obj.get("matrix") or obj.get("array"), dtype=float)
            return arr, dimnames
    return None, None


def _as_matrix(
    obj: Any,
) -> tuple[np.ndarray | None, List[str] | None, List[str] | None]:
    arr, dimnames = _as_array_with_dimnames(obj)
    if arr is None:
        return None, None, None
    row_names = None
    col_names = None
    if dimnames and len(dimnames) >= 1:
        row_names = [str(v) for v in dimnames[0]]
    if dimnames and len(dimnames) >= 2:
        col_names = [str(v) for v in dimnames[1]]
    if arr.ndim == 1:
        arr = arr.reshape((-1, 1))
    return arr, row_names, col_names


def _normalize_dimnames(dimnames: Any) -> List[List[str]] | None:
    if dimnames is None:
        return None
    if isinstance(dimnames, dict) and "data" in dimnames:
        dimnames = dimnames.get("data")
    if isinstance(dimnames, (list, tuple)):
        out: list[list[str]] = []
        for entry in dimnames:
            out.append([str(v) for v in _as_list(entry)])
        return out
    return None


def _as_named_vector(obj: Any) -> tuple[np.ndarray | None, List[str] | None]:
    if obj is None:
        return None, None
    if _is_xarray_dataarray(obj):
        arr = np.asarray(obj.to_numpy(), dtype=float).reshape(-1)
        dimnames = _xarray_dimnames(obj)
        names = None
        if dimnames and len(dimnames) >= 1 and len(dimnames[0]) == arr.size:
            names = dimnames[0]
        return arr, names
    value = _r_value(obj)
    if value is not None and value is not obj:
        if isinstance(value, np.ndarray):
            arr = np.asarray(value, dtype=float).reshape(-1)
            names = _as_str_list(_r_attr(obj, "names"))
            return arr, names or None
    if isinstance(obj, np.ndarray):
        arr = np.asarray(obj, dtype=float).reshape(-1)
        names = getattr(obj, "names", None)
        if names is not None:
            names = [str(v) for v in _as_list(names)]
        return arr, names
    if isinstance(obj, dict):
        if "data" in obj:
            arr = np.asarray(obj.get("data"), dtype=float).reshape(-1)
            names = _as_str_list(_r_attr(obj, "names"))
            return arr, names or None
    if isinstance(obj, list):
        arr = np.asarray(obj, dtype=float).reshape(-1)
        return arr, None
    try:
        arr = np.asarray(obj, dtype=float).reshape(-1)
        return arr, None
    except Exception:
        return None, None


def _as_list_of_str_lists(obj: Any) -> List[List[str]]:
    values = _as_list(obj)
    out: List[List[str]] = []
    for entry in values:
        out.append([str(v) for v in _as_list(entry)])
    return out


def _is_xarray_dataarray(obj: Any) -> bool:
    mod = type(obj).__module__
    return (
        mod.startswith("xarray") and hasattr(obj, "to_numpy") and hasattr(obj, "dims")
    )


def _xarray_dimnames(obj: Any) -> List[List[str]] | None:
    try:
        dims = list(getattr(obj, "dims", []))
    except Exception:
        return None
    if not dims:
        return None
    try:
        shape = tuple(getattr(obj, "shape", ()))
    except Exception:
        shape = tuple()
    try:
        coords = getattr(obj, "coords", {})
    except Exception:
        coords = {}

    out: List[List[str]] = []
    for idx, dim in enumerate(dims):
        labels: List[str] | None = None
        try:
            if dim in coords:
                coord = coords[dim]
                coord_vals = getattr(coord, "values", coord)
                labels = [str(v) for v in np.asarray(coord_vals).reshape(-1).tolist()]
        except Exception:
            labels = None
        if labels is None:
            if idx >= len(shape):
                return None
            labels = [str(i) for i in range(int(shape[idx]))]
        out.append(labels)
    return out
