#!/usr/bin/env python3
"""
Python-only BNRepository helper for benchmarking.

Supports:
  - Discrete BNs via .bif.gz (parsed in Python, sampled from scratch)
  - Gaussian / CLG BNs via generated JSON models (no R, no Docker)

Typical usage:
  python generate_bnlearn.py --all --outdir benchmarking/bnlearn_data/generated --seed 0
  python generate_bnlearn.py --name ecoli70 --type gaussian --nodes 46 --edges 70 --out benchmarking/bnlearn_data/generated/ecoli70/model.json
  python generate_bnlearn.py --name healthcare --type clgaussian --spec benchmarking/bnlearn_data/specs/healthcare.yaml --out benchmarking/bnlearn_data/generated/healthcare/model.json
  python generate_bnlearn.py --save_samples --n 1000 --seed 42
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import urllib.request
from collections import defaultdict, deque
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml


# ----------------------------
# IO helpers
# ----------------------------
def download(url: str, dst: Path, force: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not force and dst.exists() and dst.stat().st_size > 0:
        return
    print(f"  downloading -> {dst.name}")
    with urllib.request.urlopen(url) as r, open(dst, "wb") as f:
        shutil.copyfileobj(r, f)


def gunzip(src_gz: Path, dst_bif: Path, force: bool = False) -> None:
    dst_bif.parent.mkdir(parents=True, exist_ok=True)
    if not force and dst_bif.exists() and dst_bif.stat().st_size > 0:
        return
    print(f"  extracting  -> {dst_bif.name}")
    with gzip.open(src_gz, "rb") as f_in, open(dst_bif, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


def load_bif_model(bif_path: Path):
    """
    Parse a .bif file and return the model object.
    We try multiple import paths to be robust across installations.
    """
    try:
        from pgmpy.readwrite import BIFReader  # type: ignore

        reader = BIFReader(str(bif_path))
        return reader.get_model()
    except Exception as e_pgmpy:
        try:
            import bnlearn as bn  # noqa: F401

            dag = bn.import_DAG(str(bif_path), verbose=0)
            if isinstance(dag, dict) and "model" in dag:
                return dag["model"]
            if hasattr(dag, "nodes") and hasattr(dag, "edges"):
                return dag
            raise RuntimeError(
                f"bn.import_DAG(path) did not return a model-like object: {type(dag)}"
            )
        except Exception as e_bn:
            raise RuntimeError(
                f"Failed to parse BIF via pgmpy and bnlearn.\n"
                f"pgmpy error: {type(e_pgmpy).__name__}: {e_pgmpy}\n"
                f"bnlearn error: {type(e_bn).__name__}: {e_bn}"
            )


# ----------------------------
# Discrete sampling (scratch)
# ----------------------------
def topological_sort(nodes: List[str], edges: List[Tuple[str, str]]) -> List[str]:
    graph = defaultdict(list)
    indegree = {n: 0 for n in nodes}
    for parent, child in edges:
        if parent not in indegree or child not in indegree:
            raise ValueError(f"Edge references unknown node: {(parent, child)}")
        graph[parent].append(child)
        indegree[child] += 1

    q = deque([n for n in nodes if indegree[n] == 0])
    order = []
    while q:
        n = q.popleft()
        order.append(n)
        for ch in graph[n]:
            indegree[ch] -= 1
            if indegree[ch] == 0:
                q.append(ch)

    if len(order) != len(nodes):
        raise ValueError("Graph is not a DAG (cycle detected) or nodes mismatch.")
    return order


def _mixed_radix_index(assign: Tuple[int, ...], cards: Tuple[int, ...]) -> int:
    idx = 0
    mult = 1
    for v, c in zip(reversed(assign), reversed(cards)):
        idx += int(v) * mult
        mult *= int(c)
    return idx


def _get_parents_from_cpd(cpd) -> List[str]:
    if hasattr(cpd, "get_evidence"):
        ev = cpd.get_evidence()
        return list(ev) if ev else []
    if hasattr(cpd, "variables") and cpd.variables is not None:
        return list(cpd.variables[1:])
    if hasattr(cpd, "evidence") and cpd.evidence is not None:
        return list(cpd.evidence)
    return []


def extract_cpds_from_model(model) -> Dict[str, Dict[str, Any]]:
    if not hasattr(model, "get_cpds"):
        raise TypeError("Model does not expose get_cpds(); unexpected object.")
    model_cpds = model.get_cpds()
    if not model_cpds:
        raise ValueError("No CPDs found inside model.get_cpds().")

    if not hasattr(model, "get_cardinality"):
        raise TypeError(
            "Model does not expose get_cardinality(); cannot infer cards robustly."
        )

    cpds_out: Dict[str, Dict[str, Any]] = {}

    for cpd in model_cpds:
        var = cpd.variable
        var_card = int(model.get_cardinality(var))

        parents = _get_parents_from_cpd(cpd)
        parent_cards = [int(model.get_cardinality(p)) for p in parents]

        values = np.asarray(cpd.values, dtype=float)
        if values.ndim == 1:
            values = values.reshape(var_card, -1)
        elif values.ndim != 2:
            values = values.reshape(var_card, -1)

        if parents:
            expected_cols = int(np.prod(parent_cards))
            if values.shape[1] != expected_cols:
                raise ValueError(
                    f"CPD shape mismatch for {var}: values has {values.shape[1]} cols "
                    f"but expected {expected_cols} from parent_cards={parent_cards} parents={parents}"
                )

        pmf: Dict[Tuple[int, ...], np.ndarray] = {}
        if not parents:
            probs = values[:, 0].copy()
            probs = probs / probs.sum()
            pmf[()] = probs
        else:
            for parent_vals in product(*[range(c) for c in parent_cards]):
                col = _mixed_radix_index(tuple(parent_vals), tuple(parent_cards))
                probs = values[:, col].copy()
                probs = probs / probs.sum()
                pmf[tuple(parent_vals)] = probs

        cpds_out[var] = {
            "parents": parents,
            "parent_cards": parent_cards,
            "var_card": var_card,
            "pmf": pmf,
        }

    return cpds_out


def ancestral_sample_discrete(
    n_samples: int,
    nodes: List[str],
    edges: List[Tuple[str, str]],
    cpds: Dict[str, Dict[str, Any]],
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    order = topological_sort(nodes, edges)

    rows = []
    for _ in range(n_samples):
        s: Dict[str, int] = {}
        for node in order:
            info = cpds[node]
            parents = info["parents"]
            pmf_map = info["pmf"]
            parent_vals = tuple(int(s[p]) for p in parents) if parents else ()
            probs = pmf_map[parent_vals]
            s[node] = int(rng.choice(len(probs), p=probs))
        rows.append(s)

    return pd.DataFrame(rows)


# ----------------------------
# Generated Gaussian / CLG models
# ----------------------------
def _stable_seed(seed: int, key: str) -> int:
    h = hashlib.sha256(key.encode("utf-8")).digest()
    return (int(seed) + int.from_bytes(h[:4], "little")) % (2**32)


def _random_cpt(
    cardinality: int, parent_cards: List[int], rng: np.random.Generator
) -> List[List[float]]:
    n_cols = int(np.prod(parent_cards)) if parent_cards else 1
    values = np.zeros((cardinality, n_cols), dtype=float)
    for col in range(n_cols):
        probs = rng.dirichlet(np.ones(cardinality))
        values[:, col] = probs
    return values.tolist()


def _generate_random_dag(
    nodes: List[str],
    n_edges: int,
    rng: np.random.Generator,
    allow_edge: Callable[[str, str], bool] | None = None,
) -> List[Tuple[str, str]]:
    if n_edges < 0:
        raise ValueError("n_edges must be >= 0")
    order = list(rng.permutation(nodes))
    possible: List[Tuple[str, str]] = []
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            parent, child = order[i], order[j]
            if allow_edge is not None and not allow_edge(parent, child):
                continue
            possible.append((parent, child))
    if n_edges > len(possible):
        raise ValueError(
            f"Cannot sample {n_edges} edges; only {len(possible)} allowed under constraints."
        )
    if n_edges == 0:
        return []
    idx = rng.choice(len(possible), size=n_edges, replace=False)
    edges = [possible[i] for i in idx]
    edges.sort(key=lambda e: (e[0], e[1]))
    return edges


def _choose_discrete_count(
    n_nodes: int, n_edges: int, target_ratio: float = 0.3
) -> int:
    if n_nodes <= 1:
        return 0
    target = int(round(n_nodes * target_ratio))
    target = max(1, min(n_nodes - 1, target))
    candidates: List[int] = []
    for d in range(1, n_nodes):
        c = n_nodes - d
        allowed = (n_nodes * (n_nodes - 1)) // 2 - c * d
        if allowed >= n_edges:
            candidates.append(d)
    if not candidates:
        return target
    return min(candidates, key=lambda d: (abs(d - target), d))


def _parents_by_node(
    nodes: List[str], edges: List[Tuple[str, str]]
) -> Dict[str, List[str]]:
    parents = {n: [] for n in nodes}
    for p, c in edges:
        parents[c].append(p)
    idx = {n: i for i, n in enumerate(nodes)}
    for n in nodes:
        parents[n] = sorted(parents[n], key=lambda x: idx[x])
    return parents


def generate_gaussian_model(
    name: str,
    n_nodes: int,
    n_edges: int,
    rng: np.random.Generator,
    nodes: List[str] | None = None,
    edges: List[Tuple[str, str]] | None = None,
) -> Dict[str, Any]:
    if nodes is None:
        nodes = [f"X{i + 1}" for i in range(n_nodes)]
    if edges is None:
        edges = _generate_random_dag(nodes, n_edges, rng)

    parents = _parents_by_node(nodes, edges)
    node_specs: Dict[str, Any] = {}
    for node in nodes:
        parents_cont = parents[node]
        coeffs = {p: float(rng.normal(0.0, 0.5)) for p in parents_cont}
        intercept = float(rng.normal(0.0, 1.0))
        sd = float(np.clip(rng.lognormal(mean=0.0, sigma=0.35), 0.2, 2.0))
        node_specs[node] = {
            "kind": "continuous",
            "parents_cont": parents_cont,
            "parents_disc": [],
            "linear_gaussian": {
                "intercept": intercept,
                "coeffs": coeffs,
                "sd": sd,
            },
        }

    return {
        "name": name,
        "type": "gaussian",
        "nodes": nodes,
        "edges": [[p, c] for p, c in edges],
        "node_specs": node_specs,
    }


def generate_clg_model(
    name: str,
    n_nodes: int,
    n_edges: int,
    rng: np.random.Generator,
    nodes: List[str] | None = None,
    edges: List[Tuple[str, str]] | None = None,
    node_kinds: Dict[str, str] | None = None,
    card_map: Dict[str, int] | None = None,
) -> Dict[str, Any]:
    if nodes is None:
        n_disc = _choose_discrete_count(n_nodes, n_edges)
        n_cont = n_nodes - n_disc
        disc_nodes = [f"D{i + 1}" for i in range(n_disc)]
        cont_nodes = [f"X{i + 1}" for i in range(n_cont)]
        nodes = disc_nodes + cont_nodes
    else:
        disc_nodes = []
        cont_nodes = []
        for n in nodes:
            kind = (node_kinds or {}).get(n, "continuous")
            if kind == "discrete":
                disc_nodes.append(n)
            else:
                cont_nodes.append(n)

    node_kinds = node_kinds or {
        n: ("discrete" if n in disc_nodes else "continuous") for n in nodes
    }
    card_map = card_map or {}

    for n in disc_nodes:
        if n not in card_map:
            card_map[n] = int(rng.integers(2, 5))

    def allow_edge(parent: str, child: str) -> bool:
        return not (
            node_kinds.get(child) == "discrete"
            and node_kinds.get(parent) == "continuous"
        )

    if edges is None:
        edges = _generate_random_dag(nodes, n_edges, rng, allow_edge=allow_edge)

    # Validate CLG constraint for provided edges
    for p, c in edges:
        if node_kinds.get(c) == "discrete" and node_kinds.get(p) == "continuous":
            raise ValueError(
                f"Invalid CLG edge: continuous parent {p} -> discrete child {c}"
            )

    parents = _parents_by_node(nodes, edges)
    node_specs: Dict[str, Any] = {}

    for node in nodes:
        kind = node_kinds.get(node, "continuous")
        if kind == "discrete":
            parent_list = parents[node]
            for p in parent_list:
                if node_kinds.get(p) != "discrete":
                    raise ValueError(
                        f"Discrete node {node} has non-discrete parent {p}"
                    )
            parent_cards = [card_map[p] for p in parent_list]
            values = _random_cpt(card_map[node], parent_cards, rng)
            node_specs[node] = {
                "kind": "discrete",
                "cardinality": int(card_map[node]),
                "cpt": {
                    "parents": parent_list,
                    "parent_cards": parent_cards,
                    "values": values,
                },
            }
        else:
            parent_list = parents[node]
            parents_disc = [p for p in parent_list if node_kinds.get(p) == "discrete"]
            parents_cont = [p for p in parent_list if node_kinds.get(p) != "discrete"]
            disc_cards = {p: int(card_map[p]) for p in parents_disc}
            disc_card_list = [disc_cards[p] for p in parents_disc]
            n_states = int(np.prod(disc_card_list)) if disc_card_list else 1
            params_by_disc_state: Dict[str, Any] = {}
            for idx in range(n_states):
                coeffs = {p: float(rng.normal(0.0, 0.5)) for p in parents_cont}
                intercept = float(rng.normal(0.0, 1.0))
                sd = float(np.clip(rng.lognormal(mean=0.0, sigma=0.35), 0.2, 2.0))
                params_by_disc_state[str(idx)] = {
                    "intercept": intercept,
                    "coeffs": coeffs,
                    "sd": sd,
                }

            node_specs[node] = {
                "kind": "continuous",
                "parents_cont": parents_cont,
                "parents_disc": parents_disc,
                "clg": {
                    "disc_parent_cards": disc_cards,
                    "params_by_disc_state": params_by_disc_state,
                },
            }

    return {
        "name": name,
        "type": "clgaussian",
        "nodes": nodes,
        "edges": [[p, c] for p, c in edges],
        "node_specs": node_specs,
    }


def save_model_json(model: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, indent=2))


def load_generated_model(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def ancestral_sample_generated(
    model: Dict[str, Any], n_samples: int, seed: int
) -> pd.DataFrame:
    nodes = list(model["nodes"])
    edges = [(p, c) for p, c in model["edges"]]
    node_specs = model["node_specs"]

    rng = np.random.default_rng(seed)
    order = topological_sort(nodes, edges)

    rows: List[Dict[str, Any]] = []
    for _ in range(n_samples):
        s: Dict[str, Any] = {}
        for node in order:
            spec = node_specs[node]
            kind = spec.get("kind")
            if kind == "discrete":
                cpt = spec["cpt"]
                parents = cpt.get("parents", [])
                parent_cards = cpt.get("parent_cards", [])
                values = np.asarray(cpt.get("values"), dtype=float)
                if not parents:
                    probs = values[:, 0]
                else:
                    parent_vals = tuple(int(s[p]) for p in parents)
                    idx = _mixed_radix_index(parent_vals, tuple(parent_cards))
                    probs = values[:, idx]
                probs = probs / probs.sum()
                s[node] = int(rng.choice(len(probs), p=probs))
            elif kind == "continuous":
                if "linear_gaussian" in spec:
                    lg = spec["linear_gaussian"]
                    coeffs = lg.get("coeffs", {})
                    intercept = float(lg.get("intercept", 0.0))
                    sd = float(lg.get("sd", 1.0))
                    mean = intercept + sum(
                        float(coeffs[p]) * float(s[p]) for p in coeffs
                    )
                    s[node] = float(rng.normal(mean, sd))
                elif "clg" in spec:
                    clg = spec["clg"]
                    parents_disc = spec.get("parents_disc", [])
                    parents_cont = spec.get("parents_cont", [])
                    disc_cards = clg.get("disc_parent_cards", {})
                    if parents_disc:
                        cards = [int(disc_cards[p]) for p in parents_disc]
                        parent_vals = tuple(int(s[p]) for p in parents_disc)
                        idx = _mixed_radix_index(parent_vals, tuple(cards))
                        key = str(idx)
                    else:
                        key = "0"
                    params = clg["params_by_disc_state"].get(key)
                    if params is None:
                        raise KeyError(
                            f"Missing CLG parameters for disc state {key} in node {node}"
                        )
                    coeffs = params.get("coeffs", {})
                    intercept = float(params.get("intercept", 0.0))
                    sd = float(params.get("sd", 1.0))
                    mean = intercept + sum(
                        float(coeffs[p]) * float(s[p]) for p in parents_cont
                    )
                    s[node] = float(rng.normal(mean, sd))
                else:
                    raise KeyError(
                        f"Continuous node {node} missing linear_gaussian/clg"
                    )
            else:
                raise KeyError(f"Node {node} missing or invalid kind")
        rows.append(s)

    return pd.DataFrame(rows)


# ----------------------------
# Spec loading
# ----------------------------
def load_spec(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Spec must be a mapping")

    name = data.get("name")
    typ = data.get("type")

    node_kinds: Dict[str, str] = {}
    card_map: Dict[str, int] = {}

    nodes: List[str] = []
    if "nodes" in data:
        for item in data["nodes"]:
            if isinstance(item, str):
                nodes.append(item)
            elif isinstance(item, dict):
                n = item.get("name")
                if not n:
                    raise ValueError("Spec node dict missing 'name'")
                nodes.append(n)
                kind = item.get("kind", "continuous")
                node_kinds[n] = kind
                if kind == "discrete":
                    card = int(item.get("cardinality", 2))
                    card_map[n] = card
            else:
                raise ValueError("Spec nodes must be strings or dicts")
    else:
        disc = data.get("discrete_nodes", [])
        cont = data.get("continuous_nodes", [])
        if disc or cont:
            nodes = list(disc) + list(cont)
            for n in disc:
                node_kinds[n] = "discrete"
                card_map[n] = int(data.get("cardinalities", {}).get(n, 2))
            for n in cont:
                node_kinds[n] = "continuous"

    edges = data.get("edges", [])
    if not isinstance(edges, list):
        raise ValueError("Spec edges must be a list")
    edge_list: List[Tuple[str, str]] = []
    for e in edges:
        if isinstance(e, (list, tuple)) and len(e) == 2:
            edge_list.append((str(e[0]), str(e[1])))
        else:
            raise ValueError("Spec edge must be a 2-item list")

    return {
        "name": name,
        "type": typ,
        "nodes": nodes,
        "edges": edge_list,
        "node_kinds": node_kinds,
        "card_map": card_map,
    }


# ----------------------------
# Registry helpers
# ----------------------------
def load_registry(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def resolve_urls(meta: Dict[str, Any]) -> Dict[str, str]:
    if "urls" in meta and isinstance(meta["urls"], dict):
        return meta["urls"]
    if "url" in meta and isinstance(meta["url"], str):
        return {"url": meta["url"]}
    return {}


def _suggest_generate_cmd(
    key: str, typ: str, stats: Dict[str, Any], model_json: Path
) -> str:
    nodes = stats.get("nodes")
    edges = stats.get("arcs")
    if nodes is None or edges is None:
        return f"python generate_bnlearn.py --name {key} --type {typ} --out {model_json.as_posix()}"
    return (
        f"python generate_bnlearn.py --name {key} --type {typ} "
        f"--nodes {int(nodes)} --edges {int(edges)} --out {model_json.as_posix()}"
    )


# ----------------------------
# Main
# ----------------------------
def _run_generation(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)

    if args.all:
        registry = load_registry(Path(args.registry))
        for key, meta in registry.items():
            typ = meta.get("type")
            if typ not in ("gaussian", "clgaussian"):
                continue
            stats = meta.get("stats", {})
            n_nodes = stats.get("nodes")
            n_edges = stats.get("arcs")
            if n_nodes is None or n_edges is None:
                print(f"[{key}] missing stats.nodes/arcs; skipping")
                continue
            rng = np.random.default_rng(_stable_seed(args.seed, key))
            if typ == "gaussian":
                model = generate_gaussian_model(
                    name=meta.get("name", key),
                    n_nodes=int(n_nodes),
                    n_edges=int(n_edges),
                    rng=rng,
                )
            else:
                model = generate_clg_model(
                    name=meta.get("name", key),
                    n_nodes=int(n_nodes),
                    n_edges=int(n_edges),
                    rng=rng,
                )

            model_path = outdir / key / "model.json"
            save_model_json(model, model_path)
            print(f"[{key}] wrote {model_path}")

            if args.save_samples:
                df = ancestral_sample_generated(model, args.n, args.seed)
                sample_csv = outdir / key / f"samples_n{args.n}_seed{args.seed}.csv"
                df.to_csv(sample_csv, index=False)
                print(f"[{key}] samples -> {sample_csv}")
        return

    if args.spec:
        spec = load_spec(Path(args.spec))
        name = args.name or spec.get("name")
        typ = args.type or spec.get("type")
        if not name or not typ:
            raise ValueError("Spec must define name/type, or pass --name/--type")
        nodes = spec.get("nodes") or []
        edges = spec.get("edges") or []
        node_kinds = spec.get("node_kinds") or {}
        card_map = spec.get("card_map") or {}
        rng = np.random.default_rng(_stable_seed(args.seed, name))
        if typ == "gaussian":
            model = generate_gaussian_model(
                name=name,
                n_nodes=len(nodes),
                n_edges=len(edges),
                rng=rng,
                nodes=nodes,
                edges=edges,
            )
        elif typ == "clgaussian":
            model = generate_clg_model(
                name=name,
                n_nodes=len(nodes),
                n_edges=len(edges),
                rng=rng,
                nodes=nodes,
                edges=edges,
                node_kinds=node_kinds,
                card_map=card_map,
            )
        else:
            raise ValueError(f"Unsupported type: {typ}")
    else:
        if not args.name or not args.type:
            raise ValueError("Manual generation requires --name and --type (or --spec)")
        if args.nodes is None or args.edges is None:
            raise ValueError(
                "Manual generation requires --nodes and --edges (or --spec)"
            )
        rng = np.random.default_rng(_stable_seed(args.seed, args.name))
        if args.type == "gaussian":
            model = generate_gaussian_model(
                name=args.name,
                n_nodes=int(args.nodes),
                n_edges=int(args.edges),
                rng=rng,
            )
        elif args.type == "clgaussian":
            model = generate_clg_model(
                name=args.name,
                n_nodes=int(args.nodes),
                n_edges=int(args.edges),
                rng=rng,
            )
        else:
            raise ValueError(f"Unsupported type: {args.type}")

    out_path: Path
    if args.out:
        out_path = Path(args.out)
        if out_path.suffix.lower() != ".json":
            out_path = out_path / "model.json"
    else:
        out_path = outdir / model["name"] / "model.json"

    save_model_json(model, out_path)
    print(f"wrote {out_path}")

    if args.save_samples:
        df = ancestral_sample_generated(model, args.n, args.seed)
        sample_csv = out_path.parent / f"samples_n{args.n}_seed{args.seed}.csv"
        df.to_csv(sample_csv, index=False)
        print(f"samples -> {sample_csv}")


def _run_registry_sampling(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    base_bif = outdir / "bif"
    base_samples = outdir / "samples"

    registry = load_registry(Path(args.registry))
    results: Dict[str, Any] = {}

    print(f"Registry: {Path(args.registry).resolve()}")
    print(f"Output dir: {outdir.resolve()}")
    print(f"n={args.n} seed={args.seed} save_samples={args.save_samples}")

    for key, meta in registry.items():
        meta = dict(meta)
        meta["key"] = key

        print(f"\n[{key}] {meta.get('name', key)} ({meta.get('category', 'n/a')})")
        try:
            typ = meta.get("type")
            if typ not in ("discrete", "gaussian", "clgaussian"):
                raise ValueError(
                    f"Unsupported or missing meta['type'] for {key}: {typ}"
                )

            if typ == "discrete":
                urls = resolve_urls(meta)
                url = urls.get("bif_gz") or urls.get("url")
                if not url:
                    raise KeyError("Missing urls.bif_gz for discrete dataset.")

                gz_path = base_bif / f"{key}.bif.gz"
                bif_path = base_bif / f"{key}.bif"

                download(url, gz_path, force=args.force_download)
                gunzip(gz_path, bif_path, force=args.force_download)

                model = load_bif_model(bif_path)

                stats = meta.get("stats", {})
                n_nodes = len(list(model.nodes()))
                n_edges = len(list(model.edges()))
                ok_nodes = (stats.get("nodes") is None) or (
                    int(stats["nodes"]) == n_nodes
                )
                ok_edges = (stats.get("arcs") is None) or (
                    int(stats["arcs"]) == n_edges
                )

                has_cpds = True
                if hasattr(model, "get_cpds"):
                    has_cpds = bool(model.get_cpds())

                sampled = False
                sample_csv = None
                if has_cpds and args.save_samples:
                    nodes = list(model.nodes())
                    edges = list(model.edges())
                    cpds = extract_cpds_from_model(model)
                    df = ancestral_sample_discrete(
                        args.n, nodes, edges, cpds, seed=args.seed
                    )

                    sample_csv = (
                        base_samples
                        / "discrete"
                        / f"{key}_n{args.n}_seed{args.seed}.csv"
                    )
                    sample_csv.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(sample_csv, index=False)
                    sampled = True

                results[key] = {
                    "available": True,
                    "type": "discrete",
                    "bif_path": str(bif_path),
                    "nodes_parsed": n_nodes,
                    "edges_parsed": n_edges,
                    "nodes_match_registry": ok_nodes,
                    "edges_match_registry": ok_edges,
                    "has_cpds": has_cpds,
                    "sampled": sampled,
                    "sample_csv": str(sample_csv) if sample_csv else None,
                }

                print(
                    f"  OK  discrete nodes={n_nodes} edges={n_edges} cpds={has_cpds} "
                    f"(nodes_match={ok_nodes}, edges_match={ok_edges}) sampled={sampled}"
                )

            else:
                if meta.get("source") != "generated":
                    raise ValueError(
                        "Gaussian/CLG datasets must be generated (source=generated)."
                    )
                gen = meta.get("generated", {})
                model_json = gen.get("model_json")
                if not model_json:
                    raise KeyError(
                        "Missing generated.model_json for gaussian/clgaussian."
                    )
                model_path = Path(model_json)
                if not model_path.exists():
                    cmd = _suggest_generate_cmd(
                        key=key,
                        typ=typ,
                        stats=meta.get("stats", {}),
                        model_json=model_path,
                    )
                    print(f"  missing generated model, run: {cmd}")
                    results[key] = {
                        "available": False,
                        "type": typ,
                        "error": "missing generated model",
                        "model_json": str(model_path),
                    }
                    continue

                model = load_generated_model(model_path)
                nodes = model.get("nodes", [])
                edges = model.get("edges", [])
                stats = meta.get("stats", {})
                ok_nodes = (stats.get("nodes") is None) or (
                    int(stats["nodes"]) == len(nodes)
                )
                ok_edges = (stats.get("arcs") is None) or (
                    int(stats["arcs"]) == len(edges)
                )

                sampled = False
                sample_csv = None
                if args.save_samples:
                    df = ancestral_sample_generated(model, args.n, args.seed)
                    sample_csv = (
                        base_samples / typ / f"{key}_n{args.n}_seed{args.seed}.csv"
                    )
                    sample_csv.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(sample_csv, index=False)
                    sampled = True

                results[key] = {
                    "available": True,
                    "type": typ,
                    "model_json": str(model_path),
                    "nodes_parsed": len(nodes),
                    "edges_parsed": len(edges),
                    "nodes_match_registry": ok_nodes,
                    "edges_match_registry": ok_edges,
                    "sampled": sampled,
                    "sample_csv": str(sample_csv) if sample_csv else None,
                }

                print(
                    f"  OK  {typ} nodes={len(nodes)} edges={len(edges)} "
                    f"(nodes_match={ok_nodes}, edges_match={ok_edges}) sampled={sampled}"
                )

        except Exception as e:
            results[key] = {
                "available": False,
                "error": f"{type(e).__name__}: {e}",
            }
            print(f"  NO  {type(e).__name__}: {e}")

    ok = [k for k, v in results.items() if v.get("available")]
    no = [k for k, v in results.items() if not v.get("available")]

    print("\n=== Summary ===")
    print(f"Available [{len(ok)}/{len(results)}]: {ok}")
    if no:
        print(f"Not available [{len(no)}/{len(results)}]: {no}")

    out = outdir / "availability_from_registry.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote -> {out.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry", type=str, default="benchmarking/bnlearn_data/metadata.json"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="benchmarking/bnlearn_data",
        help="Base output dir. In generation mode, this is the generated models dir.",
    )
    parser.add_argument("--n", type=int, default=1000, help="samples per dataset")
    parser.add_argument(
        "--seed", type=int, default=42, help="seed for sampling and generation"
    )
    parser.add_argument(
        "--save_samples", action="store_true", help="write sampled CSVs to disk"
    )
    parser.add_argument(
        "--force_download", action="store_true", help="redownload even if file exists"
    )

    # Generation flags
    parser.add_argument(
        "--all", action="store_true", help="generate all gaussian/clg from registry"
    )
    parser.add_argument(
        "--name", type=str, default=None, help="dataset key/name for generation"
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["gaussian", "clgaussian"],
        default=None,
        help="model type for generation",
    )
    parser.add_argument("--nodes", type=int, default=None, help="number of nodes")
    parser.add_argument("--edges", type=int, default=None, help="number of edges")
    parser.add_argument(
        "--spec", type=str, default=None, help="YAML spec for generation"
    )
    parser.add_argument(
        "--out", type=str, default=None, help="output path for model.json"
    )
    parser.add_argument(
        "--format", type=str, default="json", help="model format (json only)"
    )

    args = parser.parse_args()

    if args.format.lower() != "json":
        raise ValueError("Only --format json is supported")

    generation_mode = any(
        [
            args.all,
            args.name,
            args.spec,
            args.type,
            args.nodes is not None,
            args.edges is not None,
        ]
    )

    if generation_mode:
        _run_generation(args)
    else:
        _run_registry_sampling(args)


if __name__ == "__main__":
    main()
