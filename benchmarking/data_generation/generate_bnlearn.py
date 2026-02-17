from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import urllib.request
from collections import defaultdict, deque
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .base import BaseDataGenerator
from .registry import register_generator

# ----------------------------
# IO helpers
# ----------------------------


def download(url: str, dst: Path, force: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not force and dst.exists() and dst.stat().st_size > 0:
        return
    with urllib.request.urlopen(url) as r, open(dst, "wb") as f:
        shutil.copyfileobj(r, f)


def gunzip(src_gz: Path, dst_bif: Path, force: bool = False) -> None:
    dst_bif.parent.mkdir(parents=True, exist_ok=True)
    if not force and dst_bif.exists() and dst_bif.stat().st_size > 0:
        return
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
            import bnlearn as bn  # type: ignore

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
                "Failed to parse BIF via pgmpy and bnlearn.\n"
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
# Metadata helpers
# ----------------------------


def _stable_seed(seed: int, key: str) -> int:
    h = hashlib.sha256(key.encode("utf-8")).digest()
    return (int(seed) + int.from_bytes(h[:4], "little")) % (2**32)


def _get_nodes_edges(model) -> Tuple[List[str], List[Tuple[str, str]]]:
    nodes = (
        list(model.nodes())
        if callable(getattr(model, "nodes", None))
        else list(model.nodes)
    )
    edges = (
        list(model.edges())
        if callable(getattr(model, "edges", None))
        else list(model.edges)
    )
    return nodes, edges


def _extract_state_names(model) -> Dict[str, List[str]]:
    state_names: Dict[str, List[str]] = {}
    if hasattr(model, "get_cpds"):
        for cpd in model.get_cpds() or []:
            names = None
            if hasattr(cpd, "state_names") and cpd.state_names:
                names = cpd.state_names.get(cpd.variable)
            if names:
                state_names[cpd.variable] = [str(n) for n in names]
    return state_names


def _build_encoding_map(
    nodes: List[str],
    state_names: Dict[str, List[str]],
    cardinalities: Dict[str, int],
) -> Dict[str, Dict[str, float]]:
    encoding: Dict[str, Dict[str, float]] = {}
    for node in nodes:
        names = state_names.get(node)
        card = int(cardinalities[node])
        if not names or len(names) != card:
            names = [str(i) for i in range(card)]
        encoding[node] = {name: float(i) for i, name in enumerate(names)}
    return encoding


@register_generator
class BNLearnGenerator(BaseDataGenerator):
    name = "bnlearn"
    test_networks = ["asia", "cancer"]

    def generate(
        self,
        n_samples: int,
        networks: list[str] | None = None,
        force: bool = False,
        **kwargs: Any,
    ) -> None:
        if int(n_samples) <= 0:
            raise ValueError("n_samples must be a positive integer")

        bnlearn_root = self.root_path / "benchmarking" / "bnlearn_data"
        registry_path = bnlearn_root / "metadata.json"
        if not registry_path.exists():
            raise FileNotFoundError(f"Missing registry: {registry_path}")

        registry = json.loads(registry_path.read_text())
        available_networks = [
            key for key, meta in registry.items() if meta.get("type") == "discrete"
        ]

        if networks:
            networks = list(dict.fromkeys(networks))
            missing = sorted(set(networks) - set(registry))
            if missing:
                raise KeyError(f"Unknown network(s): {missing}")
            non_discrete = [n for n in networks if n not in available_networks]
            if non_discrete:
                raise ValueError(
                    "Requested networks are not discrete: "
                    + ", ".join(sorted(non_discrete))
                )
            selected = list(networks)
        else:
            selected = sorted(available_networks)

        existing_metadata: Dict[str, Any] = {}
        metadata_path = self.dataset_path / "metadata.json"
        if metadata_path.exists():
            try:
                existing_metadata = json.loads(metadata_path.read_text())
            except Exception:
                existing_metadata = {}

        existing_variables = existing_metadata.get("variables", {})
        existing_encoding = existing_metadata.get("encoding", {})
        existing_cardinalities = existing_metadata.get("cardinalities", {})

        variables: Dict[str, list[str]] = {}
        encoding: Dict[str, Dict[str, Dict[str, float]]] = {}
        cardinalities_all: Dict[str, Dict[str, int]] = {}

        for network in self.progress(selected, desc="Generating bnlearn"):
            data_path = self.dataset_file(network)
            rows, cols = (
                self.inspect_parquet(data_path) if data_path.exists() else (None, None)
            )
            up_to_date = self.should_skip(network, n_samples, force)

            if not force and up_to_date:
                vars_existing = existing_variables.get(network)
                enc_existing = existing_encoding.get(network)
                cards_existing = existing_cardinalities.get(network)
                if vars_existing and enc_existing:
                    variables[network] = vars_existing
                    encoding[network] = enc_existing
                    if cards_existing:
                        cardinalities_all[network] = cards_existing
                    continue

            meta = registry.get(network, {})
            url = meta.get("urls", {}).get("bif_gz")
            if not url:
                raise ValueError(f"No BIF URL for network '{network}'")

            bif_dir = bnlearn_root / "bif"
            gz_path = bif_dir / f"{network}.bif.gz"
            bif_path = bif_dir / f"{network}.bif"

            download(url, gz_path, force=force)
            gunzip(gz_path, bif_path, force=force)

            model = load_bif_model(bif_path)
            nodes, edges = _get_nodes_edges(model)
            state_names = _extract_state_names(model)
            cardinalities = {n: int(model.get_cardinality(n)) for n in nodes}
            encoding_map = _build_encoding_map(nodes, state_names, cardinalities)

            if force or not up_to_date:
                cpds = extract_cpds_from_model(model)
                seed_samples = _stable_seed(self.seed, f"{network}-samples")
                df = ancestral_sample_discrete(
                    int(n_samples), nodes, edges, cpds, seed=seed_samples
                )
                df = df.astype(float)
                df.to_parquet(data_path, index=False)
                variables[network] = list(df.columns)
            else:
                variables[network] = cols or nodes

            encoding[network] = encoding_map
            cardinalities_all[network] = cardinalities

        metadata = self.build_metadata(
            n_samples=int(n_samples),
            networks=selected,
            variables=variables,
            encoding=encoding,
            cardinalities=cardinalities_all,
        )
        self.save_metadata(metadata)
