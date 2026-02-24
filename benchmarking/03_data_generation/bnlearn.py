from __future__ import annotations

import gzip
import json
import logging
import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from benchmarking.paths import (
    ensure_dir,
    get_dataset_domain_metadata_path,
    get_dataset_download_metadata_path,
)
from .base import BaseDataGenerator, DataGenResult, save_dataframe
from .registry import register_data_generator


@dataclass(frozen=True)
class CPDEntries:
    entries: Dict[Tuple[str, ...], List[float]]
    default: List[float] | None


@dataclass(frozen=True)
class NodeCPD:
    node: str
    parents: List[str]
    parent_states: List[List[str]]
    target_states: List[str]
    table: np.ndarray
    multipliers: np.ndarray


def _clean_token(token: str) -> str:
    token = token.strip()
    if token.startswith('"') and token.endswith('"'):
        token = token[1:-1]
    return token.strip()


def _strip_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        out: list[str] = []
        in_quote: str | None = None
        i = 0
        while i < len(line):
            ch = line[i]
            if in_quote:
                out.append(ch)
                if ch == "\\" and i + 1 < len(line):
                    i += 1
                    out.append(line[i])
                elif ch == in_quote:
                    in_quote = None
                i += 1
                continue
            if ch in {'"', "'"}:
                in_quote = ch
                out.append(ch)
                i += 1
                continue
            if ch == "/" and i + 1 < len(line) and line[i + 1] == "/":
                break
            if ch == "#":
                break
            out.append(ch)
            i += 1
        lines.append("".join(out))
    return "\n".join(lines)


def _tokenize_values(raw: str) -> List[str]:
    cleaned = raw.strip().strip("{}").strip(";")
    tokens = re.findall(r'"[^"]*"|\'[^\']*\'|[^,\s|{};]+', cleaned)
    return [_clean_token(t) for t in tokens if _clean_token(t)]


def _parse_states(raw: str) -> List[str]:
    return _tokenize_values(raw)


def _parse_floats(raw: str) -> List[float]:
    cleaned = raw.replace("(", " ").replace(")", " ").replace(";", " ")
    tokens = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cleaned)
    return [float(t) for t in tokens]


def _parse_parent_tuple(raw: str) -> Tuple[str, ...]:
    parts = _tokenize_values(raw)
    return tuple(parts)


def _read_bif_text(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return f.read()
    return path.read_text()


def _find_matching_brace(text: str, start: int) -> int:
    depth = 0
    in_quote: str | None = None
    i = start
    while i < len(text):
        ch = text[i]
        if in_quote:
            if ch == "\\" and i + 1 < len(text):
                i += 2
                continue
            if ch == in_quote:
                in_quote = None
            i += 1
            continue
        if ch in {'"', "'"}:
            in_quote = ch
            i += 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError("Unmatched '{' in BIF content")


def _iter_blocks(
    text: str, header_re: re.Pattern[str]
) -> Iterable[Tuple[re.Match[str], str]]:
    for match in header_re.finditer(text):
        brace_idx = match.end() - 1
        if brace_idx < 0 or brace_idx >= len(text) or text[brace_idx] != "{":
            brace_idx = text.find("{", match.end())
        if brace_idx == -1:
            continue
        end = _find_matching_brace(text, brace_idx)
        body = text[brace_idx + 1 : end]
        yield match, body


def _extract_discrete_states(body: str) -> List[str] | None:
    match = re.search(r"type\s+discrete", body, re.IGNORECASE)
    if not match:
        return None
    brace_idx = body.find("{", match.end())
    if brace_idx == -1:
        return None
    end = _find_matching_brace(body, brace_idx)
    return _parse_states(body[brace_idx + 1 : end])


def _parse_bif(
    path: Path,
) -> tuple[
    List[str],
    Dict[str, List[str]],
    Dict[str, str],
    Dict[str, str],
    Dict[str, List[str]],
    Dict[str, CPDEntries],
]:
    text = _strip_comments(_read_bif_text(path))

    var_re = re.compile(
        r"variable\s+(?P<name>\"[^\"]+\"|[^\s{]+)\s*\{",
        re.IGNORECASE,
    )
    prob_re = re.compile(
        r"probability\s*\(\s*(?P<header>[^\)]*)\)\s*\{",
        re.IGNORECASE,
    )
    continuous_re = re.compile(r"type\s+continuous", re.IGNORECASE)

    nodes: List[str] = []
    node_set = set()
    node_states: Dict[str, List[str]] = {}
    node_types: Dict[str, str] = {}
    node_sources: Dict[str, str] = {}

    for m, body in _iter_blocks(text, var_re):
        name = _clean_token(m.group("name"))
        if not name:
            continue
        if name not in node_set:
            nodes.append(name)
            node_set.add(name)
        node_sources[name] = "bif"
        states = _extract_discrete_states(body)
        if states is not None:
            node_types[name] = "discrete"
            node_states[name] = states
            continue
        if continuous_re.search(body):
            node_types[name] = "continuous"
            continue
        node_types.setdefault(name, "discrete")

    parents_map: Dict[str, List[str]] = {}
    cpds: Dict[str, CPDEntries] = {}

    for m, body in _iter_blocks(text, prob_re):
        header = m.group("header")
        if "|" in header:
            target_raw, parents_raw = header.split("|", 1)
            parents = _tokenize_values(parents_raw)
        else:
            target_raw = header
            parents = []
        target_tokens = _tokenize_values(target_raw)
        target = target_tokens[0] if target_tokens else _clean_token(target_raw)
        if not target:
            continue
        if target not in node_set:
            nodes.append(target)
            node_set.add(target)
        for parent in parents:
            if parent not in node_set:
                nodes.append(parent)
                node_set.add(parent)
        parents_map[target] = parents

        default_probs = None
        default_match = re.search(
            r"default\s+([^;]+);", body, flags=re.IGNORECASE | re.DOTALL
        )
        if default_match:
            default_probs = _parse_floats(default_match.group(1))

        entries: Dict[Tuple[str, ...], List[float]] = {}
        for entry in re.finditer(
            r"\(\s*([^\)]+)\s*\)\s*([^;]+);",
            body,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            parents_tuple = _parse_parent_tuple(entry.group(1))
            probs = _parse_floats(entry.group(2))
            entries[parents_tuple] = probs

        if not entries:
            table_match = re.search(
                r"table\s+([^;]+);", body, flags=re.IGNORECASE | re.DOTALL
            )
            if table_match:
                entries[tuple()] = _parse_floats(table_match.group(1))

        cpds[target] = CPDEntries(entries=entries, default=default_probs)

    for node in nodes:
        parents_map.setdefault(node, [])
        node_types.setdefault(node, "discrete")
        node_sources.setdefault(node, "probability")

    return nodes, node_states, node_types, node_sources, parents_map, cpds


def _infer_missing_states(
    nodes: Iterable[str],
    node_states: Dict[str, List[str]],
    node_types: Dict[str, str],
    node_sources: Dict[str, str],
    cpds: Dict[str, CPDEntries],
    parents_map: Dict[str, List[str]],
) -> tuple[List[str], List[str]]:
    inferred: List[str] = []
    missing: List[str] = []
    for node in nodes:
        if node_types.get(node, "discrete") != "discrete":
            continue
        if node_states.get(node):
            continue
        entries = cpds.get(node)
        probs: List[float] | None = None
        if entries is not None:
            if entries.entries:
                probs = next(iter(entries.entries.values()))
            elif entries.default:
                probs = entries.default
        k_from_cpd = len(probs) if probs else None

        parent_labels: List[str] = []
        seen: set[str] = set()
        for target, target_entries in cpds.items():
            parents = parents_map.get(target, [])
            if node not in parents:
                continue
            idx = parents.index(node)
            for parent_tuple in target_entries.entries.keys():
                if idx >= len(parent_tuple):
                    continue
                label = parent_tuple[idx]
                if label not in seen:
                    seen.add(label)
                    parent_labels.append(label)

        if parent_labels and (k_from_cpd is None or len(parent_labels) == k_from_cpd):
            node_states[node] = parent_labels
            node_sources[node] = "inferred_from_cpd"
            inferred.append(node)
            continue

        if k_from_cpd is not None and k_from_cpd > 0:
            node_states[node] = [f"s{i}" for i in range(k_from_cpd)]
            node_sources[node] = "inferred_from_cpd"
            inferred.append(node)
            continue

        if parent_labels:
            node_states[node] = parent_labels
            node_sources[node] = "inferred_from_cpd"
            inferred.append(node)
            continue
        missing.append(node)
    return inferred, missing


def _topological_order(
    nodes: Iterable[str], parents: Dict[str, List[str]]
) -> List[str]:
    nodes = list(nodes)
    indegree: Dict[str, int] = {node: 0 for node in nodes}
    children: Dict[str, List[str]] = {node: [] for node in nodes}
    for node in nodes:
        for parent in parents.get(node, []):
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
        raise ValueError("Cycle detected in network; cannot topologically sort")
    return order


def _normalize_probs(probs: List[float], expected: int, node: str) -> np.ndarray:
    if len(probs) != expected:
        raise ValueError(
            f"CPD for {node} has {len(probs)} entries; expected {expected}"
        )
    arr = np.asarray(probs, dtype=float)
    total = float(arr.sum())
    if total <= 0:
        raise ValueError(f"CPD for {node} has non-positive total probability")
    return arr / total


def _build_node_cpd(
    node: str,
    parents: List[str],
    entries: CPDEntries,
    node_states: Dict[str, List[str]],
) -> NodeCPD:
    target_states = node_states.get(node, [])
    if not target_states:
        raise ValueError(f"Missing states for node '{node}'")

    if not parents:
        probs = entries.entries.get(tuple())
        if probs is None:
            raise ValueError(f"Missing CPD table for node '{node}'")
        table = _normalize_probs(probs, len(target_states), node).reshape(1, -1)
        return NodeCPD(
            node=node,
            parents=[],
            parent_states=[],
            target_states=target_states,
            table=table,
            multipliers=np.array([], dtype=int),
        )

    parent_states = [node_states.get(parent, []) for parent in parents]
    for parent, states in zip(parents, parent_states):
        if not states:
            raise ValueError(f"Missing states for parent '{parent}'")

    combos = list(product(*parent_states))
    total_rows = len(combos)
    table = np.zeros((total_rows, len(target_states)), dtype=float)

    default_probs = entries.default
    for idx, combo in enumerate(combos):
        probs = entries.entries.get(tuple(combo))
        if probs is None:
            if default_probs is None:
                raise ValueError(f"Missing CPD entry for {node} with parents {combo}")
            probs = default_probs
        table[idx] = _normalize_probs(probs, len(target_states), node)

    multipliers = []
    prod = 1
    for size in reversed([len(states) for states in parent_states]):
        multipliers.append(prod)
        prod *= size
    multipliers = list(reversed(multipliers))

    return NodeCPD(
        node=node,
        parents=parents,
        parent_states=parent_states,
        target_states=target_states,
        table=table,
        multipliers=np.asarray(multipliers, dtype=int),
    )


def _build_domain(
    *,
    dataset_id: str,
    generator: str,
    dataset_type: str | None,
    nodes: List[str],
    node_states: Dict[str, List[str]],
    node_types: Dict[str, str],
    node_sources: Dict[str, str],
) -> tuple[dict, bool, str | None]:
    domain_nodes: Dict[str, dict] = {}
    missing_states: List[str] = []
    for node in sorted(nodes):
        ntype = node_types.get(node)
        if ntype is None:
            ntype = "discrete" if dataset_type == "discrete" else "continuous"
        if ntype == "discrete":
            states = list(node_states.get(node, []))
            if not states:
                missing_states.append(node)
            index_to_code = [idx for idx, _ in enumerate(states)]
            domain_nodes[node] = {
                "type": "discrete",
                "states": states,
                "codes": {state: idx for idx, state in enumerate(states)},
                "index_to_code": index_to_code,
                "code_to_state": {idx: state for idx, state in enumerate(states)},
                "source": node_sources.get(node, "metadata"),
            }
        else:
            domain_nodes[node] = {
                "type": "continuous",
                "dtype": None,
                "range": None,
                "source": node_sources.get(node, "metadata"),
            }

    unsupported = False
    reason = None
    if missing_states:
        unsupported = True
        reason = f"Missing discrete states for nodes: {sorted(missing_states)}"

    domain = {
        "dataset_id": dataset_id,
        "generator": generator,
        "type": dataset_type,
        "nodes": domain_nodes,
        "unsupported": unsupported,
        "reason": reason,
    }

    return domain, unsupported, reason


def _domain_path(root: Path, generator: str, dataset_id: str) -> Path:
    return get_dataset_domain_metadata_path(root, generator, dataset_id)


def _load_or_create_domain(
    *,
    root_path: Path,
    generator: str,
    dataset_id: str,
    dataset_type: str | None,
    nodes: List[str],
    node_states: Dict[str, List[str]],
    node_types: Dict[str, str],
    node_sources: Dict[str, str],
    logger: logging.Logger,
) -> tuple[Path, dict]:
    path = _domain_path(root_path, generator, dataset_id)
    if path.exists():
        domain = json.loads(path.read_text())
        return path, domain

    domain, unsupported, reason = _build_domain(
        dataset_id=dataset_id,
        generator=generator,
        dataset_type=dataset_type,
        nodes=nodes,
        node_states=node_states,
        node_types=node_types,
        node_sources=node_sources,
    )
    ensure_dir(path.parent)
    path.write_text(json.dumps(domain, indent=2, sort_keys=True))
    if unsupported:
        logger.warning("Domain marked unsupported: %s", reason)
    return path, domain


def _schema_from_domain(
    columns: List[str],
    domain: dict,
    dtypes: Dict[str, str],
) -> dict:
    variables: Dict[str, dict] = {}
    nodes = domain.get("nodes", {}) if isinstance(domain, dict) else {}
    for col in columns:
        meta = nodes.get(col, {})
        dtype = dtypes.get(col)
        entry = {
            "type": meta.get("type"),
            "dtype": dtype,
        }
        if meta.get("type") == "discrete":
            states = meta.get("states") or []
            codes = meta.get("codes") or {}
            index_to_code = [codes.get(state, idx) for idx, state in enumerate(states)]
            code_to_state = {int(code): state for state, code in codes.items()}
            entry["domain"] = states
            entry["codes"] = codes
            entry["index_to_code"] = index_to_code
            entry["code_to_state"] = code_to_state
        variables[col] = entry
    return {
        "columns": list(columns),
        "dtypes": dict(dtypes),
        "variables": variables,
    }


@register_data_generator
class BNLearnDataGenerator(BaseDataGenerator):
    name = "bnlearn"

    def generate(
        self,
        dataset_id: str,
        dataset_dir: Path,
        out_dir: Path,
        meta_dir: Path,
        logger: logging.Logger,
    ) -> DataGenResult | None:
        download_meta_path = get_dataset_download_metadata_path(
            self.root_path, self.name, dataset_id
        )
        download_meta = None
        if download_meta_path.exists():
            download_meta = json.loads(download_meta_path.read_text())

        dataset_type = None
        if isinstance(download_meta, dict):
            dataset_type = download_meta.get("type")
        if dataset_type is None:
            dataset_type = "discrete"
        approx_domain = False
        approx_domain_reason = None

        bif_path = None
        if isinstance(download_meta, dict):
            downloaded = download_meta.get("downloaded_files") or {}
            if isinstance(downloaded, dict) and downloaded.get("bif"):
                candidate = dataset_dir / downloaded["bif"]
                if candidate.exists():
                    bif_path = candidate

        if bif_path is None:
            candidate = dataset_dir / "model.bif"
            if candidate.exists():
                bif_path = candidate

        if bif_path is None:
            gz_candidate = dataset_dir / "model.bif.gz"
            if gz_candidate.exists():
                bif_path = gz_candidate

        if bif_path is None:
            reason = "Missing BIF file; cannot sample BN"
            if isinstance(download_meta, dict) and download_meta.get("reason"):
                reason = download_meta.get("reason")
            logger.warning("Skipping %s: %s", dataset_id, reason)
            return DataGenResult(
                data_path=None,
                format=None,
                schema=None,
                capabilities={
                    "can_generate_data": False,
                    "uses_bn_sampling": False,
                    "supports_discrete": True,
                    "supports_continuous": False,
                },
                notes={
                    "approx_on_manifold": None,
                    "approx_reason": reason,
                    "approx_domain": approx_domain,
                    "approx_domain_reason": approx_domain_reason,
                },
                skipped=True,
                reason=reason,
            )

        if self.generation_strategy != "default":
            reason = f"Unsupported generation strategy '{self.generation_strategy}'"
            logger.warning("Skipping %s: %s", dataset_id, reason)
            return DataGenResult(
                data_path=None,
                format=None,
                schema=None,
                capabilities={
                    "can_generate_data": False,
                    "uses_bn_sampling": False,
                    "supports_discrete": True,
                    "supports_continuous": False,
                },
                notes={
                    "approx_on_manifold": None,
                    "approx_reason": reason,
                    "approx_domain": approx_domain,
                    "approx_domain_reason": approx_domain_reason,
                },
                skipped=True,
                reason=reason,
            )

        nodes, node_states, node_types, node_sources, parents_map, cpds = _parse_bif(
            bif_path
        )

        if any(ntype == "continuous" for ntype in node_types.values()):
            reason = "Continuous variables in BIF are not supported yet"
            logger.warning("Skipping %s: %s", dataset_id, reason)
            return DataGenResult(
                data_path=None,
                format=None,
                schema=None,
                capabilities={
                    "can_generate_data": False,
                    "uses_bn_sampling": False,
                    "supports_discrete": True,
                    "supports_continuous": False,
                },
                notes={
                    "approx_on_manifold": None,
                    "approx_reason": reason,
                    "approx_domain": approx_domain,
                    "approx_domain_reason": approx_domain_reason,
                },
                skipped=True,
                reason=reason,
            )

        for node in nodes:
            if node not in cpds:
                reason = f"Missing CPD for node '{node}'"
                logger.warning("Skipping %s: %s", dataset_id, reason)
                return DataGenResult(
                    data_path=None,
                    format=None,
                    schema=None,
                    capabilities={
                        "can_generate_data": False,
                        "uses_bn_sampling": False,
                        "supports_discrete": True,
                        "supports_continuous": False,
                    },
                    notes={
                        "approx_on_manifold": None,
                        "approx_reason": reason,
                        "approx_domain": approx_domain,
                        "approx_domain_reason": approx_domain_reason,
                    },
                    skipped=True,
                    reason=reason,
                )

        inferred_nodes, missing_nodes = _infer_missing_states(
            nodes, node_states, node_types, node_sources, cpds, parents_map
        )
        if inferred_nodes:
            approx_domain = True
            approx_domain_reason = (
                "Inferred discrete states from CPD row lengths for: "
                + ", ".join(sorted(inferred_nodes))
            )
            logger.warning("%s", approx_domain_reason)
        if missing_nodes:
            reason = "Missing discrete states for nodes: " + ", ".join(
                sorted(missing_nodes)
            )
            logger.warning("Skipping %s: %s", dataset_id, reason)
            return DataGenResult(
                data_path=None,
                format=None,
                schema=None,
                capabilities={
                    "can_generate_data": False,
                    "uses_bn_sampling": False,
                    "supports_discrete": True,
                    "supports_continuous": False,
                },
                notes={
                    "approx_on_manifold": None,
                    "approx_reason": reason,
                    "approx_domain": approx_domain,
                    "approx_domain_reason": approx_domain_reason,
                },
                skipped=True,
                reason=reason,
            )

        domain_path, domain = _load_or_create_domain(
            root_path=self.root_path,
            generator=self.name,
            dataset_id=dataset_id,
            dataset_type=dataset_type,
            nodes=nodes,
            node_states=node_states,
            node_types=node_types,
            node_sources=node_sources,
            logger=logger,
        )

        if isinstance(domain, dict) and domain.get("unsupported"):
            reason = domain.get("reason") or "Domain metadata marked unsupported"
            logger.warning("Skipping %s: %s", dataset_id, reason)
            return DataGenResult(
                data_path=None,
                format=None,
                schema=None,
                capabilities={
                    "can_generate_data": False,
                    "uses_bn_sampling": False,
                    "supports_discrete": True,
                    "supports_continuous": False,
                },
                notes={
                    "approx_on_manifold": None,
                    "approx_reason": reason,
                    "approx_domain": approx_domain,
                    "approx_domain_reason": approx_domain_reason,
                },
                skipped=True,
                reason=reason,
                domain_path=domain_path,
            )

        topo_order = _topological_order(nodes, parents_map)

        node_cpds: Dict[str, NodeCPD] = {}
        for node in topo_order:
            node_cpds[node] = _build_node_cpd(
                node, parents_map.get(node, []), cpds[node], node_states
            )

        dataset_seed = self._stable_seed(dataset_id)
        rng = np.random.default_rng(dataset_seed)

        samples: Dict[str, np.ndarray] = {}
        for node in topo_order:
            cpd = node_cpds[node]
            k = len(cpd.target_states)
            if not cpd.parents:
                draws = rng.choice(k, size=self.n_samples, p=cpd.table[0])
                samples[node] = draws.astype(np.int32)
                continue
            parent_arrays = [samples[parent] for parent in cpd.parents]
            parent_index = np.zeros(self.n_samples, dtype=np.int64)
            for arr, mult in zip(parent_arrays, cpd.multipliers):
                parent_index += arr.astype(np.int64) * int(mult)
            out = np.empty(self.n_samples, dtype=np.int32)
            for idx in np.unique(parent_index):
                mask = parent_index == idx
                out[mask] = rng.choice(k, size=int(mask.sum()), p=cpd.table[int(idx)])
            samples[node] = out

        columns = list(nodes)
        data: Dict[str, np.ndarray] = {}
        domain_nodes = domain.get("nodes", {}) if isinstance(domain, dict) else {}
        for col in columns:
            values = samples[col]
            meta = domain_nodes.get(col, {})
            if meta.get("type") == "discrete":
                states = meta.get("states") or []
                codes = meta.get("codes") or {}
                if states:
                    index_to_code = [codes.get(s, i) for i, s in enumerate(states)]
                    if any(code != idx for idx, code in enumerate(index_to_code)):
                        values = np.asarray(index_to_code, dtype=np.int64)[values]
                values = values.astype(np.int32)
            data[col] = values

        df = pd.DataFrame(data, columns=columns)

        path_prefix = out_dir / (
            f"data_{self.generation_strategy}_n{self.n_samples}_seed{self.seed}"
        )
        data_path, fmt = save_dataframe(df, path_prefix, prefer="parquet")

        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        schema = _schema_from_domain(columns, domain, dtypes)

        return DataGenResult(
            data_path=data_path,
            format=fmt,
            schema=schema,
            capabilities={
                "can_generate_data": True,
                "uses_bn_sampling": True,
                "supports_discrete": True,
                "supports_continuous": False,
            },
            notes={
                "approx_on_manifold": False,
                "approx_reason": None,
                "approx_domain": approx_domain,
                "approx_domain_reason": approx_domain_reason,
            },
            domain_path=domain_path,
            skipped=False,
            reason=None,
        )
