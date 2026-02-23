from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

from benchmarking.paths import (
    ensure_dir,
    get_dataset_domain_metadata_path,
    get_dataset_download_metadata_path,
    get_static_metadata_dir,
)
from .base import BaseQueryGenerator
from .registry import register_query_generator


@dataclass(frozen=True)
class EvidenceSpec:
    mode: str
    vars: List[str]
    values: dict | None

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "vars": list(self.vars),
            "values": self.values,
        }


@dataclass(frozen=True)
class CPDQuery:
    target: str
    target_category: str
    evidence_strategy: str
    evidence_vars: List[str]
    seed: int
    dataset_id: str

    def to_dict(self) -> dict:
        evidence_vars = list(self.evidence_vars)
        return {
            "query_type": "cpd",
            "target": self.target,
            "target_category": self.target_category,
            "evidence_strategy": self.evidence_strategy,
            "evidence_vars": evidence_vars,
            "evidence_values": None,
            "evidence": {"vars": evidence_vars, "values": None},
            "seed": int(self.seed),
            "dataset_id": self.dataset_id,
        }


@dataclass(frozen=True)
class InferenceQuery:
    task: str
    target: str
    target_category: str
    evidence_strategy: str
    effective_evidence_strategy: str
    evidence_mode: str
    evidence_vars: List[str]
    evidence: EvidenceSpec
    skeleton_id: str
    mc_id: int | None
    approx_on_manifold: bool
    seed: int
    dataset_id: str
    generator: str
    generator_kwargs: dict

    def to_dict(self) -> dict:
        evidence_vars = list(self.evidence_vars)
        payload = {
            "query_type": "inference",
            "task": self.task,
            "target": self.target,
            "target_category": self.target_category,
            "evidence_strategy": self.evidence_strategy,
            "evidence_mode": self.evidence_mode,
            "evidence_vars": evidence_vars,
            "evidence_values": self.evidence.values,
            "evidence": self.evidence.to_dict(),
            "skeleton_id": self.skeleton_id,
            "mc_id": self.mc_id,
            "seed": int(self.seed),
            "dataset_id": self.dataset_id,
            "generator": self.generator,
            "generator_kwargs": dict(sorted(self.generator_kwargs.items())),
        }
        if self.effective_evidence_strategy != self.evidence_strategy:
            payload["effective_evidence_strategy"] = self.effective_evidence_strategy
        if self.approx_on_manifold:
            payload["approx_on_manifold"] = True
        return payload


def _clean_token(token: str) -> str:
    token = token.strip()
    if token.startswith('"') and token.endswith('"'):
        token = token[1:-1]
    return token.strip()


def _parse_bif(
    path: Path,
) -> tuple[
    List[str],
    Dict[str, Set[str]],
    Dict[str, Set[str]],
    Dict[str, List[str]],
    Dict[str, str],
    Dict[str, str],
]:
    var_re = re.compile(r"^\s*variable\s+(\"[^\"]+\"|[^\s{]+)\s*\{", re.IGNORECASE)
    prob_re = re.compile(
        r"^\s*probability\s*\(\s*(\"[^\"]+\"|[^\s\|\)]+)\s*(\|[^\)]*)?\)\s*\{",
        re.IGNORECASE,
    )
    discrete_re = re.compile(
        r"type\s+discrete\s*\[[^\]]*\]\s*\{([^}]*)\}", re.IGNORECASE
    )
    continuous_re = re.compile(r"type\s+continuous", re.IGNORECASE)

    nodes: List[str] = []
    node_set: Set[str] = set()
    parents: Dict[str, Set[str]] = defaultdict(set)
    node_states: Dict[str, List[str]] = {}
    node_types: Dict[str, str] = {}
    node_sources: Dict[str, str] = {}

    current_var: str | None = None
    var_buf: List[str] = []

    def parse_states(raw: str) -> List[str]:
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if len(parts) == 1 and " " in parts[0]:
            parts = [p for p in re.split(r"\s+", parts[0]) if p]
        return [_clean_token(p) for p in parts if _clean_token(p)]

    def finalize_var(var_name: str, buf: List[str]) -> None:
        block = " ".join(buf)
        m = discrete_re.search(block)
        if m:
            states = parse_states(m.group(1))
            node_types[var_name] = "discrete"
            node_states[var_name] = states
            return
        if continuous_re.search(block):
            node_types[var_name] = "continuous"
            return
        # Default to discrete when unspecified; downstream may override.
        node_types.setdefault(var_name, "discrete")

    for line in path.read_text().splitlines():
        if current_var is None:
            m = var_re.match(line)
            if m:
                name = _clean_token(m.group(1))
                if name:
                    current_var = name
                    var_buf = [line]
                    node_sources[name] = "bif"
                    if name not in node_set:
                        nodes.append(name)
                        node_set.add(name)
                    if "}" in line:
                        finalize_var(current_var, var_buf)
                        current_var = None
                        var_buf = []
                continue
        else:
            var_buf.append(line)
            if "}" in line:
                finalize_var(current_var, var_buf)
                current_var = None
                var_buf = []
            continue

        m = prob_re.match(line)
        if m:
            target = _clean_token(m.group(1))
            if target and target not in node_set:
                nodes.append(target)
                node_set.add(target)
            parents_str = m.group(2)
            if parents_str:
                parents_str = parents_str.strip()
                if parents_str.startswith("|"):
                    parents_str = parents_str[1:]
                for parent in parents_str.split(","):
                    name = _clean_token(parent)
                    if not name:
                        continue
                    parents[target].add(name)
                    if name not in node_set:
                        nodes.append(name)
                        node_set.add(name)

    for node in nodes:
        parents.setdefault(node, set())

    children: Dict[str, Set[str]] = {n: set() for n in nodes}
    for child in sorted(parents):
        for parent in sorted(parents[child]):
            children[parent].add(child)

    return nodes, parents, children, node_states, node_types, node_sources


def _build_domain(
    *,
    dataset_id: str,
    generator: str,
    dataset_type: str,
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
            domain_nodes[node] = {
                "type": "discrete",
                "states": states,
                "codes": {state: idx for idx, state in enumerate(states)},
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


def _write_domain(
    root_path: Path, generator: str, dataset_id: str, domain: dict
) -> Path:
    path = get_dataset_domain_metadata_path(root_path, generator, dataset_id)
    ensure_dir(path.parent)
    path.write_text(json.dumps(domain, indent=2, sort_keys=True))
    return path


def _skeleton_id(parts: List[str]) -> str:
    payload = "|".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sample_value(
    *,
    node: str,
    domain_nodes: Dict[str, dict],
    rng: random.Random,
) -> int | float:
    meta = domain_nodes.get(node)
    if meta is None:
        raise ValueError(f"Missing domain metadata for node '{node}'")
    if meta.get("type") == "discrete":
        states = meta.get("states") or []
        if not states:
            raise ValueError(f"Missing states for discrete node '{node}'")
        return int(rng.randrange(len(states)))
    return float(rng.gauss(0.0, 1.0))


def _markov_blanket(
    node: str, parents: Dict[str, Set[str]], children: Dict[str, Set[str]]
) -> Set[str]:
    pa = set(parents.get(node, set()))
    ch = set(children.get(node, set()))
    spouses: Set[str] = set()
    for c in sorted(ch):
        spouses.update(parents.get(c, set()))
    spouses.discard(node)
    mb = pa | ch | spouses
    mb.discard(node)
    return mb


def _moralized_undirected(
    nodes: Iterable[str],
    parents: Dict[str, Set[str]],
    children: Dict[str, Set[str]],
) -> Dict[str, Set[str]]:
    adj: Dict[str, Set[str]] = {n: set() for n in nodes}
    for parent in sorted(children):
        for child in sorted(children[parent]):
            adj[parent].add(child)
            adj[child].add(parent)
    for child in sorted(parents):
        pars_list = sorted(parents[child])
        for i, u in enumerate(pars_list):
            for v in pars_list[i + 1 :]:
                adj[u].add(v)
                adj[v].add(u)
    return adj


def _connected_component(adj: Dict[str, Set[str]], start: str) -> Set[str]:
    if start not in adj:
        return set()
    visited = {start}
    q = deque([start])
    while q:
        node = q.popleft()
        for nbr in sorted(adj[node]):
            if nbr not in visited:
                visited.add(nbr)
                q.append(nbr)
    return visited


def _bfs_distances(adj: Dict[str, Set[str]], start: str) -> Dict[str, int]:
    if start not in adj:
        return {}
    dist = {start: 0}
    q = deque([start])
    while q:
        node = q.popleft()
        for nbr in sorted(adj[node]):
            if nbr not in dist:
                dist[nbr] = dist[node] + 1
                q.append(nbr)
    return dist


def _eccentricities(adj: Dict[str, Set[str]]) -> Dict[str, int]:
    ecc: Dict[str, int] = {}
    for node in sorted(adj):
        dist = _bfs_distances(adj, node)
        ecc[node] = max(dist.values()) if dist else 0
    return ecc


def _articulation_points(adj: Dict[str, Set[str]]) -> Set[str]:
    time = 0
    disc: Dict[str, int] = {}
    low: Dict[str, int] = {}
    parent: Dict[str, str | None] = {}
    ap: Set[str] = set()

    def dfs(u: str) -> None:
        nonlocal time
        disc[u] = time
        low[u] = time
        time += 1
        children = 0
        for v in sorted(adj[u]):
            if v not in disc:
                parent[v] = u
                children += 1
                dfs(v)
                low[u] = min(low[u], low[v])
                if parent.get(u) is None and children > 1:
                    ap.add(u)
                if parent.get(u) is not None and low[v] >= disc[u]:
                    ap.add(u)
            elif v != parent.get(u):
                low[u] = min(low[u], disc[v])

    for node in sorted(adj):
        if node not in disc:
            parent[node] = None
            dfs(node)
    return ap


def _betweenness_centrality(
    adj: Dict[str, Set[str]], rng: random.Random
) -> Dict[str, float]:
    nodes = sorted(adj.keys())
    if not nodes:
        return {}

    max_sources = 200
    if len(nodes) <= max_sources:
        sources = nodes
    else:
        sources = rng.sample(nodes, k=max_sources)

    bc = {v: 0.0 for v in nodes}

    for s in sources:
        stack: List[str] = []
        pred: Dict[str, List[str]] = {v: [] for v in nodes}
        sigma: Dict[str, float] = {v: 0.0 for v in nodes}
        dist: Dict[str, int] = {v: -1 for v in nodes}
        sigma[s] = 1.0
        dist[s] = 0
        q = deque([s])

        while q:
            v = q.popleft()
            stack.append(v)
            for w in sorted(adj[v]):
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    q.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)

        delta = {v: 0.0 for v in nodes}
        while stack:
            w = stack.pop()
            for v in pred[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                bc[w] += delta[w]

    scale = 1.0
    if len(sources) < len(nodes):
        scale = float(len(nodes)) / float(len(sources))
    for v in bc:
        bc[v] = (bc[v] * scale) / 2.0
    return bc


def _split_budget(
    total: int, categories: List[str], rng: random.Random
) -> Dict[str, int]:
    total = max(0, int(total))
    base = total // max(1, len(categories))
    remainder = total % max(1, len(categories))
    budgets = {c: base for c in categories}
    order = list(categories)
    rng.shuffle(order)
    for c in order[:remainder]:
        budgets[c] += 1
    return budgets


def _jaccard_distance(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return 1.0 - float(inter) / float(union)


def _pac_diverse_select(
    candidates: List[str],
    context_sets: Dict[str, Set[str]],
    k: int,
    rng: random.Random,
    forbidden: Set[str] | None = None,
) -> List[str]:
    forbidden = forbidden or set()
    remaining = [c for c in sorted(candidates) if c not in forbidden]
    rng.shuffle(remaining)
    selected: List[str] = []
    min_dists: List[float] = []

    for cand in remaining:
        if len(selected) >= k:
            break
        if not selected:
            selected.append(cand)
            continue
        min_dist = min(
            _jaccard_distance(context_sets[cand], context_sets[s]) for s in selected
        )
        threshold = sum(min_dists) / len(min_dists) if min_dists else 0.0
        # Accept if candidate increases diversity relative to the running mean
        # (parameter-free PAC-style rule).
        if min_dist >= threshold:
            selected.append(cand)
            min_dists.append(min_dist)

    if len(selected) < k:
        for cand in remaining:
            if len(selected) >= k:
                break
            if cand in selected:
                continue
            selected.append(cand)

    return selected


def _allocate_targets(
    *,
    total: int,
    categories: List[str],
    candidate_map: Dict[str, List[str]],
    rng: random.Random,
    context_sets: Dict[str, Set[str]] | None = None,
) -> List[Tuple[str, str]]:
    budgets = _split_budget(total, categories, rng)
    candidate_map = {k: sorted(v) for k, v in candidate_map.items()}
    selected: List[Tuple[str, str]] = []
    selected_set: Set[str] = set()
    spill = 0

    for category in categories:
        take = budgets.get(category, 0) + spill
        spill = 0
        if take <= 0:
            continue
        if category == "random_pac":
            if context_sets is None:
                context_sets = {n: {n} for n in candidate_map.get(category, [])}
            chosen = _pac_diverse_select(
                candidate_map.get(category, []),
                context_sets,
                take,
                rng,
                forbidden=selected_set,
            )
        else:
            candidates = [
                n for n in candidate_map.get(category, []) if n not in selected_set
            ]
            chosen = candidates[:take]
        if len(chosen) < take:
            spill = take - len(chosen)
        for node in chosen:
            if node in selected_set:
                continue
            selected.append((node, category))
            selected_set.add(node)

    if spill > 0:
        remaining = [
            n for n in candidate_map.get("random_pac", []) if n not in selected_set
        ]
        if context_sets is None:
            context_sets = {n: {n} for n in remaining}
        chosen = _pac_diverse_select(
            remaining, context_sets, spill, rng, forbidden=selected_set
        )
        for node in chosen:
            if node in selected_set:
                continue
            selected.append((node, categories[-1]))
            selected_set.add(node)

    return selected


def _allocate_targets_exact(
    *,
    total: int,
    categories: List[str],
    candidate_map: Dict[str, List[str]],
    rng: random.Random,
    context_sets: Dict[str, Set[str]] | None = None,
) -> List[Tuple[str, str]]:
    if total <= 0:
        return []

    labels = _assign_balanced(categories, total, rng)
    candidate_map = {k: sorted(v) for k, v in candidate_map.items()}
    fallback = candidate_map.get("random_pac") or sorted(
        {node for nodes in candidate_map.values() for node in nodes}
    )
    if not fallback:
        raise ValueError("No candidates available for target selection.")

    per_category_order: Dict[str, List[str]] = {}
    per_category_index: Dict[str, int] = {}

    for category in categories:
        candidates = list(candidate_map.get(category, [])) or list(fallback)
        if category == "random_pac":
            if context_sets is None:
                context_sets = {n: {n} for n in candidates}
            base = _pac_diverse_select(candidates, context_sets, len(candidates), rng)
            if len(base) < len(candidates):
                remaining = [n for n in candidates if n not in base]
                base.extend(sorted(remaining))
            order = base
        else:
            order = list(candidates)
        rng.shuffle(order)
        per_category_order[category] = order
        per_category_index[category] = 0

    selected: List[Tuple[str, str]] = []
    for label in labels:
        order = per_category_order.get(label) or list(fallback)
        idx = per_category_index.get(label, 0)
        node = order[idx % len(order)]
        per_category_index[label] = idx + 1
        selected.append((node, label))

    return selected


def _assign_balanced(labels: List[str], n: int, rng: random.Random) -> List[str]:
    if n <= 0:
        return []
    budgets = _split_budget(n, labels, rng)
    out: List[str] = []
    for label in labels:
        out.extend([label] * budgets.get(label, 0))
    rng.shuffle(out)
    return out


def _ancestors_descendants(
    nodes: List[str],
    parents: Dict[str, Set[str]],
    children: Dict[str, Set[str]],
) -> tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    ancestors: Dict[str, Set[str]] = {}
    descendants: Dict[str, Set[str]] = {}

    for node in nodes:
        anc: Set[str] = set()
        stack = list(sorted(parents.get(node, set())))
        while stack:
            v = stack.pop()
            if v in anc:
                continue
            anc.add(v)
            stack.extend(sorted(parents.get(v, set())))
        ancestors[node] = anc

        desc: Set[str] = set()
        stack = list(sorted(children.get(node, set())))
        while stack:
            v = stack.pop()
            if v in desc:
                continue
            desc.add(v)
            stack.extend(sorted(children.get(v, set())))
        descendants[node] = desc

    return ancestors, descendants


def _sample_evidence(pool: List[str], rng: random.Random) -> List[str]:
    m = len(pool)
    k = rng.randint(0, m)
    if k == 0:
        return []
    return sorted(rng.sample(pool, k=k))


def _pool_for_strategy(
    *,
    target: str,
    strategy: str,
    nodes: List[str],
    mb_map: Dict[str, Set[str]],
    component_cache: Dict[str, Set[str]],
) -> List[str]:
    if strategy == "markov_blanket":
        pool = sorted(mb_map.get(target, set()))
    elif strategy == "paths":
        comp = component_cache.get(target, set())
        pool = sorted([n for n in comp if n != target])
    else:
        pool = [n for n in nodes if n != target]
    return pool


def _full_evidence(nodes: List[str], target: str) -> List[str]:
    return [n for n in nodes if n != target]


def _compute_cpd_evidence_metrics(
    *,
    pool_union: Set[str],
    evidence_union: Set[str],
    evidence_sizes: List[int],
    pool_sizes: List[int],
    observed_sizes: Set[int],
    possible_sizes: Set[int],
    path_possible_distances: Set[int],
    path_observed_distances: Set[int],
) -> dict:
    size_coverage = float(len(observed_sizes) / max(1, len(possible_sizes)))
    evidence_var_coverage = float(len(evidence_union) / max(1, len(pool_union)))
    path_distance_coverage = (
        float(len(path_observed_distances) / max(1, len(path_possible_distances)))
        if path_possible_distances
        else 0.0
    )

    return {
        "evidence_size_coverage": size_coverage,
        "evidence_var_coverage": evidence_var_coverage,
        "path_distance_coverage": path_distance_coverage,
        "avg_pool_size": (
            float(sum(pool_sizes) / max(1, len(pool_sizes))) if pool_sizes else 0.0
        ),
        "avg_evidence_size": (
            float(sum(evidence_sizes) / max(1, len(evidence_sizes)))
            if evidence_sizes
            else 0.0
        ),
    }


def _compute_inference_evidence_metrics(
    *,
    mode_counts: Dict[str, int],
    skeleton_counts: Dict[str, int],
    on_pool_union: Set[str],
    on_evidence_union: Set[str],
    on_observed_sizes: Set[int],
    on_possible_sizes: Set[int],
    off_full_count: int,
    total_queries: int,
    n_skeletons: int,
    avg_size_by_mode: Dict[str, float],
) -> dict:
    empty = mode_counts.get("empty", 0)
    on_manifold = mode_counts.get("on_manifold", 0)
    off_manifold = mode_counts.get("off_manifold", 0)

    empty_rate = float(empty / max(1, total_queries))
    on_var_coverage = float(len(on_evidence_union) / max(1, len(on_pool_union)))
    on_size_coverage = float(len(on_observed_sizes) / max(1, len(on_possible_sizes)))
    off_full_rate = float(off_full_count / max(1, off_manifold))

    return {
        "mode_counts": {
            "empty": int(empty),
            "on_manifold": int(on_manifold),
            "off_manifold": int(off_manifold),
        },
        "skeleton_mode_counts": {
            "empty": int(skeleton_counts.get("empty", 0)),
            "on_manifold": int(skeleton_counts.get("on_manifold", 0)),
            "off_manifold": int(skeleton_counts.get("off_manifold", 0)),
        },
        "n_skeletons": int(n_skeletons),
        "n_instantiated": int(total_queries),
        "instantiation_factor": float(total_queries / max(1, n_skeletons)),
        "empty_rate": empty_rate,
        "on_manifold_var_coverage": on_var_coverage,
        "on_manifold_size_coverage": on_size_coverage,
        "off_manifold_full_rate": off_full_rate,
        "avg_evidence_size_by_mode": avg_size_by_mode,
    }


def _generate_cpd_queries(
    *,
    dataset_id: str,
    seed: int,
    targets: List[Tuple[str, str]],
    strategies: List[str],
    nodes: List[str],
    mb_map: Dict[str, Set[str]],
    component_cache: Dict[str, Set[str]],
    undirected: Dict[str, Set[str]],
    dist_cache: Dict[str, Dict[str, int]],
    rng: random.Random,
) -> tuple[List[CPDQuery], dict]:
    pool_union: Set[str] = set()
    evidence_union: Set[str] = set()
    evidence_sizes: List[int] = []
    pool_sizes: List[int] = []
    observed_sizes: Set[int] = set()
    possible_sizes: Set[int] = set()
    path_possible_distances: Set[int] = set()
    path_observed_distances: Set[int] = set()

    queries: List[CPDQuery] = []

    for (target, category), strategy in zip(targets, strategies):
        pool = _pool_for_strategy(
            target=target,
            strategy=strategy,
            nodes=nodes,
            mb_map=mb_map,
            component_cache=component_cache,
        )
        evidence_vars = _sample_evidence(pool, rng)

        queries.append(
            CPDQuery(
                target=target,
                target_category=category,
                evidence_strategy=strategy,
                evidence_vars=evidence_vars,
                seed=seed,
                dataset_id=dataset_id,
            )
        )

        pool_set = set(pool)
        evidence_set = set(evidence_vars)
        pool_union |= pool_set
        evidence_union |= evidence_set
        evidence_sizes.append(len(evidence_vars))
        pool_sizes.append(len(pool))
        observed_sizes.add(len(evidence_vars))
        possible_sizes.update(range(0, len(pool) + 1))

        if strategy == "paths" and pool:
            dist_map = dist_cache.get(target)
            if dist_map is None:
                dist_map = _bfs_distances(undirected, target)
                dist_cache[target] = dist_map
            path_possible_distances |= {
                dist_map.get(v, 0) for v in pool if v in dist_map
            }
            path_observed_distances |= {
                dist_map.get(v, 0) for v in evidence_vars if v in dist_map
            }

    evidence_metrics = _compute_cpd_evidence_metrics(
        pool_union=pool_union,
        evidence_union=evidence_union,
        evidence_sizes=evidence_sizes,
        pool_sizes=pool_sizes,
        observed_sizes=observed_sizes,
        possible_sizes=possible_sizes,
        path_possible_distances=path_possible_distances,
        path_observed_distances=path_observed_distances,
    )

    queries = sorted(
        queries,
        key=lambda q: (
            q.target,
            q.target_category,
            q.evidence_strategy,
            len(q.evidence_vars),
            q.evidence_vars,
        ),
    )

    return queries, evidence_metrics


def _generate_inference_queries(
    *,
    dataset_id: str,
    seed: int,
    targets: List[Tuple[str, str]],
    strategies: List[str],
    tasks: List[str],
    modes: List[str],
    block_sizes: List[int],
    nodes: List[str],
    mb_map: Dict[str, Set[str]],
    children: Dict[str, Set[str]],
    ancestors: Dict[str, Set[str]],
    descendants: Dict[str, Set[str]],
    component_cache: Dict[str, Set[str]],
    rng: random.Random,
    n_mc: int,
    domain_nodes: Dict[str, dict],
    generator: str,
    generator_kwargs: dict,
) -> tuple[List[InferenceQuery], dict]:
    mode_counts = {"empty": 0, "on_manifold": 0, "off_manifold": 0}
    skeleton_counts = {"empty": 0, "on_manifold": 0, "off_manifold": 0}
    on_pool_union: Set[str] = set()
    on_evidence_union: Set[str] = set()
    on_observed_sizes: Set[int] = set()
    on_possible_sizes: Set[int] = set()
    off_full_count = 0

    size_by_mode: Dict[str, List[int]] = {
        "empty": [],
        "on_manifold": [],
        "off_manifold": [],
    }

    queries: List[InferenceQuery] = []
    approx_on_manifold_any = False

    for idx, ((target, category), strategy, task, mode) in enumerate(
        zip(targets, strategies, tasks, modes)
    ):
        block_size = block_sizes[idx] if idx < len(block_sizes) else 0
        skeleton_counts[mode] += 1
        evidence_vars: List[str]
        effective_strategy = strategy
        approx_on_manifold = False

        if mode == "empty":
            evidence_vars = []
        elif mode == "off_manifold":
            evidence_vars = _full_evidence(nodes, target)
            effective_strategy = "off_manifold_full"
        else:
            base_pool = _pool_for_strategy(
                target=target,
                strategy=strategy,
                nodes=nodes,
                mb_map=mb_map,
                component_cache=component_cache,
            )
            if task == "prediction":
                pref = set(ancestors.get(target, set()))
                if not pref:
                    pref = set(mb_map.get(target, set()))
            else:
                pref = set(descendants.get(target, set()))
                if not pref:
                    pref = set(children.get(target, set()))
                if not pref:
                    pref = set(mb_map.get(target, set()))
            pref.discard(target)

            if pref:
                pool = [n for n in base_pool if n in pref]
                if not pool:
                    pool = sorted(pref)
            else:
                pool = base_pool

            evidence_vars = _sample_evidence(pool, rng)
            approx_on_manifold = True
            approx_on_manifold_any = True

            pool_set = set(pool)
            evidence_set = set(evidence_vars)
            on_pool_union |= pool_set
            on_evidence_union |= evidence_set
            on_observed_sizes.add(len(evidence_vars))
            on_possible_sizes.update(range(0, len(pool) + 1))

        evidence_vars = sorted(evidence_vars)
        skeleton_id = _skeleton_id(
            [
                target,
                ",".join(evidence_vars),
                mode,
                task,
                category,
                str(seed),
            ]
        )

        if block_size <= 0:
            continue

        for mc_id in range(block_size):
            if mode == "empty":
                values = {}
            else:
                values = {
                    var: _sample_value(node=var, domain_nodes=domain_nodes, rng=rng)
                    for var in evidence_vars
                }
            evidence = EvidenceSpec(
                mode=mode,
                vars=evidence_vars,
                values=values,
            )
            queries.append(
                InferenceQuery(
                    task=task,
                    target=target,
                    target_category=category,
                    evidence_strategy=strategy,
                    effective_evidence_strategy=effective_strategy,
                    evidence_mode=mode,
                    evidence_vars=evidence_vars,
                    evidence=evidence,
                    skeleton_id=skeleton_id,
                    mc_id=mc_id,
                    approx_on_manifold=approx_on_manifold,
                    seed=seed,
                    dataset_id=dataset_id,
                    generator=generator,
                    generator_kwargs=generator_kwargs,
                )
            )
            mode_counts[mode] += 1
            size_by_mode[mode].append(len(evidence_vars))
            if mode == "off_manifold" and len(evidence_vars) == len(nodes) - 1:
                off_full_count += 1

    avg_size_by_mode = {
        mode: (sum(sizes) / len(sizes) if sizes else 0.0)
        for mode, sizes in size_by_mode.items()
    }

    evidence_metrics = _compute_inference_evidence_metrics(
        mode_counts=mode_counts,
        skeleton_counts=skeleton_counts,
        on_pool_union=on_pool_union,
        on_evidence_union=on_evidence_union,
        on_observed_sizes=on_observed_sizes,
        on_possible_sizes=on_possible_sizes,
        off_full_count=off_full_count,
        total_queries=len(queries),
        n_skeletons=len(targets),
        avg_size_by_mode=avg_size_by_mode,
    )
    if approx_on_manifold_any:
        evidence_metrics["approx_on_manifold"] = True

    queries = sorted(
        queries,
        key=lambda q: (
            q.target,
            q.task,
            q.evidence_mode,
            q.target_category,
            q.evidence_strategy,
            len(q.evidence_vars),
            q.evidence_vars,
            q.mc_id if q.mc_id is not None else -1,
        ),
    )

    return queries, evidence_metrics


@register_query_generator
class BNLearnQueryGenerator(BaseQueryGenerator):
    name = "bnlearn"

    def _find_bif(self, dataset_dir: Path) -> Path:
        candidates = sorted(dataset_dir.glob("*.bif"))
        if not candidates:
            candidates = sorted(dataset_dir.glob("**/*.bif"))
        if not candidates:
            raise FileNotFoundError(f"No .bif file found in {dataset_dir}")
        for candidate in candidates:
            if candidate.name == "model.bif":
                return candidate
        return candidates[0]

    def _resolve_dataset_type(self, dataset_id: str, download_meta: dict) -> str:
        dataset_type = download_meta.get("type") if download_meta else None
        if dataset_type:
            return str(dataset_type)
        static_path = get_static_metadata_dir(self.root_path) / "bnlearn.json"
        if static_path.exists():
            try:
                static_meta = json.loads(static_path.read_text())
                dataset_type = static_meta.get(dataset_id, {}).get("type")
            except Exception:
                dataset_type = None
        return str(dataset_type) if dataset_type else "discrete"

    def generate_payload(
        self,
        dataset_id: str,
        dataset_dir: Path,
        rng: random.Random,
        logger: logging.Logger,
    ) -> dict:
        download_meta_path = get_dataset_download_metadata_path(
            self.root_path, self.name, dataset_id
        )
        download_meta: dict = {}
        if download_meta_path.exists():
            try:
                download_meta = json.loads(download_meta_path.read_text())
            except Exception as exc:
                logger.warning(
                    "Failed to read download metadata for %s: %s", dataset_id, exc
                )
                download_meta = {}
            capabilities = download_meta.get("capabilities", {})
            if not capabilities.get("can_generate_queries", True):
                reason = download_meta.get("reason") or "unsupported dataset format"
                logger.warning("Skipping %s: %s", dataset_id, reason)
                return None

        try:
            bif_path = self._find_bif(dataset_dir)
        except FileNotFoundError as exc:
            logger.warning("Skipping %s: %s", dataset_id, exc)
            return None
        nodes, parents, children, node_states, node_types, node_sources = _parse_bif(
            bif_path
        )
        if not nodes:
            raise ValueError(f"No variables found in BIF file: {bif_path}")

        nodes = sorted(nodes)
        dataset_type = self._resolve_dataset_type(dataset_id, download_meta)
        domain, unsupported, reason = _build_domain(
            dataset_id=dataset_id,
            generator=self.name,
            dataset_type=dataset_type,
            nodes=nodes,
            node_states=node_states,
            node_types=node_types,
            node_sources=node_sources,
        )
        domain_path = _write_domain(self.root_path, self.name, dataset_id, domain)
        logger.info("Wrote domain metadata to %s", domain_path)
        if unsupported:
            logger.warning("Skipping %s: %s", dataset_id, reason)
            return None

        generator_kwargs = dict(self.generator_kwargs)
        generator_kwargs.setdefault("n_mc", int(self.n_mc))
        mb_map = {n: _markov_blanket(n, parents, children) for n in nodes}
        parent_sizes = {n: len(parents.get(n, set())) for n in nodes}
        mb_sizes = {n: len(mb_map[n]) for n in nodes}

        undirected = _moralized_undirected(nodes, parents, children)
        eccentricity = _eccentricities(undirected)
        articulation = _articulation_points(undirected)
        betweenness = _betweenness_centrality(undirected, rng)
        degree = {n: len(undirected.get(n, set())) for n in nodes}

        ancestors, descendants = _ancestors_descendants(nodes, parents, children)

        component_cache: Dict[str, Set[str]] = {}
        for node in nodes:
            if node in component_cache:
                continue
            comp = _connected_component(undirected, node)
            for member in sorted(comp):
                component_cache[member] = comp

        dist_cache: Dict[str, Dict[str, int]] = {}

        mb_context = {n: set(mb_map[n]) | {n} for n in nodes}

        cpd_categories = ["big_markov_blanket", "big_parent_set", "random_pac"]
        inf_categories = [
            "central_hub",
            "separator_cut",
            "peripheral_terminal",
            "random_pac",
        ]

        if self.n_queries_cpds < 0 or self.n_queries_inference < 0:
            raise ValueError("n_queries_cpds and n_queries_inference must be >= 0")

        cpd_budget = int(self.n_queries_cpds)
        total_inference = int(self.n_queries_inference)
        if total_inference == 0:
            inf_skeletons = 0
            block_sizes: List[int] = []
        else:
            n_full_blocks = total_inference // self.n_mc
            remainder = total_inference % self.n_mc
            if n_full_blocks == 0:
                inf_skeletons = 1
                block_sizes = [total_inference]
            else:
                inf_skeletons = n_full_blocks + (1 if remainder > 0 else 0)
                block_sizes = [self.n_mc] * n_full_blocks
                if remainder > 0:
                    block_sizes.append(remainder)

            logger.info(
                "Inference counts: N=%s M=%s skeletons=%s remainder=%s",
                total_inference,
                self.n_mc,
                inf_skeletons,
                remainder,
            )

        cpd_candidates = {
            "big_markov_blanket": sorted(nodes, key=lambda n: (-mb_sizes[n], n)),
            "big_parent_set": sorted(nodes, key=lambda n: (-parent_sizes[n], n)),
            "random_pac": list(nodes),
        }

        hub_sorted = sorted(
            nodes, key=lambda n: (-betweenness.get(n, 0.0), -degree[n], n)
        )
        separator_sorted = sorted(articulation) or []
        ecc_values = sorted(eccentricity.get(n, 0) for n in nodes)
        if ecc_values:
            mid = len(ecc_values) // 2
            if len(ecc_values) % 2 == 1:
                ecc_median = ecc_values[mid]
            else:
                ecc_median = 0.5 * (ecc_values[mid - 1] + ecc_values[mid])
        else:
            ecc_median = 0.0
        peripheral_nodes = [n for n in nodes if eccentricity.get(n, 0) >= ecc_median]
        peripheral_sorted = sorted(
            peripheral_nodes, key=lambda n: (-eccentricity.get(n, 0), n)
        )
        terminal_nodes = [n for n in nodes if degree.get(n, 0) <= 1]
        peripheral_terminal = list(dict.fromkeys(peripheral_sorted + terminal_nodes))

        inf_candidates = {
            "central_hub": hub_sorted,
            "separator_cut": separator_sorted,
            "peripheral_terminal": peripheral_terminal,
            "random_pac": list(nodes),
        }

        cpd_targets = _allocate_targets_exact(
            total=cpd_budget,
            categories=cpd_categories,
            candidate_map=cpd_candidates,
            rng=rng,
            context_sets=mb_context,
        )
        inf_targets = _allocate_targets_exact(
            total=inf_skeletons,
            categories=inf_categories,
            candidate_map=inf_candidates,
            rng=rng,
            context_sets=mb_context,
        )

        evidence_strategies = ["paths", "markov_blanket", "random"]
        cpd_strategy_assign = _assign_balanced(
            evidence_strategies, len(cpd_targets), rng
        )
        inf_strategy_assign = _assign_balanced(
            evidence_strategies, len(inf_targets), rng
        )
        task_assign = _assign_balanced(
            ["prediction", "diagnosis"], len(inf_targets), rng
        )
        mode_assign = _assign_balanced(
            ["empty", "on_manifold", "off_manifold"], len(inf_targets), rng
        )
        if inf_targets:
            logger.info(
                "Inference skeletons by mode: %s",
                dict(sorted(Counter(mode_assign).items())),
            )
            logger.info(
                "Inference skeletons by task: %s",
                dict(sorted(Counter(task_assign).items())),
            )
            logger.info(
                "Inference skeletons by category: %s",
                dict(sorted(Counter(cat for _, cat in inf_targets).items())),
            )
            logger.info("Inference block sizes: %s", block_sizes)

        cpd_queries, cpd_evidence_metrics = _generate_cpd_queries(
            dataset_id=dataset_id,
            seed=self.seed,
            targets=cpd_targets,
            strategies=cpd_strategy_assign,
            nodes=nodes,
            mb_map=mb_map,
            component_cache=component_cache,
            undirected=undirected,
            dist_cache=dist_cache,
            rng=rng,
        )

        inference_queries, inference_evidence_metrics = _generate_inference_queries(
            dataset_id=dataset_id,
            seed=self.seed,
            targets=inf_targets,
            strategies=inf_strategy_assign,
            tasks=task_assign,
            modes=mode_assign,
            block_sizes=block_sizes,
            nodes=nodes,
            mb_map=mb_map,
            children=children,
            ancestors=ancestors,
            descendants=descendants,
            component_cache=component_cache,
            rng=rng,
            n_mc=self.n_mc,
            domain_nodes=domain.get("nodes", {}),
            generator=self.name,
            generator_kwargs=generator_kwargs,
        )

        cpd_target_set = {q.target for q in cpd_queries}
        inf_target_set = {t for (t, _) in inf_targets}

        all_pairs: Set[Tuple[str, str]] = set()
        for n in nodes:
            for m in sorted(mb_map.get(n, set())):
                a, b = (n, m) if n < m else (m, n)
                all_pairs.add((a, b))

        selected_pairs: Set[Tuple[str, str]] = set()
        for n in sorted(cpd_target_set):
            for m in sorted(mb_map.get(n, set())):
                a, b = (n, m) if n < m else (m, n)
                selected_pairs.add((a, b))

        avg_mb_size_all = sum(mb_sizes.values()) / max(1, len(nodes))
        avg_mb_size_selected = (
            sum(mb_sizes[n] for n in cpd_target_set) / max(1, len(cpd_target_set))
            if cpd_target_set
            else 0.0
        )
        mb_size_ratio = (
            avg_mb_size_selected / avg_mb_size_all if avg_mb_size_all > 0 else 0.0
        )

        parent_union: Set[str] = set()
        for n in sorted(cpd_target_set):
            parent_union |= parents.get(n, set())

        cpd_metrics = {
            "target_coverage": float(len(cpd_target_set) / max(1, len(nodes))),
            "parent_coverage": float(len(parent_union) / max(1, len(nodes))),
            "mb_size_ratio": float(mb_size_ratio),
            "mb_pair_coverage": float(len(selected_pairs) / max(1, len(all_pairs))),
            "evidence": cpd_evidence_metrics,
        }

        betweenness_values = sorted(betweenness.get(n, 0.0) for n in nodes)
        if betweenness_values:
            mid = len(betweenness_values) // 2
            if len(betweenness_values) % 2 == 1:
                median = betweenness_values[mid]
            else:
                median = 0.5 * (betweenness_values[mid - 1] + betweenness_values[mid])
        else:
            median = 0.0
        hub_set = {n for n in nodes if betweenness.get(n, 0.0) >= median}
        separator_set = set(separator_sorted)
        peripheral_set = set(peripheral_sorted)
        terminal_set = set(terminal_nodes)

        def coverage(selected: Set[str], available: Set[str]) -> float:
            if not available:
                return 0.0
            return float(len(selected & available) / len(available))

        inference_metrics = {
            "target_coverage": float(len(inf_target_set) / max(1, len(nodes))),
            "hub_coverage": coverage(inf_target_set, hub_set),
            "separator_coverage": coverage(inf_target_set, separator_set),
            "peripheral_coverage": coverage(inf_target_set, peripheral_set),
            "terminal_coverage": coverage(inf_target_set, terminal_set),
            "evidence": inference_evidence_metrics,
        }

        """approx_on_manifold = any(
            q.approx_on_manifold
            for q in inference_queries
            if q.evidence_mode == "on_manifold"
        )"""

        logger.info(
            "Nodes=%s Edges=%s", len(nodes), sum(len(v) for v in children.values())
        )
        logger.info(
            "Inference skeletons=%s instantiated=%s",
            len(inf_targets),
            len(inference_queries),
        )
        logger.info("CPD metrics: %s", json.dumps(cpd_metrics, sort_keys=True))
        logger.info(
            "CPD evidence metrics: %s",
            json.dumps(cpd_evidence_metrics, sort_keys=True),
        )
        logger.info(
            "Inference metrics: %s", json.dumps(inference_metrics, sort_keys=True)
        )
        logger.info(
            "Inference evidence metrics: %s",
            json.dumps(inference_evidence_metrics, sort_keys=True),
        )
        if inference_evidence_metrics.get("approx_on_manifold"):
            logger.info("On-manifold sampling approximated with independent draws.")

        if len(cpd_queries) != self.n_queries_cpds:
            logger.warning(
                "Skipping %s: generated %s CPD queries but expected %s",
                dataset_id,
                len(cpd_queries),
                self.n_queries_cpds,
            )
            return None
        if len(inference_queries) != self.n_queries_inference:
            logger.warning(
                "Skipping %s: generated %s inference queries but expected %s",
                dataset_id,
                len(inference_queries),
                self.n_queries_inference,
            )
            return None

        payload = {
            "dataset_id": dataset_id,
            "generator": self.name,
            "seed": int(self.seed),
            "n_mc": int(self.n_mc),
            "generator_kwargs": dict(sorted(self.generator_kwargs.items())),
            "n_queries": {
                "cpds": len(cpd_queries),
                "inference": len(inference_queries),
            },
            "n_skeletons": {"inference": len(inf_targets)},
            "cpd_queries": [q.to_dict() for q in cpd_queries],
            "inference_queries": [q.to_dict() for q in inference_queries],
            "coverage": {
                "cpds": cpd_metrics,
                "inference": inference_metrics,
            },
            "notes": {},
        }

        has_continuous = any(
            meta.get("type") == "continuous"
            for meta in domain.get("nodes", {}).values()
        )
        payload["notes"]["off_manifold_distribution"] = (
            "standard_normal" if has_continuous else "uniform_state"
        )
        if inference_evidence_metrics.get("approx_on_manifold"):
            payload["notes"]["approx_on_manifold"] = True
            payload["notes"]["on_manifold_sampling"] = "approx_independent"

        return payload
