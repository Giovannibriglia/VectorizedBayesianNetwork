from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from vbn.core.base import Query
from vbn.core.utils import ensure_2d


@dataclass(frozen=True)
class InferenceState:
    topo_order: Tuple[str, ...]
    parent_idx: Tuple[Tuple[int, ...], ...]
    evidence_mask: Tuple[bool, ...]
    do_mask: Tuple[bool, ...]
    target_idx: int
    node_to_idx: Dict[str, int]
    node_slices: Tuple[slice, ...]
    parent_slices: Tuple[Tuple[slice, ...], ...]
    total_dim: int
    children_idx: Tuple[Tuple[int, ...], ...]
    child_parent_idx: Tuple[Tuple[Tuple[int, ...], ...], ...]


def query_signature(vbn, query: Query) -> Tuple:
    return (
        id(vbn),
        query.target,
        tuple(sorted(query.evidence.keys())),
        tuple(sorted(query.do.keys())),
    )


def resolve_dtype(vbn, query: Query) -> torch.dtype:
    if query.evidence:
        return next(iter(query.evidence.values())).dtype
    if query.do:
        return next(iter(query.do.values())).dtype
    for cpd in vbn.nodes.values():
        for param in cpd.parameters():
            return param.dtype
        for buf in cpd.buffers():
            return buf.dtype
        extra = getattr(cpd, "get_extra_state", None)
        if callable(extra):
            state = extra()
            if isinstance(state, dict):
                for value in state.values():
                    if isinstance(value, torch.Tensor):
                        return value.dtype
    return torch.float32


def get_inference_state(vbn, query: Query, cache: Dict) -> InferenceState:
    sig = query_signature(vbn, query)
    if sig in cache:
        return cache[sig]

    topo = tuple(vbn.dag.topological_order())
    node_to_idx = {node: idx for idx, node in enumerate(topo)}
    parent_idx: List[Tuple[int, ...]] = []
    for node in topo:
        parents = vbn.dag.parents(node)
        parent_idx.append(tuple(node_to_idx[p] for p in parents))
    evidence_set = set(query.evidence.keys())
    do_set = set(query.do.keys())
    evidence_mask = tuple(node in evidence_set for node in topo)
    do_mask = tuple(node in do_set for node in topo)
    target_idx = node_to_idx[query.target]

    node_slices: List[slice] = []
    total_dim = 0
    for node in topo:
        dim = int(vbn.nodes[node].output_dim)
        node_slices.append(slice(total_dim, total_dim + dim))
        total_dim += dim
    parent_slices = tuple(
        tuple(node_slices[p] for p in parents) for parents in parent_idx
    )

    children_map = {node: [] for node in topo}
    for parent, child in vbn.dag.edges():
        if parent in children_map:
            children_map[parent].append(child)
    children_idx = tuple(
        tuple(node_to_idx[c] for c in children_map[node]) for node in topo
    )
    child_parent_idx = tuple(
        tuple(parent_idx[c] for c in children_idx[i]) for i in range(len(topo))
    )

    state = InferenceState(
        topo_order=topo,
        parent_idx=tuple(parent_idx),
        evidence_mask=evidence_mask,
        do_mask=do_mask,
        target_idx=target_idx,
        node_to_idx=node_to_idx,
        node_slices=tuple(node_slices),
        parent_slices=parent_slices,
        total_dim=total_dim,
        children_idx=children_idx,
        child_parent_idx=child_parent_idx,
    )
    cache[sig] = state
    return state


def clamp_evidence(x: torch.Tensor) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
    return x.clamp(min=-1e6, max=1e6)


def prepare_fixed_values(
    query: Query,
    state: InferenceState,
    device: torch.device,
    dtype: torch.dtype,
    *,
    clamp_obs: bool = False,
) -> List[Optional[torch.Tensor]]:
    values: List[Optional[torch.Tensor]] = [None] * len(state.topo_order)
    for node, value in query.do.items():
        idx = state.node_to_idx[node]
        values[idx] = ensure_2d(value).to(device=device, dtype=dtype)
    for node, value in query.evidence.items():
        idx = state.node_to_idx[node]
        v = ensure_2d(value).to(device=device, dtype=dtype)
        if clamp_obs:
            v = clamp_evidence(v)
        values[idx] = v
    return values
