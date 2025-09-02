from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import networkx as nx
import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# Shared metadata + parameter containers
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class BNMeta:
    G: nx.DiGraph
    types: Dict[str, Literal["discrete", "continuous"]]
    cards: Optional[Dict[str, int]]  # required for discrete nodes
    order: List[str]  # topological order
    parents: Dict[str, List[str]]

    @staticmethod
    def from_graph(
        G: nx.DiGraph, types: Dict[str, str], cards: Optional[Dict[str, int]] = None
    ) -> "BNMeta":
        order = list(nx.topological_sort(G))
        parents = {n: list(G.predecessors(n)) for n in G.nodes()}
        return BNMeta(G=G, types=types, cards=cards, order=order, parents=parents)


@dataclass
class DiscreteCPDTable:
    """
    Tabular CPD for a single discrete node.
    probs shape: [n_parent_configs, child_card] (or [1, C] if no parents)
    """

    probs: torch.Tensor
    parent_names: List[str]
    parent_cards: List[int]
    child_card: int
    strides: torch.Tensor  # [n_parents], to map parent assignment -> row index


@dataclass
class LGParams:
    """
    Linear-Gaussian params for the continuous subgraph only.
    Names and indices refer only to continuous nodes (respecting topo order).
    """

    order: List[str]
    name2idx: Dict[str, int]
    W: torch.Tensor  # [nc, nc], strictly upper-triangular under topo order
    b: torch.Tensor  # [nc]
    sigma2: torch.Tensor  # [nc]


@dataclass
class LearnParams:
    """
    A single container for all families; you can fill any subset.
    """

    meta: BNMeta
    # Discrete CPDs (tabular)
    discrete_tables: Optional[Dict[str, DiscreteCPDTable]] = None
    # Discrete neural CPDs (per-node models)
    discrete_mlps: Optional[Dict[str, nn.Module]] = None
    # Continuous: linear-Gaussian parameters (exact)
    lg: Optional[LGParams] = None
    # Continuous MLP CPDs (mean/logvar heads) + their parent meta
    cont_mlps: Optional[Dict[str, nn.Module]] = None
    cont_mlp_meta: Optional[Dict[str, Dict]] = None


# ─────────────────────────────────────────────────────────────────────────────
# Thin BN facade wiring learners and inference backends
# ─────────────────────────────────────────────────────────────────────────────


class CausalBayesNet:

    def __init__(
        self,
        G: nx.DiGraph,
        types: Dict[str, str],
        cards: Optional[Dict[str, int]] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.meta = BNMeta.from_graph(G, types, cards)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = dtype

    # Learning front-ends (import lazily to avoid circulars)
    def fit_discrete_mle(self, data: Dict[str, torch.Tensor], **kw) -> LearnParams:
        from .learning.discrete_mle import DiscreteMLELearner

        return DiscreteMLELearner(
            self.meta, device=self.device, dtype=self.dtype, **kw
        ).fit(data)

    def fit_discrete_mlp(self, data: Dict[str, torch.Tensor], **kw) -> LearnParams:
        from .learning.discrete_mlp import DiscreteMLPLearner

        return DiscreteMLPLearner(
            self.meta, device=self.device, dtype=self.dtype, **kw
        ).fit(data)

    def materialize_discrete_mlp(self, lp: LearnParams) -> LearnParams:
        from .learning.discrete_mlp import DiscreteMLPLearner

        return DiscreteMLPLearner(
            self.meta, device=self.device, dtype=self.dtype
        ).materialize_tables(lp)

    def fit_continuous_gaussian(
        self, data: Dict[str, torch.Tensor], **kw
    ) -> LearnParams:
        from .learning.gaussian_linear import GaussianLinearLearner

        return GaussianLinearLearner(
            self.meta, device=self.device, dtype=self.dtype, **kw
        ).fit(data)

    def fit_continuous_mlp(self, data: Dict[str, torch.Tensor], **kw) -> LearnParams:
        from .learning.continuous_mlp import ContinuousMLPLearner

        return ContinuousMLPLearner(
            self.meta, device=self.device, dtype=self.dtype, **kw
        ).fit(data)

    def materialize_lg_from_cont_mlp(
        self,
        lp: LearnParams,
        pivot: Optional[Dict[str, torch.Tensor]] = None,
        data: Optional[Dict[str, torch.Tensor]] = None,
    ) -> LearnParams:
        from .learning.continuous_mlp import materialize_lg_from_cont_mlp

        return materialize_lg_from_cont_mlp(lp, pivot=pivot, data=data)

    def infer_discrete_exact(
        self,
        lp: LearnParams,
        evidence: Dict[str, torch.Tensor],
        query: List[str],
        do: Optional[Dict[str, torch.Tensor]] = None,
    ):
        from .inference.discrete_exact import DiscreteExactVEInference

        return DiscreteExactVEInference(
            self.meta, device=self.device, dtype=self.dtype
        ).posterior(lp, evidence, query, do=do)

    def infer_discrete_approx(
        self,
        lp: LearnParams,
        evidence: Dict[str, torch.Tensor],
        query: List[str],
        do: Optional[Dict[str, torch.Tensor]] = None,
        **kw,
    ):
        from .inference.discrete_approx import DiscreteApproxInference

        return DiscreteApproxInference(
            self.meta, device=self.device, dtype=self.dtype, **kw
        ).posterior(lp, evidence, query, do=do)

    def infer_continuous_gaussian(
        self,
        lp: LearnParams,
        evidence: Dict[str, torch.Tensor],
        query: List[str],
        do: Optional[Dict[str, torch.Tensor]] = None,
    ):
        from .inference.continuous_gaussian import ContinuousLGInference

        return ContinuousLGInference(
            self.meta, device=self.device, dtype=self.dtype
        ).posterior(lp, evidence, query, do=do)

    def infer_continuous_approx(
        self,
        lp: LearnParams,
        evidence: Dict[str, torch.Tensor],
        query: List[str],
        do: Optional[Dict[str, torch.Tensor]] = None,
        **kw,
    ):
        from .inference.continuous_approx import ContinuousApproxInference

        return ContinuousApproxInference(
            self.meta, device=self.device, dtype=self.dtype, **kw
        ).posterior(lp, evidence, query, do=do)

        # --- IO helpers ---

    def save_params(self, lp: LearnParams, path: str) -> None:
        from .io import save_learnparams

        save_learnparams(path, lp)

    def load_params(
        self, path: str, map_location: Optional[torch.device] = None
    ) -> LearnParams:
        from .io import load_learnparams

        lp = load_learnparams(path, map_location=map_location or self.device)
        return lp


# ─────────────────────────────────────────────────────────────────────────────
# Merge helper: combine multiple LearnParams into one
# ─────────────────────────────────────────────────────────────────────────────
def merge_learnparams(*lps: LearnParams) -> LearnParams:
    assert len(lps) >= 1, "Provide at least one LearnParams"
    meta = lps[0].meta
    out = LearnParams(
        meta=meta,
        discrete_tables={},
        discrete_mlps={},
        lg=None,
        cont_mlps={},
        cont_mlp_meta={},
    )
    for lp in lps:
        assert lp.meta.order == meta.order, "All LearnParams must share the same BN"
        if lp.discrete_tables:
            out.discrete_tables.update(lp.discrete_tables)
        if lp.discrete_mlps:
            out.discrete_mlps.update(lp.discrete_mlps)
        if lp.cont_mlps:
            out.cont_mlps.update(lp.cont_mlps)
        if lp.cont_mlp_meta:
            out.cont_mlp_meta.update(lp.cont_mlp_meta)
        if lp.lg is not None:
            out.lg = lp.lg  # last-wins if multiple
    # Normalize empties to None
    if not out.discrete_tables:
        out.discrete_tables = None
    if not out.discrete_mlps:
        out.discrete_mlps = None
    if not out.cont_mlps:
        out.cont_mlps = None
    if not out.cont_mlp_meta:
        out.cont_mlp_meta = None
    return out
