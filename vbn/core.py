from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Union

import networkx as nx
import pandas as pd
import torch
import torch.nn as nn
from tensordict import TensorDict

TDLike = Union[TensorDict, Dict[str, torch.Tensor], pd.DataFrame]

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

    @property
    def variables(self) -> List[str]:
        # choose topo order to keep parents-before-children where needed
        return self.meta.order

    @property
    def types(self) -> Dict[str, str]:
        return self.meta.types

    @property
    def parents(self) -> Dict[str, List[str]]:
        return self.meta.parents

    @property
    def cards(self) -> Optional[Dict[str, int]]:
        return self.meta.cards

    def _normalize_fit_input(self, data: TDLike) -> Dict[str, torch.Tensor]:
        """Accepts DataFrame or dict; returns dict[str, Tensor] on self.device with proper dtypes."""
        if isinstance(data, pd.DataFrame):
            cols = [k for k in self.variables if k in data.columns]
            if not cols:
                raise ValueError("DataFrame has no columns matching BN variables.")
            mapping = {k: torch.as_tensor(data[k].values) for k in cols}
        elif isinstance(data, dict):
            mapping = {
                k: (v if isinstance(v, torch.Tensor) else torch.as_tensor(v))
                for k, v in data.items()
            }
        else:
            raise TypeError("fit_* expects a dict[str, Tensor] or a pandas.DataFrame")

        # dtype/device normalization
        out = {}
        for k, t in mapping.items():
            if self.types.get(k, "discrete") == "discrete":
                out[k] = t.to(self.device).long()
            else:
                out[k] = t.to(self.device).float()
        return out

    def fit_discrete_mle(self, data: TDLike, **kw) -> LearnParams:
        from .learning.discrete_mle import DiscreteMLELearner

        data = self._normalize_fit_input(data)
        return DiscreteMLELearner(
            self.meta, device=self.device, dtype=self.dtype, **kw
        ).fit(data)

    def fit_discrete_mlp(self, data: TDLike, **kw) -> LearnParams:
        from .learning.discrete_mlp import DiscreteMLPLearner

        data = self._normalize_fit_input(data)
        return DiscreteMLPLearner(
            self.meta, device=self.device, dtype=self.dtype, **kw
        ).fit(data)

    def fit_continuous_gaussian(self, data: TDLike, **kw) -> LearnParams:
        from .learning.gaussian_linear import GaussianLinearLearner

        data = self._normalize_fit_input(data)
        return GaussianLinearLearner(
            self.meta, device=self.device, dtype=self.dtype, **kw
        ).fit(data)

    def fit_continuous_mlp(self, data: TDLike, **kw) -> LearnParams:
        from .learning.continuous_mlp import ContinuousMLPLearner

        data = self._normalize_fit_input(data)
        return ContinuousMLPLearner(
            self.meta, device=self.device, dtype=self.dtype, **kw
        ).fit(data)

    def materialize_discrete_mlp(self, lp: LearnParams) -> LearnParams:
        from .learning.discrete_mlp import DiscreteMLPLearner

        return DiscreteMLPLearner(
            self.meta, device=self.device, dtype=self.dtype
        ).materialize_tables(lp)

    def materialize_lg_from_cont_mlp(
        self,
        lp: LearnParams,
        pivot: Optional[Dict[str, torch.Tensor]] = None,
        data: Optional[Dict[str, torch.Tensor] | pd.DataFrame] = None,
    ) -> LearnParams:
        from .learning.continuous_mlp import materialize_lg_from_cont_mlp as _mat

        norm_pivot = (
            None
            if pivot is None
            else {
                k: (v if isinstance(v, torch.Tensor) else torch.as_tensor(v))
                for k, v in pivot.items()
            }
        )
        norm_data = None if data is None else self._normalize_fit_input(data)
        return _mat(lp, pivot=norm_pivot, data=norm_data)

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

    # ---- 1) canonical ingest -> tensordict ---------------------------------
    def _to_tensordict(
        self, data: TDLike, device: Optional[torch.device] = None
    ) -> TensorDict:
        if isinstance(data, TensorDict):
            td = data
        elif isinstance(data, pd.DataFrame):
            # only take columns that belong to the BN; preserve topo order
            cols = [k for k in self.variables if k in data.columns]
            if not cols:
                raise ValueError("DataFrame has no columns matching BN variables.")
            mapping = {k: torch.as_tensor(data[k].values) for k in cols}
            td = TensorDict(mapping, batch_size=[len(data)])
        elif isinstance(data, dict):
            mapping = {}
            n = None
            for k in self.variables:
                if k not in data:
                    continue
                t = torch.as_tensor(data[k])
                n = t.shape[0] if n is None else n
                assert t.shape[0] == n, f"Batch mismatch for key {k}"
                mapping[k] = t
            if not mapping:
                raise ValueError("No valid keys found to build a TensorDict.")
            td = TensorDict(mapping, batch_size=[n])
        else:
            raise TypeError(
                "Unsupported data type; use TensorDict, dict, or DataFrame."
            )

        dev = device or self.device
        out = {}
        for k, t in td.items():
            if self.types.get(k, "discrete") == "discrete":
                out[k] = t.to(dev).long()
            else:
                out[k] = t.to(dev).float()
        return TensorDict(out, batch_size=[next(iter(out.values())).shape[0]])

    # ---- 2) extend the in-memory dataset -----------------------------------

    @torch.no_grad()
    def add_data(
        self,
        new_data: TDLike,
        validate: bool = True,
        update_params: bool = False,
        **partial_fit_kwargs,
    ):
        """
        Append rows to the model's dataset.
        If update_params=True, also perform an incremental parameter update.
        """
        td_new = self._to_tensordict(new_data, device=self.device)
        if validate:
            self._validate_batch(td_new)

        # ensure buffers exist before optional online update
        self._ensure_incremental_buffers()

        if not hasattr(self, "data") or self.data is None:
            self.data = td_new.clone()
        else:
            self.data = TensorDict.cat([self.data, td_new], dim=0)

        if update_params:
            self.partial_fit(td_new, **partial_fit_kwargs)

    def _validate_batch(self, td: TensorDict):
        for var in self.variables:
            if var not in td:
                continue
            if self.types.get(var, "discrete") == "discrete":
                if td[var].dtype.is_floating_point:
                    raise ValueError(
                        f"{var}: expected integer tensor for discrete var."
                    )
                if self.cards and var in self.cards:
                    vmax = int(td[var].max().item())
                    assert (
                        vmax < self.cards[var]
                    ), f"{var}: value {vmax} ≥ card {self.cards[var]}"
            else:
                if not td[var].dtype.is_floating_point:
                    raise ValueError(
                        f"{var}: expected float tensor for continuous var."
                    )

    # ---- 3) incremental parameter update ("partial_fit") --------------------
    def partial_fit(
        self,
        td_new: Optional[TensorDict] = None,
        epochs: int = 1,
        batch_size: int = 8192,
        lr: Optional[float] = None,
    ):
        self._ensure_incremental_buffers()
        td_iter = self._batch_iterator(
            self.data if td_new is None else td_new, batch_size
        )
        for _ in range(epochs):
            for td_b in td_iter:
                self._update_discrete_sufficient_stats(td_b)
                self._update_continuous_sufficient_stats(td_b, lr=lr)
        self._finalize_incremental_updates()

    # ---- 4) helpers for incremental updates --------------------------------
    @torch.no_grad()
    def _update_discrete_sufficient_stats(self, td_b: TensorDict):
        for node, ntype in self.types.items():
            if ntype != "discrete" or node not in td_b:
                continue
            pars = self.parents.get(node, [])
            if any(p not in td_b for p in pars):
                continue
            y = td_b[node].view(-1).long()
            if pars:
                parent_cards = [self.cards[p] for p in pars]
                parent_vals = [td_b[p].view(-1).long() for p in pars]
                parent_idx = self._ravel_multi_index(parent_vals, parent_cards)
                counts = self.tabular_counts[node]
                Pcfg, card_y = counts.shape
                idx = parent_idx * card_y + y
                flat = torch.bincount(idx, minlength=Pcfg * card_y).view(Pcfg, card_y)
                self.tabular_counts[node] = counts + flat
            else:
                card_y = self.cards[node]
                flat = torch.bincount(y, minlength=card_y).to(
                    self.tabular_counts[node].dtype
                )
                self.tabular_counts[node] = self.tabular_counts[node] + flat

    @torch.no_grad()
    def _update_continuous_sufficient_stats(
        self, td_b: TensorDict, lr: Optional[float]
    ):
        for node, ntype in self.types.items():
            if ntype != "continuous" or node not in td_b:
                continue
            pars = self.parents.get(node, [])
            if any(p not in td_b for p in pars):
                continue
            y = td_b[node].float().view(-1, 1)
            X = (
                torch.cat(
                    [torch.ones_like(y)] + [td_b[p].float().view(-1, 1) for p in pars],
                    dim=1,
                )
                if pars
                else torch.ones_like(y)
            )
            self.XTX[node] += X.T @ X
            self.XTy[node] += X.T @ y
            self.yy[node] += (y.T @ y).squeeze()
            self.N[node] += X.shape[0]

    @torch.no_grad()
    def _finalize_incremental_updates(self, ridge: float = 1e-6):
        """
        Normalize discrete counts to probabilities; solve linear-Gaussian params.
        """
        # Discrete: normalize counts row-wise
        for node, ntype in self.types.items():
            if ntype == "discrete":
                C = self.tabular_counts[
                    node
                ]  # [Pcfg, card_y] or [card_y] if no parents
                if C.ndim == 1:
                    Z = C.sum().clamp_min(1.0)
                    self.tabular_probs[node] = C / Z
                else:
                    Z = C.sum(dim=1, keepdim=True).clamp_min(1.0)
                    self.tabular_probs[node] = C / Z

        # Continuous: solve (XTX + λI)β = XTy, then estimate noise σ^2
        for node, ntype in self.types.items():
            if ntype == "continuous":
                XTX = self.XTX[node]
                XTy = self.XTy[node]
                N = max(int(self.N[node]), 1)
                D = XTX.shape[0]
                beta = torch.linalg.solve(
                    XTX + ridge * torch.eye(D, device=XTX.device, dtype=XTX.dtype), XTy
                )  # [D, 1]
                self.lin_weights[node] = beta.squeeze(1)  # store

                # noise variance estimate (σ^2) using accumulated stats:
                # σ^2 = (yy - 2β^T XTy + β^T XTX β) / N
                yy = self.yy[node]
                s2 = (yy - 2 * (beta.T @ XTy).item() + (beta.T @ XTX @ beta).item()) / N
                self.lin_noise_var[node] = max(float(s2), 1e-9)

    # ---- 5) utilities -------------------------------------------------------
    @staticmethod
    def _ravel_multi_index(vals_list, dims):
        """
        Convert list of integer tensors [v0, v1, ...] with 0<=vk<dims[k]
        into a single flat index tensor, row-major.
        """
        idx = vals_list[0]
        for v, d in zip(vals_list[1:], dims[1:]):
            idx = idx * d + v
        return idx

    def _batch_iterator(self, td: TensorDict, batch_size: int):
        N = td.batch_size[0]
        for i in range(0, N, batch_size):
            j = min(i + batch_size, N)
            yield td[i:j]

    def _ensure_incremental_buffers(self):
        # discrete
        if not hasattr(self, "tabular_counts"):
            self.tabular_counts = {}
        if not hasattr(self, "tabular_probs"):
            self.tabular_probs = {}
        # continuous (linear-Gaussian)
        if not hasattr(self, "XTX"):
            self.XTX, self.XTy, self.yy, self.N = {}, {}, {}, {}
        if not hasattr(self, "lin_weights"):
            self.lin_weights, self.lin_noise_var = {}, {}
        # allocate per node if missing
        for node, ntype in self.types.items():
            if ntype == "discrete":
                pars = self.parents.get(node, [])
                if self.cards is None or node not in self.cards:
                    continue  # can’t size counts without card
                child_card = self.cards[node]
                if pars and self.cards:
                    pcfg = 1
                    for p in pars:
                        pcfg *= self.cards[p]
                    shape = (pcfg, child_card)
                else:
                    shape = (child_card,)
                if node not in self.tabular_counts:
                    self.tabular_counts[node] = torch.zeros(
                        *shape, device=self.device, dtype=torch.float64
                    )
            else:
                pars = self.parents.get(node, [])
                D = 1 + len(pars)  # affine
                if node not in self.XTX:
                    self.XTX[node] = torch.zeros(
                        (D, D), device=self.device, dtype=self.dtype
                    )
                    self.XTy[node] = torch.zeros(
                        (D, 1), device=self.device, dtype=self.dtype
                    )
                    self.yy[node] = torch.tensor(
                        0.0, device=self.device, dtype=self.dtype
                    )
                    self.N[node] = 0


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
