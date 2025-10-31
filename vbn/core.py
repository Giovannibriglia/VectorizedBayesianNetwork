from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn

from .inference import INFERENCE_BACKENDS
from .learning import CPD_REGISTRY
from .learning.base import BaseCPD
from .utils import Tensor, topo_sort


class VBN:
    """Minimal BN core with *integrated* parallel learning.

    nodes: Dict[name, {"type": "discrete"|"gaussian", "card"|"dim": int}]
    parents: Dict[name, List[parent_name]]
    cpd: Dict[name, CPD]
    """

    def __init__(
        self,
        nodes: Dict[str, Dict],
        parents: Dict[str, List[str]],
        device=None,
        inference_method: str | None = None,
        seed: int | None = None,
        # ── NEW: learning knobs (optional) ─────────────────────────────────────
        learner_map: Optional[
            Dict[str, str]
        ] = None,  # node -> kind ("mle","linear_gaussian","kde","gp_svgp")
        default_batch_size: int = 2048,
        default_steps_svgp_kde: int = 500,  # run minibatch steps automatically if gp_svgp/kde appear
        default_steps_others: int = 0,  # keep 0 to preserve classic behavior for MLE/linear_gaussian
        default_lr: float = 1e-3,
        default_clip_grad: float = 1.0,
        default_shuffle: bool = True,
        **inf_kw,
    ):
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.nodes = nodes
        self.parents = parents
        self.topo_order = topo_sort({n: parents.get(n, []) for n in nodes})
        self.cpd: Dict[str, object] = {}
        self._inference_obj = None
        self.inference_method = inference_method

        # learning config
        self._learner_map = learner_map or {}  # node -> kind
        self._trainer: Optional[VBNParallelTrainer] = None
        self._dl_defaults = dict(
            batch_size=default_batch_size,
            steps_svgp_kde=default_steps_svgp_kde,
            steps_others=default_steps_others,
            lr=default_lr,
            clip_grad=default_clip_grad,
            shuffle=default_shuffle,
        )

        if seed is not None:
            self.seed = seed
            self.set_seed(seed)

        if inference_method is not None:
            self.set_inference(inference_method, **inf_kw)

    # ─────────────────────────────────────────────────────────────────────────
    # RNG / inference wiring
    # ─────────────────────────────────────────────────────────────────────────
    def set_seed(self, seed):
        self.seed = seed
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

    def set_inference(self, method: str = "lw", **kwargs):
        cls = INFERENCE_BACKENDS.get(method)
        if cls is None:
            raise ValueError(f"Unknown inference method: {method}")
        self._inference_obj = cls(device=self.device, **kwargs)
        self.inference_method = method
        return self

    # ─────────────────────────────────────────────────────────────────────────
    # CPD / learner wiring
    # ─────────────────────────────────────────────────────────────────────────
    def set_cpd(self, name: str, cpd) -> None:
        """Manual CPD injection (kept for back-compat)."""
        self.cpd[name] = cpd
        # If a trainer already exists, mirror the CPD into it next time we build.

    def set_learners(self, mapping: Dict[str, str]) -> None:
        """Override auto learner selection per node (kind ∈ CPD_REGISTRY)."""
        for node, kind in mapping.items():
            if kind not in CPD_REGISTRY:
                raise ValueError(f"Unknown CPD kind '{kind}' for node '{node}'.")
        self._learner_map.update(mapping)
        # Invalidate trainer so it can be rebuilt with the new mapping
        self._trainer = None

    def _default_kind_for(self, node: str) -> str:
        info = self.nodes[node]
        t = info.get("type", "gaussian")
        if t == "discrete":
            return "mle"
        # continuous
        return "linear_gaussian"

    def _node_specs(self) -> List[NodeSpec]:
        specs: List[NodeSpec] = []
        for n in self.topo_order:
            info = self.nodes[n]
            kind = self._learner_map.get(n, self._default_kind_for(n))
            if kind not in CPD_REGISTRY:
                raise ValueError(f"Node '{n}': unknown learner kind '{kind}'.")
            y_shape = int(
                info.get("card")
                if info.get("type") == "discrete"
                else info.get("dim", 1)
            )
            specs.append(NodeSpec(n, kind=kind, y_shape=y_shape))
        return specs

    def _ensure_trainer(self):
        if self._trainer is not None:
            return
        specs = self._node_specs()
        self._trainer = VBNParallelTrainer(self, specs, device=self.device)

    # ─────────────────────────────────────────────────────────────────────────
    # LEARNING — now *integrates* parallel init + (optional) minibatch training
    # ─────────────────────────────────────────────────────────────────────────
    def fit(
        self,
        data: Dict[str, torch.Tensor],
        steps: Optional[int] = None,
        batch_size: Optional[int] = None,
        lr: Optional[float] = None,
        clip_grad: Optional[float] = None,
        shuffle: Optional[bool] = None,
    ) -> None:
        """
        Learn all CPDs in parallel. Behavior:
          • Builds CPDs from nodes if missing (auto mapping).
          • Offline init on full data (always).
          • Then *optionally* runs joint minibatch optimization:
                - If `steps` is provided, use that.
                - Else, if any node uses 'gp_svgp' or 'kde', default to steps_svgp_kde.
                - Else, default to steps_others (0 by default for back-compat).
        """
        # If the user manually set *all* CPDs, keep them; else build trainer CPDs.
        self._ensure_trainer()

        # Good defaults
        batch_size = batch_size or self._dl_defaults["batch_size"]
        lr = lr if lr is not None else self._dl_defaults["lr"]
        clip_grad = (
            clip_grad if clip_grad is not None else self._dl_defaults["clip_grad"]
        )
        shuffle = shuffle if shuffle is not None else self._dl_defaults["shuffle"]

        # Decide a default `steps` if not explicitly given
        if steps is None:
            kinds = {spec.kind for spec in self._trainer.cpds.values()}
            # `kinds` is actually modules; grab their class-bound “kind” by re-deriving from specs:
            kinds = {s.kind for s in self._node_specs()}
            if any(k in {"gp_svgp", "kde"} for k in kinds):
                steps = int(self._dl_defaults["steps_svgp_kde"])
            else:
                steps = int(self._dl_defaults["steps_others"])

        # 1) Always offline initialize from the full batch
        self._trainer.fit(data)

        # 2) Optional joint minibatch optimization
        if steps and steps > 0:
            loader = DictDataLoader(data, batch_size=batch_size, shuffle=shuffle)
            self._trainer.train_minibatch(
                loader, steps=steps, lr=lr, clip_grad=clip_grad
            )

    def partial_fit(self, data: Dict[str, torch.Tensor]) -> None:
        """Streaming update (EMA/count updates, or small ELBO/NLL steps depending on CPD)."""
        self._ensure_trainer()
        self._trainer.partial_fit(data)

    # ─────────────────────────────────────────────────────────────────────────
    # SAMPLING / INFERENCE (unchanged)
    # ─────────────────────────────────────────────────────────────────────────
    def sample(
        self,
        n_samples: int = 1024,
        do: Optional[Dict[str, torch.Tensor]] = None,
        method: str = "ancestral",
        **kw,
    ):
        from .sampling import sample as _sample

        return _sample(
            self, n=n_samples, do=do, method=method, device=self.device, **kw
        )

    def sample_conditional(
        self,
        evidence: Dict[str, torch.Tensor],
        n_samples: int = 1024,
        do: Optional[Dict[str, torch.Tensor]] = None,
        **kw,
    ):
        from .sampling import sample_conditional as _samplec

        return _samplec(
            self, evidence=evidence, n=n_samples, do=do, device=self.device, **kw
        )

    def posterior(
        self,
        query,
        evidence=None,
        do=None,
        n_samples: int = 4096,
        method: str | None = None,
        **kwargs,
    ):
        if method is not None and (
            self._inference_obj is None or method != self.inference_method
        ):
            self.set_inference(method, **kwargs)
        elif self._inference_obj is None:
            self.set_inference("lw", **kwargs)

        backend = self._inference_obj
        call_kw = dict(bn=self, query=query, evidence=evidence, do=do)
        if "n_samples" in backend.posterior.__code__.co_varnames:
            call_kw["n_samples"] = n_samples
        for k, v in kwargs.items():
            if k not in ("device",):
                call_kw[k] = v
        return backend.posterior(**call_kw)


def build_cpd(
    name: str, kind: str, parents: Dict[str, int], y_shape: int, **kw
) -> BaseCPD:
    cls = CPD_REGISTRY[kind]
    return cls(
        name=name,
        parents=parents,
        **(
            {"card_y": y_shape}
            if kind == "mle"
            else {
                (
                    "out_dim" if kind in ("linear_gaussian", "gp_svgp") else "y_dim"
                ): y_shape
            }
        ),
        **kw,
    )


@dataclass
class NodeSpec:
    name: str
    kind: str  # one of CPD_REGISTRY keys
    y_shape: int  # card for discrete / dimension for continuous


class VBNParallelTrainer(nn.Module):
    def __init__(
        self,
        vbn,
        node_specs: List[NodeSpec],
        device: Optional[str | torch.device] = None,
    ):
        super().__init__()
        self.vbn = vbn
        self.device = torch.device(device or vbn.device)
        self.cpds = nn.ModuleDict()
        # build CPDs per node
        for spec in node_specs:
            par_names = vbn.parents.get(spec.name, [])
            par_dims: Dict[str, int] = {}
            for p in par_names:
                pinfo = vbn.nodes[p]
                if pinfo.get("type", "gaussian") == "discrete":
                    par_dims[p] = int(pinfo.get("card", 1))
                else:
                    par_dims[p] = int(pinfo.get("dim", 1))
            cpd = CPD_REGISTRY[spec.kind](
                name=spec.name,
                parents=par_dims,
                card_y=spec.y_shape if spec.kind == "mle" else None,
                out_dim=(
                    spec.y_shape
                    if spec.kind in ("linear_gaussian", "gp_svgp")
                    else None
                ),
                y_dim=spec.y_shape if spec.kind == "kde" else None,
                device=self.device,
            )
            self.cpds[spec.name] = cpd
            self.vbn.set_cpd(spec.name, cpd)
        self.to(self.device)

    def _split_parents(self, batch: Dict[str, Tensor], node: str) -> Dict[str, Tensor]:
        par = {}
        for p in self.vbn.parents.get(node, []):
            par[p] = batch[p]
        return par

    def nll(self, batch: Dict[str, Tensor]) -> Tensor:
        losses = []
        for node, cpd in self.cpds.items():
            par = self._split_parents(batch, node)
            y = batch[node]
            losses.append(-cpd.log_prob(y, par).mean())
        return torch.stack(losses).sum()

    @torch.no_grad()
    def fit(self, data: Dict[str, Tensor]) -> None:
        # initialize each CPD from full batch
        for node, cpd in self.cpds.items():
            par = self._split_parents(data, node)
            cpd.fit(par, data[node])

    @torch.no_grad()
    def partial_fit(self, data: Dict[str, Tensor]) -> None:
        for node, cpd in self.cpds.items():
            par = self._split_parents(data, node)
            cpd.update(par, data[node])

    def train_minibatch(
        self, iterator, steps: int = 1000, lr: float = 1e-3, clip_grad: float = 1.0
    ) -> None:
        self.train()
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        it = iter(iterator)
        for _ in range(steps):
            try:
                batch = next(it)
            except StopIteration:
                break
            # ensure device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            loss = self.nll(batch)
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad)
            opt.step()


class DictDataLoader:
    def __init__(
        self, data: Dict[str, Tensor], batch_size: int = 1024, shuffle: bool = True
    ):
        N = next(iter(data.values())).shape[0]
        self.data = data
        self.N = N
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        idx = torch.randperm(self.N) if self.shuffle else torch.arange(self.N)
        for s in range(0, self.N, self.batch_size):
            sel = idx[s : s + self.batch_size]
            yield {k: v[sel] for k, v in self.data.items()}
