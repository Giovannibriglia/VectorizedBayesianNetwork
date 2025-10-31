from __future__ import annotations

import random
from typing import Dict, List, Optional

import numpy as np
import torch

from .inference import INFERENCE_BACKENDS
from .learning import CPD_REGISTRY

from .learning.trainer import DictDataLoader, NodeSpec, VBNParallelTrainer

from .utils import topo_sort


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
        learner_map: Optional[Dict[str, str]] = None,
        default_batch_size: int = 2048,
        default_steps_svgp_kde: int = 500,
        default_steps_others: int = 0,
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
        self._learner_map = learner_map or {}
        self._trainer = None
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
        for n, k in mapping.items():
            if k not in CPD_REGISTRY:
                raise ValueError(f"Unknown CPD kind '{k}' for node '{n}'")
        self._learner_map.update(mapping)
        self._trainer = None

    def _default_kind_for(self, node: str) -> str:
        return (
            "mle" if self.nodes[node].get("type") == "discrete" else "linear_gaussian"
        )

    def _node_specs(self):
        specs = []
        for n in self.topo_order:
            info = self.nodes[n]
            kind = self._learner_map.get(n, self._default_kind_for(n))
            y_shape = int(
                info["card"] if info.get("type") == "discrete" else info.get("dim", 1)
            )
            specs.append(NodeSpec(n, kind, y_shape))
        return specs

    def _ensure_trainer(self):
        if self._trainer is None:
            self._trainer = VBNParallelTrainer(
                self, self._node_specs(), device=self.device
            )

    def fit(
        self,
        data: Dict[str, torch.Tensor],
        steps: Optional[int] = None,
        batch_size: Optional[int] = None,
        lr: Optional[float] = None,
        clip_grad: Optional[float] = None,
        shuffle: Optional[bool] = None,
    ) -> None:
        self._ensure_trainer()
        bs = batch_size or self._dl_defaults["batch_size"]
        lr = self._dl_defaults["lr"] if lr is None else lr
        cg = self._dl_defaults["clip_grad"] if clip_grad is None else clip_grad
        sh = self._dl_defaults["shuffle"] if shuffle is None else shuffle
        if steps is None:
            kinds = {s.kind for s in self._node_specs()}

            steps = (
                self._dl_defaults["steps_svgp_kde"]
                if any(k in {"kde", "gp_svgp"} for k in kinds)
                else self._dl_defaults["steps_others"]
            )
        self._trainer.fit(data)
        if steps and steps > 0:
            loader = DictDataLoader(data, batch_size=bs, shuffle=sh)
            self._trainer.train_minibatch(loader, steps=steps, lr=lr, clip_grad=cg)

    def partial_fit(self, data: Dict[str, torch.Tensor]) -> None:
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
