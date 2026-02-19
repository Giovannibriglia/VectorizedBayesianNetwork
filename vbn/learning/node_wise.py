from __future__ import annotations

import os
from typing import Dict, Optional

import torch
from tqdm import tqdm

from vbn.config_cast import coerce_numbers, CPD_SCHEMAS, FIT_SCHEMA, UPDATE_SCHEMA
from vbn.core.base import BaseCPD
from vbn.core.registry import CPD_REGISTRY, register_learning
from vbn.core.utils import resolve_verbosity
from vbn.utils import concat_parents

TRAINING_KEYS = {
    "epochs",
    "lr",
    "batch_size",
    "weight_decay",
    "n_steps",
    "max_grad_norm",
}
FIT_REQUIRED_KEYS = {"epochs", "batch_size"}
FIT_ALLOWED_KEYS = set(FIT_SCHEMA.keys()) - {"show_progress", "verbosity"}
UPDATE_ALLOWED_KEYS = set(UPDATE_SCHEMA.keys())
ORCHESTRATION_KEYS = {"show_progress", "verbosity"}


def _validate_learning_kwargs(kwargs: Dict) -> None:
    if not kwargs:
        return
    training_keys = sorted(set(kwargs) & TRAINING_KEYS)
    if training_keys:
        raise ValueError(
            "node_wise learning config cannot include training hyperparameters "
            f"({', '.join(training_keys)}). Move them into each node CPD config under "
            "'fit'/'update' (use defaults.learning('node_wise') for orchestration)."
        )
    unknown = sorted(set(kwargs) - ORCHESTRATION_KEYS)
    if unknown:
        raise ValueError(
            "node_wise learning config only supports orchestration keys "
            f"{sorted(ORCHESTRATION_KEYS)}. Unknown keys: {unknown}. "
            "Move CPD init and training parameters into each node CPD config."
        )


def _validate_node_conf(node: str, conf: Dict) -> None:
    if not isinstance(conf, dict):
        raise ValueError(
            f"CPD config for node '{node}' must be a dict with keys "
            "'cpd', 'fit', and optional 'update'."
        )
    # Reject legacy flat training keys; per-CPD 'fit'/'update' is required.
    legacy = sorted(set(conf) & (TRAINING_KEYS | UPDATE_ALLOWED_KEYS))
    if legacy:
        raise ValueError(
            f"Legacy training keys {legacy} are not allowed at the CPD top level for "
            f"node '{node}'. Use conf['fit'] and conf['update'] dicts instead."
        )
    if "cpd" not in conf:
        raise ValueError(f"CPD config for node '{node}' must include a 'cpd' key.")
    if "fit" not in conf:
        raise ValueError(f"CPD config for node '{node}' must include a 'fit' dict.")
    fit = conf.get("fit")
    if not isinstance(fit, dict):
        raise ValueError(f"CPD 'fit' config for node '{node}' must be a dict.")
    missing = sorted(FIT_REQUIRED_KEYS - set(fit))
    if missing:
        raise ValueError(
            f"CPD 'fit' config for node '{node}' is missing required keys: {missing}."
        )
    unknown_fit = sorted(set(fit) - FIT_ALLOWED_KEYS)
    if unknown_fit:
        raise ValueError(
            f"Unknown keys in CPD 'fit' config for node '{node}': {unknown_fit}. "
            f"Allowed keys: {sorted(FIT_ALLOWED_KEYS)}."
        )
    if "update" in conf and conf["update"] is not None:
        update = conf["update"]
        if not isinstance(update, dict):
            raise ValueError(f"CPD 'update' config for node '{node}' must be a dict.")
        unknown_update = sorted(set(update) - UPDATE_ALLOWED_KEYS)
        if unknown_update:
            raise ValueError(
                f"Unknown keys in CPD 'update' config for node '{node}': {unknown_update}. "
                f"Allowed keys: {sorted(UPDATE_ALLOWED_KEYS)}."
            )


@register_learning("node_wise")
class NodeWiseLearner:
    def __init__(
        self,
        nodes_cpds: Optional[Dict[str, Dict]] = None,
        default_cpd: str = "gaussian_nn",
        **kwargs,
    ):
        _validate_learning_kwargs(kwargs)
        self.nodes_cpds = nodes_cpds or {}
        self.default_cpd = default_cpd
        self.default_show_progress = kwargs.get("show_progress")
        self.default_verbosity = kwargs.get("verbosity")

    def fit(
        self,
        vbn,
        data: Dict[str, torch.Tensor],
        verbosity: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, BaseCPD]:
        if verbosity is None:
            verbosity = kwargs.pop("verbosity", None)
        else:
            kwargs.pop("verbosity", None)
        if "verbose" in kwargs:
            if verbosity is None:
                verbosity = kwargs.pop("verbose")
            else:
                kwargs.pop("verbose")
        show_progress = kwargs.pop("show_progress", None)
        if verbosity is None:
            verbosity = self.default_verbosity
        if kwargs:
            bad = sorted(set(kwargs) & TRAINING_KEYS)
            if bad:
                raise ValueError(
                    "node_wise.fit(...) does not accept global training hyperparameters "
                    f"({', '.join(bad)}). Move them into each node CPD config under "
                    "'fit'/'update'."
                )
            raise ValueError(
                f"Unknown node_wise.fit(...) arguments: {sorted(kwargs)}. "
                "Use per-CPD 'fit'/'update' dicts instead."
            )
        verbosity = resolve_verbosity(verbosity)
        if show_progress is None:
            show_progress = self.default_show_progress
        progress_enabled = verbosity > 0
        if show_progress is not None:
            progress_enabled = progress_enabled and bool(show_progress)
        if os.getenv("CI"):
            progress_enabled = False
        # Update policies are handled by VBN.update(...); node_wise only wires per-CPD fit configs.
        nodes: Dict[str, BaseCPD] = {}
        for node in tqdm(
            vbn.dag.topological_order(),
            desc="Fitting CPDs",
            disable=not progress_enabled,
        ):
            if node not in self.nodes_cpds:
                raise ValueError(
                    f"Missing CPD config for node '{node}'. Provide a per-node "
                    "config with 'cpd' and 'fit' keys."
                )
            node_conf = self.nodes_cpds.get(node, {})
            _validate_node_conf(node, node_conf)
            parents = vbn.dag.parents(node)
            parent_tensor = concat_parents(data, parents)
            x = data[node]
            init_kwargs = {
                k: v for k, v in node_conf.items() if k not in {"cpd", "fit", "update"}
            }
            fit_kwargs = dict(node_conf["fit"])
            cpd_name = node_conf.get("cpd")
            if isinstance(cpd_name, str):
                key = cpd_name.lower().strip()
                if key not in CPD_REGISTRY:
                    raise ValueError(
                        f"Unknown CPD '{cpd_name}'. Available: {list(CPD_REGISTRY.keys())}"
                    )
                cpd_cls = CPD_REGISTRY[key]
            else:
                cpd_cls = cpd_name
            output_dim = int(x.shape[-1])
            input_dim = int(parent_tensor.shape[-1]) if parent_tensor is not None else 0
            key = cpd_name.lower().strip() if isinstance(cpd_name, str) else None
            if key and key in CPD_SCHEMAS:
                init_kwargs = coerce_numbers(init_kwargs, CPD_SCHEMAS[key])
            fit_kwargs = coerce_numbers(fit_kwargs, FIT_SCHEMA)

            cpd = cpd_cls(
                input_dim=input_dim,
                output_dim=output_dim,
                device=vbn.device,
                seed=vbn.seed,
                **init_kwargs,
            )
            cpd.fit(parent_tensor, x, **{**fit_kwargs, "verbosity": verbosity})
            nodes[node] = cpd
        return nodes
