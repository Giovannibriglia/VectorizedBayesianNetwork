from __future__ import annotations

from typing import Dict, Optional

import torch
from tqdm import tqdm

from vbn.config_cast import coerce_numbers, CPD_SCHEMAS, FIT_SCHEMA
from vbn.core.base import BaseCPD
from vbn.core.registry import CPD_REGISTRY, register_learning
from vbn.utils import concat_parents

FIT_KEYS = {
    "epochs",
    "lr",
    "batch_size",
    "weight_decay",
    "n_steps",
    "show_progress",
}


def _split_kwargs(config: Dict) -> tuple[Dict, Dict]:
    init_kwargs = {}
    fit_kwargs = {}
    for k, v in config.items():
        if k in FIT_KEYS:
            fit_kwargs[k] = v
        elif k != "cpd":
            init_kwargs[k] = v
    return init_kwargs, fit_kwargs


@register_learning("node_wise")
class NodeWiseLearner:
    def __init__(
        self,
        nodes_cpds: Optional[Dict[str, Dict]] = None,
        default_cpd: str = "softmax_nn",
        **kwargs,
    ):
        self.nodes_cpds = nodes_cpds or {}
        self.default_cpd = default_cpd
        self.default_fit_kwargs = {k: v for k, v in kwargs.items() if k in FIT_KEYS}
        self.default_init_kwargs = {
            k: v for k, v in kwargs.items() if k not in FIT_KEYS
        }

    def fit(self, vbn, data: Dict[str, torch.Tensor], **kwargs) -> Dict[str, BaseCPD]:
        nodes: Dict[str, BaseCPD] = {}
        for node in tqdm(vbn.dag.topological_order(), desc="Fitting CPDs"):
            parents = vbn.dag.parents(node)
            parent_tensor = concat_parents(data, parents)
            x = data[node]
            node_conf = dict(self.default_init_kwargs)
            node_fit_conf = dict(self.default_fit_kwargs)
            override = self.nodes_cpds.get(node, {})
            init_kwargs, fit_kwargs = _split_kwargs(override)
            node_conf.update(init_kwargs)
            node_fit_conf.update(fit_kwargs)
            cpd_name = override.get("cpd", self.default_cpd)
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
                node_conf = coerce_numbers(node_conf, CPD_SCHEMAS[key])
            node_fit_conf = coerce_numbers(node_fit_conf, FIT_SCHEMA)

            cpd = cpd_cls(
                input_dim=input_dim,
                output_dim=output_dim,
                device=vbn.device,
                seed=vbn.seed,
                **node_conf,
            )
            cpd.fit(parent_tensor, x, **node_fit_conf)
            nodes[node] = cpd
        return nodes
