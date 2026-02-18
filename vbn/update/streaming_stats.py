from __future__ import annotations

from typing import Dict

import torch

from vbn.config_cast import coerce_numbers, UPDATE_SCHEMA
from vbn.core.registry import register_update
from vbn.update.base_update import BaseUpdatePolicy
from vbn.utils import concat_parents


@register_update("streaming_stats")
class StreamingStatsUpdate(BaseUpdatePolicy):
    def update(self, vbn, data: Dict[str, torch.Tensor], **kwargs):
        verbosity = kwargs.pop("verbosity", None)
        params = coerce_numbers(dict(kwargs), UPDATE_SCHEMA)
        if verbosity is not None:
            params["verbosity"] = verbosity
        for node in vbn.dag.topological_order():
            parents = vbn.dag.parents(node)
            parent_tensor = concat_parents(data, parents)
            vbn.nodes[node].update(parent_tensor, data[node], **params)
        return vbn.nodes
