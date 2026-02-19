from __future__ import annotations

from typing import Dict

import torch

from vbn.core.registry import register_update
from vbn.update.base_update import _resolve_node_update, BaseUpdatePolicy
from vbn.utils import concat_parents


@register_update("streaming_stats")
class StreamingStatsUpdate(BaseUpdatePolicy):
    def update(self, vbn, data: Dict[str, torch.Tensor], **kwargs):
        verbosity = kwargs.pop("verbosity", None)
        # Update hyperparameters are pulled from each node's CPD 'update' config.
        if verbosity is not None:
            kwargs["verbosity"] = verbosity
        for node in vbn.dag.topological_order():
            params = _resolve_node_update(vbn, node)
            if verbosity is not None:
                params["verbosity"] = verbosity
            parents = vbn.dag.parents(node)
            parent_tensor = concat_parents(data, parents)
            vbn.nodes[node].update(parent_tensor, data[node], **params)
        return vbn.nodes
