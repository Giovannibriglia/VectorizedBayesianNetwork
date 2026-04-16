from __future__ import annotations

from typing import Dict

import torch

from vbn.core.registry import register_update
from vbn.update.base_update import _resolve_node_update, BaseUpdatePolicy
from vbn.utils import concat_parents


@register_update("online_sgd")
class OnlineSGDUpdate(BaseUpdatePolicy):
    def update(
        self,
        vbn,
        data: Dict[str, torch.Tensor],
        lr: float = 1e-3,
        n_steps: int = 1,
        batch_size: int = 128,
        weight_decay: float = 0.0,
        **kwargs,
    ):
        # Update hyperparameters are pulled from each node's CPD 'update' config.
        for node in vbn.dag.topological_order():
            params = _resolve_node_update(vbn, node)
            parents = vbn.dag.parents(node)
            parent_tensor = concat_parents(data, parents)
            self._update_cpd(vbn.nodes[node], parent_tensor, data[node], params)
        return vbn.nodes

    def _update_cpd(self, cpd, parents, x, params):
        # Delegate updates to the CPD implementation so non-gradient CPDs/root
        # fast paths (e.g. cached logits) can update safely without autograd.
        cpd.update(parents, x, **params)
