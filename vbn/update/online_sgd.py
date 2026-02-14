from __future__ import annotations

from typing import Dict

import torch

from vbn.core.registry import register_update
from vbn.update.base_update import BaseUpdatePolicy
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
        for node in vbn.dag.topological_order():
            parents = vbn.dag.parents(node)
            parent_tensor = concat_parents(data, parents)
            self._update_cpd(
                vbn.nodes[node],
                parent_tensor,
                data[node],
                lr,
                n_steps,
                batch_size,
                weight_decay,
            )
        return vbn.nodes

    def _update_cpd(self, cpd, parents, x, lr, n_steps, batch_size, weight_decay):
        if not any(p.requires_grad for p in cpd.parameters()):
            raise NotImplementedError(
                "CPD has no trainable parameters for online SGD update"
            )
        if parents is None:
            parents = torch.zeros(x.shape[0], 0, device=cpd.device)
        dataset = torch.utils.data.TensorDataset(parents, x)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        optimizer = getattr(cpd, "_optimizer", None)
        if optimizer is None:
            optimizer = torch.optim.Adam(
                cpd.parameters(), lr=lr, weight_decay=weight_decay
            )
            cpd._optimizer = optimizer
        for _ in range(int(n_steps)):
            for batch_parents, batch_x in loader:
                log_prob = cpd.log_prob(batch_x, batch_parents)
                loss = -log_prob.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
