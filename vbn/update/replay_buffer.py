from __future__ import annotations

from typing import Dict, Tuple

import torch

from vbn.core.registry import register_update
from vbn.update.base_update import _resolve_node_update, BaseUpdatePolicy
from vbn.utils import concat_parents


@register_update("replay_buffer")
class ReplayBufferUpdate(BaseUpdatePolicy):
    def __init__(self, max_size: int = 2000, replay_ratio: float = 0.5) -> None:
        self.max_size = int(max_size)
        self.replay_ratio = float(replay_ratio)
        self._buffer: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

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
            self._update_buffer(node, parent_tensor, data[node])
            parents_mix, x_mix = self._mix_with_replay(node, parent_tensor, data[node])
            self._online_update(vbn.nodes[node], parents_mix, x_mix, params)
        return vbn.nodes

    def _update_buffer(
        self, node: str, parents: torch.Tensor | None, x: torch.Tensor
    ) -> None:
        if parents is None:
            parents = torch.zeros(x.shape[0], 0, device=x.device)
        if node not in self._buffer:
            self._buffer[node] = (parents.clone(), x.clone())
            return
        p_buf, x_buf = self._buffer[node]
        p_buf = torch.cat([p_buf, parents], dim=0)
        x_buf = torch.cat([x_buf, x], dim=0)
        if p_buf.shape[0] > self.max_size:
            p_buf = p_buf[-self.max_size :]
            x_buf = x_buf[-self.max_size :]
        self._buffer[node] = (p_buf, x_buf)

    def _mix_with_replay(
        self, node: str, parents: torch.Tensor | None, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if parents is None:
            parents = torch.zeros(x.shape[0], 0, device=x.device)
        if node not in self._buffer:
            return parents, x
        p_buf, x_buf = self._buffer[node]
        if p_buf.shape[0] == 0:
            return parents, x
        n_replay = int(max(1, self.replay_ratio * x.shape[0]))
        idx = torch.randint(0, p_buf.shape[0], (n_replay,), device=x.device)
        p_replay = p_buf[idx]
        x_replay = x_buf[idx]
        return torch.cat([parents, p_replay], dim=0), torch.cat([x, x_replay], dim=0)

    def _online_update(self, cpd, parents, x, params):
        # Delegate to CPD update implementation to support replay with both
        # differentiable and closed-form/non-gradient CPDs.
        cpd.update(parents, x, **params)

    def get_state(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        return {"buffer": self._buffer}

    def set_state(self, state: Dict[str, Tuple[torch.Tensor, torch.Tensor]]) -> None:
        buffer = state.get("buffer", {})
        self._buffer = buffer
