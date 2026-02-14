from __future__ import annotations

from typing import Optional

import torch

from vbn.core.base import CPDOutput
from vbn.core.utils import ensure_2d, ensure_tensor


class CPDHandle:
    def __init__(self, vbn, node: str) -> None:
        if node not in vbn.nodes:
            raise ValueError(f"Unknown node '{node}'.")
        self._vbn = vbn
        self._node = node
        self._cpd = vbn.nodes[node]
        self._parents = list(vbn.dag.parents(node))

    @property
    def node(self) -> str:
        return self._node

    @property
    def name(self) -> str:
        return self._node

    @property
    def cpd(self):
        return self._cpd

    @property
    def parents(self) -> list[str]:
        return list(self._parents)

    @property
    def x_dim(self) -> int:
        return int(self._cpd.output_dim)

    @property
    def parents_dim(self) -> int:
        return int(self._cpd.input_dim)

    def _parents_tensor(self, parents) -> Optional[torch.Tensor]:
        if self.parents_dim == 0:
            if parents is None:
                return None
            if isinstance(parents, dict) and not parents:
                return None
            if isinstance(parents, torch.Tensor):
                t = ensure_tensor(parents, device=self._vbn.device)
                if t.dim() == 2 and t.shape[-1] == 0:
                    return t
            raise ValueError(f"Node '{self._node}' has no parents.")

        if parents is None:
            raise ValueError(f"Parents required for node '{self._node}'.")

        if isinstance(parents, dict):
            tensors = []
            for parent in self._parents:
                if parent not in parents:
                    raise ValueError(
                        f"Missing parent '{parent}' for node '{self._node}'."
                    )
                t = ensure_tensor(parents[parent], device=self._vbn.device)
                t = ensure_2d(t)
                tensors.append(t)
            if not tensors:
                return None
            parents_tensor = torch.cat(tensors, dim=-1)
            if parents_tensor.shape[-1] != self.parents_dim:
                raise ValueError(
                    f"Expected parents_dim {self.parents_dim}, got {parents_tensor.shape[-1]}"
                )
            return parents_tensor

        if isinstance(parents, torch.Tensor):
            t = ensure_tensor(parents, device=self._vbn.device)
            if t.dim() == 1:
                t = ensure_2d(t)
            if t.dim() not in (2, 3):
                raise ValueError(
                    f"Expected parents with 2D or 3D shape, got {tuple(t.shape)}"
                )
            if t.shape[-1] != self.parents_dim:
                raise ValueError(
                    f"Expected parents_dim {self.parents_dim}, got {t.shape[-1]}"
                )
            return t

        raise TypeError("parents must be a tensor or dict")

    def _x_tensor(self, x) -> torch.Tensor:
        t = ensure_tensor(x, device=self._vbn.device)
        if t.dim() == 1:
            t = ensure_2d(t)
        if t.dim() not in (2, 3):
            raise ValueError(f"Expected x with 2D or 3D shape, got {tuple(t.shape)}")
        return t

    def sample(self, parents, n_samples: int) -> torch.Tensor:
        parents_tensor = self._parents_tensor(parents)
        samples = self._cpd.sample(parents_tensor, int(n_samples))
        return samples.detach()

    def log_prob(self, x, parents) -> torch.Tensor:
        x_tensor = self._x_tensor(x)
        parents_tensor = self._parents_tensor(parents)
        log_prob = self._cpd.log_prob(x_tensor, parents_tensor)
        return log_prob.detach()

    def pdf(self, x, parents) -> torch.Tensor:
        return torch.exp(self.log_prob(x, parents))

    def forward(self, parents, n_samples: int) -> CPDOutput:
        parents_tensor = self._parents_tensor(parents)
        output = self._cpd.forward(parents_tensor, int(n_samples))
        return CPDOutput(
            samples=output.samples.detach(),
            log_prob=output.log_prob.detach(),
            pdf=output.pdf.detach(),
        )
