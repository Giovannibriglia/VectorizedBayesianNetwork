from __future__ import annotations

from typing import Any, Dict, List

import torch

from ..base import BaseCPDModule


class KDECpd(BaseCPDModule):
    def __init__(
        self,
        node_name: str,
        parents: List[str],
        node_config: Dict[str, Any],
        parent_configs: Dict[str, Any],
        device: torch.device,
    ):
        super().__init__(node_name, parents, node_config, parent_configs, device)
        # ------------------------------------------
        self._data_buffer = None
        self._kde_bandwidth = 1.0

    def fit_kde(self, data: Dict[str, torch.Tensor]) -> None:
        """Stores the training data for later density estimation."""
        # For simplicity, we model P(X_i) here (unconditional)
        self._data_buffer = data[self.node_name].float().to(self.device)

    def forward(
        self, parent_data: Dict[str, torch.Tensor], target_data: torch.Tensor
    ) -> torch.Tensor:
        # Non-parametric, non-differentiable: does not participate in backprop training
        # We return a dummy loss of 0 to avoid errors in the BNModule.forward loop.
        return torch.zeros_like(target_data.squeeze(-1))

    def update(self, data: Dict[str, torch.Tensor], node_configs: Dict[str, Any]):
        """
        Appends new data to the internal buffer for kernel density estimation.
        """
        # Assuming _data_buffer holds all past training data for this node's KDE
        target_data = data[self.node_name]

        if self._data_buffer is None:
            self._data_buffer = target_data
        else:
            # Use torch.cat to append the new samples
            self._data_buffer = torch.cat([self._data_buffer, target_data], dim=0)

    def log_prob(
        self, parent_data: Dict[str, torch.Tensor], target_data: torch.Tensor
    ) -> torch.Tensor:
        if self._data_buffer is None:
            raise RuntimeError("KDE CPD must be fit_kde() before inference.")

        # Naive KDE implementation P(x) = 1/N * sum_i K(x, x_i)
        # 1. Calculate squared distance (B, N_data)
        N_samples = self._data_buffer.size(0)

        # Target (B, 1) - Buffer (N_data, 1) -> (B, N_data)
        diff = target_data.float().unsqueeze(1) - self._data_buffer.unsqueeze(0)

        # Gaussian kernel: K(x, x_i) = exp(-0.5 * (diff/h)^2) / (h * sqrt(2*pi))
        kernel_val = torch.exp(-0.5 * (diff / self._kde_bandwidth) ** 2)
        kernel_val /= self._kde_bandwidth * (2 * torch.pi) ** 0.5

        # Sum kernels and divide by N to get the density P(X)
        density = kernel_val.sum(dim=1) / N_samples

        # Log probability
        return torch.log(density + 1e-10)  # Add epsilon for stability
