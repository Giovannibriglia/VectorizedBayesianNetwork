from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch import nn

Tensor = torch.Tensor


class BaseCPDModule(nn.Module):
    def __init__(
        self,
        node_name: str,
        parents: List[str],
        node_config: Dict[str, Any],
        parent_configs: Dict[str, Any],
        device: torch.device,
    ):
        super().__init__()
        self.node_name = node_name
        self.parents = parents
        self.node_config = node_config
        self.parent_configs = (
            parent_configs  # Store it if derived classes need it later
        )
        self.device = device
        self.to(device)

    def forward(
        self, parent_data: Dict[str, torch.Tensor], target_data: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the Negative Log-Likelihood (NLL) for the target node given its parents' data.
        This is the LOSS function for training.
        Returns: A Tensor representing the per-sample average NLL (loss).
        """
        raise NotImplementedError(
            "Subclasses must implement the forward pass for loss computation."
        )

    def log_prob(
        self, parent_data: Dict[str, torch.Tensor], target_data: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the per-sample log-likelihood P(X_i | Pa(X_i)).
        This is the standardized output for INFERENCE.
        Returns: A Tensor of shape (Batch_size) containing log-probabilities.
        """
        raise NotImplementedError(
            "Subclasses must implement the log_prob method for inference."
        )
