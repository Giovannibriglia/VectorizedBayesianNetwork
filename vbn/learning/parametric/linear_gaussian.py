# vbn/learning/parametric/linear_gaussian.py
from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn

from vbn.learning import BaseCPDModule

Tensor = torch.Tensor


class GaussianCPD(BaseCPDModule):
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
        dim = self.node_config.get("dim", 1)

        if self.parents:
            input_dim = sum(
                self.parent_configs[p].get("dim", self.parent_configs[p].get("card", 1))
                for p in self.parents
            )
            self.mean_layer = nn.Linear(input_dim, dim).to(self.device)
        else:
            # Unconditioned Gaussian: Mean is a learned constant
            self.mean_param = nn.Parameter(torch.zeros(dim, device=device))

        # Variance (Precision): Learned constant (log_std for stability)
        self.log_std = nn.Parameter(torch.zeros(dim, device=device))

    def _get_mean_and_std(
        self, parent_data: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if self.parents:
            # --- START FIX: Feature preparation for linear layer ---
            input_features = []
            for p in self.parents:
                parent_config = self.parent_configs[p]
                p_data = parent_data[p]

                # If discrete, use one-hot encoding
                if "card" in parent_config:
                    cardinality = parent_config["card"]
                    # Convert (B, 1) indices to (B, card) one-hot vector
                    one_hot = nn.functional.one_hot(
                        p_data.long().squeeze(-1), num_classes=cardinality
                    ).float()
                    input_features.append(one_hot)

                # If continuous, use the raw value
                else:
                    input_features.append(p_data)

            # Concatenate all prepared features
            x = torch.cat(input_features, dim=-1)  # Should now be (200 x 3)
            # --- END FIX ---

            mean = self.mean_layer(x)  # Conditional Mean
        else:
            mean = self.mean_param.expand(
                next(iter(parent_data.values())).size(0) if parent_data else 1, -1
            )

        std = torch.exp(self.log_std)  # Learned constant standard deviation
        return mean, std

    def forward(
        self, parent_data: Dict[str, torch.Tensor], target_data: torch.Tensor
    ) -> torch.Tensor:
        mean, std = self._get_mean_and_std(parent_data)

        # Compute NLL = -log(P) using the Normal distribution log_prob
        dist = torch.distributions.Normal(mean, std)

        # log_prob returns shape (B, Dim). We sum over the dimensions to get (B,)
        log_prob_tensor = dist.log_prob(target_data.float()).sum(dim=-1)
        nll = -log_prob_tensor
        return nll

    def log_prob(
        self, parent_data: Dict[str, torch.Tensor], target_data: torch.Tensor
    ) -> torch.Tensor:
        mean, std = self._get_mean_and_std(parent_data)
        dist = torch.distributions.Normal(mean, std)

        # Sum log-probs over the dimensions (Dim) for the final P(X|Pa)
        log_p = dist.log_prob(target_data.float()).sum(dim=-1)
        return log_p
