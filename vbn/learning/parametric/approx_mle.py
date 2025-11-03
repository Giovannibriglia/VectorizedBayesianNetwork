from typing import Any, Dict, List

import torch
from torch import nn

from vbn.learning import BaseCPDModule


class CategoricalCPD(BaseCPDModule):
    def __init__(
        self,
        node_name: str,
        parents: List[str],
        node_config: Dict[str, Any],
        parent_configs: Dict[str, Any],
        device: torch.device,
    ):
        super().__init__(node_name, parents, node_config, parent_configs, device)
        cardinality = self.node_config.get("card", 2)

        # --- FIX: Use the correct input dimension calculation ---
        input_dim = 0
        if self.parents:
            for p in self.parents:
                parent_config = self.parent_configs[p]

                # If discrete, input dimension is the cardinality (for one-hot encoding)
                if "card" in parent_config:
                    input_dim += parent_config["card"]

                # If continuous, input dimension is the dimension (usually 1)
                else:
                    input_dim += parent_config.get("dim", 1)  # Default to 1
        else:
            input_dim = 1
        # --------------------------------------------------------

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, cardinality)
        ).to(device)

    def forward(
        self, parent_data: Dict[str, torch.Tensor], target_data: torch.Tensor
    ) -> torch.Tensor:

        input_features = []

        if self.parents:
            for p in self.parents:
                parent_config = self.parent_configs[p]
                p_data = parent_data[p]

                # If discrete, use one-hot encoding for proper input to MLP
                if "card" in parent_config:
                    cardinality = parent_config["card"]
                    # Convert (B, 1) indices to (B, card) one-hot vector
                    one_hot = nn.functional.one_hot(
                        p_data.long().squeeze(-1), num_classes=cardinality
                    ).float()
                    input_features.append(one_hot)

                # If continuous (Gaussian/KDE), use the raw value
                else:
                    input_features.append(p_data)

            x = torch.cat(input_features, dim=-1)
        else:
            # Unconditioned node: use a dummy feature (N x 1)
            x = torch.ones_like(target_data, dtype=torch.float32)

        logits = self.mlp(x)  # Should now be (200 x 3) @ (3 x 32) -> (200 x 32)
        # ... (rest of the NLL calculation) ...

        target_indices = target_data.long().squeeze(-1)
        loss = nn.functional.cross_entropy(logits, target_indices, reduction="none")
        return loss

    def log_prob(
        self, parent_data: Dict[str, torch.Tensor], target_data: torch.Tensor
    ) -> torch.Tensor:

        input_features = []

        if self.parents:
            for p in self.parents:
                parent_config = self.parent_configs[p]
                p_data = parent_data[p]

                # If discrete, use one-hot encoding
                if "card" in parent_config:
                    cardinality = parent_config["card"]
                    one_hot = nn.functional.one_hot(
                        p_data.long().squeeze(-1), num_classes=cardinality
                    ).float()
                    input_features.append(one_hot)

                # If continuous, use the raw value
                else:
                    input_features.append(p_data)

            x = torch.cat(input_features, dim=-1)
        else:
            x = torch.ones_like(target_data, dtype=torch.float32)

        logits = self.mlp(x)  # Now receives the correct 3-dimensional input!

        # LogSoftmax converts logits to log probabilities
        log_probs = nn.functional.log_softmax(logits, dim=-1)  # Shape: (B, Card)

        # Use gather to select the log-prob for the observed class
        target_indices = target_data.long().squeeze(-1).unsqueeze(-1)
        log_p = log_probs.gather(1, target_indices).squeeze(-1)
        return log_p

    def _calculate_batch_counts(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Computes the joint counts for a single batch and returns a tensor of shape table_shape.
        This implementation uses PyTorch advanced indexing for 1D and assumes no continuous parents.
        """
        target_data = data[self.node_name].long().squeeze(-1)

        if not self.parents:
            # Unconditioned case
            counts = torch.bincount(target_data, minlength=self.cardinality).float()
            return counts

        # Conditioned case: Calculate the linear index for joint states
        # This requires knowing the order of parents used in self.parent_cardinalities

        linear_index = target_data
        # Stride calculation: total_states_until_this_parent / current_parent_cardinality
        stride = self.cardinality

        for parent in reversed(self.parents):
            parent_index = data[parent].long().squeeze(-1)
            # linear_index = parent_index * stride + current_index (reversing the construction)

            linear_index += parent_index * stride

            # Update stride for the next parent
            stride *= self.parent_cardinalities[parent]

            # Total number of possible parent-target states
        total_states = stride

        # Use bincount to get the total counts for all joint states
        # Reshape the 1D counts into the N-dimensional table shape
        count_table = torch.bincount(linear_index, minlength=total_states).float()

        table_shape = [self.parent_cardinalities[p] for p in self.parents] + [
            self.cardinality
        ]
        return count_table.reshape(table_shape).to(self.device)

    """def update(self, data: Dict[str, torch.Tensor], node_configs: Dict[str, Any]):
        if self._log_cpd_table is None:
            # Ensure initial fit has run
            self.fit_exact(data, node_configs)
            return

        # 1. Calculate counts for the new batch
        new_counts_tensor = self._calculate_batch_counts(data)

        # 2. Add new counts to the existing table (self.count_table must be a parameter/buffer)
        # Use .data.add_() for in-place update if count_table is registered as a parameter/buffer
        self.count_table.data.add_(new_counts_tensor.to(self.count_table.device))

        # 3. Recalculate log_cpd_table (re-normalize)
        # Assume Laplace smoothing (alpha=1) was applied initially or is maintained.
        smoothed_counts = self.count_table.data

        # Normalize: log(Count / Sum(Count))
        log_sum = torch.logsumexp(smoothed_counts, dim=-1, keepdim=True)
        self._log_cpd_table = smoothed_counts.log() - log_sum"""
