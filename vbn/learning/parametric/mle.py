from typing import Any, Dict, List

import numpy as np
import torch

from vbn.learning import BaseCPDModule


class ExactCategoricalCPD(BaseCPDModule):
    """
    Computes the exact Maximum Likelihood Estimate (MLE) for a Categorical CPD
    using frequency counting (non-differentiable).
    This method should be called *outside* the PyTorch optimization loop (fit).
    """

    def __init__(
        self,
        node_name: str,
        parents: List[str],
        node_config: Dict[str, Any],
        parent_configs: Dict[str, Any],
        device: torch.device,
    ):
        super().__init__(node_name, parents, node_config, parent_configs, device)

        self.cardinality = self.node_config.get("card", 2)
        self.parent_cardinalities = {}

        # --- THE FIX: Register as persistent buffers ---
        self.register_buffer("_log_cpd_table", None)
        self.register_buffer("count_table", None)
        # ---------------------------------------------

    def fit_exact(
        self, data: Dict[str, torch.Tensor], parent_configs: Dict[str, Any]
    ) -> None:
        """
        Calculates the CPD table using frequency counting.
        """

        # 1. Gather parent cardinalities
        for parent in self.parents:
            self.parent_cardinalities[parent] = parent_configs[parent].get("card")

        # Convert data to numpy/CPU for efficient counting
        target_np = data[self.node_name].cpu().numpy().squeeze()
        parent_data_np = {p: data[p].cpu().numpy().squeeze() for p in self.parents}

        # 2. Determine the shape of the CPD table
        # Shape: (Card_Parent1, Card_Parent2, ..., Card_Target)
        table_shape = tuple(self.parent_cardinalities.values()) + (self.cardinality,)

        # 3. Initialize count table and apply Laplace smoothing (add 1)
        count_table = np.ones(table_shape, dtype=float)  # Laplace smoothing

        # 4. Perform Frequency Counting

        # Convert all discrete data to flat array of tuples/rows (N_samples, N_parents+1)
        # columns = self.parents + [self.node_name]
        data_combined = np.stack(
            [parent_data_np[p] for p in self.parents] + [target_np], axis=1
        )

        # Simple iterative counting (can be optimized with np.bincount or pd.groupby)
        for row in data_combined:
            # Create a tuple index (parent1_value, parent2_value, ..., target_value)
            idx = tuple(row.astype(int))
            count_table[idx] += 1

        # 5. Compute Probabilities (MLE)

        # Sum counts over the last dimension (the target node) to get marginal parent counts
        # This represents the denominator in the MLE formula: Count(Pa=pa)
        parent_marginal_counts = count_table.sum(axis=-1, keepdims=True)

        # Divide to get conditional probabilities: P(Target | Parents)
        cpd_table = count_table / parent_marginal_counts

        # 6. Store the results as log probabilities on the PyTorch device
        self._log_cpd_table = torch.tensor(np.log(cpd_table), dtype=torch.float32).to(
            self.device
        )

    def forward(
        self, parent_data: Dict[str, torch.Tensor], target_data: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the NLL using the pre-calculated, fixed CPD table.
        """
        if self._log_cpd_table is None:
            raise RuntimeError(
                "CPD table must be calculated via fit_exact() before calling forward."
            )

        # 1. Prepare parent indices
        # We need the parent data to look up the correct slice of the CPD table.

        # Stack parent data (B x N_parents)
        parent_values = [parent_data[p].long().squeeze(-1) for p in self.parents]

        if not parent_values:
            # Unconditioned case: P(X) - index the last dimension (Target)
            log_probs_slice = self._log_cpd_table
        else:
            # Conditional case: Use parent values to index the table dimensions
            # indices_tuple = (B, B, ...) where B is the batch index. This is complex indexing.

            # Simple approach: Reshape target to match log_probs_slice and use torch.gather
            # The indices for the look-up are the combined indices of the parents.

            # Use torch.index_select on the flattened parent states for simplified lookup

            # --- Robust Indexing using Parent States ---
            # 1. Compute the linear index (state) for each parent configuration in the batch.
            multipliers = []
            cumulative_multiplier = 1
            for parent in reversed(self.parents):
                multipliers.insert(0, cumulative_multiplier)
                cumulative_multiplier *= self.parent_cardinalities[parent]

            linear_parent_index = torch.zeros_like(parent_values[0])
            for idx, p_val in enumerate(parent_values):
                linear_parent_index += p_val * multipliers[idx]

            # 2. Flatten the CPD table to (N_Parent_States, Card_Target)
            flat_log_cpd = self._log_cpd_table.view(-1, self.cardinality)

            # 3. Look up the log probabilities for each sample's parent state
            log_probs_slice = flat_log_cpd.index_select(
                0, linear_parent_index
            )  # Shape: (B, Card_Target)

        # 2. Final NLL computation (Loss)

        # target_data must be indices (B, 1 or B)
        target_indices = target_data.long().squeeze(-1)

        # Use torch.gather to select the log-probability P(X=x | Pa=pa) for the observed sample
        # gather needs indices of shape (B, 1) to match log_probs_slice shape (B, C)
        log_prob = log_probs_slice.gather(1, target_indices.unsqueeze(-1)).squeeze(-1)

        # NLL is -log(P)
        nll = -log_prob
        return nll
