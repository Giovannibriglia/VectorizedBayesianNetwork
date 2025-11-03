from __future__ import annotations

from typing import Any, Dict, Type

import torch
from torch import nn

from .base import BaseCPDModule

Tensor = torch.Tensor


class BNModule(nn.Module):
    # ... (Keep the definition as provided in the previous response) ...
    def __init__(
        self,
        dag,
        node_configs: Dict[str, Any],
        cpd_mapping: Dict[str, Type[BaseCPDModule]],
        device: torch.device,
    ):
        super().__init__()
        self.dag = dag
        self.node_configs = node_configs
        self.cpd_modules = nn.ModuleDict()

        # Initialize a CPD module for every node in the DAG
        for node in dag.nodes():
            parents = list(
                dag.predecessors(node)
            )  # Get the parents of the current node
            node_type = node_configs[node].get("type")

            if node_type not in cpd_mapping:
                raise ValueError(
                    f"No CPD module mapping found for node type: {node_type} of node {node}"
                )

            cpd_class = cpd_mapping[node_type]

            # --- START FIX: Prepare Parent Configurations ---

            # Create a dictionary containing the configurations (type, card/dim)
            # for all parent nodes.
            parent_configs = {parent: node_configs[parent] for parent in parents}

            # --- END FIX ---

            # Instantiate the specific CPD module, passing the parent_configs
            self.cpd_modules[node] = cpd_class(
                node_name=node,
                parents=parents,
                node_config=node_configs[node],
                parent_configs=parent_configs,  # Required for calculating input_dim
                device=device,
            )

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Computes the total Negative Log-Likelihood (NLL) for the entire network.
        This is the loss function for fitting.

        Args:
            data: A dict of node_name -> Tensor (batch of data for each node).

        Returns:
            A Tensor representing the total average NLL (loss).
        """

        if not self.cpd_modules:
            return torch.tensor(0.0, device=data[list(data.keys())[0]].device)

        # 1. ⚙️ Determine the correct device for the loss tensor
        # Get the first registered parameter from the first CPD module
        try:
            first_module = next(iter(self.cpd_modules.values()))
            module_device = next(first_module.parameters()).device
        except StopIteration:
            # Fallback if a CPD module has no parameters (unlikely in this context)
            module_device = data[list(data.keys())[0]].device

        # Initialize the total NLL on the determined device
        total_nll = torch.tensor(0.0, device=module_device)

        # 2. 🚀 Compute and aggregate NLL from all CPDs in parallel
        # This loop iterates over all nodes and their CPDs. Since each CPD
        # is independent, PyTorch computes them in parallel.
        for node in self.dag.nodes():
            cpd_module = self.cpd_modules[node]

            # Prepare parent data for the CPD module
            parent_data = {parent: data[parent] for parent in cpd_module.parents}

            # Compute the loss (NLL) for the current node's CPD
            # The CPD forward pass must return a NLL tensor (per-sample loss)
            node_nll = cpd_module(parent_data=parent_data, target_data=data[node])

            # Aggregate the loss: The BN's loss is the sum of the average NLLs
            # We use .mean() to average across the batch size (B) for stable training.
            total_nll += node_nll.mean()

        return total_nll
