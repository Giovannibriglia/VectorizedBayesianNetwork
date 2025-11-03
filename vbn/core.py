from __future__ import annotations

import random
from typing import Dict, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .inference import INFERENCE_BACKENDS
from .learning import LEARNING_METHODS
from .learning.trainer import BNModule
from .sampling import SAMPLING_METHODS


class VBN:
    """
    nodes: Dict[name, {"type": "discrete"|"gaussian", "card"|"dim": int}]
    parents: Dict[name, List[parent_name]]
    cpd: Dict[name, CPD]
    """

    def __init__(
        self,
        dag: nx.DiGraph,
        device=None,
        seed: int | None = None,
    ):

        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dag = dag

        if seed is not None:
            self.seed = seed
            self.set_seed(seed)

        self._trainer: BNModule | None = None

        self._learning_methods = {}
        self._node_configs = None
        self._optimizer = None

        self._sampling_method = None
        self._sampling_obj = None

        self._inference_method = None
        self._inference_obj = None

    def set_seed(self, seed):
        self.seed = seed
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

    def set_learning_method(self, methods: Dict, **kwargs):
        """
        Defines the configuration (type, cardinality/dimension) for each node
        and initializes the main BNModule for parallel computation.

        Args:
            methods: Dict[node_name, {"type": "discrete"|"gaussian", "card"|"dim": int, ...}]
        """
        required_keys = ["type"]
        node_configs = {}

        for node in self.dag.nodes():
            if node not in methods:
                raise ValueError(
                    f"Node *{node}* has no learning method/configuration defined."
                )

            config = methods[node]

            # 1. Validate required keys
            for key in required_keys:
                if key not in config:
                    raise ValueError(
                        f"Node *{node}* configuration must include '{key}'."
                    )

            # 2. Validate type-specific keys (e.g., 'card' for discrete)
            node_type = config["type"].lower()
            if node_type == "discrete" and "card" not in config:
                raise ValueError(
                    f"Discrete node *{node}* requires 'card' (cardinality) in its configuration."
                )
            if node_type == "gaussian" and "dim" not in config:
                # Although unconditioned Gaussian usually has dim=1, it's good practice
                config["dim"] = config.get("dim", 1)

            node_configs[node] = config

        self._learning_methods = methods
        self._node_configs = node_configs

        # 3. Initialize the BNModule using the gathered configurations
        print("✅ All node configurations validated. Initializing BNModule...")
        try:
            self._trainer = BNModule(
                dag=self.dag,
                node_configs=self._node_configs,
                cpd_mapping=LEARNING_METHODS,
                device=self.device,
            )
            print("✨ BNModule created successfully. Ready for parallel training.")
        except Exception as e:
            print(f"❌ Error initializing BNModule: {e}")
            self._trainer = None
            raise

    def set_sampling_method(self, method: str, **kwargs):
        cls = SAMPLING_METHODS.get(method)
        if cls is None:
            raise ValueError(f"Unknown inference method: {method}")

        self._sampling_method = method
        self._sampling_obj = cls(device=self.device, **kwargs)

    def set_inference(self, method: str, **kwargs):
        cls = INFERENCE_BACKENDS.get(method)
        if cls is None:
            raise ValueError(f"Unknown inference method: {method}")

        self._inference_method = method
        self._inference_obj = cls(device=self.device, **kwargs)

    def _run_nondiff_fitting(self, data: Dict[str, torch.Tensor]) -> None:
        """
        Internal helper to run non-differentiable fitting (Exact MLE and KDE).
        This must be called BEFORE the gradient descent loop.
        """
        if self._trainer is None:
            raise RuntimeError(
                "BNModule not initialized. Call VBN.set_learning_method() first."
            )

        tensor_data = {
            k: v.to(self.device).unsqueeze(-1) if v.ndim == 1 else v.to(self.device)
            for k, v in data.items()
        }

        print("\n--- Auto-running Non-Differentiable CPD Fitting (Exact/KDE) ---")

        for node in self.dag.nodes():
            cpd_module = self._trainer.cpd_modules[node]

            # Check if the module has a non-differentiable fitting method
            if hasattr(cpd_module, "fit_exact"):
                print(f"-> Running Exact MLE for node: {node}")
                cpd_module.fit_exact(tensor_data, self._node_configs)

            elif hasattr(cpd_module, "fit_kde"):
                print(f"-> Storing data for KDE node: {node}")
                cpd_module.fit_kde(tensor_data)

    def _prepare_data(
        self, data: Dict[str, torch.Tensor] | pd.DataFrame
    ) -> Dict[str, torch.Tensor]:
        """
        Converts input data (DataFrame or dict of Tensors) into a standardized
        dictionary of Tensors, ensuring correct device placement and shape (N x 1).

        Args:
            data: The input training data.

        Returns:
            A dict of node_name -> torch.Tensor on the correct device.
        """

        if isinstance(data, pd.DataFrame):
            # 1. Handle Pandas DataFrame: Convert each column to a Tensor
            tensor_data = {}
            for col in data.columns:
                # Convert Series to Tensor
                tensor = torch.tensor(data[col].values, dtype=torch.float32)

                # Ensure shape is at least 2D (N x 1)
                if tensor.ndim == 1:
                    tensor = tensor.unsqueeze(-1)

                tensor_data[col] = tensor.to(self.device)

        elif isinstance(data, dict):
            # 2. Handle Dictionary of Tensors: Ensure correct device and shape
            tensor_data = {}
            for k, v in data.items():
                tensor = v.to(self.device).float()

                # Ensure shape is at least 2D (N x 1)
                if tensor.ndim == 1:
                    tensor = tensor.unsqueeze(-1)

                tensor_data[k] = tensor

        else:
            raise TypeError(
                "Input data must be a pandas DataFrame or a dictionary of torch.Tensors."
            )

        # Final check: Ensure all nodes in the DAG have corresponding data
        missing_nodes = [node for node in self.dag.nodes() if node not in tensor_data]
        if missing_nodes:
            raise ValueError(
                f"Missing data for required nodes in the DAG: {missing_nodes}"
            )

        return tensor_data

    def fit(
        self,
        data: Dict[str, torch.Tensor] | pd.DataFrame,
        epochs: int = 100,
        **kwargs,
    ) -> None:
        """
        Unified method for training the Bayesian Network.
        Automatically handles non-differentiable CPDs before starting gradient descent.
        """
        if self._trainer is None:
            raise RuntimeError("BNModule not initialized.")

        # 1. Prepare Data and check for non-differentiable requirements
        tensor_data = self._prepare_data(data)

        # 2. ⚡️ THE FIX: Run non-differentiable fitting ONCE ⚡️
        self._run_nondiff_fitting(tensor_data)
        # Now D_exact and E_kde are ready for the forward pass.

        # 3. Setup Optimizer for gradient-based nodes
        # Only train parameters that require gradients (GaussianCPD, CategoricalCPD, GPSVGPCPD)
        trainable_params = [p for p in self._trainer.parameters() if p.requires_grad]
        kwargs_optimizer = kwargs.get("optimizer", {})
        optimizer = torch.optim.Adam(trainable_params, **kwargs_optimizer)
        self._optimizer = optimizer

        # 4. Start Gradient Descent Loop
        pbar = tqdm(range(epochs), desc="Training Differentiable estimators...")
        for _ in pbar:
            optimizer.zero_grad()

            # BNModule forward pass runs parallel computation
            loss = self._trainer(data=tensor_data)

            # Backpropagation only affects gradient-based CPDs
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": loss.item()})

    def update(self, data: Dict[str, torch.Tensor] | pd.DataFrame, **kwargs) -> None:
        """
        Updates the CPD parameters with a new batch of data (online learning).
        """
        if self._trainer is None:
            raise RuntimeError("BNModule not initialized.")
        if not hasattr(self, "_optimizer"):
            # This handles cases where update() is called before fit()
            raise RuntimeError(
                "Optimizer not initialized. Call VBN.fit() first or initialize _optimizer."
            )

        tensor_data = self._prepare_data(data)

        # 1. Update non-differentiable CPDs
        self._run_nondiff_update(tensor_data)

        # 2. Perform one step of gradient descent for differentiable CPDs
        # --- FIX 2: Use the stored optimizer ---
        self._optimizer.zero_grad()
        loss = self._trainer(data=tensor_data)
        loss.backward()
        self._optimizer.step()
        # --------------------------------------

        # print(f"VBN Update completed. Batch Loss: {loss.item():.4f}")

    def _run_nondiff_update(self, data: Dict[str, torch.Tensor]) -> None:
        for node in self.dag.nodes():
            cpd_module = self._trainer.cpd_modules[node]

            # Only run update on modules that implement the .update() method (i.e., Exact and KDE)
            if hasattr(cpd_module, "update"):
                parents = list(self.dag.predecessors(node))

                # Create a data dict containing only the required tensors
                node_data = {p: data[p] for p in parents}
                node_data[node] = data[node]

                # Pass node_data and all node configurations (as some CPDs need parent configs)
                cpd_module.update(data=node_data, node_configs=self._node_configs)

    def get_log_prob(
        self,
        target_node: str,
        evidence: Optional[Dict[str, torch.Tensor]] = None,
        do: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Union[Dict[str, torch.Tensor], None]]:
        """
        Computes the Log Probability P(Target | Evidence, do(Intervention)).

        This method currently only returns the CPD for the target node,
        assuming the provided evidence/do-interventions cover all parents.
        For full inference (marginalization over unobserved parents),
        a separate engine is required.

        Args:
            target_node: The name of the node (X) for which to compute P(X | Pa_X).
            evidence: A dictionary of observed node states {node_name: tensor_data}.
                      Only needs to contain states for the PARENTS of the target_node.
            do: A dictionary of intervened node states {node_name: tensor_data}.
                             Treats intervened nodes as fixed parent values for any of their children.

        Returns:
            A tuple: (log_prob_tensor, parent_data_dict)
            log_prob_tensor: Log probability (Log-Prob or Log-NLL) of the target node,
                             conditioned on its parents (or do-interventions on parents).
            parent_data_dict: The final dictionary of parent states used for conditioning.
        """
        if self._trainer is None:
            raise RuntimeError(
                "BNModule not initialized. Run set_learning_method() first."
            )

        if target_node not in self._trainer.cpd_modules:
            raise ValueError(f"Target node '{target_node}' not found in BNModule.")

        cpd_module = self._trainer.cpd_modules[target_node]

        if cpd_module is None:
            raise ValueError(f"Target node '{target_node}' not found in BNModule.")

        # 1. Identify required parent nodes
        required_parents = list(self.dag.predecessors(target_node))

        # 2. Compile parent conditioning data, prioritizing 'do' over 'evidence'
        parent_data = {}

        # Check if we have the target data itself (for log_prob)
        target_data = None
        if evidence and target_node in evidence:
            target_data = self._standardize_tensor(evidence[target_node])

        for parent in required_parents:
            # Check for do-intervention first (stronger manipulation of the graph)
            if do and parent in do:
                # Intervention: Parent value is fixed by the 'do' operation
                parent_data[parent] = self._standardize_tensor(do[parent])

            # Check for observed evidence second
            elif evidence and parent in evidence:
                # Observation: Parent value is fixed by evidence
                parent_data[parent] = self._standardize_tensor(evidence[parent])

            else:
                # If the parent is not observed or intervened upon, we cannot
                # compute the conditional probability P(Target | Pa) fully.
                # In a full inference engine, this would trigger marginalization.
                raise ValueError(
                    f"Parent node '{parent}' of '{target_node}' is neither observed nor intervened upon. "
                    "Provide evidence/do for all parents to compute the CPD."
                )

        # 3. Compute Log Probability using the CPD module
        if target_data is not None:
            # Compute the specific log probability P(Target=t | Pa=pa)
            log_prob_tensor = cpd_module.log_prob(
                parent_data=parent_data, target_data=target_data
            )

        else:
            # Requesting the CPD itself (the full distribution)
            # Since log_prob requires target_data, we may need a separate cpd_dist() method
            # or rely on the module's forward method if it returns the full distribution.

            # For simplicity and robust integration with existing methods:
            raise NotImplementedError(
                "To get the full CPD, you need a specific 'cpd_dist' method in the CPD module. "
                "For now, only computing log_prob for observed target is supported. "
                "Please provide target_node's value in the 'evidence' dict."
            )

        return log_prob_tensor, parent_data

    # Assume self._standardize_tensor is a helper in VBN for device/dtype alignment.
    def _standardize_tensor(
        self, data: Union[torch.Tensor, pd.DataFrame]
    ) -> torch.Tensor:
        # Basic helper to convert input data to a standardized tensor format
        if isinstance(data, pd.DataFrame):
            data = torch.tensor(data.values, dtype=torch.float32)
        elif not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)

        return data.to(self.device).view(-1, 1)

    # ─────────────────────────────────────────────────────────────────────────
    # SAMPLING / INFERENCE (unchanged)
    # ─────────────────────────────────────────────────────────────────────────
    def sample(
        self,
        n_samples: int = 1024,
        do: Optional[Dict[str, torch.Tensor]] = None,
        method: str = "ancestral",
        **kw,
    ):
        from .sampling import sample as _sample

        return _sample(
            self, n=n_samples, do=do, method=method, device=self.device, **kw
        )

    def sample_conditional(
        self,
        evidence: Dict[str, torch.Tensor],
        n_samples: int = 1024,
        do: Optional[Dict[str, torch.Tensor]] = None,
        **kw,
    ):
        from .sampling import sample_conditional as _samplec

        # NOTE: pass n=..., not n_samples=...
        return _samplec(
            self, evidence=evidence, n=n_samples, do=do, device=self.device, **kw
        )

    def _is_linear_gaussian_node(self, n: str) -> bool:
        cpd = self.cpd.get(n)
        # Robust check: either class name or required attributes
        return (
            cpd is not None
            and hasattr(cpd, "W")
            and hasattr(cpd, "b")
            and hasattr(cpd, "sigma2")
        )

    def _vars_kind(self, vars_set):
        kinds = []
        for v in vars_set:
            s = self.nodes.get(v, {})
            t = s.get("type")
            if t == "discrete":
                kinds.append("discrete")
            elif t == "gaussian" and s.get("dim", 1) == 1:
                kinds.append("gaussian_scalar")
            else:
                kinds.append("other")
        return set(kinds)

    def posterior(
        self,
        query,
        evidence=None,
        do=None,
        n_samples: int = 4096,
        method: str | None = None,
        **kwargs,
    ):
        involved = set(query) | set(evidence or {}) | set(do or {})

        if method is None:
            kinds = self._vars_kind(involved)

            if kinds <= {"discrete"}:
                method = "ve"
            elif kinds <= {"gaussian_scalar"}:
                # NEW: only use GaussianExact if EVERY involved continuous node
                # that could appear in the Gaussian solve is truly linear-Gaussian.
                all_lg = all(
                    (self.nodes.get(v, {}).get("type") != "gaussian")
                    or self._is_linear_gaussian_node(v)
                    for v in involved
                )
                method = "gaussian" if all_lg else (self.inference_method or "lw")
            else:
                method = self.inference_method or "lw"

        if self._inference_obj is None or method != self.inference_method:
            self.set_inference(method, **kwargs)

        backend = self._inference_obj
        call_kw = dict(bn=self, query=query, evidence=evidence, do=do)
        if "n_samples" in backend.posterior.__code__.co_varnames:
            call_kw["n_samples"] = n_samples
        for k, v in kwargs.items():
            if k not in ("device",):
                call_kw[k] = v
        return backend.posterior(**call_kw)
