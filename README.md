# VectorizedBayesianNetwork (VBN)

[![Tests](https://github.com/Giovannibriglia/VectorizedBayesianNetwork/actions/workflows/tests.yml/badge.svg)](https://github.com/Giovannibriglia/VectorizedBayesianNetwork/actions/workflows/tests.yml)
[![Coverage](https://img.shields.io/codecov/c/github/Giovannibriglia/VectorizedBayesianNetwork/main)](https://codecov.io/gh/Giovannibriglia/VectorizedBayesianNetwork)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen.svg)](https://pre-commit.com/)
[![License](https://img.shields.io/github/license/Giovannibriglia/VectorizedBayesianNetwork)](LICENSE)

Vectorized Bayesian Networks is a **continuous-only**, **torch-native** Bayesian Network library built for **batched learning, inference, and sampling**. The goal is a research-grade, extensible framework that stays differentiable end-to-end and scales to modern estimators.

## Philosophy
- Continuous variables only (no discrete special casing in core).
- Batched operations everywhere.
- PyTorch-first: all core math in torch.
- Global device invariant: `VBN(device=...)` is the single source of truth for device placement.
- Modular and registry-driven: CPDs, learning, inference, sampling, and update policies are pluggable.

## Implemented Methods
**CPDs**
- `softmax_nn`: neural Gaussian CPD
- `kde`: (conditional) Gaussian KDE CPD
- `mdn`: mixture density network CPD

**Learning**
- `node_wise`

**Inference**
- Monte Carlo marginalization
- Importance sampling
- SVGP (placeholder, MC fallback)

**Sampling**
- Ancestral sampling
- Gibbs sampling (simple conditional sampling)
- HMC (placeholder, ancestral fallback)

**Update policies**
- `streaming_stats`
- `online_sgd`
- `ema`
- `replay_buffer`

## Planned Methods
- Amortized learning
- Temporal/Dynamic DAGs
- Additional CPD families and inference backends

## Installation
```bash
pip install -e .
```

## Minimal Usage
```python
import networkx as nx
import torch
from vbn import VBN

G = nx.DiGraph()
G.add_edges_from([("feature_0", "feature_2"), ("feature_1", "feature_2")])

vbn = VBN(G, seed=0, device="cpu")

vbn.set_learning_method(
    method=vbn.config.learning.node_wise,
    nodes_cpds={
        "feature_0": {"cpd": "softmax_nn"},
        "feature_1": {"cpd": "softmax_nn"},
        "feature_2": {"cpd": "mdn", "n_components": 3},
    },
)

vbn.fit(df)

vbn.set_inference_method(vbn.config.inference.monte_carlo_marginalization, n_samples=200)
query = {
    "target": "feature_2",
    "evidence": {
        "feature_0": torch.tensor([[0.3]]),
        "feature_1": torch.tensor([[-0.2]]),
    },
}
pdf, samples = vbn.infer_posterior(query)

vbn.set_sampling_method(vbn.config.sampling.gibbs)
samples = vbn.sample(query, n_samples=200)

new_df = df.sample(64)
vbn.update(new_df, update_method="online_sgd", lr=1e-4, n_steps=5)
```

## Config Loading
Config YAMLs are packaged under `vbn/configs/**` and loaded via `importlib.resources` from the installed package.

## Repository Layout
```
vbn/
  vbn.py
  utils.py
  core/
  cpds/
  learning/
  inference/
  sampling/
  update/
  configs/
examples/
  01_basic_fit.py
  02_infer_posterior.py
  03_sampling.py
  04_update_online.py
tests/
```

## Testing
```bash
pytest -q
```

## Contribution Guidelines
- Keep all core computations torch-only.
- Respect the global device invariant.
- Register new components in the registries.
