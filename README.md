# VectorizedBayesianNetwork (VBN)

[![Tests](https://github.com/Giovannibriglia/VectorizedBayesianNetwork/actions/workflows/tests.yml/badge.svg)](https://github.com/Giovannibriglia/VectorizedBayesianNetwork/actions/workflows/tests.yml)
![GitHub stars](https://img.shields.io/github/stars/Giovannibriglia/VectorizedBayesianNetwork?style=social)
[![Coverage](https://img.shields.io/codecov/c/github/Giovannibriglia/VectorizedBayesianNetwork/main)](https://codecov.io/gh/Giovannibriglia/VectorizedBayesianNetwork)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen.svg)](https://pre-commit.com/)
[![License](https://img.shields.io/github/license/Giovannibriglia/VectorizedBayesianNetwork)](LICENSE)

Vectorized Bayesian Networks is a **continuous-first**, **torch-native** Bayesian Network library built for **batched learning, inference, and sampling**. The goal is a research-grade, extensible framework that stays differentiable end-to-end and scales to modern estimators.

## Philosophy
- Continuous variables by default, with optional binned categorical CPDs.
- Batched operations everywhere.
- PyTorch-first: all core math in torch.
- Global device invariant: `VBN(device=...)` is the single source of truth for device placement.
- Modular and registry-driven: CPDs, learning, inference, sampling, and update policies are pluggable.

## Implemented Methods
**Learning**
- `node_wise`
- `amortized` (in progress)

**CPDs**
- `gaussian_nn`: neural Gaussian CPD
- `linear_gaussian`: linear Gaussian CPD (ridge regression)
- `softmax_nn`: binned categorical CPD (softmax classifier)
- `kde`: (conditional) Gaussian KDE CPD
- `mdn`: mixture density network CPD


**Inference**
- `monte_carlo_marginalization`
- `importance_sampling`
- SVGP (placeholder, MC fallback)

**Sampling**
- `ancestral`
- `gibbs` (simple conditional sampling)
- HMC (placeholder, ancestral fallback)

**Update policies**
- `streaming_stats`
- `online_sgd`
- `ema`
- `replay_buffer`

## Planned Methods
- [methods] Causal inference: instrumental variables, counterfactuals, soft- and back-door adjustments
- [methods] Amortized learning and inference
- [methods] Temporal/Dynamic DAGs
- [methods] Additional CPD families and inference backends
- [methods] Handling missing values in training data
- [benchmarking] Introducing sensitive analysis on *n_queries*
- [benchmarking] New generators (rl, cv, protein...)
- [library] Saving to .onnx
- [library] pip install
- [library] installation with clone and with pip
- [library] check dependencies between vbn and benchmarking

## Installation
```bash
pip install -e .
python setup.py install
```

## Minimal Usage
```python
import networkx as nx
import torch
import pandas as pd
from vbn import VBN, defaults

def make_df(n, seed=0):
    gen = torch.Generator().manual_seed(seed)
    x0 = torch.randn(n, generator=gen)
    x1 = torch.randn(n, generator=gen)
    x2 = 0.5 * x0 - 0.2 * x1 + 0.1 * torch.randn(n, generator=gen)
    return pd.DataFrame(
        {"feature_0": x0.numpy(), "feature_1": x1.numpy(), "feature_2": x2.numpy()}
    )

G = nx.DiGraph()
G.add_edges_from([("feature_0", "feature_2"), ("feature_1", "feature_2")])

df = make_df(1000)

vbn = VBN(G, seed=0, device="cpu")

learning_conf = defaults.learning("node_wise")
vbn.set_learning_method(
    method=learning_conf,
    nodes_cpds={
        "feature_0": defaults.cpd("gaussian_nn"),
        "feature_1": defaults.cpd("gaussian_nn"),
        "feature_2": {**defaults.cpd("mdn"), "n_components": 3},
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
vbn.update(new_df, update_method="online_sgd")
```

## Per-CPD Training Hyperparameters
Training hyperparameters are defined per CPD under `fit` and `update`. The learning config is orchestration-only.

```python
from vbn import VBN, defaults

learning_conf = defaults.learning("node_wise")
nodes_cpds = {
    "x1": {
        **defaults.cpd("gaussian_nn"),
        "min_scale": 0.0001,
        "fit": {"epochs": 50, "batch_size": 256},
        "update": {"n_steps": 1, "batch_size": 256, "lr": 1e-3, "weight_decay": 0.0},
    },
    "x2": {
        **defaults.cpd("softmax_nn"),
        "binning": "uniform",
        "fit": {"epochs": 200, "batch_size": 128},
        "update": {"n_steps": 1, "batch_size": 128, "lr": 1e-3, "weight_decay": 0.0},
    },
    "y": {
        **defaults.cpd("kde"),
        "fit": {"epochs": 1, "batch_size": 1024},
        "update": {"n_steps": 1, "batch_size": 512, "lr": 1e-3, "weight_decay": 0.0},
    },
}
vbn.set_learning_method(method=learning_conf, nodes_cpds=nodes_cpds)
```

## Learner Update Policies
Update policies (`online_sgd`, `ema`, `replay_buffer`, `streaming_stats`) are learner-level scheduling/data policies used in `vbn.update(...)`.

- They only affect CPDs via the data passed into `cpd.update(...)` (batching, replay sampling, streaming stats),
- and by applying optimizer steps to CPD parameters in the update policy implementation (e.g., `online_sgd`, `ema`, `replay_buffer`).

Update hyperparameters (`lr`, `n_steps`, `batch_size`, `weight_decay`) are defined per CPD under `update` and are used by update policies. Update method configs only carry policy-level parameters (for example `alpha`, `max_size`, `replay_ratio`).

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
  __init__.py
  01_basic_fit.py
  02_infer_posterior.py
  03_sampling.py
  04_update_online.py
tests/
```

## Testing

Install the project in editable mode (ensures tests use local sources):

```bash
pip uninstall -y vbn
pip install -e ".[test]"
```

Quick Run:

```bash
pytest -q
```

Verbose (recommended for dev):

```bash
pytest -vv
```

`-q` = quiet, `-vv` = verbose (shows each test function).

Benchmarking-only tests:

```bash
pytest benchmarking/ -vv
```

With Coverage:

```bash
pytest --cov=vbn --cov-report=term-missing
```

Focus Single Test:

```bash
pytest tests/test_learning.py -vv
```

## Running Examples (Module Execution)
Run examples from the repo root using `python -m`:

```bash
python -m examples.03_sampling
```

## Benchmarking

See `benchmarking/README.md` for the full benchmarking workflow.
The reporting script now produces plots vs `n_nodes` and `n_edges` plus a flat results table with network sizes.

## Why We Use `python -m`
Module execution avoids `PYTHONPATH`/`sys.path` hacks, ensures consistent imports from the repo root, and works cleanly with editable installs and CI. Registries remain in place to keep the system extensible without modifying core scripts.

## Contribution Guidelines
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pre-commit install
```

Run formatting checks *before* pushing:

```bash
pre-commit run --all-files
```

- Keep all core computations torch-only.
- Respect the global device invariant.
- Register new components in the registries.
