# VectorizedBayesianNetwork (VBN)

Fast, modular Bayesian Networks for **discrete** and **continuous** data — with **vectorized** exact & approximate inference, **neural CPDs**, `do(·)` interventions, plotting, and lightweight save/load.

* **Learners**

  * Discrete: **MLE (tabular)**, **Categorical MLP**
  * Continuous: **Linear-Gaussian (ridge)**, **Gaussian MLP** (mean/logvar)
* **Inference**

  * Discrete: **Exact** (variable elimination), **Approximate** (likelihood weighting)
  * Continuous: **Exact Gaussian** (canonical form), **Approximate** (vectorized sampling + LW)
* **Extras**

  * `do(...)` **interventions** in *all* inference backends
  * **Batched evidence** (vectorized over data points)
  * **TensorDict-compatible** save/load (`torch.save` / `torch.load` fallback)
  * **Plotting**: DAG, CPDs, marginals, Gaussian posteriors, LG params, sample diagnostics

---

## Table of Contents

* [Install](#install)
* [Why VBN?](#why-vbn)
* [Quickstart](#quickstart)
* [Repo Layout](#repo-layout)
* [Learning](#learning)
* [Inference (with `do`)](#inference-with-do)
* [Save / Load](#save--load)
* [Plotting](#plotting)
* [Advanced](#advanced)
* [Troubleshooting](#troubleshooting)
* [Citation](#citation)

---

## Install

```bash
  pip install -r requirements.txt
  python setup.py install
```

> GPU is auto-detected by PyTorch; all heavy ops are vectorized and GPU-friendly.

---

## Why VBN?

* **One pass per node, not per data point**: continuous exact inference uses canonical Gaussians (single Cholesky).
* **Vectorized**: batched evidences and sampled particles fly through the graph in parallel.
* **Neural CPDs**: categorical MLPs for discrete nodes; Gaussian MLPs (mean/logvar) for continuous nodes.
* **Interventions everywhere**: `do(...)` cuts edges and clamps values consistently across backends.

---

## Quickstart

```python
import networkx as nx, torch
from vbn.core import CausalBayesNet, merge_learnparams

# Graph
G = nx.DiGraph([("X","Y"), ("Z","Y"), ("Y","A"), ("X","A")])
types = {"X":"discrete","Z":"discrete","Y":"discrete","A":"continuous"}
cards = {"X":3, "Z":2, "Y":4}
bn = CausalBayesNet(G, types, cards)

# Data
N = 20_000
data = {
    "X": torch.randint(0,3,(N,)),
    "Z": torch.randint(0,2,(N,)),
    "Y": torch.randint(0,4,(N,)),
    "A": torch.randn(N) + 0.3,
}

# Learn
lp_disc = bn.fit_discrete_mle(data, laplace_alpha=0.5)       # tabular CPDs
lp_cont = bn.fit_continuous_mlp(data, epochs=10, hidden=64)  # Gaussian MLPs
lp_all  = merge_learnparams(lp_disc, lp_cont)

# Discrete exact (with do)
postY = bn.infer_discrete_exact(lp_disc,
                                evidence={"Z": torch.tensor(1)},
                                query=["Y"],
                                do={"X": torch.tensor(2)})

# Continuous exact Gaussian (linearize the MLPs → LG once)
lp_lg = bn.materialize_lg_from_cont_mlp(lp_cont, data=data)
mu, cov = bn.infer_continuous_gaussian(lp_lg,
                                       evidence={"A": torch.tensor(1.0)},
                                       query=["A"],
                                       do={"A": torch.tensor(0.5)})

# Continuous approximate (works directly on MLPs + discrete CPDs)
postA = bn.infer_continuous_approx(lp_all,
                                   evidence={"Z": torch.tensor(1)},
                                   query=["A"],
                                   do={"X": torch.tensor(2)},
                                   num_samples=4096)
```

---

## Repo Layout

```
vbn/
  core.py           # BNMeta, LearnParams, CausalBayesNet facade, merge_learnparams(...)
  utils.py          # vectorized helpers (factor ops, pivots, pdfs)
  io.py             # save/load LearnParams (TensorDict-friendly)
  plot.py           # DAG, CPDs, posteriors, LG params, diagnostics

  learning/
    discrete_mle.py       # Maximum-likelihood tabular CPDs (Laplace smoothing)
    discrete_mlp.py       # Categorical MLP CPDs (+ materialize to tables)
    gaussian_linear.py    # Linear-Gaussian (ridge) per-node regression
    continuous_mlp.py     # Gaussian MLP CPDs + linearization → LG

  inference/
    discrete_exact.py     # Variable elimination (batched evidence, supports do)
    discrete_approx.py    # Likelihood weighting (tables or discrete MLPs, supports do)
    continuous_gaussian.py# Canonical-form exact inference (supports do)
    continuous_approx.py  # Vectorized ancestral sampling + LW (supports do)
```

---

## Troubleshooting

* **PyTorch `.to(...)` TypeError**
  Always pass **named** args when moving tensors:
  `x = x.to(device=self.device, dtype=self.dtype)`

* **Broadcast / shape errors during exact inference**
  We align and insert singleton axes internally. If you still see errors, check that **cards** match your data ranges and that evidence values are within `0..card-1`.

* **Continuous approximate needs discrete CPDs**
  `infer_continuous_approx` must sample/weight discrete parents.
  Either `merge_learnparams(lp_disc, lp_cmlp)` or ensure `lp.discrete_mlps` is present (we fall back to MLPs if tables are missing).

* **Empty continuous query**
  Asking Gaussian inference for variables that aren’t continuous in your model returns empty results. Query only the continuous nodes present in `LGParams.order`.

---

## Citation
```
@article{
}
```
