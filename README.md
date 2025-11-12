# TODO
- temporal bn
- dynamic bn


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


## Quickstart

We provide three example scripts in `examples/`:

1. **01\_fit\_and\_infer.py** – learn discrete and continuous CPDs (MLE, MLP, linear-Gaussian) on synthetic data and run exact/approximate inference.
2. **02\_add\_data\_and\_refit.py** – update a fitted model with new data (DataFrame, dict, or TensorDict) and incrementally refit parameters.
3. **03\_save\_and\_load.py** – save learned parameters to disk, reload them, and plot CPDs from the saved model.

Run any example with:

```bash
python examples/01_fit_and_infer.py
```

---

## Repo Layout

```
examples/
  01_fit_and_infer.py
  02_add_data_and_refit.py
  03_save_and_load.py
vbn/
  core.py           # BNMeta, LearnParams, CausalBayesNet facade, merge_learnparams(...)
  utils.py          # vectorized helpers (factor ops, pivots, pdfs)
  io.py             # save/load LearnParams (TensorDict-friendly)
  plotting.py           # DAG, CPDs, posteriors, LG params, diagnostics

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


# Unified Class Hierarchy & Compatibility Map

```
                    ┌────────────────────────────────────────────────────────┐
                    │                CAPABILITY CONTRACT                    │
                    │  Capabilities{ has_sample, has_log_prob,              │
                    │                 is_reparameterized, supports_do }     │
                    └────────────────────────────────────────────────────────┘
                                       ▲
                                       │ (queried by samplers & inferencers)
                                       │
┌──────────────────────────┐           │           ┌──────────────────────────┐
│        LEARNING          │           │           │         SAMPLING         │
│  (models + trainers)     │           │           │   (per graph/topology)   │
├──────────────────────────┤           │           ├──────────────────────────┤
│ BaseConditionalModel     │───────────┼──────────▶│ BaseSampler              │
│  ├─ ParametricCPD        │ has_log_prob✓  sample✓│  ├─ ParametricSampler    │
│  │   (Categorical, Gauss)│ is_reparam✗            │  │  (ancestral, IS, LW, │
│  │                        │                       │  │   MCMC; log_prob✓)    │
│  ├─ DifferentiableCPD    │ has_log_prob✓  sample✓ │  ├─ DifferentiableSampler│
│  │   (Flows, MoG, NDE)   │ is_reparam✓            │  │  (reparam; VI-ready)  │
│  │                        │                       │  └─ ImplicitSampler      │
│  └─ ImplicitGenerator    │ has_log_prob✗  sample✓ │     (generator only;     │
│      (Sim/GAN/Diffusion) │ is_reparam✗            │      ABC/ratio-only)     │
├──────────────────────────┤           │           └──────────────────────────-┘
│ Trainers                 │           │                     │
│  ├─ MLE / MAP / EM       │           │                     ▼ samples (+weights?)
│  ├─ Bayesian (θ posterior)│          │        ┌──────────────────────────────┐
│  ├─ Variational (ELBO/VI)│           │        │            INFERENCE         │
│  └─ ABC / Ratio (LF)     │           │        │ (select by graph size & caps)│
└──────────────────────────┘           │        ├──────────────────────────────┤
                                       │        │ ExactFactorInference (VE/JT) │
                                       │  needs │  needs: has_log_prob✓        │
                                       │ logpdf │  out: exact marginals        │
                                       │        ├──────────────────────────────┤
                                       │        │ LoopyBeliefPropagation       │
                                       │        │  needs: has_log_prob✓        │
                                       │        │  out: approx marginals       │
                                       │        ├──────────────────────────────┤
                                       │        │ MonteCarloInference          │
                                       │        │  needs: sample✓ (+log_prob✓) │
                                       │        │  out: samples + weights      │
                                       │        ├──────────────────────────────┤
                                       │        │ VariationalInference (ELBO)  │
                                       │        │  needs: has_log_prob✓ &      │
                                       │        │         is_reparameterized✓  │
                                       │        │  out: qφ + amortized samples │
                                       │        ├──────────────────────────────┤
                                       │        │ LikelihoodFreeInference (ABC)│
                                       │        │  needs: sample✓ (no log_prob)│
                                       │        │  out: empirical posteriors   │
                                       │        ├──────────────────────────────┤
                                       │        │ SMC / Particle Filtering     │
                                       │        │  needs: sample✓ (+log_prob✓) │
                                       │        │  out: sequential posteriors  │
                                       │        └──────────────────────────────┘
```

## Compatibility Table (✓ works • ~ partial • ✗ no)

| Inference \ Learning         |      ParametricCPD      | DifferentiableCPD |        ImplicitGenerator       |
| ---------------------------- | :---------------------: | :---------------: | :----------------------------: |
| ExactFactor (VE/JT)          |            ✓            |         ✓         |                ✗               |
| Loopy BP                     |            ✓            |         ✓         |                ✗               |
| Monte Carlo (IS/LW/MH/Gibbs) |            ✓            |         ✓         | ~ *(no pdf → rejection/ratio)* |
| Variational (ELBO)           | ~ *(if differentiable)* |    **✓ ideal**    |                ✗               |
| Likelihood-Free (ABC/ratio)  |      ~ *(unneeded)*     |   ~ *(unneeded)*  |              **✓**             |
| SMC / Particle Filter        |            ✓            |         ✓         |          ~ *(ABC‑SMC)*         |

## Routing Heuristics (factory)

* If `has_log_prob` and low treewidth → **ExactFactor**.
* Else if `has_log_prob` and `is_reparameterized` → **Variational**.
* Else if `has_sample` and `has_log_prob` → **MonteCarlo**.
* Else if `has_sample` only → **Likelihood‑Free**.
* Temporal graphs → prefer **SMC** when available.

## Minimal Implementations

* **Models**: `ParametricCPD`, `DifferentiableCPD`, `ImplicitGenerator` (all subclass `BaseConditionalModel`).
* **Samplers**: `ParametricSampler`, `DifferentiableSampler`, `ImplicitSampler` (all subclass `BaseSampler`).
* **Inference**: `ExactFactorInference`, `MonteCarloInference`, `VariationalInference`, `LikelihoodFreeInference` (+ optional `SMCInference`).

## Notes

* Do‑operator: samplers and inferencers must support **edge‑cut + node‑override** (`D=d`).
* Evidence: implement **masking + weight accumulation** (log‑space) where relevant.
* GPU: prioritize **DifferentiableCPD + VI**, and **vectorized Monte Carlo**.
