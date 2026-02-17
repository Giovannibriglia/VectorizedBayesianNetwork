# Benchmarking Framework

This benchmarking suite compares VBN and other probabilistic graphical model libraries across a curated set of Bayesian networks. It is designed as a staged, reproducible pipeline so each step can be validated independently and extended over time.

---

## 1. Data Generation

**Purpose.** Generate deterministic observational datasets from registered sources so downstream stages can reuse the same inputs.

**Supported sources.** Currently: bnlearn BIF repository.

**How to run.**

```bash
python -m benchmarking.scripts.01_generate_data \
    --generator bnlearn \
    --n_samples 10000
```

**Where data is stored.**

```
benchmarking/data/{generator}/
```

Example output:

```
benchmarking/data/bnlearn/
    asia.parquet
    alarm.parquet
    metadata.json
```

Developer guide for adding new generators: `benchmarking/scripts/01_generate_data_readme.md`.

**Currently supported datasets (bnlearn).**

- asia (small)
- alarm (medium)
- hailfinder (large)
- andes (very_large)
- barley (stress)

---

## 2. Benchmark Setup (Coming Next)

This stage will configure model combinations, generate queries, run inference and sampling, and define metrics. The goal is to make setup fully declarative so new models and datasets can be added without changing core benchmarking code.

---

## 3. Running the Benchmark (Coming Next)

Planned entry point:

```bash
python -m benchmarking.scripts.02_run_all
```

This stage will support batch execution, automatic model combinations, and fit caching to avoid repeated training on identical datasets.

---

## 4. Summarizing Results (Coming Next)

Planned entry point:

```bash
python -m benchmarking.scripts.03_summarize
```

This stage will produce tables and plots, and split results into learning, CPD, inference, and sampling performance categories.

---

## 5. Current Benchmark Results

To be updated after the first full benchmark run.

- Tables and plots will be published here.
- Comparisons will include VBN vs pgmpy and other baselines.
- Analysis will highlight speed vs accuracy tradeoffs.

---

## Running Benchmarking Tests

Run the benchmarking pipeline tests from the project root:

```bash
pytest benchmarking/ -vv
```

Optional (single stage):

```bash
pytest benchmarking/tests/test_01_data_generation.py -vv
```

---

## Why We Use `python -m`

All scripts are executed as modules (e.g., `python -m benchmarking.scripts.00_generate_data`) to avoid `PYTHONPATH` and `sys.path` hacks. This makes execution consistent from the repo root and works cleanly with editable installs and CI. Registries remain in place because they provide extensibility: new generators or models can be added without changing core scripts.
