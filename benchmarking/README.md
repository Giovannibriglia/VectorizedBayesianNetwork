# Benchmarking Framework

This benchmarking suite compares VBN and other probabilistic graphical model libraries across a curated set of Bayesian networks. It is designed as a staged, reproducible pipeline so each step can be validated independently and extended over time.

---

## 1. Data Download

**Purpose.** Download and prepare dataset artifacts from registered sources so downstream stages can reuse the same inputs.

**Supported sources.** Currently: bnlearn BIF repository.

**How to run.**

```bash
python -m benchmarking.scripts.01_download_data \
    --generator bnlearn
```

**Where data is stored.**

```
benchmarking/data/datasets/<generator>/<problem>/
benchmarking/data/metadata/<generator>/<generator>.json
```

**Metadata locations.**

- Static metadata shipped with the repo lives in `benchmarking/metadata/`.
- Generated/run metadata lives in `benchmarking/data/metadata/`.

Example output:

```
benchmarking/data/datasets/bnlearn/asia/
    model.bif
    dataset.json
```

Dataset IDs are the problem/network name (for bnlearn: `<network>`). The generator name is stored separately.

Developer guide for adding new downloaders: `benchmarking/scripts/01_download_data_readme.md`.
Query generation guide: `benchmarking/02_generate_benchmark_queries_readme.md`.
If you have legacy data folders named `<generator>__<problem>`, run `python -m benchmarking.migrate_data_layout`.

**Currently supported datasets (bnlearn).**

- asia (small)
- alarm (medium)
- hailfinder (large)
- andes (very_large)
- barley (stress)

---

## 2. Query Generation

**Purpose.** Generate deterministic benchmark query sets for each dataset under `benchmarking/data/datasets/`.

**How to run.**

```bash
python -m benchmarking.scripts.02_generate_benchmark_queries \
    --generator bnlearn \
    --seed 42 \
    --n_queries_cpds 64 \
    --n_queries_inference 128 \
    --generator-kwargs '{"n_mc": 32}'
```

**Where queries are stored.**

```
benchmarking/data/queries/<generator>/<problem>/queries.json
benchmarking/data/queries/log/<generator>/<problem>_seed<seed>.log
benchmarking/data/metadata/<generator>/<problem>/domain.json
```

**Query JSON schema (stable).**

```
{
  "dataset_id": "<problem>",
  "generator": "<name>",
  "seed": 42,
  "n_mc": 32,
  "generator_kwargs": {"n_mc": 32},
  "n_queries": {"cpds": 64, "inference": 128},
  "n_skeletons": {"inference": 4},
  "cpd_queries": [
    {"query_type": "cpd", "target": "X", "evidence_vars": ["A", "B"]}
  ],
  "inference_queries": [
    {
      "query_type": "inference",
      "task": "prediction",
      "target": "Y",
      "evidence": {"mode": "on_manifold", "vars": ["A"], "values": {"A": 0}},
      "skeleton_id": "<sha256>",
      "mc_id": 0
    }
  ],
  "coverage": {"cpds": {}, "inference": {}}
}
```

---

## Data Encoding (Preprocessing Helpers)

The benchmarking package provides one-hot encoding helpers for datasets with non-numeric variables. Encoding metadata is stored under:

```
benchmarking/data/metadata/<generator>/<problem>/encoding.json
```

The helper functions live in `benchmarking/datasets.py` and expose:

- `encode_dataframe(df)` for deterministic one-hot encoding.
- `decode_assignment(encoded_assignment, encoding_meta)` to map back to original variables.
- `lift_query(original_query, encoding_meta)` to map queries into encoded space.

The encoding pipeline uses a stable category ordering (sorted labels) and a deterministic column naming scheme: `<var>__<category>`. Missing values are mapped to the `<NA>` category.

---

## 3. Benchmark Setup (Coming Next)

This stage will configure model combinations, generate queries, run inference and sampling, and define metrics. The goal is to make setup fully declarative so new models and datasets can be added without changing core benchmarking code.

---

## 4. Running the Benchmark (Coming Next)

Planned entry point:

```bash
python -m benchmarking.scripts.02_run_all
```

This stage will support batch execution, automatic model combinations, and fit caching to avoid repeated training on identical datasets.

---

## 5. Summarizing Results (Coming Next)

Planned entry point:

```bash
python -m benchmarking.scripts.03_summarize
```

This stage will produce tables and plots, and split results into learning, CPD, inference, and sampling performance categories.

---

## 6. Current Benchmark Results

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
pytest benchmarking/tests/test_01_data_download.py -vv
```

---

## Why We Use `python -m`

All scripts are executed as modules (e.g., `python -m benchmarking.scripts.01_download_data`) to avoid `PYTHONPATH` and `sys.path` hacks. This makes execution consistent from the repo root and works cleanly with editable installs and CI. Registries remain in place because they provide extensibility: new generators or models can be added without changing core benchmarking code.
