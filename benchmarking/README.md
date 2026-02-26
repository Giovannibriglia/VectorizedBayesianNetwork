# Benchmarking Framework

This benchmarking suite compares VBN and other probabilistic graphical model libraries across a curated set of Bayesian networks. It is designed as a staged, reproducible pipeline so each step can be validated independently and extended over time.

---

## Pipeline Overview

```
01_download_data
        ↓
02_generate_benchmark_queries
        ↓
03_generate_data
        ↓
04_run_benchmark
```

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
Query generation guide: `benchmarking/scripts/02_generate_benchmark_queries_readme.md`.
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
    --mode cpds \
    --seed 42 \
    --n_queries_cpds 64 \
    --generator-kwargs '{"n_mc": 1}'
```

For inference queries, switch to `--mode inference` and provide `--n_queries_inference`.

**Where queries are stored.**

```
benchmarking/data/queries/<generator>/<problem>/cpds.jsonl
benchmarking/data/queries/<generator>/<problem>/inference.jsonl
benchmarking/data/queries/<generator>/<problem>/queries.json
benchmarking/data/queries/<generator>/<problem>/ground_truth.jsonl
benchmarking/data/queries/log/<generator>/<problem>_seed<seed>.log
benchmarking/data/metadata/<generator>/<problem>/domain.json
```

Ground truth distributions are computed during query generation (pgmpy exact inference) and stored once per dataset in `ground_truth.jsonl`. The `queries.json` metadata records a pointer under `ground_truth.path` along with a status/reason if GT could not be computed.
The full query sets are stored in `cpds.jsonl` / `inference.jsonl`.

**Query metadata JSON schema.**

```
{
  "dataset_id": "<problem>",
  "generator": "<name>",
  "seed": 42,
  "n_mc": 32,
  "generator_kwargs": {"n_mc": 32},
  "n_queries": {"cpds": 64, "inference": 128},
  "n_skeletons": {"inference": 4},
  "queries": {
    "cpds": {"path": ".../cpds.jsonl", "count": 64},
    "inference": {"path": ".../inference.jsonl", "count": 128}
  },
  "coverage": {"cpds": {}, "inference": {}}
}
```

**Query JSONL schema (per line).**

```
{"query_type": "cpd", "target": "X", "evidence_vars": ["A", "B"], "...": "..."}
{"query_type": "inference", "task": "prediction", "target": "Y", "evidence": {"mode": "on_manifold", "vars": ["A"], "values": {"A": 0}}, "...": "..."}
```

---

## 3. Learning Data Generation

**Purpose.** Generate tabular datasets (i.i.d. samples) for model fitting / CPD learning.

**How to run.**

```bash
python -m benchmarking.scripts.03_generate_data \
    --generator bnlearn \
    --n_samples 10240 \
    --seed 0 \
```

**Where data is stored.**

```
benchmarking/data/datasets/<generator>/<problem>/data_<strategy>_n<n_samples>_seed<seed>.(parquet|csv|pkl)
benchmarking/data/datasets/log/<generator>/<problem>_seed<seed>.log
benchmarking/data/metadata/<generator>/<problem>/data_generation.json
benchmarking/data/metadata/<generator>/<problem>/domain.json
```

**Numeric coding for discrete variables.**

Discrete variables are stored as integer codes. The mapping from state label to code is stored in `domain.json` under each node’s `codes` mapping, and `data_generation.json` records the schema used for each generated file.

---

## 4. Run Benchmark

**Purpose.** Fit registered models on learning data and run CPD + inference query workloads.

**How to run.**

```bash
python -m benchmarking.scripts.04_run_benchmark \
    --generator bnlearn \
    --seed 0 \
    --mode cpds \
    --models vbn:vbn_linear_gauss_is,pgmpy:pgmpy_mle_ei
```

### Parameters

- `--generator` (required): dataset generator name (e.g., `bnlearn`).
- `--seed` (required): seed used for dataset selection and model init.
- `--mode` (required): `cpds` or `inference`.
- `--models` (required): comma-separated or repeatable list of models.
  - Example: `--models vbn,pgmpy` or `--models vbn --models pgmpy`.
  - To run multiple configs for the same model in one run, use aliases:
    - `--models vbn:vbn_softmax_is,vbn:vbn_gauss_mcm`
- `--config` (optional): model config preset selector.
  - Single value applies to all models (default: `default`).
  - Or per-model pairs: `model:config_id` (comma-separated).
- `--config-overrides` (optional): JSON dict of component overrides (learning/cpd/inference).
  - Keys can be the base model name (apply to all aliases) or the alias name (apply to one).
- `--model-kwargs` (optional): JSON dict forwarded to model constructors.
- `--max_problems` (optional): limit number of problems (debugging).
- `--store_full_query` (optional): store full query payloads in each JSONL record.

**Preset YAMLs**

Presets are defined in backend-specific YAML files:

- `benchmarking/models/presets_vbn.yaml`
- `benchmarking/models/presets_pgmpy.yaml`

Minimal schemas:

```yaml
# pgmpy.yaml
cpds:
  <preset_name>:
    cpds:
      estimator: <string>
      kwargs: {}
inference:
  <preset_name>:
    cpds:
      estimator: <string>
      kwargs: {}
    inference:
      method: <string>
      kwargs: {}
```

```yaml
# vbn.yaml
cpds:
  <preset_name>:
    learning:
      method: <string>
      kwargs: {}
    cpds:
      default:
        method: <string>
        kwargs: {}
      per_node:
        <node_name>:
          method: <string>
          kwargs: {}
inference:
  <preset_name>:
    learning:
      method: <string>
      kwargs: {}
    cpds:
      default:
        method: <string>
        kwargs: {}
      per_node:
        <node_name>:
          method: <string>
          kwargs: {}
    inference:
      method: <string>
      kwargs: {}
```

**Config example**

```bash
python -m benchmarking.scripts.04_run_benchmark \
    --generator bnlearn \
    --seed 0 \
    --mode inference \
    --models vbn \
    --config vbn_softmax_is \
    --config-overrides '{"vbn":{"inference":{"kwargs":{"n_particles":512}}}}'
```

**Multiple configs in one run**

```bash
python -m benchmarking.scripts.04_run_benchmark \
    --generator bnlearn \
    --seed 0 \
    --mode cpds \
    --models vbn:vbn_softmax_is,vbn:vbn_gauss_mcm
```

**Where outputs are stored.**

```
benchmarking/out/<generator>/benchmark_<mode>_<timestamp>/
  cpds/<model>.jsonl
  inference/<model>.jsonl
  run_metadata.json
  ground_truth_sources.json
  configs/<model>.json
  summary.json
  logs/run.log
```

Each query result includes model metadata (config id + component keys + config hash), the query payload (compact by default), `ok/error`, and `timing_ms`. Results are deterministic given the same seed and model configuration.

---

## 5. Report Results

**Purpose.** Join predictions with ground truth and generate summary tables/plots (KL/Wasserstein with robust IQM ± IQR-STD).

**How to run.**

```bash
python -m benchmarking.scripts.05_report_results \
    --run_dir benchmarking/out/<generator>/benchmark_<timestamp>
```

**Where outputs are stored.**

```
benchmarking/out/<generator>/benchmark_<timestamp>/report/
  tables/
  figures/
  report.md
```

See `benchmarking/scripts/05_report_results_readme.md` for full flag details and plotting outputs.

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

## 5. Benchmark Setup (Coming Next)

This stage will configure model combinations, generate queries, run inference and sampling, and define metrics. The goal is to make setup fully declarative so new models and datasets can be added without changing core benchmarking code.

---

## 6. Running the Benchmark (Coming Next)

Planned entry point:

```bash
python -m benchmarking.scripts.02_run_all
```

This stage will support batch execution, automatic model combinations, and fit caching to avoid repeated training on identical datasets.

---

## 7. Summarizing Results (Coming Next)

Planned entry point:

```bash
python -m benchmarking.scripts.03_summarize
```

This stage will produce tables and plots, and split results into learning, CPD, inference, and sampling performance categories.

---

## 8. Current Benchmark Results

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

## Folder Layout (Current)

```
benchmarking/data/
  datasets/
    <generator>/<problem>/...
    log/<generator>/<problem>_seed<seed>.log
  queries/
    <generator>/<problem>/cpds.jsonl
    <generator>/<problem>/inference.jsonl
    <generator>/<problem>/queries.json
    log/<generator>/<problem>_seed<seed>.log
  metadata/
    <generator>/<problem>/download.json
    <generator>/<problem>/domain.json
    <generator>/<problem>/data_generation.json
benchmarking/out/
  <generator>/benchmark_<mode>_<timestamp>/...
```

---

## Generator Kwargs

Both query and data generation accept generator-specific keyword arguments:

- Query generation (e.g., Monte Carlo samples):

```bash
python -m benchmarking.scripts.02_generate_benchmark_queries \
    --generator bnlearn \
    --mode inference \
    --n_queries_inference 128 \
    --seed 42 \
    --generator-kwargs '{"n_mc": 32}'
```

- Data generation (forwarded to the generator implementation):

```bash
python -m benchmarking.scripts.03_generate_data \
    --generator bnlearn \
    --n_samples 50000 \
    --seed 7 \
    --generator-kwargs '{"batch_size": 4096}'
```

---

## Why We Use `python -m`

All scripts are executed as modules (e.g., `python -m benchmarking.scripts.01_download_data`) to avoid `PYTHONPATH` and `sys.path` hacks. This makes execution consistent from the repo root and works cleanly with editable installs and CI. Registries remain in place because they provide extensibility: new generators or models can be added without changing core benchmarking code.
