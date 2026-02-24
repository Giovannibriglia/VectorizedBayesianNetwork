# 04: Run Benchmark

This step executes the benchmark by fitting registered models on the learning datasets and running the query workloads. It connects the earlier pipeline stages:

- `01_download_data`: downloads model artifacts (e.g., BIF) into `benchmarking/data/datasets/<generator>/<problem>/`.
- `02_generate_benchmark_queries`: creates CPD and inference query payloads under `benchmarking/data/queries/<generator>/<problem>/queries.json`.
- `03_generate_data`: generates training datasets under `benchmarking/data/datasets/<generator>/<problem>/data_*.{parquet,csv,pkl}`.

Step 04 orchestrates the full execution by combining data + queries + model implementations, producing benchmark outputs under `benchmarking/out/`.

The implementation is split into three layers:

- **Models abstraction** (`benchmarking/models/`): standardized model interface across libraries.
- **Generator-specific runners** (`benchmarking/IIII_run_benchmark/`): dataset/asset loading per generator (e.g., bnlearn).
- **CLI script** (`benchmarking/scripts/04_run_benchmark.py`): entry point and argument parsing.

---

## Architecture Overview

### 1) `benchmarking/models/`

This folder defines a minimal model API for benchmarking:

- `base.py`: `BaseBenchmarkModel` defines:
  - `fit(data_df)`
  - `answer_cpd_query(query)`
  - `answer_inference_query(query)`
  - `supports()`
- `registry.py`: model registry (`register_benchmark_model`, `get_benchmark_model`, `list_benchmark_models`).
- `vbn.py`: VBN implementation of the base interface.
- `config.py`: model benchmark config dataclasses (`ModelBenchmarkConfig`, `ComponentSpec`).
- `presets.py`: default config presets per model (VBN presets included).

**Per-query timing** is mandatory. Each query result must include:

```
{
  "ok": bool,
  "error": str | null,
  "timing_ms": float,
  "result": dict | null
}
```

For discrete targets, results must expose a probability vector aligned with integer-coded states (`0..K-1`).

---

### 2) `benchmarking/IIII_run_benchmark/`

Runner layer loads assets and executes models:

- `base.py`: `BaseBenchmarkRunner` orchestrates dataset loading, model initialization, fitting, query execution, and output writing.
- `bnlearn.py`: generator-specific runner for bnlearn datasets.

Responsibilities include:

- Locating assets (DAG/BIF, `domain.json`, datasets, `queries.json`).
- Deterministic ordering of problems, models, and queries.
- Recording per-query results and timings.
- Writing outputs under `benchmarking/out/`.
- Resolving model benchmark configs + writing config snapshots.

---

## CLI Usage

```bash
python -m benchmarking.scripts.04_run_benchmark \
  --generator bnlearn \
  --seed 0 \
  --models vbn:vbn_softmax_is,vbn:vbn_gauss_mcm
```

### Flags

- `--generator` (required): dataset generator name (e.g., `bnlearn`).
- `--seed` (required): seed used for deterministic dataset selection and model init.
- `--models` (required): comma-separated or repeatable list of models.
  - Example: `--models vbn,pgmpy` or `--models vbn --models pgmpy`.
  - To run multiple configs for the same model in one run, use aliases:
    - `--models vbn:vbn_softmax_is,vbn:vbn_gauss_mcm`
- `--config` (optional): model config preset selector.
  - Single value applies to all models (default: `default`).
  - Or per-model pairs: `model:config_id` (comma-separated).
- `--config-overrides` (optional): JSON dict of component overrides (learning/cpd/inference).
- `--model-kwargs` (optional): JSON dict forwarded to model constructors.
- `--max_problems` (optional): limit number of problems (debugging).
- `--store_full_query` (optional): store full query payloads in each JSONL record.

### Config Example

```bash
python -m benchmarking.scripts.04_run_benchmark \
    --generator bnlearn \
    --seed 0 \
    --models vbn \
    --config vbn_softmax_is \
    --config-overrides '{"vbn":{"inference":{"kwargs":{"n_particles":512}}}}'
```

### Multiple Configs In One Run

```bash
python -m benchmarking.scripts.04_run_benchmark \
    --generator bnlearn \
    --seed 0 \
    --models vbn:vbn_softmax_is,vbn:vbn_gauss_mcm
```

Override payload shape:

```json
{
  "vbn": {
    "learning": {"kwargs": {"batch_size": 2048}},
    "cpd": {"name": "softmax_nn", "kwargs": {"hidden_sizes": [128, 128]}},
    "inference": {"name": "importance_sampling", "kwargs": {"n_particles": 256}}
  }
}
```

Override rules:

- Only `learning`, `cpd`, `inference` keys allowed per model.
- If `name` changes, `key` becomes `"<component>:<name>"` unless explicitly provided.
  - Override keys may refer to the base model name (apply to all aliases) or the alias
    name (apply to that alias only).

---

## Output Structure

```
benchmarking/out/<generator>/benchmark_<timestamp>/
  cpds/<model>.jsonl
  inference/<model>.jsonl
  configs/<model>.json
  ground_truth_sources.json
  summary.json
  logs/run.log
```

Each query record includes:

- `model` metadata: model name, config id, component keys, config hash
- `query` payload (compact by default, full if `--store_full_query`)
- `ok`, `error`, `timing_ms`, `result`

Per-query `timing_ms` is always recorded. Output is deterministic given the same seed, dataset selection, and model configuration.

`ground_truth_sources.json` records paths to the per-dataset ground truth files generated during query generation so reporting can reuse them without copying.

---

## Extensibility Guide

### Add a New Model

1. Create `benchmarking/models/<model>.py` implementing `BaseBenchmarkModel`.
2. Register it via `register_benchmark_model`.
3. Run with `--models <model>`.

### Add a New Generator Runner

1. Create `benchmarking/04_run_benchmark/<generator>.py` inheriting `BaseBenchmarkRunner`.
2. Implement asset loading and register it via `register_benchmark_runner`.
3. Run with `--generator <generator>`.

### Add New Metrics Later

Metrics should consume the saved `cpds/<model>.json`, `inference/<model>.json`, and `summary.json` outputs without changing the runner. Add a post-processing step or metrics module that reads these outputs deterministically.
