# 03: Learning Dataset Generation

This step generates tabular datasets for training/fit (learning CPDs or model parameters) in the benchmark. It samples i.i.d. rows from each dataset's Bayesian network (when possible) and stores the resulting data alongside metadata describing schema, provenance, and reproducibility.

## CLI Usage

Generate bnlearn data (parquet preferred):

```bash
python -m benchmarking.scripts.03_generate_data \
    --generator bnlearn \
    --n_samples 100000 \
    --seed 0
```

Specify a strategy and pass generator kwargs:

```bash
python -m benchmarking.scripts.03_generate_data \
    --generator bnlearn \
    --n_samples 50000 \
    --seed 123 \
    --generation_strategy default \
    --generator-kwargs '{"batch_size": 4096}'
```

You can also pass repeatable key/value pairs (overrides JSON):

```bash
python -m benchmarking.scripts.03_generate_data \
    --generator bnlearn \
    --n_samples 10000 \
    --seed 7 \
    --kw batch_size=2048
```

## Generator Structure

Data generators live under:

- `benchmarking/03_data_generation/base.py` (base interface + helpers)
- `benchmarking/03_data_generation/registry.py` (registry + decorator)
- `benchmarking/03_data_generation/<generator>.py` (per-generator logic)

Each generator implements:

```
generate(dataset_id, dataset_dir, out_dir, meta_dir, logger) -> DataGenResult
```

## Output Layout

Generated data and metadata follow the `{generator}/{problem}` convention:

```
benchmarking/data/datasets/<generator>/<problem>/
  data_<strategy>_n<n_samples>_seed<seed>.parquet   (preferred)
  data_<strategy>_n<n_samples>_seed<seed>.csv       (fallback)
  data_<strategy>_n<n_samples>_seed<seed>.pkl       (fallback)
benchmarking/data/datasets/log/<generator>/
  <problem>_seed<seed>.log
benchmarking/data/metadata/<generator>/<problem>/
  domain.json
  data_generation.json
```

## Data Schema Rules

- Columns correspond to BN variables.
- Values are numeric.
  - Discrete variables are stored as integer codes (0..K-1 or per `domain.json`).
  - Continuous variables (if supported) are stored as floats.
- Domain mappings are stored in `domain.json` and referenced in `data_generation.json`.
- `data_generation.json` also records column names, dtypes, domains, and code maps.

## Reproducibility

- Deterministic output given the same generator, seed, `n_samples`, and strategy.
- `data_generation.json` stores the output path, SHA-256 hash, and full schema.
- Per-dataset logs capture generation details and any skip reasons.

## Limitations

- bnlearn CLG/gaussian problems without BIF are skipped (no BN sampling available).
- Continuous variables inside BIF are not supported yet.

## Relationship to Query Generation

This dataset is the learning/fit input for later benchmark steps. Queries generated in `02_generate_benchmark_queries.py` rely on the same `domain.json` so training data and query payloads share a consistent encoding.
