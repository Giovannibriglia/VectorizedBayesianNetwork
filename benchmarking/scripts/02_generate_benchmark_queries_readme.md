# Benchmark Query Generation (Step 02)

## Overview

Query generation produces deterministic CPD or inference query sets for each dataset in a bundle.

Bundle layout:

```
benchmarking/data/benchmarks/benchmark_<mode>_<timestamp>/
  queries/<generator>/<problem>/
    cpds.jsonl | inference.jsonl
    queries.json
  ground_truth/<generator>/<problem>/ground_truth.jsonl
```

Logs are written under:

```
benchmarking/data/benchmarks/benchmark_<mode>_<timestamp>/logs/queries/<generator>/
```

## CLI

```bash
python -m benchmarking.scripts.02_generate_benchmark_queries \
  --generator bnlearn \
  --mode cpds \
  --seed 42 \
  --n_queries_cpds 64 \
  --generator-kwargs '{"n_mc": 32}' \
  --bundle benchmark_cpds_YYYYMMDD_HHMMSS
```

Required flags:

- `--generator`
- `--mode` (`cpds` or `inference`)
- `--n_queries_cpds` for `--mode cpds`
- `--n_queries_inference` for `--mode inference`

Bundle selection:

- `--bundle_dir <path>` (explicit)
- `--bundle <benchmark_cpds_...>` + `--bundle_root <root>`

If no bundle is provided, the script uses the most recent bundle for the generator/mode.

## Ground Truth

Exact ground truth distributions are computed during query generation and stored once per dataset:

```
ground_truth/<generator>/<problem>/ground_truth.jsonl
```

The `queries.json` file records the ground-truth path and status for downstream use.
