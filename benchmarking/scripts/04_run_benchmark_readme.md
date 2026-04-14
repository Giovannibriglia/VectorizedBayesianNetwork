# Run Benchmark (Step 04)

This step fits registered models on the generated data and runs CPD/inference query workloads from a selected bundle.

## Inputs (from bundle)

```
benchmarking/data/benchmarks/benchmark_<mode>_<timestamp>/
  datasets/<generator>/<problem>/...
  queries/<generator>/<problem>/cpds.jsonl|inference.jsonl
  ground_truth/<generator>/<problem>/ground_truth.jsonl
```

## CLI

```bash
python -m benchmarking.scripts.04_run_benchmark \
  --generator bnlearn \
  --seed 0 \
  --mode cpds \
  --bundle benchmark_cpds_YYYYMMDD_HHMMSS \
  --models vbn,pgmpy
```

Bundle selection:

- `--bundle_dir <path>` (explicit)
- `--bundle <benchmark_cpds_...>` + `--bundle_root <root>`

Use `--dry_run` to print the resolved bundle, models, and output dir without running.

## Output Structure

```
benchmarking/out/<generator>/benchmark_<mode>_<timestamp>/
  run_metadata.json
  configs/<model>.json
  logs/run.log
  errors/...
  results/<problem_id>/<method_id>.jsonl
  ground_truth_sources.json
  summary.json
```

Each JSONL record contains model metadata, compact query info, timing, and result/error fields. Errors are recorded per query and the run continues.

## Model Configuration

- `--models` accepts a comma-separated list or repeatable flag.
- Use aliases to run multiple configs for the same model: `--models vbn:vbn_softmax_is,vbn:vbn_gauss_mcm`.
- `--config` sets a default preset; per-model overrides are allowed via `model:config_id`.
- `--config-overrides` accepts JSON for component-level overrides.

Presets live in:

- `benchmarking/models/presets_vbn.yaml`
- `benchmarking/models/presets_pgmpy.yaml`
