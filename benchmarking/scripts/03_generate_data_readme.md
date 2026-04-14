# Learning Data Generation (Step 03)

## Overview

This step generates tabular datasets (i.i.d. samples) for model fitting and CPD learning. Outputs are written inside the selected bundle.

Bundle layout:

```
benchmarking/data/benchmarks/benchmark_<mode>_<timestamp>/
  datasets/<generator>/<problem>/
    data_<strategy>_n<n_samples>_seed<seed>.(parquet|csv|pkl)
    data_generation.json
    domain.json
```

Logs are written under:

```
benchmarking/data/benchmarks/benchmark_<mode>_<timestamp>/logs/datasets/<generator>/
```

## CLI

```bash
python -m benchmarking.scripts.03_generate_data \
  --generator bnlearn \
  --n_samples 10240 \
  --seed 0 \
  --bundle benchmark_cpds_YYYYMMDD_HHMMSS
```

Bundle selection:

- `--bundle_dir <path>` (explicit)
- `--bundle <benchmark_cpds_...>` + `--bundle_root <root>`

If no bundle is provided, the script uses the most recent bundle for the generator.
