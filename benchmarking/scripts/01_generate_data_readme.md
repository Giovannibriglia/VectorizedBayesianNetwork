# Data Generation Developer Guide

## A) Overview

`00_generate_data` is the entry point for creating observational datasets used by the benchmarking pipeline. It generates data for each selected network and stores it under a generator-specific folder:

```
benchmarking/data/{generator}/
    <network>.parquet
    metadata.json
```

The generator is responsible for:
- Producing a deterministic dataset (seeded).
- Writing parquet files for each network.
- Writing a machine-readable `metadata.json` describing variables and encodings.

**Metadata contract (minimum):**

```json
{
  "generator": "<name>",
  "n_samples": 10000,
  "seed": 42,
  "networks": ["asia", "alarm"],
  "timestamp": "...",
  "variables": {"asia": ["A", "B"], "alarm": ["..."]},
  "encoding": {"asia": {"A": {"a0": 0.0, "a1": 1.0}}, "alarm": {...}}
}
```

## B) Implementing a New Generator

### Step 1: Create a new file

```
benchmarking/data_generation/generate_<name>.py
```

### Step 2: Implement the generator class

```python
from benchmarking.data_generation.base import BaseDataGenerator
from benchmarking.data_generation.registry import register_generator

@register_generator
class MySourceGenerator(BaseDataGenerator):
    name = "my_source"

    def generate(self, n_samples: int, networks: list[str] | None = None, force: bool = False, **kwargs):
        # 1) Resolve networks list
        # 2) For each network:
        #    - if should skip and not force, continue
        #    - load source model
        #    - generate n_samples deterministically
        #    - write parquet to self.dataset_file(network)
        # 3) Build metadata and save via self.save_metadata(...)
        pass
```

Key requirements:
- Use `self.dataset_path` (created in `BaseDataGenerator`).
- Use deterministic sampling (seed + stable per-network seed).
- Parquet output per network.
- Metadata must include `variables` and `encoding` mappings.

### Step 3: Register the generator

Decorate the class with `@register_generator`. The registry auto-discovers modules under `benchmarking.data_generation` at import time. No changes are needed in the CLI script.

### Step 4: Run via the CLI

```bash
python -m benchmarking.scripts.00_generate_data --generator my_source --n_samples 10000
```

## C) Required Outputs

For each network:
- `<network>.parquet` in `benchmarking/data/<generator>/`
- `metadata.json` (single file for the generator)

The metadata must include:
- `generator`, `n_samples`, `seed`, `networks`, `timestamp`
- `variables` per network
- `encoding` map per network

## D) Testing Checklist

Run:

```bash
pytest benchmarking/ -vv
```

The Step‑01 test validates:
- Dataset folder creation
- Parquet file creation for selected networks
- Metadata contract
- Row count equals `n_samples`
- Determinism with the same seed
- Skip and force logic

If you add a new generator, update the Step‑01 test fixture to create a minimal local test dataset for your generator.

## E) Example: bnlearn

Minimal skeleton:

```python
@register_generator
class BNLearnGenerator(BaseDataGenerator):
    name = "bnlearn"

    def generate(self, n_samples: int, networks: list[str] | None = None, force: bool = False, **kwargs):
        # download BIFs, sample deterministically, write parquet, save metadata
        ...
```

Run:

```bash
python -m benchmarking.scripts.00_generate_data --generator bnlearn --n_samples 10000
```
