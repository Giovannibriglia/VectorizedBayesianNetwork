# Data Download Developer Guide

## A) Overview

`01_download_data` is the entry point for fetching and preparing dataset artifacts used by the benchmarking pipeline. It downloads dataset files for each selected network and stores them under a dataset-specific folder:

```
benchmarking/data/datasets/<generator>/<problem>/
    model.bif
    dataset.json
```

The downloader is responsible for:
- Fetching/extracting dataset files (no simulation or sampling).
- Writing artifacts under `benchmarking/data/datasets/<generator>/<problem>/`.
- Writing run metadata under `benchmarking/data/metadata/<generator>/`.

**Generated metadata contract (minimum):**

```json
{
  "generator": "<name>",
  "seed": 42,
  "timestamp": "...",
  "datasets": [
    {
      "dataset_id": "<network>",
      "network": "asia",
      "generator": "<name>",
      "files": {"bif": "benchmarking/data/datasets/<generator>/<problem>/model.bif"}
    }
  ]
}
```

## B) Implementing a New Downloader

### Step 1: Create a new file

```
benchmarking/data_download/download_<name>.py
```

### Step 2: Implement the downloader class

```python
from benchmarking.data_download.base import BaseDataDownloader
from benchmarking.data_download.registry import register_downloader

@register_downloader
class MySourceDownloader(BaseDataDownloader):
    name = "my_source"

    def download(self, datasets: list[str] | None = None, force: bool = False, **kwargs):
        # 1) Resolve dataset list
        # 2) For each dataset:
        #    - download/extract files
        #    - write artifacts under benchmarking/data/datasets/<generator>/<problem>/
        #    - write dataset.json manifest (optional but recommended)
        # 3) Build and save generated metadata via self.save_metadata(...)
        pass
```

Key requirements:
- Use `self.datasets_dir` and `self.dataset_dir(dataset_id)`.
- Do not generate/simulate datasets in this step.
- Write generated metadata to `benchmarking/data/metadata/<generator>/`.

### Step 3: Register the downloader

Decorate the class with `@register_downloader`. The registry auto-discovers modules under `benchmarking.data_download` at import time. No changes are needed in the CLI script.

### Step 4: Run via the CLI

```bash
python -m benchmarking.scripts.01_download_data --generator my_source
```

## C) Required Outputs

For each dataset:
- Artifact files under `benchmarking/data/datasets/<generator>/<problem>/`
- Optional `dataset.json` manifest

Generated metadata:
- One JSON file in `benchmarking/data/metadata/<generator>/` (e.g., `benchmarking/data/metadata/<generator>/<generator>.json`)

## D) Testing Checklist

Run:

```bash
pytest benchmarking/ -vv
```

The Step‑01 test validates:
- Dataset folder creation
- Required artifacts present
- Generated metadata file creation
- Registry wiring
- CLI rejects the old `--n_samples` argument

If you add a new downloader, update the Step‑01 test fixture to create a minimal local dataset for your downloader.

## E) Example: bnlearn

Minimal skeleton:

```python
@register_downloader
class BNLearnDownloader(BaseDataDownloader):
    name = "bnlearn"

    def download(self, datasets: list[str] | None = None, force: bool = False, **kwargs):
        # download BIFs and write dataset manifests + generated metadata
        ...
```

Run:

```bash
python -m benchmarking.scripts.01_download_data --generator bnlearn
```
