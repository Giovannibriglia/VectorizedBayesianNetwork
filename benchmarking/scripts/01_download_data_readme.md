# Data Download (Step 01)

## Overview

`01_download_data` fetches and prepares dataset artifacts for a generator and writes them into a **benchmark bundle** under `benchmarking/data/benchmarks/`.

Bundle layout:

```
benchmarking/data/benchmarks/benchmark_<mode>_<timestamp>/
  metadata.json
  datasets/<generator>/<problem>/
    model.bif
    dataset.json
    download.json
```

`metadata.json` records generator, mode, seeds, dataset_ids, and download configuration.

## CLI

```bash
python -m benchmarking.scripts.01_download_data \
  --generator bnlearn \
  --mode cpds
```

Bundle selection:

- `--bundle_dir <path>` (explicit bundle folder)
- `--bundle <benchmark_cpds_YYYYMMDD_HHMMSS>` + `--bundle_root <root>`

If no bundle is provided, a new bundle is created under `benchmarking/data/benchmarks/`.

## Implementing a New Downloader

Create a new file:

```
benchmarking/I_data_download/download_<name>.py
```

Implement the class:

```python
from benchmarking.I_data_download.base import BaseDataDownloader
from benchmarking.I_data_download.registry import register_downloader

@register_downloader
class MySourceDownloader(BaseDataDownloader):
    name = "my_source"

    def download(self, datasets: list[str] | None = None, force: bool = False, **kwargs):
        # 1) Resolve dataset list
        # 2) Download/extract files into self.dataset_dir(dataset_id)
        # 3) Write dataset.json and download.json
        # 4) Call self.save_metadata(...)
        pass
```

Key requirements:

- Use `self.datasets_dir` and `self.dataset_dir(dataset_id)`.
- Do not generate sample data here (Step 03 handles that).
- Call `self.save_metadata(...)` to update the bundle metadata.

## Testing

```bash
pytest benchmarking/tests/test_01_data_download.py -vv
```

If you add a new downloader, extend the test fixture to create minimal local artifacts.
