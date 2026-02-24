from __future__ import annotations

import gzip
import shutil
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

from benchmarking.utils import get_static_metadata_dir, read_json
from .base import BaseDataDownloader
from .registry import register_downloader

# ----------------------------
# IO helpers
# ----------------------------


def download(url: str, dst: Path, force: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not force and dst.exists() and dst.stat().st_size > 0:
        return
    with urllib.request.urlopen(url) as r, open(dst, "wb") as f:
        shutil.copyfileobj(r, f)


def gunzip(src_gz: Path, dst_bif: Path, force: bool = False) -> None:
    dst_bif.parent.mkdir(parents=True, exist_ok=True)
    if not force and dst_bif.exists() and dst_bif.stat().st_size > 0:
        return
    with gzip.open(src_gz, "rb") as f_in, open(dst_bif, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


@register_downloader
class BNLearnDownloader(BaseDataDownloader):
    name = "bnlearn"
    test_datasets = ["asia", "cancer"]

    def download(
        self,
        datasets: list[str] | None = None,
        force: bool = False,
        **kwargs: Any,
    ) -> None:
        registry_path = get_static_metadata_dir(self.root_path) / "bnlearn.json"
        if not registry_path.exists():
            raise FileNotFoundError(f"Missing registry: {registry_path}")

        registry: Dict[str, Dict[str, Any]] = read_json(registry_path)

        if datasets:
            datasets = list(dict.fromkeys(datasets))
            missing = sorted(set(datasets) - set(registry))
            if missing:
                raise KeyError(f"Unknown network(s): {missing}")
            selected = list(datasets)
        else:
            selected = sorted(registry.keys())

        dataset_entries: List[dict] = []

        for network in self.progress(selected, desc="Downloading bnlearn"):
            meta = registry.get(network, {})
            urls = dict(meta.get("urls") or {})
            generated = meta.get("generated") or {}
            model_json = meta.get("model_json") or generated.get("model_json")
            for key in ("bif", "bif_gz", "net", "dsc", "rds", "rda"):
                if key in generated and key not in urls:
                    urls[key] = generated[key]
            if (
                model_json
                and "bif_gz" not in urls
                and str(model_json).endswith(".bif.gz")
            ):
                urls["bif_gz"] = model_json

            artifacts = set(meta.get("artifacts") or [])
            for key in sorted(urls):
                if key.startswith("bif"):
                    artifacts.add("bif")
                elif key == "net":
                    artifacts.add("net")
                elif key == "dsc":
                    artifacts.add("dsc")
                elif key == "rds":
                    artifacts.add("rds")
                elif key == "rda":
                    artifacts.add("rda")
            artifacts_list = sorted(artifacts)

            dataset_id = self.dataset_id(network)
            dataset_dir = self.dataset_dir(dataset_id)
            downloaded_files: Dict[str, str] = {}
            structure_available = False
            reason = None

            try:
                bif_gz_url = urls.get("bif_gz")
                bif_url = urls.get("bif")
                rds_url = urls.get("rds")
                rda_url = urls.get("rda")

                bif_error = None
                if bif_gz_url or bif_url:
                    try:
                        if bif_gz_url:
                            gz_path = dataset_dir / "model.bif.gz"
                            bif_path = dataset_dir / "model.bif"
                            download(bif_gz_url, gz_path, force=force)
                            gunzip(gz_path, bif_path, force=force)
                        else:
                            bif_path = dataset_dir / "model.bif"
                            download(bif_url, bif_path, force=force)
                        downloaded_files["bif"] = bif_path.name
                        structure_available = True
                    except Exception as exc:
                        bif_error = exc

                if not structure_available:
                    if rds_url or rda_url:
                        url = rds_url or rda_url
                        suffix = ".rds" if rds_url else ".rda"
                        r_path = dataset_dir / f"model{suffix}"
                        download(url, r_path, force=force)
                        downloaded_files[suffix.lstrip(".")] = r_path.name
                        reason = (
                            "bnlearn CLG networks are distributed as R bn.fit objects "
                            "(RDS/RDA) and require export to BIF/NET/DSC; "
                            "currently unsupported without R."
                        )
                        if bif_error is not None:
                            reason = (
                                f"{reason} (BIF download failed: "
                                f"{type(bif_error).__name__}: {bif_error})"
                            )
                    elif bif_error is not None:
                        if meta.get("type") in {"gaussian", "clgaussian"}:
                            reason = (
                                "bnlearn CLG networks are distributed as R bn.fit objects "
                                "(RDS/RDA) and require export to BIF/NET/DSC; "
                                "currently unsupported without R."
                            )
                            reason = (
                                f"{reason} (BIF download failed: "
                                f"{type(bif_error).__name__}: {bif_error})"
                            )
                        else:
                            reason = f"Download failed: {type(bif_error).__name__}: {bif_error}"
                    else:
                        if meta.get("type") in {"gaussian", "clgaussian"}:
                            reason = (
                                "bnlearn CLG networks are distributed as R bn.fit objects "
                                "(RDS/RDA) and require export to BIF/NET/DSC; "
                                "currently unsupported without R."
                            )
                        else:
                            reason = "No supported artifact URLs in metadata."
            except Exception as exc:
                reason = f"Download failed: {type(exc).__name__}: {exc}"

            capabilities = {
                "can_parse_bif": structure_available,
                "can_generate_queries": structure_available,
                "can_compute_ground_truth": structure_available,
            }

            manifest = {
                "dataset_id": dataset_id,
                "generator": self.name,
                "problem": network,
                "network": network,
                "files": downloaded_files,
                "structure_available": structure_available,
                "reason": reason,
                "artifacts": artifacts_list,
            }
            self.write_dataset_manifest(dataset_dir, manifest)
            self.write_dataset_metadata(
                dataset_id,
                "download.json",
                {
                    "dataset_id": dataset_id,
                    "generator": self.name,
                    "problem": network,
                    "network": network,
                    "type": meta.get("type"),
                    "category": meta.get("category"),
                    "urls": urls,
                    "artifacts": artifacts_list,
                    "downloaded_files": downloaded_files,
                    "structure_available": structure_available,
                    "reason": reason,
                    "capabilities": capabilities,
                },
            )

            dataset_entries.append(
                {
                    "dataset_id": dataset_id,
                    "network": network,
                    "generator": self.name,
                    "problem": network,
                    "files": {
                        key: str((dataset_dir / value).relative_to(self.root_path))
                        for key, value in downloaded_files.items()
                    },
                    "category": meta.get("category"),
                    "type": meta.get("type"),
                    "artifacts": artifacts_list,
                    "capabilities": capabilities,
                    "structure_available": structure_available,
                    "reason": reason,
                }
            )

        metadata = self.build_metadata(
            datasets=dataset_entries,
            static_registry=str(registry_path.relative_to(self.root_path)),
            selected_networks=selected,
        )
        self.save_metadata(metadata, filename=f"{self.name}.json")
