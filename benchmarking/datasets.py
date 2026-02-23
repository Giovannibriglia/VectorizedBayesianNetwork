from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from benchmarking.paths import ensure_dir, get_dataset_encoding_metadata_path

ENCODING_PIPELINE_VERSION = "one_hot_v1"
ONE_HOT_SEPARATOR = "__"
MISSING_CATEGORY = "<NA>"


@dataclass(frozen=True)
class DatasetArtifact:
    dataset_id: str
    raw_path: Path
    encoded_path: Path | None
    encoding_path: Path | None
    encoding_meta: dict | None

    def view(self, space: str) -> DatasetView:
        if space == "original":
            return DatasetView(
                dataset_id=self.dataset_id,
                path=self.raw_path,
                space=space,
                encoding_meta=self.encoding_meta,
            )
        if space == "encoded":
            if self.encoded_path is None:
                raise ValueError("Encoded path is not available for this dataset.")
            return DatasetView(
                dataset_id=self.dataset_id,
                path=self.encoded_path,
                space=space,
                encoding_meta=self.encoding_meta,
            )
        raise ValueError(f"Unknown dataset view space '{space}'")

    def load_raw(self) -> pd.DataFrame:
        return load_dataframe(self.raw_path)

    def load_encoded(self) -> pd.DataFrame:
        if self.encoded_path is None:
            raise ValueError("Encoded path is not available for this dataset.")
        return load_dataframe(self.encoded_path)


@dataclass(frozen=True)
class DatasetView:
    dataset_id: str
    path: Path
    space: str
    encoding_meta: dict | None

    def load(self) -> pd.DataFrame:
        return load_dataframe(self.path)


def _is_numeric_column(series: pd.Series) -> bool:
    if is_bool_dtype(series):
        return False
    return is_numeric_dtype(series)


def _category_label(value: Any) -> str:
    if pd.isna(value):
        return MISSING_CATEGORY
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    return str(value)


def detect_feature_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric: list[str] = []
    categorical: list[str] = []
    for column in df.columns:
        if _is_numeric_column(df[column]):
            numeric.append(column)
        else:
            categorical.append(column)
    return numeric, categorical


def encode_dataframe(
    df: pd.DataFrame,
    encoding_meta: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    if encoding_meta is None:
        numeric_cols, categorical_cols = detect_feature_types(df)
        categorical_meta: dict[str, dict[str, list[str]]] = {}
        for column in categorical_cols:
            labels = df[column].map(_category_label)
            categories = sorted(set(labels))
            one_hot_cols = [f"{column}{ONE_HOT_SEPARATOR}{cat}" for cat in categories]
            categorical_meta[column] = {
                "categories": categories,
                "one_hot_columns": one_hot_cols,
            }
        encoding_meta = {
            "encoding": "one_hot",
            "version": 1,
            "original_columns": list(df.columns),
            "categorical": categorical_meta,
            "numeric": list(numeric_cols),
        }
    else:
        if "original_columns" not in encoding_meta:
            encoding_meta["original_columns"] = list(df.columns)
        numeric_cols = list(encoding_meta.get("numeric", []))
        categorical_cols = list(encoding_meta.get("categorical", {}).keys())

    encoded_frames: list[pd.DataFrame] = []
    encoded_columns: list[str] = []

    if numeric_cols:
        numeric_df = df[numeric_cols].copy()
        encoded_frames.append(numeric_df)
        encoded_columns.extend(numeric_cols)

    categorical_meta = encoding_meta.get("categorical", {})
    for column in categorical_cols:
        meta = categorical_meta.get(column)
        if not meta:
            raise ValueError(f"Missing categorical metadata for column '{column}'")
        categories = list(meta.get("categories", []))
        if not categories:
            raise ValueError(f"No categories defined for column '{column}'")
        labels = df[column].map(_category_label)
        unknown = sorted(set(labels) - set(categories))
        if unknown:
            raise ValueError(f"Unknown categories for '{column}': {unknown}")

        one_hot_cols = [f"{column}{ONE_HOT_SEPARATOR}{cat}" for cat in categories]
        meta["one_hot_columns"] = one_hot_cols
        categorical_meta[column] = meta

        for cat in categories:
            col_name = f"{column}{ONE_HOT_SEPARATOR}{cat}"
            encoded_frames.append(
                pd.DataFrame({col_name: (labels == cat).astype(int)}, index=df.index)
            )
            encoded_columns.append(col_name)

    df_encoded = (
        pd.concat(encoded_frames, axis=1)
        if encoded_frames
        else pd.DataFrame(index=df.index)
    )
    if encoded_columns:
        df_encoded = df_encoded[encoded_columns]

    encoding_meta["categorical"] = categorical_meta
    encoding_meta["transform"] = {
        "n_original_features": int(len(encoding_meta.get("original_columns", []))),
        "n_encoded_features": int(len(encoded_columns)),
        "n_dropped": 0,
    }

    return df_encoded, encoding_meta


def decode_assignment(
    encoded_assignment: dict[str, Any], encoding_meta: dict
) -> dict[str, Any]:
    decoded: dict[str, Any] = {}
    numeric_cols = list(encoding_meta.get("numeric", []))
    categorical_meta = encoding_meta.get("categorical", {})

    for column in numeric_cols:
        if column in encoded_assignment:
            decoded[column] = encoded_assignment[column]

    for column in sorted(categorical_meta.keys()):
        meta = categorical_meta[column]
        categories = list(meta.get("categories", []))
        one_hot_cols = list(meta.get("one_hot_columns", []))
        if not categories or not one_hot_cols:
            continue
        best_idx = None
        best_val = None
        for idx, col_name in enumerate(one_hot_cols):
            if col_name not in encoded_assignment:
                continue
            value = encoded_assignment[col_name]
            if value is None:
                continue
            if best_val is None or value > best_val:
                best_val = value
                best_idx = idx
        if best_idx is not None and best_val is not None and best_val > 0:
            decoded[column] = categories[best_idx]

    return decoded


def lift_query(original_query: dict, encoding_meta: dict) -> dict:
    categorical_meta = encoding_meta.get("categorical", {})

    def lift_var(var: str) -> list[str]:
        if var in categorical_meta:
            return list(categorical_meta[var].get("one_hot_columns", []))
        return [var]

    def lift_vars(vars_list: list[str]) -> list[str]:
        lifted: list[str] = []
        for v in vars_list:
            lifted.extend(lift_var(v))
        return sorted(lifted)

    target = original_query.get("target")
    if isinstance(target, list):
        target_vars = target
    else:
        target_vars = [target] if target is not None else []
    lifted_target = lift_vars(target_vars)

    evidence = original_query.get("evidence")
    if evidence is None:
        lifted_evidence = {"vars": [], "values": None}
    elif isinstance(evidence, list):
        lifted_evidence = {"vars": lift_vars(evidence), "values": None}
    elif isinstance(evidence, dict):
        encoded_values: dict[str, Any] = {}
        encoded_vars: list[str] = []
        for var, value in evidence.items():
            if var in categorical_meta:
                if value is None:
                    continue
                label = _category_label(value)
                categories = list(categorical_meta[var].get("categories", []))
                if label not in categories:
                    raise ValueError(
                        f"Unknown category '{value}' for '{var}'. Known: {categories}"
                    )
                col_name = f"{var}{ONE_HOT_SEPARATOR}{label}"
                encoded_vars.append(col_name)
                encoded_values[col_name] = 1
            else:
                encoded_vars.append(var)
                encoded_values[var] = value
        lifted_evidence = {
            "vars": sorted(encoded_vars),
            "values": encoded_values if encoded_values else None,
        }
    else:
        raise TypeError("evidence must be dict, list, or None")

    lifted = dict(original_query)
    lifted["target"] = lifted_target
    lifted["evidence"] = lifted_evidence
    return lifted


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported dataset format: {path}")


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
        return
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
        return
    raise ValueError(f"Unsupported dataset format: {path}")


def encode_dataset_file(
    *,
    dataset_id: str,
    raw_path: Path,
    encoded_path: Path | None = None,
    root_path: Path | None = None,
    encoding_meta: dict | None = None,
    generator: str | None = None,
    problem: str | None = None,
) -> DatasetArtifact:
    raw_path = Path(raw_path)
    if encoded_path is None:
        encoded_path = raw_path.with_name(f"{raw_path.stem}_encoded{raw_path.suffix}")

    df = load_dataframe(raw_path)
    df_encoded, meta = encode_dataframe(df, encoding_meta=encoding_meta)
    save_dataframe(df_encoded, encoded_path)

    meta = dict(meta)
    meta["dataset_id"] = dataset_id
    meta["pipeline_version"] = ENCODING_PIPELINE_VERSION
    meta["hashes"] = {
        "raw_sha256": hash_file(raw_path),
        "encoded_sha256": hash_file(encoded_path),
    }

    encoding_path = None
    if root_path is not None:
        if generator is None or problem is None:
            raise ValueError("generator and problem are required when root_path is set")
        encoding_path = get_dataset_encoding_metadata_path(
            root_path, generator, problem
        )
        ensure_dir(encoding_path.parent)
        encoding_path.write_text(json.dumps(meta, indent=2, sort_keys=True))

    return DatasetArtifact(
        dataset_id=dataset_id,
        raw_path=raw_path,
        encoded_path=encoded_path,
        encoding_path=encoding_path,
        encoding_meta=meta,
    )


def load_encoding_metadata(
    root_path: Path, generator: str, problem: str
) -> dict | None:
    path = get_dataset_encoding_metadata_path(root_path, generator, problem)
    if not path.exists():
        return None
    return json.loads(path.read_text())
