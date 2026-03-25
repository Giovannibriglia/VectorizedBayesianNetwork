from __future__ import annotations

from typing import Dict, Iterable, Optional

import pandas as pd
import torch

from vbn.core.utils import ensure_2d, ensure_tensor

__all__ = [
    "df_to_tensor_dict",
    "dict_to_device",
    "concat_parents",
    "infer_batch_size",
]


def df_to_tensor_dict(
    df: pd.DataFrame, device: torch.device
) -> Dict[str, torch.Tensor]:
    data: Dict[str, torch.Tensor] = {}
    for col in df.columns:
        tensor = ensure_tensor(df[col].to_numpy(), device=device)
        data[col] = ensure_2d(tensor)
    return data


def dict_to_device(
    data: Dict[str, torch.Tensor], device: torch.device
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in data.items():
        out[k] = ensure_2d(ensure_tensor(v, device=device))
    return out


def concat_parents(
    data: Dict[str, torch.Tensor], parents: Iterable[str]
) -> Optional[torch.Tensor]:
    parent_list = list(parents)
    if not parent_list:
        return None
    return torch.cat([data[p] for p in parent_list], dim=-1)


def infer_batch_size(
    evidence: Dict[str, torch.Tensor],
    do: Optional[Dict[str, torch.Tensor]] = None,
) -> int:
    evidence = evidence or {}
    do = do or {}
    if evidence:
        batch = int(next(iter(evidence.values())).shape[0])
        if do:
            do_batch = int(next(iter(do.values())).shape[0])
            if batch != do_batch:
                raise ValueError("Evidence and do batch sizes must match.")
        return batch
    if do:
        return int(next(iter(do.values())).shape[0])
    return 1
