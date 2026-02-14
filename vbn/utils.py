from __future__ import annotations

from typing import Dict, Iterable, Optional

import pandas as pd
import torch

from vbn.core.utils import ensure_2d, ensure_tensor


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


def infer_batch_size(evidence: Dict[str, torch.Tensor]) -> int:
    if not evidence:
        return 1
    first = next(iter(evidence.values()))
    return int(first.shape[0])
