from __future__ import annotations

import copy
from functools import lru_cache
from importlib import resources
from typing import Dict

import yaml


@lru_cache(maxsize=None)
def _load_category(category: str) -> Dict[str, Dict]:
    items: Dict[str, Dict] = {}
    base = resources.files("vbn.configs")
    cat_dir = base / category
    if cat_dir.is_dir():
        for path in sorted(cat_dir.iterdir(), key=lambda p: p.name):
            if path.name.endswith(".yaml"):
                data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                name = data.pop("name", path.stem)
                items[path.stem] = {"name": name, "params": data}
    return items


def _resolve_name(name_or_item) -> str:
    if isinstance(name_or_item, str):
        return name_or_item
    if hasattr(name_or_item, "name"):
        return getattr(name_or_item, "name")
    raise TypeError("name_or_item must be a string or ConfigItem")


def _get_item(category: str, name_or_item) -> Dict:
    items = _load_category(category)
    name = _resolve_name(name_or_item)
    if name in items:
        return items[name]
    for entry in items.values():
        if entry["name"] == name:
            return entry
    raise ValueError(
        f"Unknown {category} config '{name}'. Available: {list(items.keys())}"
    )


class Defaults:
    def cpd(self, name_or_item) -> Dict:
        entry = _get_item("cpds", name_or_item)
        params = copy.deepcopy(entry["params"])
        training_keys = {
            "epochs",
            "lr",
            "batch_size",
            "weight_decay",
            "n_steps",
            "max_grad_norm",
        }
        legacy = sorted(set(params) & training_keys)
        if legacy:
            raise ValueError(
                "CPD defaults must not include training keys at top level "
                f"({legacy}). Move them under 'fit'/'update'."
            )
        if "fit" not in params or "update" not in params:
            raise ValueError(
                "CPD defaults must include explicit 'fit' and 'update' dicts."
            )
        fit = params.pop("fit")
        update = params.pop("update")
        if not isinstance(fit, dict):
            raise TypeError("CPD 'fit' defaults must be a dict.")
        if not isinstance(update, dict):
            raise TypeError("CPD 'update' defaults must be a dict.")
        return {"cpd": entry["name"], **params, "fit": fit, "update": update}

    def learning(self, name_or_item) -> Dict:
        entry = _get_item("learning", name_or_item)
        params = copy.deepcopy(entry["params"])
        if entry["name"] == "node_wise":
            training_keys = {
                "epochs",
                "lr",
                "batch_size",
                "weight_decay",
                "n_steps",
                "max_grad_norm",
            }
            bad = sorted(set(params) & training_keys)
            if bad:
                raise ValueError(
                    "node_wise learning defaults must not include training keys "
                    f"({bad}). Move them into per-CPD 'fit'/'update' configs."
                )
        return {"name": entry["name"], **params}

    def inference(self, name_or_item) -> Dict:
        entry = _get_item("inference", name_or_item)
        params = copy.deepcopy(entry["params"])
        return {"name": entry["name"], **params}

    def sampling(self, name_or_item) -> Dict:
        entry = _get_item("sampling", name_or_item)
        params = copy.deepcopy(entry["params"])
        return {"name": entry["name"], **params}

    def update(self, name_or_item) -> Dict:
        entry = _get_item("update", name_or_item)
        params = copy.deepcopy(entry["params"])
        return {"name": entry["name"], **params}


defaults = Defaults()
