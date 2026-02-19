from __future__ import annotations

from typing import Dict

import torch

from vbn.config_cast import coerce_numbers, UPDATE_SCHEMA

UPDATE_REQUIRED_KEYS = {"lr", "n_steps", "batch_size"}
UPDATE_ALLOWED_KEYS = set(UPDATE_SCHEMA.keys())


def _resolve_node_update(vbn, node: str) -> Dict:
    learning_cfg = getattr(vbn, "_learning_config", None) or {}
    nodes_cpds = learning_cfg.get("nodes_cpds") or {}
    if not isinstance(nodes_cpds, dict) or node not in nodes_cpds:
        raise ValueError(
            f"Missing CPD config for node '{node}'. Provide an 'update' dict per node."
        )
    conf = nodes_cpds.get(node) or {}
    if not isinstance(conf, dict):
        raise ValueError(f"CPD config for node '{node}' must be a dict.")
    if "update" not in conf:
        raise ValueError(f"CPD config for node '{node}' must include an 'update' dict.")
    update_conf = conf["update"]
    if not isinstance(update_conf, dict):
        raise ValueError(f"CPD 'update' config for node '{node}' must be a dict.")
    missing = sorted(UPDATE_REQUIRED_KEYS - set(update_conf))
    if missing:
        raise ValueError(
            f"CPD 'update' config for node '{node}' is missing required keys: {missing}."
        )
    unknown = sorted(set(update_conf) - UPDATE_ALLOWED_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown keys in CPD 'update' config for node '{node}': {unknown}. "
            f"Allowed keys: {sorted(UPDATE_ALLOWED_KEYS)}."
        )
    return coerce_numbers(update_conf, UPDATE_SCHEMA)


class BaseUpdatePolicy:
    def update(self, vbn, data: Dict[str, torch.Tensor], **kwargs):
        raise NotImplementedError

    def get_state(self) -> Dict[str, object]:
        return {}

    def set_state(self, state: Dict[str, object]) -> None:
        return None
