from __future__ import annotations

import torch


def save_state(bn, path: str):
    torch.save(
        {
            "nodes": bn.nodes,
            "parents": bn.parents,
            "cpd_state": {k: bn.cpd[k].__dict__ for k in bn.cpd},
            "cpd_types": {k: bn.cpd[k].__class__.__name__ for k in bn.cpd},
        },
        path,
    )


def load_state(bn, path: str):
    state = torch.load(path, map_location=bn.device, weights_only=True)
    bn.nodes = state["nodes"]
    bn.parents = state["parents"]
    # caller must have created CPDs; we restore __dict__ safely
    for k, d in state["cpd_state"].items():
        bn.cpd[k].__dict__.update(d)
