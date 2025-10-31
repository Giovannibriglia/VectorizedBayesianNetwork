from __future__ import annotations

import torch


def save_state(bn, path: str):
    torch.save(
        {
            "version": 1,
            "nodes": bn.nodes,
            "parents": bn.parents,
            "topo_order": bn.topo_order,
            "device_str": str(bn.device),
            # learning / inference wiring
            "learner_map": getattr(bn, "_learner_map", {}),
            "inference_method": getattr(bn, "inference_method", None),
            # CPD payloads
            "cpd_types": {k: bn.cpd[k].__class__.__name__ for k in bn.cpd},
            "cpd_state": {k: dict(bn.cpd[k].__dict__) for k in bn.cpd},
        },
        path,
    )


def load_state(bn, path: str):
    # NOTE: weights_only=True is available in recent PyTorch; if you’re on older
    # versions, drop that kwarg.
    state = torch.load(path, map_location=bn.device, weights_only=False)

    # Restore structure (defensive: keep the instance’s nodes/parents unless identical)
    bn.nodes = state["nodes"]
    bn.parents = state["parents"]
    bn.topo_order = state.get("topo_order", bn.topo_order)

    # Make sure CPD *instances* exist before stuffing in their state.
    # If you used set_learners before saving, restore it so trainer can build CPDs.
    learner_map = state.get("learner_map", {})
    if learner_map:
        bn.set_learners(learner_map)

    # Force-trainable objects to be materialized (no-op training is fine);
    # this ensures bn.cpd[name] instances exist.
    bn._ensure_trainer()
    if not bn.cpd:  # if your trainer populates CPDs lazily, call its initializer
        # Many trainers build CPDs on first fit/partial_fit; if that’s your case,
        # do a “dry” materialization pass:
        (
            bn._trainer.ensure_modules_created()
            if hasattr(bn._trainer, "ensure_modules_created")
            else None
        )

    # Now restore the raw __dict__ of each CPD (safe: only tensors/buffers/ints/floats)
    for name, cpd_payload in state["cpd_state"].items():
        if name not in bn.cpd:
            raise RuntimeError(f"CPD for node '{name}' not initialized before load.")
        bn.cpd[name].__dict__.update(cpd_payload)

    # Restore inference backend, if any
    inf = state.get("inference_method")
    if inf:
        bn.set_inference(inf)
