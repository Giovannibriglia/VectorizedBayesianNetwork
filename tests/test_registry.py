import vbn  # noqa: F401
from vbn.core.registry import (
    CPD_REGISTRY,
    INFERENCE_REGISTRY,
    LEARNING_REGISTRY,
    SAMPLING_REGISTRY,
    UPDATE_REGISTRY,
)


def test_registries_have_defaults():
    assert "softmax_nn" in CPD_REGISTRY
    assert "kde" in CPD_REGISTRY
    assert "mdn" in CPD_REGISTRY

    assert "node_wise" in LEARNING_REGISTRY
    assert "amortized" in LEARNING_REGISTRY

    assert "monte_carlo_marginalization" in INFERENCE_REGISTRY
    assert "importance_sampling" in INFERENCE_REGISTRY
    assert "svgp" in INFERENCE_REGISTRY

    assert "ancestral" in SAMPLING_REGISTRY
    assert "gibbs" in SAMPLING_REGISTRY
    assert "hmc" in SAMPLING_REGISTRY

    assert "streaming_stats" in UPDATE_REGISTRY
    assert "online_sgd" in UPDATE_REGISTRY
    assert "ema" in UPDATE_REGISTRY
    assert "replay_buffer" in UPDATE_REGISTRY
