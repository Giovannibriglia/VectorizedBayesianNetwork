from __future__ import annotations

import pytest
from benchmarking.models import presets as presets_mod


def test_vbn_cpds_requires_learning_and_default_cpd_method() -> None:
    with pytest.raises(ValueError):
        presets_mod.validate_preset(
            "vbn",
            "cpds",
            {
                "cpds": {"default": {"method": "linear_gaussian"}},
            },
        )
    with pytest.raises(ValueError):
        presets_mod.validate_preset(
            "vbn",
            "cpds",
            {
                "learning": {"method": "node_wise"},
                "cpds": {"default": {}},
            },
        )

    preset = presets_mod.validate_preset(
        "vbn",
        "cpds",
        {
            "learning": {"method": "node_wise"},
            "cpds": {"default": {"method": "linear_gaussian"}},
        },
    )
    assert preset["learning"]["kwargs"] == {}
    assert preset["cpds"]["default"]["kwargs"] == {}


def test_vbn_inference_requires_inference_method() -> None:
    with pytest.raises(ValueError):
        presets_mod.validate_preset(
            "vbn",
            "inference",
            {
                "learning": {"method": "node_wise"},
                "cpds": {"default": {"method": "gaussian_nn"}},
            },
        )
    with pytest.raises(ValueError):
        presets_mod.validate_preset(
            "vbn",
            "inference",
            {
                "learning": {"method": "node_wise"},
                "cpds": {"default": {"method": "gaussian_nn"}},
                "inference": {},
            },
        )

    preset = presets_mod.validate_preset(
        "vbn",
        "inference",
        {
            "learning": {"method": "node_wise"},
            "cpds": {"default": {"method": "gaussian_nn"}},
            "inference": {"method": "importance_sampling"},
        },
    )
    assert preset["inference"]["kwargs"] == {}


def test_pgmpy_cpds_requires_estimator() -> None:
    with pytest.raises(ValueError):
        presets_mod.validate_preset("pgmpy", "cpds", {"cpds": {}})

    preset = presets_mod.validate_preset(
        "pgmpy",
        "cpds",
        {"cpds": {"estimator": "mle"}},
    )
    assert preset["cpds"]["kwargs"] == {}


def test_pgmpy_inference_requires_inference_method() -> None:
    with pytest.raises(ValueError):
        presets_mod.validate_preset(
            "pgmpy",
            "inference",
            {"cpds": {"estimator": "mle"}},
        )
    with pytest.raises(ValueError):
        presets_mod.validate_preset(
            "pgmpy",
            "inference",
            {"cpds": {"estimator": "mle"}, "inference": {}},
        )

    preset = presets_mod.validate_preset(
        "pgmpy",
        "inference",
        {
            "cpds": {"estimator": "mle"},
            "inference": {"method": "exact_variable_elimination"},
        },
    )
    assert preset["inference"]["kwargs"] == {}
