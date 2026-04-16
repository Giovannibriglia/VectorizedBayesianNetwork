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


def test_numpyro_cpds_requires_learning_and_cpd_method() -> None:
    with pytest.raises(ValueError):
        presets_mod.validate_preset(
            "numpyro",
            "cpds",
            {
                "cpds": {"method": "dirichlet_table"},
            },
        )
    with pytest.raises(ValueError):
        presets_mod.validate_preset(
            "numpyro",
            "cpds",
            {
                "learning": {"method": "dirichlet_table"},
                "cpds": {},
            },
        )

    preset = presets_mod.validate_preset(
        "numpyro",
        "cpds",
        {
            "learning": {"method": "dirichlet_table"},
            "cpds": {"method": "dirichlet_table"},
        },
    )
    assert preset["learning"]["kwargs"] == {}
    assert preset["cpds"]["kwargs"] == {}


def test_numpyro_inference_requires_method() -> None:
    with pytest.raises(ValueError):
        presets_mod.validate_preset(
            "numpyro",
            "inference",
            {
                "learning": {"method": "dirichlet_table"},
                "cpds": {"method": "dirichlet_table"},
            },
        )
    with pytest.raises(ValueError):
        presets_mod.validate_preset(
            "numpyro",
            "inference",
            {
                "learning": {"method": "dirichlet_table"},
                "cpds": {"method": "dirichlet_table"},
                "inference": {},
            },
        )

    preset = presets_mod.validate_preset(
        "numpyro",
        "inference",
        {
            "learning": {"method": "dirichlet_table"},
            "cpds": {"method": "dirichlet_table"},
            "inference": {"method": "likelihood_weighting"},
        },
    )
    assert preset["inference"]["kwargs"] == {}


def test_gpytorch_cpds_requires_learning_and_cpd_method() -> None:
    with pytest.raises(ValueError):
        presets_mod.validate_preset(
            "gpytorch",
            "cpds",
            {
                "cpds": {"method": "gp_posterior"},
            },
        )
    with pytest.raises(ValueError):
        presets_mod.validate_preset(
            "gpytorch",
            "cpds",
            {
                "learning": {"method": "exact_gp"},
                "cpds": {},
            },
        )

    preset = presets_mod.validate_preset(
        "gpytorch",
        "cpds",
        {
            "learning": {"method": "exact_gp"},
            "cpds": {"method": "gp_posterior"},
        },
    )
    assert preset["learning"]["kwargs"] == {}
    assert preset["cpds"]["kwargs"] == {}


def test_gpytorch_inference_requires_method() -> None:
    with pytest.raises(ValueError):
        presets_mod.validate_preset(
            "gpytorch",
            "inference",
            {
                "learning": {"method": "exact_gp"},
                "cpds": {"method": "gp_posterior"},
            },
        )
    with pytest.raises(ValueError):
        presets_mod.validate_preset(
            "gpytorch",
            "inference",
            {
                "learning": {"method": "exact_gp"},
                "cpds": {"method": "gp_posterior"},
                "inference": {},
            },
        )

    preset = presets_mod.validate_preset(
        "gpytorch",
        "inference",
        {
            "learning": {"method": "exact_gp"},
            "cpds": {"method": "gp_posterior"},
            "inference": {"method": "gp_forward_sample"},
        },
    )
    assert preset["inference"]["kwargs"] == {}


def test_pyro_cpds_requires_learning_and_cpd_method() -> None:
    with pytest.raises(ValueError):
        presets_mod.validate_preset(
            "pyro",
            "cpds",
            {
                "cpds": {"method": "dirichlet_table"},
            },
        )
    with pytest.raises(ValueError):
        presets_mod.validate_preset(
            "pyro",
            "cpds",
            {
                "learning": {"method": "dirichlet_table"},
                "cpds": {},
            },
        )

    preset = presets_mod.validate_preset(
        "pyro",
        "cpds",
        {
            "learning": {"method": "dirichlet_table"},
            "cpds": {"method": "dirichlet_table"},
        },
    )
    assert preset["learning"]["kwargs"] == {}
    assert preset["cpds"]["kwargs"] == {}


def test_pyro_inference_requires_method() -> None:
    with pytest.raises(ValueError):
        presets_mod.validate_preset(
            "pyro",
            "inference",
            {
                "learning": {"method": "dirichlet_table"},
                "cpds": {"method": "dirichlet_table"},
            },
        )
    with pytest.raises(ValueError):
        presets_mod.validate_preset(
            "pyro",
            "inference",
            {
                "learning": {"method": "dirichlet_table"},
                "cpds": {"method": "dirichlet_table"},
                "inference": {},
            },
        )

    preset = presets_mod.validate_preset(
        "pyro",
        "inference",
        {
            "learning": {"method": "dirichlet_table"},
            "cpds": {"method": "dirichlet_table"},
            "inference": {"method": "likelihood_weighting"},
        },
    )
    assert preset["inference"]["kwargs"] == {}
