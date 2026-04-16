from __future__ import annotations

from benchmarking.models.presets import get_preset, get_preset_config


def test_gpytorch_cpds_preset_exists() -> None:
    preset = get_preset("gpytorch", "cpds", "gpytorch_rbf")
    assert preset["learning"]["method"] == "exact_gp"
    assert preset["cpds"]["method"] == "gp_posterior"


def test_gpytorch_inference_presets_exist() -> None:
    fs = get_preset_config("gpytorch", "inference", "gpytorch_forward")
    assert fs.learning.name == "exact_gp"
    assert fs.cpd.name == "gp_posterior"
    assert fs.inference.name == "gp_forward_sample"

    posterior = get_preset_config("gpytorch", "inference", "gpytorch_posterior")
    assert posterior.learning.name == "exact_gp"
    assert posterior.cpd.name == "gp_posterior"
    assert posterior.inference.name == "gp_posterior"
