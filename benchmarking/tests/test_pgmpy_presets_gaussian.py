from __future__ import annotations

from benchmarking.models.presets import get_preset, get_preset_config


def test_pgmpy_gaussian_cpds_preset_exists() -> None:
    preset = get_preset("pgmpy", "cpds", "pgmpy_gaussian")
    assert preset["cpds"]["estimator"] == "gaussian"


def test_pgmpy_gaussian_inference_presets_exist() -> None:
    exact = get_preset_config("pgmpy", "inference", "pgmpy_gaussian_exact")
    assert exact.learning.name == "gaussian"
    assert exact.cpd.name == "gaussian"
    assert exact.inference.name == "gaussian_exact"

    forward = get_preset_config("pgmpy", "inference", "pgmpy_gaussian_fs")
    assert forward.learning.name == "gaussian"
    assert forward.cpd.name == "gaussian"
    assert forward.inference.name == "gaussian_forward_sample"
    assert int(forward.inference.kwargs["n_samples_infer"]) == 4096
    assert int(forward.inference.kwargs["n_resample"]) == 1024
