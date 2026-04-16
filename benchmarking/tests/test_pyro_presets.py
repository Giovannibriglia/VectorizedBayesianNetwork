from __future__ import annotations

from benchmarking.models.presets import get_preset, get_preset_config


def test_pyro_cpds_preset_exists() -> None:
    preset = get_preset("pyro", "cpds", "pyro_dirichlet")
    assert preset["learning"]["method"] == "dirichlet_table"
    assert preset["cpds"]["method"] == "dirichlet_table"


def test_pyro_inference_presets_exist() -> None:
    lw = get_preset_config("pyro", "inference", "pyro_lw")
    assert lw.learning.name == "dirichlet_table"
    assert lw.cpd.name == "dirichlet_table"
    assert lw.inference.name == "likelihood_weighting"

    ais = get_preset_config("pyro", "inference", "pyro_ais")
    assert ais.learning.name == "dirichlet_table"
    assert ais.cpd.name == "dirichlet_table"
    assert ais.inference.name == "ancestral_importance_sampling"
