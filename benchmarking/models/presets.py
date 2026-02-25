from __future__ import annotations

from typing import Dict

from .config import ComponentSpec, make_component, ModelBenchmarkConfig


def _vbn_presets() -> Dict[str, ModelBenchmarkConfig]:
    model = "vbn"
    presets = {
        "vbn_softmax_is": ModelBenchmarkConfig(
            model=model,
            config_id="vbn_softmax_is",
            learning=make_component("learning", "node_wise"),
            cpd=make_component("cpd", "softmax_nn"),
            inference=make_component("inference", "importance_sampling"),
        ),
        "vbn_softmax_mcm": ModelBenchmarkConfig(
            model=model,
            config_id="vbn_softmax_mcm",
            learning=make_component("learning", "node_wise"),
            cpd=make_component("cpd", "softmax_nn"),
            inference=make_component("inference", "monte_carlo_marginalization"),
        ),
        "vbn_gauss_is": ModelBenchmarkConfig(
            model=model,
            config_id="vbn_gauss_is",
            learning=make_component("learning", "node_wise"),
            cpd=make_component("cpd", "gaussian_nn"),
            inference=make_component("inference", "importance_sampling"),
        ),
        "vbn_gauss_mcm": ModelBenchmarkConfig(
            model=model,
            config_id="vbn_gauss_mcm",
            learning=make_component("learning", "node_wise"),
            cpd=make_component("cpd", "gaussian_nn"),
            inference=make_component("inference", "monte_carlo_marginalization"),
        ),
        "vbn_linear_gauss_is": ModelBenchmarkConfig(
            model=model,
            config_id="vbn_linear_gauss_is",
            learning=make_component("learning", "node_wise"),
            cpd=make_component("cpd", "linear_gaussian"),
            inference=make_component("inference", "importance_sampling"),
        ),
        "vbn_linear_gauss_mcm": ModelBenchmarkConfig(
            model=model,
            config_id="vbn_linear_gauss_mcm",
            learning=make_component("learning", "node_wise"),
            cpd=make_component("cpd", "linear_gaussian"),
            inference=make_component("inference", "monte_carlo_marginalization"),
        ),
    }
    return presets


def pgmpy_presets() -> Dict[str, ModelBenchmarkConfig]:
    model = "pgmpy"
    return {
        "pgmpy_mle_ei": ModelBenchmarkConfig(
            model=model,
            config_id="pgmpy_mle_ei",
            learning=ComponentSpec(
                name="mle",
                key="learn:mle",
                kwargs={},
            ),
            cpd=ComponentSpec(
                name="tabular_mle",
                key="cpd:tabular_mle",
                kwargs={},
            ),
            inference=ComponentSpec(
                name="exact_variable_elimination",
                key="inf:exact_ve",
                kwargs={},
            ),
        ),
        "pgmpy_bdeu_ei": ModelBenchmarkConfig(
            model=model,
            config_id="pgmpy_bdeu_ei",
            learning=ComponentSpec(
                name="bdeu",
                key="learn:bdeu",
                kwargs={"equivalent_sample_size": 10},
            ),
            cpd=ComponentSpec(
                name="tabular_bdeu",
                key="cpd:tabular_bdeu",
                kwargs={},
            ),
            inference=ComponentSpec(
                name="exact_variable_elimination",
                key="inf:exact_ve",
                kwargs={},
            ),
        ),
    }


MODEL_PRESETS: Dict[str, Dict[str, ModelBenchmarkConfig]] = {
    "vbn": _vbn_presets(),
    "pgmpy": pgmpy_presets(),
}


def get_model_presets(model: str) -> Dict[str, ModelBenchmarkConfig]:
    try:
        return MODEL_PRESETS[model]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_PRESETS)) or "<none>"
        raise KeyError(
            f"Unknown model '{model}'. Available models with presets: {available}"
        ) from exc


def get_preset_config(model: str, config_id: str) -> ModelBenchmarkConfig:
    presets = get_model_presets(model)
    if config_id not in presets:
        available = ", ".join(sorted(presets)) or "<none>"
        raise KeyError(
            f"Unknown config_id '{config_id}' for model '{model}'. "
            f"Available: {available}"
        )
    return presets[config_id]


def get_default_config(model: str) -> ModelBenchmarkConfig:
    return get_preset_config(model, "default")
