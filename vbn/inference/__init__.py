from .approximate.gibbs import ParallelGibbs
from .approximate.likelyhood_weighting import LikelihoodWeighting
from .approximate.loopy_belief_propagation import LoopyBP

from .approximate.sequential_mc import SMC
from .exact.gaussian_exact import GaussianExact
from .exact.variable_elimination import VariableElimination

INFERENCE_BACKENDS = {
    "lw": LikelihoodWeighting,
    "ve": VariableElimination,
    "gaussian": GaussianExact,
    "lbp": LoopyBP,
    "smc": SMC,
    "gibbs": ParallelGibbs,
}
