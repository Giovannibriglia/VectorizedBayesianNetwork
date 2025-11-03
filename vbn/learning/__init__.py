from __future__ import annotations

from typing import Dict, Type

from vbn.learning.base import BaseCPDModule
from vbn.learning.nonparametric.gp_svgp import GPSVGPCPD
from vbn.learning.nonparametric.kde import KDECpd
from vbn.learning.parametric.approx_mle import CategoricalCPD
from vbn.learning.parametric.linear_gaussian import GaussianCPD
from vbn.learning.parametric.mle import ExactCategoricalCPD

LEARNING_METHODS: Dict[str, Type[BaseCPDModule]] = {
    "discrete_exact": ExactCategoricalCPD,
    "discrete_approx": CategoricalCPD,
    "gaussian": GaussianCPD,
    "kde": KDECpd,
    "gp_svgp": GPSVGPCPD,
}
