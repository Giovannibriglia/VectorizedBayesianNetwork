from .nonparametric.gp_svgp import SVGPRegCPD
from .nonparametric.kde import KDECPD
from .parametric.linear_gaussian import LinearGaussianCPD
from .parametric.mle import MLECategoricalCPD

CPD_REGISTRY = {
    "mle": MLECategoricalCPD,
    "linear_gaussian": LinearGaussianCPD,
    "kde": KDECPD,
    "gp_svgp": SVGPRegCPD,
}
