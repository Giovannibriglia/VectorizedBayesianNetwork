from vbn.learning.nonparametric.gp_svgp import SVGPRegCPD
from vbn.learning.nonparametric.kde import KDECPD
from vbn.learning.parametric.linear_gaussian import LinearGaussianCPD
from vbn.learning.parametric.mle import MLECategoricalCPD

CPD_REGISTRY = {
    "mle": MLECategoricalCPD,
    "linear_gaussian": LinearGaussianCPD,
    "kde": KDECPD,
    "gp_svgp": SVGPRegCPD,
}
