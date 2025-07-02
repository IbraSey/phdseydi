import openturns as ot
import numpy as np
from .sampling import stick_breaking, sample_mixture_niw


def density_dpmm_informatif(alpha, tau, means_base, weights_base, lambda_0, Psi_0, nu_0):
    """
    En chantier
    """
    weights = stick_breaking(alpha, tau)
    n_prior = len(weights)
    prior = [sample_mixture_niw(means_base, weights_base, lambda_0, Psi_0, nu_0) for _ in range(n_prior)]
    components = [ot.Normal(mu, sigma) for mu, sigma in prior]
    return ot.Mixture(components, weights)




