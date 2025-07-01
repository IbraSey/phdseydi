import openturns as ot
from .sampling import stick_breaking, sample_mixture_niw

def density_dpmm(alpha, tau, means_base, weights_base, lambda_0, nu_0, Psi_0):
    weights = stick_breaking(alpha, tau)
    prior = [sample_mixture_niw(means_base, weights_base, lambda_0, nu_0, Psi_0) for _ in range(len(weights))]
    components = [ot.Normal(mu, sigma) for mu, sigma in prior]
    return ot.Mixture(components, weights)
