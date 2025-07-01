import openturns as ot
import numpy as np

def stick_breaking(alpha, tau=1e-3):
    weights = []
    r = 1.0
    while r > tau:
        v = ot.Beta(1.0, alpha, 0.0, 1.0).getRealization()[0]
        w = v * r
        weights.append(w)
        r *= (1 - v)
    return np.array(weights) / np.sum(weights)

def sample_mixture_niw(means_base, weights_base, lambda_0, nu_0, Psi_0):
    base_idx = np.random.choice(len(means_base), p=weights_base)
    mu_0 = ot.Point(means_base[base_idx])
    Sigma = ot.InverseWishart(Psi_0, nu_0).getRealizationAsMatrix()
    mu = ot.Normal(mu_0, ot.CovarianceMatrix(Sigma / lambda_0)).getRealization()
    return mu, Sigma
