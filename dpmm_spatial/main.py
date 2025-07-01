from dpmm.density import density_dpmm
import numpy as np

alpha = 50
tau = 1e-2
means_base = [[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]]
weights_base = np.array([2.0, 1.0, 0.5, 0.1])
weights_base /= weights_base.sum()
lambda_0 = 50.0
nu_0 = 4
import openturns as ot
Psi_0 = ot.CovarianceMatrix([[0.26, 0.00], [0.00, 0.26]])

mixture = density_dpmm(alpha, tau, means_base, weights_base, lambda_0, nu_0, Psi_0)
print("DPMM générée avec succès.")
