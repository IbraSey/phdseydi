import openturns as ot
import numpy as np


def stick_breaking(alpha, tau=1e-3):
    """
    Génère des poids selon le processus de Stick-Breaking pour approximer une réalisation d’un DP.

    Paramètres :
        - alpha : paramètre de concentration du processus de Dirichlet 
        - tau : seuil pour arrêter l'approximation (par défaut 1e-3)

    Retourne :
        - np.ndarray : tableau de poids normalisés
    """

    weights = []
    r = 1.0
    while r > tau:
        v = ot.Beta(1.0, alpha, 0.0, 1.0).getRealization()[0]
        w = v * r
        weights.append(w)
        r *= (1 - v)
    return np.array(weights) / np.sum(weights)


def sample_mixture_niw(means_base, weights_base, lambda_0, Psi_0, nu_0):
    """
    Échantillonne une moyenne et une matrice de covariance à partir d’un mélange de Normal-Inverse-Wishart (NIW).

    Paramètres :
        - means_base : liste des centres moyens potentiels pour les composantes
        - weights_base : poids associés à chaque centre de moyenne 
        - lambda_0 : précision sur la moyenne 
        - Psi_0 : matrice d’échelle pour l’Inverse-Wishart
        - nu_0 : degrés de liberté de la loi Inverse-Wishart (> dimension - 1)

    Retourne :
        - mu : moyenne échantillonnée
        - Sigma : matrice de covariance échantillonnée
    """
    base_idx = np.random.choice(len(means_base), p=weights_base)
    mu_0 = ot.Point(means_base[base_idx])
    Sigma = ot.InverseWishart(Psi_0, nu_0).getRealizationAsMatrix()
    mu = ot.Normal(mu_0, ot.CovarianceMatrix(Sigma / lambda_0)).getRealization()
    return mu, Sigma


def sample_from_f0(n_samples, zones, weights, areas):
    """
    Génère des échantillons selon la densité f0 d'un zonage sismotectonique 

    Paramètres :
        - n_samples (int) : Nombre d’échantillons à générer.
        - zones (list) : Liste des sous-domaines.
        - weights (ndarray) : Poids w_j.
        - areas (ndarray) : Aires A_j.

    Retourne :
        - ndarray (n_samples, 2) : Échantillons tirés de f0.
    """

    probs = weights * areas
    probs /= probs.sum()
    samples = []
    for _ in range(n_samples):
        idx = np.random.choice(len(zones), p=probs)
        (x0, x1), (y0, y1) = zones[idx]
        x = ot.Uniform(x0, x1).getRealization()[0]
        y = ot.Uniform(y0, y1).getRealization()[0]
        samples.append([x, y])
    return np.array(samples)


def sample_from_f0tilde(n_samples, mus, covariances, weights, areas):
    """
    Tire des échantillons depuis une densité mélange de gaussiennes f0_tilde

    Paramètres :
        - n_samples (int) : Nombre de points à générer.
        - mus (list of ot.Point) : Moyennes μ_j des composantes.
        - covariances (list of ot.CovarianceMatrix) : Matrices Σ_j.
        - weights (ndarray) : Poids w_j
        - areas (ndarray) : Aires A_j des zones

    Retourne :
        - ndarray (n_samples, 2) : Échantillons de f0_tilde
    """

    probs = weights * areas
    probs /= probs.sum()
    samples = []
    for _ in range(n_samples):
        idx = np.random.choice(len(weights), p=probs)
        sample = ot.Normal(mus[idx], covariances[idx]).getRealization()
        samples.append([sample[0], sample[1]])
    return np.array(samples)


def sample_from_informative_dpmm(n_samples, alpha, tau, means_base, weights_base, lambda_0, Psi_0, nu_0):
    """
    Génère un échantillon aléatoire depuis un DPMM avec un prior informatif.

    Paramètres :
        - n_samples : int, nombre total d'échantillons à générer
        - alpha : float, paramètre de concentration du DP
        - tau : float, seuil d’arrêt pour le stick-breacking
        - means_base : list[list[float]], centres des régions de base pour le prior
        - weights_base : list[float], poids associés aux centres de base
        - lambda_0 : float, paramètre de précision du prior normal
        - Psi_0 : openturns.CovarianceMatrix, matrice d’échelle du prior Inverse-Wishart
        - nu_0 : int, degrés de liberté du prior Inverse-Wishart

    Retourne :
        - samples : np.ndarray de forme (n_samples, 2), échantillons générés
    """

    component_weights = stick_breaking(alpha, tau)

    component_params = [
        sample_mixture_niw(means_base, weights_base, lambda_0, Psi_0, nu_0)
        for _ in range(len(component_weights))
    ]

    samples = []
    for _ in range(n_samples):
        idx = np.random.choice(len(component_weights), p=component_weights)
        mu_k, sigma_k = component_params[idx]
        sample = ot.Normal(mu_k, sigma_k).getRealization()
        samples.append([sample[0], sample[1]])

    return np.array(samples)






