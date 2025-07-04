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
        - mu : np.ndarray représentant la moyenne d’une composante
        - Sigma : np.ndarray représentant la matrice de covariance de la composante
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

    Retour :
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





