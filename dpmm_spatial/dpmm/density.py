import openturns as ot
import numpy as np
from scipy import integrate
from .sampling import stick_breaking, sample_mixture_niw


def normalized_informative_dpmm_density(
    alpha,
    tau,
    means_base,
    weights_base,
    lambda_0,
    Psi_0,
    nu_0
):
    """
    Construit une densité normalisée f(x, y) à partir d'un DPMM avec prior informatif.

    Paramètres :
        - alpha : paramètre de concentration du processus de Dirichlet
        - tau : seuil pour arrêter le stick-breacking
        - means_base : liste des centres initiaux pour la moyenne du NIW
        - weights_base : poids associés aux centres initiaux
        - lambda_0 : paramètre de précision pour la moyenne dans le NIW
        - nu_0 : degrés de liberté de la loi Inverse-Wishart
        - Psi_0 : matrice d’échelle de la loi Inverse-Wishart

    Retourne :
        - f(x, y) : fonction Python représentant la densité normalisée
    """
    
    # Étape 1 : Génération des poids par stick-breaking
    component_weights = stick_breaking(alpha, tau)

    # Étape 2 : Échantillonnage des composantes (mu, Sigma) selon un mélange de NIW
    gaussian_parameters = [
        sample_mixture_niw(means_base, weights_base, lambda_0, Psi_0, nu_0)
        for _ in range(len(component_weights))
    ]

    # Étape 3 : Création des lois normales multivariées
    gaussian_components = [ot.Normal(mu, sigma) for mu, sigma in gaussian_parameters]

    # Étape 4 : Création du mélange pondéré
    dpmm_mixture = ot.Mixture(gaussian_components, component_weights)

    # Étape 5 : Fonction densité brute (non normalisée)
    def raw_density(x, y):
        """
        Évalue la densité DPMM non normalisée
        """
        points = [ot.Point([xi, yi]) for xi, yi in zip(np.ravel(x), np.ravel(y))]
        density_values = [dpmm_mixture.computePDF(pt) for pt in points]
        return np.array(density_values).reshape(np.shape(x))

    # Étape 6 : Calcul de la constante de normalisation sur [0,2] × [0,2]
    def integrand(y, x):
        return dpmm_mixture.computePDF(ot.Point([x, y]))
    integral_value, _ = integrate.dblquad(integrand, 0, 2, lambda x: 0, lambda x: 2)

    # Étape 7 : Densité normalisée
    def normalized_density(x, y):
        return raw_density(x, y) / integral_value

    return normalized_density


def define_zonage_grid(n_rows, n_cols, x_range=(0, 2), y_range=(0, 2)):
    """
    Définit une grille de zonage sismotectonique.

    Paramètres :
        - n_rows (int) : Nombre de lignes de la grille.
        - n_cols (int) : Nombre de colonnes de la grille.
        - x_range (tuple) : Bornes (min, max) en x.
        - y_range (tuple) : Bornes (min, max) en y.

    Retourne :
        - zones (list) : Liste de rectangles (x_bounds, y_bounds)
        - x_bounds, y_bounds (ndarray) : Coordonnées des séparations en x et y.
    """

    x_bounds = np.linspace(x_range[0], x_range[1], n_cols + 1)
    y_bounds = np.linspace(y_range[0], y_range[1], n_rows + 1)
    zones = []
    for i in range(n_rows):
        for j in range(n_cols):
            x0, x1 = x_bounds[j], x_bounds[j + 1]
            y0, y1 = y_bounds[i], y_bounds[i + 1]
            zones.append(((x0, x1), (y0, y1)))
    return zones, x_bounds, y_bounds


def compute_f0_density(x, y, zones, weights, areas):
    """
    Calcule la densité par morceaux f0(x, y), constante sur chaque zone.

    Paramètres :
        - x, y (float) : Coordonnées du point à évaluer.
        - zones (list) : Liste des sous-domaines S_j.
        - weights (ndarray) : Poids w_j associés à chaque zone.
        - areas (ndarray) : Aires A_j des zones.

    Retourne :
        - float : Valeur de la densité f0(x, y)
    """

    for idx, ((x0, x1), (y0, y1)) in enumerate(zones):
        if x0 <= x < x1 and y0 <= y < y1:
            return weights[idx] / np.sum(weights * areas)
    return 0.0


def compute_zone_gaussian_parameters(zones):
    """
    Calcule les centroïdes et les covariances associées aux zones.

    Paramètres :
        - zones (list) : Liste des zones [(x0, x1), (y0, y1)]

    Retourne :
        - mus (list of ot.Point) : Liste des centroïdes μ_j 
        - covariances (list of ot.CovarianceMatrix) : Matrices Σ_j
    """

    mus = []
    covariances = []
    for (x_bounds, y_bounds) in zones:
        x0, x1 = x_bounds
        y0, y1 = y_bounds
        center = [(x0 + x1) / 2, (y0 + y1) / 2]
        mus.append(ot.Point(center))

        # 95% du support de la gaussienne recouvre la zone
        std = (np.sqrt((x1 - x0)**2 + (y1 - y0)**2) / 2) / 1.96
        Sigma = ot.CovarianceMatrix(2)
        Sigma[0, 0] = std**2
        Sigma[1, 1] = std**2
        covariances.append(Sigma)

    return mus, covariances


def compute_f0tilde_density(x, y, mus, covariances, weights):
    """
    Calcule la densité d’un mélange de gaussiennes pondérées qui approxime un zonage.

    Paramètres :
        - x, y (float) : Coordonnées du point d’évaluation.
        - mus (list of ot.Point) : Moyennes μ_j
        - covariances (list of ot.CovarianceMatrix) : Matrices Σ_j
        - weights (ndarray) : Poids w_j

    Retour :
        - float : Valeur de la densité f0_tilde(x, y)
    """

    pt = ot.Point([x, y])
    density = 0.0
    for w, mu, Sigma in zip(weights, mus, covariances):
        density += w * ot.Normal(mu, Sigma).computePDF(pt)
    return density


def average_informative_dpmm_density(
    N,
    alpha,
    tau,
    means_base,
    weights_base,
    lambda_0,
    Psi_0,
    nu_0,
    grid_x=np.linspace(0, 2, 200),
    grid_y=np.linspace(0, 2, 200)
):
    """
    Calcule la moyenne empirique de N densités DPMM informatives.

    Retourne :
        Une grille (X, Y, Z_mean) où Z_mean est la densité moyenne.
    """
    X, Y = np.meshgrid(grid_x, grid_y)
    Z_sum = np.zeros_like(X)

    for _ in range(N):
        f_density = normalized_informative_dpmm_density(
            alpha=alpha,
            tau=tau,
            means_base=means_base,
            weights_base=weights_base,
            lambda_0=lambda_0,
            Psi_0=Psi_0,
            nu_0=nu_0
        )
        Z_sum += f_density(X, Y)

    Z_mean = Z_sum / N
    return X, Y, Z_mean


def dpmm_avg_density_constructor_factory(N, grid_x=np.linspace(0, 2, 200), grid_y=np.linspace(0, 2, 200)):
    def constructor(**kwargs):
        X, Y, Z_mean = average_informative_dpmm_density(
            N=N,
            grid_x=grid_x,
            grid_y=grid_y,
            **kwargs
        )
        return lambda X_eval, Y_eval: Z_mean  # Note : ici Z_mean est constant, adapté à la grille
    return constructor





