import openturns as ot
import numpy as np
from scipy import integrate
from .sampling import stick_breaking, sample_mixture_niw


def informative_dpmm_density(
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


def compute_f0_density(X, Y, zones, weights, areas):
    """
    Calcule la densité f0(x, y) sur une grille (X, Y), constante par zone.

    Paramètres :
        - X, Y : grilles (ndarray) générées avec np.meshgrid
        - zones : liste des sous-domaines [(x0, x1), (y0, y1)]
        - weights : tableau des poids w_j
        - areas : tableau des aires A_j associées à chaque zone

    Retour :
        - Z : ndarray de la densité f0 évaluée sur la grille
    """
    Z = np.zeros_like(X)

    for idx, ((x0, x1), (y0, y1)) in enumerate(zones):
        mask = (X >= x0) & (X < x1) & (Y >= y0) & (Y < y1)
        Z[mask] = weights[idx] / areas

    return Z


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


def compute_f0tilde_density(X, Y, mus, covariances, weights):
    """
    Calcule la densité f0_tilde(x, y) d’un mélange de gaussiennes pondérées sur une grille.

    Paramètres :
        - X, Y : grilles 2D issues de np.meshgrid
        - mus (list of ot.Point) : moyennes des gaussiennes
        - covariances (list of ot.CovarianceMatrix) : matrices de covariance
        - weights (array) : poids associés à chaque composante

    Retour :
        - Z : tableau 2D des valeurs de la densité sur la grille
    """
    Z = np.zeros_like(X)
    points = np.column_stack((X.ravel(), Y.ravel()))
    
    for w, mu, Sigma in zip(weights, mus, covariances):
        gaussian = ot.Normal(mu, Sigma)
        Z += w * np.array([gaussian.computePDF(ot.Point(p)) for p in points]).reshape(X.shape)

    return Z




