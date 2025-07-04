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


def define_zonage_grid(n_rows, n_cols, x_bounds, y_bounds):
    """
    Définie une grille de zonage sismotectonique.

    Paramètres :
        - n_rows : nombre de lignes de la grille
        - n_cols : nombre de colonnes de la grille
        - x_bounds : tuple (xmin, xmax)
        - y_bounds : tuple (ymin, ymax)

    Retourne :
        - zones : liste de tuples ((x0, x1), (y0, y1)) définissant les sous-zones
        - areas : array des aires des sous-zones
    """

    x_edges = np.linspace(*x_bounds, n_cols + 1)
    y_edges = np.linspace(*y_bounds, n_rows + 1)
    zones = []
    for i in range(n_rows):
        for j in range(n_cols):
            zones.append(((x_edges[j], x_edges[j + 1]), (y_edges[i], y_edges[i + 1])))
    area = (x_edges[1] - x_edges[0]) * (y_edges[1] - y_edges[0])
    return zones, np.full(n_rows * n_cols, area)


def compute_f0_density(x, y, zones, weights, areas):
    """
    Évalue la densité f_0 du zonage sismotectonique à un point (x, y).

    Paramètres :
        - x, y : coordonées du point
        - zones : liste des sous-zones [(x0,x1),(y0,y1)]
        - weights : poids associés à chaque zone
        - areas : aire de chaque zone

    Retourne :
        - f_0(x, y) : densité 
    """

    for idx, ((x0, x1), (y0, y1)) in enumerate(zones):
        if x0 <= x < x1 and y0 <= y < y1:
            return weights[idx] / np.sum(weights * areas)
    return 0.0



