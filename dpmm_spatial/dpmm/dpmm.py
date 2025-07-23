import numpy as np
import openturns as ot
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from numpy.linalg import pinv
from dpmm.prior_utils import (
                                compute_zone_gaussian_parameters, 
                                define_zonage_grid,
                                sample_from_f0
                                )


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


def sample_niw(mu_0, lambda_0, Psi_0, nu_0):
    Sigma = ot.InverseWishart(Psi_0, nu_0).getRealizationAsMatrix()
    mu = ot.Normal(mu_0, ot.CovarianceMatrix(Sigma / lambda_0)).getRealization()
    return mu, Sigma


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
    Sigma = ot.InverseWishart(Psi_0[base_idx], nu_0).getRealizationAsMatrix()
    mu = ot.Normal(mu_0, ot.CovarianceMatrix(Sigma / lambda_0)).getRealization()
    return mu, Sigma


class DirichletProcessMixtureModel:
    """

    """
    def __init__(self, alpha, tau, G0_sampler, G0_kwargs):
        self.alpha = alpha
        self.tau = tau
        self.G0_sampler = G0_sampler
        self.G0_kwargs = G0_kwargs
        self.weights = stick_breaking(alpha, tau)
        self.components = [G0_sampler(**G0_kwargs) for _ in range(len(self.weights))]
        self.mixture = None

        # Infos de zonage 
        self.zones = None
        self.zone_weights = None
        self.zone_areas = None
        self.zonage_defined = False


    def sample(self, n_samples):
        samples = []
        for _ in range(n_samples):
            idx = np.random.choice(len(self.weights), p=self.weights)
            mu_k, sigma_k = self.components[idx]
            pt = ot.Normal(mu_k, sigma_k).getRealization()
            samples.append([pt[0], pt[1]])
        return np.array(samples)

    def evaluate_density(self, X, Y):
        mesh_pts = np.column_stack((X.ravel(), Y.ravel()))
        mixture = ot.Mixture(
            [ot.Normal(mu, sigma) for mu, sigma in self.components],
            self.weights
        )
        Z = [mixture.computePDF(ot.Point(p)) for p in mesh_pts]
        return np.array(Z).reshape(X.shape)

    def plot_density(self, X, Y, ax=None, title='Densité du DPMM', cmap='viridis'):
        Z = self.evaluate_density(X, Y)
        if ax is None:
            ax = plt.gca()
        im = ax.imshow(Z, extent=(0, 2, 0, 2), origin='lower', cmap=cmap)
        plt.colorbar(im, ax=ax)
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.grid(True)

    def plot_samples(self, n_samples=1000, ax=None, title='Échantillons', s=5):
        samples = self.sample(n_samples)
        if ax is None:
            ax = plt.gca()
        ax.scatter(samples[:, 0], samples[:, 1], s=s, alpha=0.4)
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.grid(True)

    def get_prior(self):
        is_informative = self.G0_sampler.__name__ == 'sample_mixture_niw'
        return {
            'sampler': self.G0_sampler,
            'type': 'informatif' if is_informative else 'non-informatif',
            'params': self.G0_kwargs
        }

    @classmethod
    def from_zonage(cls, alpha, tau, n_rows, n_cols, lambda_0, nu_0, zone_weights=None, seed=0):
        np.random.seed(seed)

        # 1. Définir le zonage
        zones, zone_areas = define_zonage_grid(n_rows, n_cols)
        n_zones = len(zones)

        # 2. Poids personnalisés ou uniformes
        if zone_weights is None:
            zone_weights = np.ones(n_zones) / n_zones
        else:
            zone_weights = np.asarray(zone_weights)
            if len(zone_weights) != n_zones:
                raise ValueError(f"zone_weights doit avoir {n_zones} éléments.")
            elif np.sum(zone_weights) != 1:
                raise ValueError(f"zone_weights doit sommer à 1.")
            zone_weights = zone_weights

        # 3. Échantillons simulés depuis f₀
        n_samples = 10000
        X = sample_from_f0(n_samples=n_samples, zones=zones, weights=zone_weights, areas=zone_areas)

        # 4. Estimation des composantes gaussiennes
        mus, covs, weights_base, _ = compute_zone_gaussian_parameters(X, n_components=n_zones)

        # 5. Définir G₀ (prior informatif)

        # Calcul Psi_0 individuellement pour chaque composante
        Psi_0_list = [                                  
            ot.CovarianceMatrix(Sigma * (nu_0 - 3))  # dimension = 2
            for Sigma in covs
        ]

        G0_kwargs = dict(
            means_base=[ot.Point(mu) for mu in mus],
            weights_base=weights_base.tolist(),
            lambda_0=lambda_0,
            Psi_0=Psi_0_list,
            nu_0=nu_0
        )

        # 7. Mémoriser les infos de zonage et de l'approximation par des gaussiennes 
        instance = cls(alpha=alpha, tau=tau, G0_sampler=sample_mixture_niw, G0_kwargs=G0_kwargs)

        instance.zones = zones
        instance.zone_weights = zone_weights
        instance.zone_areas = zone_areas
        instance.zonage_defined = True
        instance.gaussian_centroids = mus
        instance.gaussian_covariances = covs
        instance.weights_base = weights_base

        return instance




