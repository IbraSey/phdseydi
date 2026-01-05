from pathlib import Path
import os, sys
ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT))
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
from shapely.geometry import Polygon


def stick_breaking(alpha, tau=1e-3):
    """
    Generate stick-breaking weights for a Dirichlet process.

    Parameters
    ----------
    alpha : float
        Concentration parameter of the Dirichlet process.

    tau : float, default=1e-3
        Threshold for stopping the stick-breaking process. The process stops when the remaining stick length is below tau.

    Returns
    -------
    weights : ndarray of shape (n_components,)
        Normalized weights sampled via the stick-breaking construction.
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
    """
    Sample a mean and covariance pair from a Normal-Inverse-Wishart (NIW) distribution.

    Parameters
    ----------
    mu_0 : array-like of shape (d,)
        Mean of the prior distribution.

    lambda_0 : float
        Precision parameter for the mean.

    Psi_0 : ndarray of shape (d, d)
        Scale matrix for the Inverse-Wishart distribution.

    nu_0 : float
        Degrees of freedom for the Inverse-Wishart distribution.

    Returns
    -------
    mu : ndarray of shape (d,)
        Sampled mean vector.

    Sigma : ndarray of shape (d, d)
        Sampled covariance matrix.
    """
    Sigma = ot.InverseWishart(Psi_0, nu_0).getRealizationAsMatrix()
    mu = ot.Normal(mu_0, ot.CovarianceMatrix(Sigma / lambda_0)).getRealization()
    return mu, Sigma


def sample_mixture_niw(means_base, weights_base, lambda_0, Psi_0, nu_0):
    """
    Sample a mean and covariance pair from a mixture of Normal-Inverse-Wishart (NIW) priors.

    Parameters
    ----------
    means_base : list of arrays
        List of mean vectors for each component in the mixture.

    weights_base : array-like of shape (n_components,)
        Mixture weights for each component.

    lambda_0 : float
        Precision parameter of the normal distribution in NIW.

    Psi_0 : list of ndarrays
        List of scale matrices for the inverse-Wishart distribution, one per component.

    nu_0 : float
        Degrees of freedom for the inverse-Wishart distribution.

    Returns
    -------
    mu : ndarray
        Sampled mean vector.

    Sigma : ndarray
        Sampled covariance matrix.
    """
    base_idx = np.random.choice(len(means_base), p=weights_base)
    mu_0 = ot.Point(means_base[base_idx])
    Sigma = ot.InverseWishart(Psi_0[base_idx], nu_0).getRealizationAsMatrix()
    mu = ot.Normal(mu_0, ot.CovarianceMatrix(Sigma / lambda_0)).getRealization()
    return mu, Sigma


class DirichletProcessMixtureModel:
    """
    Dirichlet Process Mixture Model using stick-breaking construction and Normal-Inverse-Wishart priors.

    Parameters
    ----------
    alpha : float
        Concentration parameter of the Dirichlet process.

    tau : float
        Threshold for stopping the stick-breaking construction.

    G0_sampler : callable
        Function to sample from the base distribution G₀.

    G0_kwargs : dict
        Keyword arguments passed to the base distribution sampler.

    Attributes
    ----------
    weights : ndarray
        Mixture weights obtained via stick-breaking.

    components : list of (mu, Sigma)
        List of Gaussian parameters sampled from the base distribution.

    zones : list or None
        List of zones if defined via `from_zonage`.

    zone_weights : array or None
        Associated weights for each zone.

    zone_areas : array or None
        Area of each zone.

    zonage_defined : bool
        Whether the model was constructed using a zoned prior.
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
        """
        Sample points from the Dirichlet process mixture model.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.

        Returns
        -------
        samples : ndarray of shape (n_samples, 2)
            Sampled points.
        """
        samples = []
        for _ in range(n_samples):
            idx = np.random.choice(len(self.weights), p=self.weights)
            mu_k, sigma_k = self.components[idx]
            pt = ot.Normal(mu_k, sigma_k).getRealization()
            samples.append([pt[0], pt[1]])
        return np.array(samples)

    def evaluate_density(self, X, Y):
        """
        Evaluate the mixture density on a 2D meshgrid.

        Parameters
        ----------
        X : ndarray
            Grid values along the x-axis.

        Y : ndarray
            Grid values along the y-axis.

        Returns
        -------
        Z : ndarray of shape equal to X.shape
            Density values evaluated on the meshgrid.
        """
        mesh_pts = np.column_stack((X.ravel(), Y.ravel()))
        mixture = ot.Mixture(
            [ot.Normal(mu, sigma) for mu, sigma in self.components],
            self.weights
        )
        Z = [mixture.computePDF(ot.Point(p)) for p in mesh_pts]
        return np.array(Z).reshape(X.shape)

    def plot_density(self, X, Y, ax=None, title='Densité du DPMM', cmap='viridis'):
        """
        Plot the evaluated density on a 2D grid.

        Parameters
        ----------
        X : ndarray
            Meshgrid for x-axis.

        Y : ndarray
            Meshgrid for y-axis.

        ax : matplotlib.axes.Axes or None
            Axis on which to plot. If None, uses current axis.

        title : str, default='Densité du DPMM'
            Title of the plot.

        cmap : str, default='viridis'
            Colormap used for the density plot.
        """
        Z = self.evaluate_density(X, Y)
        if ax is None:
            ax = plt.gca()
        im = ax.imshow(Z, extent=(0, 2, 0, 2), origin='lower', cmap=cmap)
        plt.colorbar(im, ax=ax)
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.grid(True)

    def plot_samples(self, n_samples=1000, ax=None, title='Échantillons', s=5):
        """
        Plot a scatter of samples drawn from the model.

        Parameters
        ----------
        n_samples : int, default=1000
            Number of samples to draw.

        ax : matplotlib.axes.Axes or None
            Axis on which to plot. If None, uses current axis.

        title : str, default='Échantillons'
            Title of the plot.

        s : float, default=5
            Marker size for scatter plot.
        """
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
        """
        Return a dictionary describing the base distribution G0.

        Returns
        -------
        prior_info : dict
            Dictionary with sampler function, prior type, and parameters.
        """
        is_informative = self.G0_sampler.__name__ == 'sample_mixture_niw'
        return {
            'sampler': self.G0_sampler,
            'type': 'informatif' if is_informative else 'non-informatif',
            'params': self.G0_kwargs
        }

    @classmethod
    def from_regular_zonage(cls, alpha, tau, n_rows, n_cols, lambda_0, nu_0, zone_weights=None):
        """
        Create a DPMM instance with an informative prior based on spatial zonation.

        Parameters
        ----------
        alpha : float
            Concentration parameter of the Dirichlet process.

        tau : float
            Truncation threshold for the stick-breaking process.

        n_rows : int
            Number of rows in the zonation grid.

        n_cols : int
            Number of columns in the zonation grid.

        lambda_0 : float
            Precision parameter for the NIW prior.

        nu_0 : float
            Degrees of freedom for the Inverse-Wishart distribution.

        zone_weights : array-like or None, default=None
            Custom weights for each zone. If None, assumes uniform weights.

        Returns
        -------
        instance : DirichletProcessMixtureModel
            Initialized DPMM with informative prior from zonation.
        """

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

        # 6. Mémoriser les infos de zonage et de l'approximation par des gaussiennes 
        instance = cls(alpha=alpha, tau=tau, G0_sampler=sample_mixture_niw, G0_kwargs=G0_kwargs)

        instance.zones = zones
        instance.zone_weights = zone_weights
        instance.zone_areas = zone_areas
        instance.zonage_defined = True
        instance.gaussian_centroids = mus
        instance.gaussian_covariances = covs
        instance.weights_base = weights_base

        return instance
    
    @classmethod
    def from_irregular_zonage(cls, alpha, tau, zones, zone_weights, lambda_0, nu_0):

        if len(zone_weights) != len(zones):
            raise ValueError(f"zone_weights doit avoir {n_zones} éléments.")
        elif np.sum(zone_weights) != 1:
            raise ValueError(f"zone_weights doit sommer à 1.")

        zone_areas = np.array([poly.area for poly in zones])
        n_zones = len(zones)

        X = sample_from_f0(n_samples=10000, zones=zones, weights=zone_weights, areas=zone_areas, irregular=True)

        mus, covs, weights_base, _ = compute_zone_gaussian_parameters(X, n_components=n_zones)

        Psi_0_list = [ot.CovarianceMatrix(Sigma * (nu_0 - 3)) for Sigma in covs]

        G0_kwargs = dict(
            means_base=[ot.Point(mu) for mu in mus],
            weights_base=weights_base.tolist(),
            lambda_0=lambda_0,
            Psi_0=Psi_0_list,
            nu_0=nu_0
        )

        instance = cls(alpha=alpha, tau=tau, G0_sampler=sample_mixture_niw, G0_kwargs=G0_kwargs)

        instance.zones = zones
        instance.zone_weights = zone_weights
        instance.zone_areas = zone_areas
        instance.zonage_defined = True
        instance.gaussian_centroids = mus
        instance.gaussian_covariances = covs
        instance.weights_base = weights_base

        return instance


def compute_empirical_mean_density(factory_fn, N, X, Y):
    """
    Compute the average density over N realizations of a DPMM model.

    Parameters
    ----------
    factory_fn : callable
        Function returning a new instance of a DPMM model with `.evaluate_density(X, Y)`.

    N : int
        Number of realizations to average over.

    X : ndarray
        Grid values along the x-axis.

    Y : ndarray
        Grid values along the y-axis.

    Returns
    -------
    Z_mean : ndarray of shape equal to X.shape
        Empirical average of the densities evaluated at each grid point.
    """
    Z_sum = np.zeros_like(X)
    for _ in range(N):
        dpmm = factory_fn()
        Z_sum += dpmm.evaluate_density(X, Y)
    return Z_sum / N






n = 10000


def hyperparameters_posterior_niw(mu_0, lambda_0, Psi_0, nu_0, k, x_bar, S):
    """
    
    """
    if k == 0:
        return mu_0, lambda_0, Psi_0, nu_0
    
    lambda_k = lambda_0 + k
    mu_k = (lambda_0*mu_0 + k*x_bar) / lambda_k
    nu_k = nu_0 + k
    delta = x_bar - mu_0
    Psi_k = Psi_0 + S + (lambda_0 * k / lambda_k) * np.outer(delta, delta)

    return mu_k, lambda_k, Psi_k, nu_k


def prob_cluster(x_i, mu_k, Sigma_k, alpha, k, xbar_k, S_k):
    """
    
    """
    pri = k / n+alpha-1
    dist = ot.Normal(mu_k, Sigma_k)
    lik = dist.computePDF(x_i)

    return pri * lik





class DPMMGibbsSampler:
    """
    
    """






# %%
