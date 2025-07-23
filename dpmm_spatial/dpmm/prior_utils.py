import numpy as np
import openturns as ot
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from numpy.linalg import pinv


def define_zonage_grid(n_rows, n_cols, x_range=(0, 2), y_range=(0, 2)):
    """
    Define a rectangular zonation grid.

    Parameters
    ----------
    n_rows : int
        Number of horizontal zones (rows) in the grid.
    n_cols : int
        Number of vertical zones (columns) in the grid.
    x_range : tuple of float, default=(0, 2)
        Range of x-axis domain as (min_x, max_x).
    y_range : tuple of float, default=(0, 2)
        Range of y-axis domain as (min_y, max_y).

    Returns
    -------
    zones : list of tuple
        List of zones, each represented by ((x_min, x_max), (y_min, y_max)).
    areas : ndarray of shape (n_rows * n_cols,)
        Area of each rectangular zone.
    """
    
    x_bounds = np.linspace(x_range[0], x_range[1], n_cols + 1)
    y_bounds = np.linspace(y_range[0], y_range[1], n_rows + 1)
    zones, areas = [], []
    for i in range(n_rows):
        for j in range(n_cols):
            x0, x1 = x_bounds[j], x_bounds[j + 1]
            y0, y1 = y_bounds[i], y_bounds[i + 1]
            zones.append(((x0, x1), (y0, y1)))
            areas.append((x1 - x0) * (y1 - y0))
    return zones, np.array(areas)


def compute_zone_gaussian_parameters(X, n_components):
    """
    Estimate Gaussian mixture parameters using KMeans initialization.

    Parameters
    ----------
    X : ndarray of shape (n_samples, 2)
        Input data points in 2D space.
    n_components : int
        Number of Gaussian components to fit.

    Returns
    -------
    means : ndarray of shape (n_components, 2)
        Estimated means of the Gaussian components.
    covariances : ndarray of shape (n_components, 2, 2)
        Estimated covariance matrices of the components.
    weights : ndarray of shape (n_components,)
        Estimated weights of each component.
    gmm : GaussianMixture object
        The fitted sklearn GaussianMixture instance.
    """

    kmeans = KMeans(n_clusters=n_components)
    labels = kmeans.fit_predict(X)
    init_means = kmeans.cluster_centers_
    covariances = []
    for i in range(n_components):
        cluster_points = X[labels == i]
        if len(cluster_points) < 2:
            cov = np.eye(2) * 1e-2  # fallback si peu de points
        else:
            cov = np.cov(cluster_points.T) + 1e-6 * np.eye(2)
        covariances.append(cov)
    init_precisions = np.array([pinv(cov) for cov in covariances])

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        means_init=init_means,
        precisions_init=init_precisions
    )
    gmm.fit(X)

    return gmm.means_, gmm.covariances_, gmm.weights_, gmm


def sample_from_f0(n_samples, zones, weights, areas):
    """
    Generate samples from a piecewise-uniform spatial density.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    zones : list of tuple
        List of rectangular zones defined by ((x0, x1), (y0, y1)).
    weights : ndarray of shape (n_zones,)
        Weight associated with each zone.
    areas : ndarray of shape (n_zones,)
        Area of each zone.

    Returns
    -------
    samples : ndarray of shape (n_samples, 2)
        Random samples drawn uniformly within the zones.
    """
    
    samples = []
    for _ in range(n_samples):
        idx = np.random.choice(len(zones), p=weights)
        (x0, x1), (y0, y1) = zones[idx]
        x = ot.Uniform(x0, x1).getRealization()[0]
        y = ot.Uniform(y0, y1).getRealization()[0]
        samples.append([x, y])
    return np.array(samples)


def sample_from_f0tilde(n_samples, mus, covariances, weights):
    """
    Generate samples from a Gaussian mixture density.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    mus : ndarray of shape (n_components, 2)
        Means of the Gaussian components.
    covariances : ndarray of shape (n_components, 2, 2)
        Covariance matrices of the components.
    weights : ndarray of shape (n_components,)
        Weights of the Gaussian components.

    Returns
    -------
    samples : ndarray of shape (n_samples, 2)
        Samples drawn from the Gaussian mixture.
    """

    samples = []
    for _ in range(n_samples):
        idx = np.random.choice(len(weights), p=weights)
        mu = ot.Point(mus[idx])  # un seul centroïde
        cov = ot.CovarianceMatrix(covariances[idx])  # une seule matrice 2x2
        sample = ot.Normal(mu, cov).getRealization()
        samples.append([sample[0], sample[1]])
    return np.array(samples)


def compute_f0_density(X, Y, zones, weights, areas):
    """
    Evaluate a piecewise-uniform density f₀ over a mesh grid.

    Parameters
    ----------
    X : ndarray of shape (n_rows, n_cols)
        Grid of x-coordinates.
    Y : ndarray of shape (n_rows, n_cols)
        Grid of y-coordinates.
    zones : list of tuple
        List of zones defined by ((x0, x1), (y0, y1)).
    weights : ndarray of shape (n_zones,)
        Weights for each zone.
    areas : ndarray of shape (n_zones,)
        Area of each zone.

    Returns
    -------
    Z : ndarray of shape (n_rows, n_cols)
        Density values of f₀ evaluated over the mesh grid.
    """

    Z = np.zeros_like(X)

    for idx, ((x0, x1), (y0, y1)) in enumerate(zones):
        mask = (X >= x0) & (X < x1) & (Y >= y0) & (Y < y1)
        Z[mask] = weights[idx] / areas[idx] 

    return Z


def compute_f0tilde_density(X, Y, mus, covariances, weights):
    """
    Evaluate the density of a Gaussian mixture on a grid.

    Parameters
    ----------
    X : ndarray of shape (n, m)
        Grid of x-values (output of np.meshgrid).

    Y : ndarray of shape (n, m)
        Grid of y-values (output of np.meshgrid).

    mus : ndarray of shape (n_components, 2)
        Gaussian component means.

    covariances : ndarray of shape (n_components, 2, 2)
        Covariance matrices of the Gaussian components.

    weights : ndarray of shape (n_components,)
        Weights associated to each Gaussian component.

    Returns
    -------
    Z : ndarray of shape (n, m)
        Density values on the grid.
    """
    Z = np.zeros_like(X)
    points = np.column_stack((X.ravel(), Y.ravel()))

    for w, mu, Sigma in zip(weights, mus, covariances):
        mu_ot = ot.Point(mu)
        Sigma_ot = ot.CovarianceMatrix(Sigma)
        gaussian = ot.Normal(mu_ot, Sigma_ot)
        Z += w * np.array([gaussian.computePDF(ot.Point(p)) for p in points]).reshape(X.shape)

    return Z







