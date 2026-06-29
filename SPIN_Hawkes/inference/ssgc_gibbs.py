import math
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import openturns as ot
import openturns.experimental as otexp
from polyagamma import random_polyagamma
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from scipy.special import expit
from shapely.geometry import Point as ShapelyPoint

from ..data.catalog import EventCatalog
from ..config import ETASInferenceConfig, ETASParameters, GPParameters, MCMCConfig
from ..spatial.domain import DomainPartition
from ..models import SPINHModel, SSGCModel

from .backends import ExactGPBackend, GPBackend, SparseGP
from .base import InferenceMethod
from .results import GibbsResults

from ..visualization import plot_field, save_figure

class SSGC_GibbsSampler:
    """Gibbs sampler for the spatially structured sigmoidal Gaussian Cox process.
    
    The background intensity is
    
        mu(x, y) = mu_tilde(x, y) * sigmoid(f(x, y)),
    
    where ``mu_tilde(x, y) = exp(eps_j)`` on zone ``S_j`` and ``f`` is a
    zero-mean Gaussian process with squared-exponential covariance. The sampler
    uses Polya-Gamma augmentation and alternates between auxiliary marks, the
    latent thinned Poisson process, the GP values, the zonal log-intensities, and
    optionally the GP hyperparameters.
    
    Parameters
    ----------
    X_bounds, Y_bounds : tuple of float
        Bounds ``(minimum, maximum)`` of the rectangular observation window.
    T : float
        Duration of the temporal observation window. Intensities are interpreted
        per unit area and per unit time.
    Areas : sequence of tuple
        Pairs ``(prepared_polygon, eps_init)`` defining the spatial partition and
        initial zonal log-intensities.
    lambda_nu : float
        Rate of the independent exponential priors on ``nu = (v_squared, ell)``.
    nu : array_like, shape (2,)
        Initial GP marginal variance and OpenTURNS length scale.
    delta : array_like, shape (2,)
        Parameters ``(delta_0, delta_1)`` of the Gaussian prior covariance of
        ``eps``: marginal variance and centroid correlation length.
    polygons : sequence of shapely.geometry.Polygon
        Unprepared polygons corresponding one-to-one with ``Areas``.
    jitter : float, optional
        Diagonal numerical regularization, by default ``1e-5``.
    rng_seed : int or None, optional
        Seed passed to the OpenTURNS random generator.
    
    Attributes
    ----------
    J : int
        Number of spatial zones.
    centroids_xy : ndarray, shape (J, 2)
        Polygon centroids.
    Sigma_eps : ndarray, shape (J, J)
        Prior covariance of the zonal log-intensities.
    Sigma_eps_cov : ot.CovarianceMatrix
        Jitter-regularized OpenTURNS representation of ``Sigma_eps`` used by the
        linear solvers."""

    def __init__(
        self,
        X_bounds,
        Y_bounds,
        T,
        Areas,
        lambda_nu,
        nu,
        delta,                  # [delta0, delta1] : variance and length-scale de Sigma_eps
        polygons,               # list of shapely polygons, même format que Areas
        jitter=1e-5,
        rng_seed=None,
    ):
        """Initialize the SSGC sampler; see the class docstring for parameters."""
        self.X_bounds = tuple(X_bounds)
        self.Y_bounds = tuple(Y_bounds)
        self.T = float(T)
        self.lambda_nu = lambda_nu
        self.nu = ot.Point(nu)
        self.delta = ot.Point(delta)
        self.jitter = jitter
        if len(Areas) != len(polygons):
            raise ValueError("Areas and polygons must have the same length.")

        initial_eps = [float(area[1]) for area in Areas]
        self.domain_partition = DomainPartition.from_polygons(polygons, initial_eps)
        self.model = SSGCModel(
            domains=self.domain_partition,
            duration=self.T,
            x_bounds=self.X_bounds,
            y_bounds=self.Y_bounds,
            gp_prior=GPParameters(float(nu[0]), float(nu[1])),
            eps_prior_variance=float(delta[0]),
            eps_prior_length_scale=float(delta[1]),
            nu_prior_rate=float(lambda_nu),
            jitter=float(jitter),
        )

        self.domains = list(self.domain_partition.polygons)
        self.prepared_domains = list(self.domain_partition.prepared_domains)
        self.domain_areas = self.domain_partition.areas
        self.eps_init = self.domain_partition.initial_log_intensities
        self.n_domains = len(self.domain_partition)

        # Backward-compatible aliases used by existing notebooks.
        self.polygons = self.domains
        self.areas = self.prepared_domains
        self.epsilons = self.eps_init.tolist()
        self.J = self.n_domains
        self.Areas = list(zip(self.prepared_domains, self.epsilons))

        if rng_seed is not None:
            ot.RandomGenerator.SetSeed(int(rng_seed))
            self.rng_state = ot.RandomGenerator.GetState()

        self.centroids_xy, self.Sigma_eps = self.compute_Sigma_eps()
        Sigma_eps_reg = ot.CovarianceMatrix(
            (self.Sigma_eps + self.jitter * np.eye(self.J)).tolist()
        )
        self.Sigma_eps_cov = Sigma_eps_reg


    # =============================================================================================
    # ----------------------------------------- Outillage -----------------------------------------
    # =============================================================================================

    @staticmethod
    def _acf(x, max_lag):
        """Compute the empirical autocorrelation function of a univariate chain.
 
        Uses the biased normalisation (dividing by n rather than n−k) which is
        standard for MCMC diagnostics and guarantees a positive-semidefinite
        estimate.
 
        Parameters
        ----------
        x : array_like, shape (n,)
            Univariate time series (typically a post-burn-in MCMC chain).
        max_lag : int
            Maximum lag at which to evaluate the ACF. Clamped to n−1 if larger.
 
        Returns
        -------
        acf_vals : ndarray, shape (min(max_lag, n−1) + 1,)
            Autocorrelation values from lag 0 (always 1.0) to lag max_lag.
        """
        x = np.asarray(x)
        x = x - x.mean()
        n = len(x)

        max_lag = min(max_lag, n - 1)       # max_lag ne peut pas dépasser n-1
        if max_lag < 0:
            return np.array([1.0])

        var = np.dot(x, x) / n
        if var == 0.0:
            return np.zeros(max_lag + 1)

        acf_vals = np.empty(max_lag + 1)
        for k in range(max_lag + 1):
            acf_vals[k] = np.dot(x[:n - k], x[k:]) / (n * var)
        return acf_vals

    def compute_Sigma_eps(self):
        """Build the structured Gaussian prior covariance Σ_ε for the zonal log-intensities.
 
        The covariance is a Gaussian (squared-exponential) kernel evaluated at
        the centroids of the Voronoï zones:
 
            Σ_{jj'} = δ₀ · exp(−‖c_j − c_{j'}‖² / (2 δ₁²))
 
        where (δ₀, δ₁) = self.delta.
 
        Returns
        -------
        centroids_xy : ndarray, shape (J, 2)
            Matrix of zone centroids, row j = (c_x^j, c_y^j).
        Sigma_eps : ndarray, shape (J, J)
            Symmetric positive-semidefinite covariance matrix.
        """
        centroids_xy = self.domain_partition.centroids
        Sigma_eps = self.model.epsilon_prior_covariance()
        Sigma_eps = 0.5 * (Sigma_eps + Sigma_eps.T)
        return centroids_xy, Sigma_eps

    def compute_kernel(self, XY_data, XY_new=None):
        """Evaluate the squared-exponential GP covariance.
        
        The covariance is parameterized by ``self.nu = (v_squared, ell)``. OpenTURNS
        expects the standard-deviation amplitude, so the implementation passes
        ``sqrt(v_squared)`` to :class:`ot.SquaredExponential`.
        
        Parameters
        ----------
        XY_data : ot.Sample or array_like, shape (N, 2)
            Training or conditioning locations.
        XY_new : ot.Sample or array_like, shape (M, 2), optional
            Prediction locations.
        
        Returns
        -------
        ot.CovarianceMatrix
            The ``N x N`` Gram matrix when ``XY_new`` is ``None``.
        (K_dd, K_new_data, K_new_new) : tuple
            Training covariance, ``M x N`` cross-covariance, and prediction covariance
            when ``XY_new`` is provided."""
        nu0, nu1 = map(float, self.nu)
        sigma_amp = np.sqrt(nu0)      # OT attend sigma, pas sigma^2

        if not isinstance(XY_data, ot.Sample):
            XY_data = ot.Sample(np.asarray(XY_data).tolist())
        N_data = XY_data.getSize()
        XY_data_arr = np.asarray(XY_data)

        kernel = ot.SquaredExponential([nu1, nu1], [sigma_amp])   

        if XY_new is None:
            K = kernel.discretize(XY_data)
            return ot.CovarianceMatrix(np.array(K).tolist())

        if not isinstance(XY_new, ot.Sample):
            XY_new = ot.Sample(np.asarray(XY_new).tolist())
        N_new = XY_new.getSize()
        XY_new_arr = np.asarray(XY_new)

        XY_all = ot.Sample(N_data + N_new, 2)
        for i in range(N_data):
            XY_all[i, 0] = float(XY_data_arr[i, 0])
            XY_all[i, 1] = float(XY_data_arr[i, 1])
        for i in range(N_new):
            XY_all[N_data + i, 0] = float(XY_new_arr[i, 0])
            XY_all[N_data + i, 1] = float(XY_new_arr[i, 1])

        K_all = np.asarray(kernel.discretize(XY_all), dtype=float)

        K_dd = ot.CovarianceMatrix(K_all[:N_data, :N_data].tolist())
        K_new_data = ot.Matrix(K_all[N_data:, :N_data].tolist())
        K_new_new = ot.CovarianceMatrix(K_all[N_data:, N_data:].tolist())

        return K_dd, K_new_data, K_new_new

    def compute_mu_tilde(self, XY, eps=None):
        """Evaluate the piecewise-constant baseline intensity μ̃(x,y) = exp(ε_j).
 
        For each query point, the method identifies the enclosing zone S_j via
        a sequential polygon containment test and returns exp(ε_j). Points that
        fall outside all zones receive μ̃ = 0.
 
        Parameters
        ----------
        XY : ot.Sample or array_like, shape (n, 2)
            Query locations.
        eps : array_like, shape (J,), optional
            Zonal log-intensities. If None, uses the initial values stored in
            self.epsilons.
 
        Returns
        -------
        mu_vals : ndarray, shape (n,)
            Baseline intensity at each query point.
        """
        points = np.asarray(XY, dtype=float)
        eps_values = self.eps_init if eps is None else np.asarray(eps)
        return self.model.baseline_intensity(
            points[:, 0], points[:, 1], eps_values
        )

    def sample_candidats(self, N):
        """Draw N points uniformly over the rectangular bounding box of the domain.
 
        This is a utility function used for rejection-based spatial sampling.
        Points may fall outside the actual study domain (union of polygons);
        the caller is responsible for subsequent containment checks.
 
        Parameters
        ----------
        N : int
            Number of candidate points to draw.
 
        Returns
        -------
        candidates : ot.Sample, shape (N, 2)
            Uniformly distributed points in [x_min, x_max] × [y_min, y_max].
        """
        xmin, xmax = self.X_bounds
        ymin, ymax = self.Y_bounds
        distribution = ot.JointDistribution(
            [ot.Uniform(xmin, xmax), ot.Uniform(ymin, ymax)]
        )
        return distribution.getSample(int(N))

    def _log_posterior_nu(self, nu_vals, f_Df, D_f_sample):
        """Evaluate the unnormalised log-posterior of the GP kernel hyperparameters.
 
        The posterior factorises as
 
            log p(ν | f, D_f) ∝ log p(f | ν, D_f) + log p(ν),
 
        where p(f | ν, D_f) = N(f; 0, K_{ff}(ν)) is the GP marginal likelihood
        and p(ν) = Exp(λ_ν) ⊗ Exp(λ_ν) is the independent exponential prior on
        each component of ν = (v², ℓ).
 
        Parameters
        ----------
        nu_vals : list or ot.Point, shape (2,)
            Candidate hyperparameters [v², ℓ].
        f_Df : ot.Point, shape (N_f,)
            Current GP realization at the augmented data locations D_f = D₀ ∪ π_S.
        D_f_sample : ot.Sample, shape (N_f, 2)
            Spatial coordinates of the augmented data locations.
 
        Returns
        -------
        log_post : float
            Unnormalised log-posterior log p(ν | f, D_f) (up to a constant).
        """
        nu0, nu1 = map(float, nu_vals)
        log_prior = -self.lambda_nu * (nu0 + nu1)
        kernel = ot.SquaredExponential([nu1, nu1], [nu0])
        N = D_f_sample.getSize()
        K_mat = kernel.discretize(D_f_sample)
        for i in range(N):
            K_mat[i, i] += self.jitter
        K_ff = ot.CovarianceMatrix(K_mat)
        m_f = ot.Point(N, 0.0)
        log_likelihood = ot.Normal(m_f, K_ff).computeLogPDF(f_Df)

        return log_likelihood + log_prior
    
    def _log_posterior_eps(self, eps_arr, N_j, M_j):
        """Evaluate the unnormalised log-conditional-posterior of the zonal log-intensities.
 
        Combines the structured Gaussian prior ε ~ N(0, Σ_ε) with the Poisson
        likelihood contribution from the augmented data:
 
            log p(ε | ·) ∝ −½ εᵀ Σ_ε⁻¹ ε + Σ_j [(N_j + M_j) ε_j − T |S_j| exp(ε_j)]
 
        where N_j counts observed background events and M_j counts latent thinned
        events in zone S_j.
 
        Parameters
        ----------
        eps_arr : ndarray, shape (J,)
            Current zonal log-intensities.
        N_j : ndarray, shape (J,)
            Number of observed background events per zone.
        M_j : ndarray, shape (J,)
            Number of latent marked Poisson process events per zone.
 
        Returns
        -------
        log_post : float
            Unnormalised log-conditional-posterior value.
        """
        areas_j = np.array([self.polygons[j].area for j in range(self.J)])
        eps_solved = np.array(self.Sigma_eps_cov.solveLinearSystem(ot.Point(eps_arr.tolist())))
        prior_term = -0.5 * eps_arr @ eps_solved
        likelihood_term = np.sum(
            (N_j + M_j) * eps_arr - self.T * areas_j * np.exp(eps_arr)
        )
        return prior_term + likelihood_term

    def _grad_log_posterior_eps(self, eps_arr, N_j, M_j):
        """Compute the gradient of the log-conditional-posterior of ε.
 
        The gradient decomposes into a prior contribution and a likelihood
        contribution:
 
            ∇_ε log p(ε | ·) = (N_j + M_j) − T |S_j| exp(ε_j) − Σ_ε⁻¹ ε
 
        This closed-form gradient is used in the MALA proposal for ε.
 
        Parameters
        ----------
        eps_arr : ndarray, shape (J,)
            Current zonal log-intensities.
        N_j : ndarray, shape (J,)
            Number of observed background events per zone.
        M_j : ndarray, shape (J,)
            Number of latent marked Poisson process events per zone.
 
        Returns
        -------
        grad : ndarray, shape (J,)
            Gradient vector ∇_ε log p(ε | ·).
        """
        areas_j = np.array([self.polygons[j].area for j in range(self.J)])
        prior_grad = np.array(self.Sigma_eps_cov.solveLinearSystem(ot.Point(eps_arr.tolist())))
        return (N_j + M_j) - self.T * areas_j * np.exp(eps_arr) - prior_grad

    def _compute_event_domain_indices(self, x, y):
        """Assign observed locations to spatial domains and cache the result.

        Returns an integer index per location; ``-1`` denotes a point outside
        the observation domain.
        """
        x_arr = np.asarray([float(value) for value in x], dtype=float)
        y_arr = np.asarray([float(value) for value in y], dtype=float)
        signature = (x_arr.shape, x_arr.tobytes(), y_arr.tobytes())
        cache = getattr(self, "_event_domain_indices_cache", None)
        if cache is not None and cache["signature"] == signature:
            return cache["domain_indices"]

        domain_indices = np.full(x_arr.size, -1, dtype=int)
        for index, (x_value, y_value) in enumerate(zip(x_arr, y_arr)):
            point = ShapelyPoint(float(x_value), float(y_value))
            for domain_index, polygon in enumerate(self.areas):
                if polygon.covers(point):
                    domain_indices[index] = domain_index
                    break
        self._event_domain_indices_cache = {
            "signature": signature,
            "domain_indices": domain_indices,
        }
        return domain_indices

    def _count_events_per_zone(self, x, y, Z, Pi_S):
        """Count observed background events and latent thinned events per zone.
 
        Iterates over all observed events and latent Poisson process points,
        assigning each to its enclosing zone via polygon containment tests.
 
        Parameters
        ----------
        x, y : array_like, shape (N,)
            Spatial coordinates of observed events.
        Z : ot.Point, shape (N,)
            Branching labels: Z[i] = 0 indicates a background event, Z[i] = j > 0
            indicates an event triggered by parent j.
        Pi_S : ot.Sample, shape (M, 3)
            Latent marked Poisson process realization. Columns are (x, y, ω).
 
        Returns
        -------
        N_j : ndarray, shape (J,)
            Number of observed background events (z_i = 0) in each zone.
        M_j : ndarray, shape (J,)
            Number of latent thinned events (from π_S) in each zone.
        """
        N_j = np.zeros(self.J)
        M_j = np.zeros(self.J)

        zones = self._compute_event_domain_indices(x, y)
        z_arr = np.asarray([float(Z[i]) for i in range(len(Z))], dtype=float)
        bg_zones = zones[(z_arr == 0.0) & (zones >= 0)]
        if bg_zones.size:
            N_j += np.bincount(bg_zones, minlength=self.J)[:self.J]

        for m in range(Pi_S.getSize()):
            pt = ShapelyPoint(float(Pi_S[m, 0]), float(Pi_S[m, 1]))
            for j, poly in enumerate(self.areas):
                if poly.covers(pt):
                    M_j[j] += 1
                    break

        return N_j, M_j


    # ==============================================================================================
    # --------------------------------- Posteriors conditionnelles ---------------------------------
    # ==============================================================================================

    def update_f(self, x, y, Z, omega_D0, Pi_S):
        """Sample the latent GP f from its Gaussian conditional posterior.
 
        Thanks to the Pólya-Gamma augmentation, the conditional posterior of f
        given the auxiliary variables is Gaussian:
 
            f | Ω, κ ~ N(Σ_post · κ,  Σ_post),
            Σ_post = (K_{ff}⁻¹ + Ω)⁻¹,
 
        where Ω = diag(ω) collects the PG auxiliary variables and κ_k = +½ for
        observed background events and −½ for latent thinned events.
 
        Parameters
        ----------
        x, y : array_like, shape (N,)
            Spatial coordinates of all observed events.
        Z : ot.Point, shape (N,)
            Current branching labels (Z[i] = 0 for background events).
        omega_D0 : ot.Point, shape (N,)
            Pólya-Gamma auxiliary variables at observed event locations.
        Pi_S : ot.Sample, shape (M, 3)
            Current latent marked Poisson process realization (x, y, ω).
 
        Returns
        -------
        f_D0 : ot.Point, shape (N₀,)
            GP draw restricted to observed background event locations.
        f_Df : ot.Point, shape (N_f,)
            GP draw at all augmented locations D_f = D₀ ∪ π_S.
        D_f : ot.Sample, shape (N_f, 2)
            Spatial coordinates of D_f.
        K_ff : ot.CovarianceMatrix, shape (N_f, N_f)
            Prior kernel matrix at D_f (with jitter).
 
        Raises
        ------
        ValueError
            If no background events are found (N₀ = 0).
        """
        idx = [i for i in range(len(Z)) if Z[i] == 0.0]
        N_0 = len(idx)
        if N_0 == 0:
            raise ValueError("N_0 = 0 : pas de background events.")

        # 1) D_0
        D_0 = ot.Sample(N_0, 2)
        omega_D_0 = ot.Point(N_0)
        for k, i in enumerate(idx):
            D_0[k, 0] = x[i]
            D_0[k, 1] = y[i]
            omega_D_0[k] = omega_D0[i]

        # 2) Pi_S
        N_Pi = Pi_S.getSize()
        if N_Pi > 0:
            PiS_xy = ot.Sample(N_Pi, 2)
            omega_Pi = ot.Point(N_Pi)
            for i in range(N_Pi):
                PiS_xy[i, 0] = Pi_S[i, 0]
                PiS_xy[i, 1] = Pi_S[i, 1]
                omega_Pi[i] = Pi_S[i, 2]
        else:
            PiS_xy = ot.Sample(0, 2)
            omega_Pi = ot.Point(0)

        # 3) D_f = D_0 ∪ Pi_S
        N_f = N_0 + N_Pi
        D_f = ot.Sample(N_f, 2)
        for i in range(N_0):
            D_f[i, 0] = D_0[i, 0]
            D_f[i, 1] = D_0[i, 1]
        for i in range(N_Pi):
            D_f[N_0 + i, 0] = PiS_xy[i, 0]
            D_f[N_0 + i, 1] = PiS_xy[i, 1]

        # 4) K_ff sur D_f
        K_ff = self.compute_kernel(D_f)
        for i in range(N_f):
            K_ff[i, i] += self.jitter
        # 5) Omega = diag(omega_{D_0}, omega_{Pi_S})
        Omega = ot.CovarianceMatrix(N_f)
        for i in range(N_0):
            Omega[i, i] = omega_D_0[i]
        for i in range(N_Pi):
            Omega[N_0 + i, N_0 + i] = omega_Pi[i]

        # 6) kappa : +1/2 pour D_0, -1/2 pour Pi_S
        kappa = ot.Point(N_f)
        for i in range(N_0):
            kappa[i] = 0.5
        for i in range(N_Pi):
            kappa[N_0 + i] = -0.5

        # 7) Posterior sur D_f.
        # (K^{-1}+Omega)^{-1} = (I + K Omega)^{-1} K, solved with OpenTURNS.
        K_arr = np.array(K_ff)
        Omega_arr = np.array(Omega)
        left_arr = np.eye(N_f) + K_arr @ Omega_arr
        left = ot.Matrix(left_arr.tolist())
        Sigma_arr = np.array(left.solveLinearSystem(ot.Matrix(K_arr.tolist())))
        Sigma_arr = 0.5 * (Sigma_arr + Sigma_arr.T) + self.jitter * np.eye(N_f)
        Sigma_post = ot.CovarianceMatrix(Sigma_arr.tolist())
        mu_post = left.solveLinearSystem(K_ff * kappa)

        # 8) Tirage sur D_f entier
        f_Df = ot.Normal(mu_post, Sigma_post).getRealization()

        # Restriction explicite à D_0 (indices 0..N_0-1)
        # Les indices N_0..N_f-1 correspondent à Pi_S et doivent pas être utilisés comme vecteur de conditionnement à l'itération suivante
        f_D0 = ot.Point([float(f_Df[i]) for i in range(N_0)])

        return f_D0, f_Df, D_f, K_ff

    def _sample_poisson_envelope(self, eps, max_candidates_per_domain=1000,
                                 max_candidates=2000):
        """Sample the piecewise-homogeneous Poisson envelope locations.

        Parameters
        ----------
        eps : array_like, shape (J,)
            Current domain log-intensities.
        max_candidates_per_domain : int or None, optional
            Per-domain safety limit. ``None`` disables this limit.
        max_candidates : int or None, optional
            Global safety limit. ``None`` disables this limit.

        Returns
        -------
        XY_cand : ot.Sample, shape (M, 2)
            Envelope locations sampled uniformly within each domain.

        Notes
        -----
        Reaching either safety limit truncates the exact Poisson envelope law.
        """
        candidates = []
        for j in range(self.J):
            raw_polygon = self.polygons[j]
            prepared_polygon = self.areas[j]
            xmin, ymin, xmax, ymax = raw_polygon.bounds
            mean_count = self.T * raw_polygon.area * np.exp(float(eps[j]))
            n_candidates = int(ot.Poisson(mean_count).getRealization()[0])
            if max_candidates_per_domain is not None:
                n_candidates = min(n_candidates, int(max_candidates_per_domain))
            if n_candidates == 0:
                continue

            accepted = []
            while len(accepted) < n_candidates:
                batch_size = max(3 * (n_candidates - len(accepted)), 16)
                points = ot.JointDistribution(
                    [ot.Uniform(xmin, xmax), ot.Uniform(ymin, ymax)]
                ).getSample(batch_size)
                for k in range(points.getSize()):
                    point = ShapelyPoint(float(points[k, 0]), float(points[k, 1]))
                    if prepared_polygon.covers(point):
                        accepted.append([float(points[k, 0]), float(points[k, 1])])
                        if len(accepted) == n_candidates:
                            break
            candidates.extend(accepted)

        if max_candidates is not None and len(candidates) > int(max_candidates):
            candidates = candidates[:int(max_candidates)]
        return ot.Sample(candidates) if candidates else ot.Sample(0, 2)

    @staticmethod
    def _validate_sparse_gp(sparse_gp):
        """Validate the minimal sparse-GP interface used by the sampler."""
        if sparse_gp is None:
            raise ValueError("sparse_gp must be provided when gp_backend='sparse'.")
        if not hasattr(sparse_gp, "m") or not callable(getattr(sparse_gp, "regressorOT", None)):
            raise TypeError("sparse_gp must expose an integer 'm' and a callable regressorOT(sample).")
        if int(sparse_gp.m) <= 0:
            raise ValueError("sparse_gp.m must be strictly positive.")

    def update_sparse_gp_coeffs(self, x, y, Z, omega_D0, Pi_S, sparse_gp):
        """Sample sparse-GP basis coefficients from their Gaussian conditional.

        The sparse model represents ``f(s) = Phi(s) @ w`` with the prior
        ``w ~ Normal(0, I)``. Polya-Gamma augmentation gives precision
        ``Q = I + Phi.T @ Omega @ Phi`` and mean ``Q^-1 Phi.T kappa``.

        Parameters
        ----------
        x, y : array_like, shape (N,)
            Observed coordinates.
        Z : array_like, shape (N,)
            Branching labels; only labels equal to zero enter the SSGC update.
        omega_D0 : array_like, shape (N,)
            Polya-Gamma marks at observed locations.
        Pi_S : ot.Sample, shape (M, 3)
            Latent locations and their Polya-Gamma marks.
        sparse_gp : object
            Object exposing ``m`` and ``regressorOT(ot.Sample)``.

        Returns
        -------
        coeffs : ot.Point, shape (m,)
            Draw from the sparse coefficient posterior.
        """
        self._validate_sparse_gp(sparse_gp)
        idx_bg = [i for i in range(len(Z)) if float(Z[i]) == 0.0]
        if not idx_bg:
            raise ValueError("No background event is available for the sparse GP update.")

        D0 = ot.Sample([[float(x[i]), float(y[i])] for i in idx_bg])
        phi_bg = np.asarray(sparse_gp.regressorOT(D0), dtype=float)
        omega_bg = np.asarray([float(omega_D0[i]) for i in idx_bg], dtype=float)

        n_pi = Pi_S.getSize()
        if n_pi:
            pi_xy = ot.Sample([[float(Pi_S[i, 0]), float(Pi_S[i, 1])] for i in range(n_pi)])
            phi_pi = np.asarray(sparse_gp.regressorOT(pi_xy), dtype=float)
            omega_pi = np.asarray([float(Pi_S[i, 2]) for i in range(n_pi)], dtype=float)
            design = np.vstack([phi_bg, phi_pi])
            omega = np.concatenate([omega_bg, omega_pi])
            kappa = np.concatenate([np.full(len(idx_bg), 0.5), np.full(n_pi, -0.5)])
        else:
            design = phi_bg
            omega = omega_bg
            kappa = np.full(len(idx_bg), 0.5)

        m = int(sparse_gp.m)
        if design.shape[1] != m:
            raise ValueError(f"regressorOT returned {design.shape[1]} columns, expected sparse_gp.m={m}.")
        precision = np.eye(m) + design.T @ (omega[:, None] * design)
        precision = 0.5 * (precision + precision.T) + self.jitter * np.eye(m)
        precision_ot = ot.CovarianceMatrix(precision.tolist())
        rhs = ot.Point((design.T @ kappa).tolist())
        mean = precision_ot.solveLinearSystem(rhs)
        covariance = np.asarray(
            precision_ot.solveLinearSystem(ot.Matrix(np.eye(m).tolist())), dtype=float
        )
        covariance = 0.5 * (covariance + covariance.T) + self.jitter * np.eye(m)
        return ot.Normal(mean, ot.CovarianceMatrix(covariance.tolist())).getRealization()

    def sample_Pi_S_sparse(self, eps, sparse_gp, gp_coeffs,
                           max_candidates_per_domain=1000, max_candidates=2000):
        """Sample ``Pi_S`` using a sparse basis representation of the GP."""
        self._validate_sparse_gp(sparse_gp)
        XY_cand = self._sample_poisson_envelope(
            eps, max_candidates_per_domain=max_candidates_per_domain,
            max_candidates=max_candidates,
        )
        n_candidates = XY_cand.getSize()
        if n_candidates == 0:
            return ot.Sample(0, 3)

        design = np.asarray(sparse_gp.regressorOT(XY_cand), dtype=float)
        f_cand = design @ np.asarray(gp_coeffs, dtype=float)
        uniforms = np.asarray(ot.Uniform(0.0, 1.0).getSample(n_candidates))[:, 0]
        mask = np.flatnonzero(uniforms < expit(-f_cand))
        if mask.size == 0:
            return ot.Sample(0, 3)

        omega = random_polyagamma(1.0, f_cand[mask])
        values = np.column_stack([np.asarray(XY_cand)[mask], omega])
        return ot.Sample(values.tolist())

    def sample_Pi_S(self, x, y, f_data, eps, LIM_CANDIDATES_DOMAINS=1000, LIM_CANDIDATES=2000):
        """Sample the latent marked thinned Poisson process ``Pi_S``.
        
        On each zone ``S_j``, the envelope intensity ``mu_tilde = exp(eps_j)`` is
        constant. Therefore the envelope count is sampled as
        
            N_j ~ Poisson(T * area(S_j) * exp(eps_j)),
        
        and its locations are uniform in ``S_j``. A conditional GP draw is evaluated
        at all envelope locations; each point is retained with probability
        ``sigmoid(-f)`` and receives a ``PG(1, f)`` mark.
        
        Parameters
        ----------
        x, y : array_like, shape (N,)
            Conditioning locations, normally the currently classified background
            events.
        f_data : ot.Point or array_like, shape (N,)
            Current GP values at those conditioning locations.
        eps : ot.Point or array_like, shape (J,)
            Current zonal log-intensities.
        LIM_CANDIDATES_DOMAINS : int, optional
            Maximum envelope count retained per zone, by default ``1000``.
        LIM_CANDIDATES : int, optional
            Maximum envelope count retained globally, by default ``2000``.
        
        Returns
        -------
        Pi_S : ot.Sample, shape (M, 3)
            Retained locations and Polya-Gamma marks in columns ``(x, y, omega)``.
        
        Notes
        -----
        Finite candidate limits protect memory but truncate the exact Poisson law when
        reached. They should be set above all plausible envelope counts for exact
        simulation."""
        N = len(x)
        XY_data = ot.Sample([[x[i], y[i]] for i in range(N)])
        XY_cand = self._sample_poisson_envelope(
            eps,
            max_candidates_per_domain=LIM_CANDIDATES_DOMAINS,
            max_candidates=LIM_CANDIDATES,
        )
        N_cand = XY_cand.getSize()
        if N_cand == 0:
            return ot.Sample(0, 3)

        # Prédiction GP conditionnelle sur D_0
        K_dd, K_star_d, K_star_star = self.compute_kernel(XY_data, XY_cand)
        K_dd_reg = ot.CovarianceMatrix(K_dd)
        for i in range(N):
            K_dd_reg[i, i] += self.jitter
        f_data_pt = f_data if isinstance(f_data, ot.Point) else ot.Point(list(f_data))
        alpha = K_dd_reg.solveLinearSystem(f_data_pt)
        mu_star = K_star_d * alpha

        K_star_d_t = ot.Matrix(np.array(K_star_d).T.tolist())
        solved_cross = K_dd_reg.solveLinearSystem(K_star_d_t)

        Sigma_arr = (
            np.array(K_star_star)
            - np.array(K_star_d * solved_cross)
        )
        Sigma_arr = 0.5 * (Sigma_arr + Sigma_arr.T) + self.jitter * np.eye(N_cand)
        Sigma_star = ot.CovarianceMatrix(Sigma_arr.tolist())

        f_star = ot.Normal(mu_star, Sigma_star).getRealization()

        # Thinning 
        accept_probs = expit(ot.Point([-float(f_star[i]) for i in range(N_cand)]))
        Uu = ot.Uniform(0.0, 1.0).getSample(N_cand)
        mask = [i for i in range(N_cand) if float(Uu[i, 0]) < float(accept_probs[i])]

        if len(mask) == 0:
            return ot.Sample(0, 3)

        XY_acc = ot.Sample(len(mask), 2)
        f_acc = np.zeros(len(mask))
        for k, i in enumerate(mask):
            XY_acc[k, 0] = XY_cand[i, 0]
            XY_acc[k, 1] = XY_cand[i, 1]
            f_acc[k] = float(f_star[i])

        omega_acc = random_polyagamma(1.0, f_acc)
        n_acc = len(omega_acc)
        Pi_S = ot.Sample(n_acc, 3)
        for i in range(n_acc):
            Pi_S[i, 0] = XY_acc[i, 0]
            Pi_S[i, 1] = XY_acc[i, 1]
            Pi_S[i, 2] = omega_acc[i]

        return Pi_S
    
    def update_eps(self, eps, N_j, M_j, step):
        """Update the zonal log-intensities ε via a MALA proposal.
 
        The Metropolis-Adjusted Langevin Algorithm exploits the closed-form
        gradient of the log-posterior to construct an informed proposal:
 
            ε* = ε + (h²/2) ∇ log p(ε | ·) + h ξ,   ξ ~ N(0, I_J).
 
        A Metropolis-Hastings correction ensures the correct stationary
        distribution. The theoretically optimal acceptance rate for MALA
        targeting a J-dimensional distribution is ≈ 57.4%.
 
        Parameters
        ----------
        eps : ot.Point or array_like, shape (J,)
            Current zonal log-intensities.
        N_j : ndarray, shape (J,)
            Number of observed background events per zone.
        M_j : ndarray, shape (J,)
            Number of latent thinned events per zone.
        step : float
            MALA step size h > 0.
 
        Returns
        -------
        eps_out : ndarray, shape (J,)
            Updated (or unchanged if rejected) zonal log-intensities.
        accepted : bool
            True if the proposal was accepted.
        """
        eps_arr = np.array(eps)

        # MALA proposaal
        grad_cur  = self._grad_log_posterior_eps(eps_arr, N_j, M_j)
        eta = np.array(ot.Normal(self.J).getRealization())
        eps_star = eps_arr + 0.5 * step**2 * grad_cur + step * eta

        grad_star = self._grad_log_posterior_eps(eps_star, N_j, M_j)
        log_p_cur  = self._log_posterior_eps(eps_arr, N_j, M_j)
        log_p_star = self._log_posterior_eps(eps_star, N_j, M_j)

        diff_fwd = eps_star - eps_arr - 0.5 * step**2 * grad_cur
        diff_bwd = eps_arr - eps_star - 0.5 * step**2 * grad_star 

        # Ratio d'Hasrtings
        log_q_ratio = (
            - 0.5 / step**2 * np.dot(diff_bwd, diff_bwd)
            + 0.5 / step**2 * np.dot(diff_fwd, diff_fwd)
        )

        log_alpha = min(0.0, (log_p_star - log_p_cur) + log_q_ratio)

        if np.log(float(ot.Uniform(0.0, 1.0).getRealization()[0])) < log_alpha:
            return eps_star, True
        else:
            return eps_arr, False

    def update_nu(self, f_Df, D_f_sample, history_log_nu, it, step_nu_init=0.1, t0=50, sd=2.38**2/2, eps_mh=1e-6):
        """Update GP kernel hyperparameters ν = (v², ℓ) via Adaptive Metropolis.
 
        Implements the AM algorithm of Haario, Saksman & Tamminen (2001). During
        a warm-up phase (it ≤ t0), an isotropic random walk with variance
        step_nu_init is used. After warm-up, the proposal covariance adapts to
        the empirical covariance of the chain history:
 
            Σ_prop = s_d · Cov(log ν^(0), …, log ν^(t−1)) + ε I_d,
 
        where s_d = 2.38² / d is the optimal scaling for d-dimensional Gaussian
        targets (Roberts et al., 1997) and ε is a small regularisation constant.
 
        The proposal is made on the log scale to enforce positivity: v², ℓ > 0.
        The Jacobian correction log|J| = Σ(log ν*_k − log ν_k) is included in
        the acceptance ratio.
 
        Parameters
        ----------
        f_Df : ot.Point, shape (N_f,)
            Current GP realization at augmented data locations.
        D_f_sample : ot.Sample, shape (N_f, 2)
            Spatial coordinates of augmented data locations.
        history_log_nu : list of ndarray
            Chain history of log ν values, modified in-place by the caller.
        it : int
            Current Gibbs iteration index.
        step_nu_init : float, optional
            Isotropic proposal variance during warm-up (default 0.1).
        t0 : int, optional
            Number of warm-up iterations before adaptation (default 50).
        sd : float, optional
            Scaling factor for the empirical covariance (default 2.38²/2).
        eps_mh : float, optional
            Regularisation constant added to the proposal covariance (default 1e-6).
 
        Returns
        -------
        nu : ot.Point, shape (2,)
            Updated (or unchanged) hyperparameters.
        accepted : bool
            True if the proposal was accepted.
        """
        nu0, nu1     = map(float, self.nu)
        log_nu_cur   = np.log([nu0, nu1])
 
        # --- Adaptive proposal covariance ---
        if it > t0 and len(history_log_nu) > t0:
            cov_emp = np.cov(np.array(history_log_nu).T)
            proposal_cov = sd * cov_emp + eps_mh * np.eye(2)
        else:
            proposal_cov = step_nu_init * np.eye(2)
 
        # --- Proposal on log scale ---
        log_nu_star = log_nu_cur + np.array(ot.Normal(ot.Point(2, 0.0), ot.CovarianceMatrix(proposal_cov.tolist())).getRealization())
 
        # Reject out-of-support proposals instead of clipping them.
        # v^2 in (1e-6, 10),  l in (1e-4, 20)
        LOG_NU_MIN = np.array([np.log(1e-6), np.log(1e-4)])
        LOG_NU_MAX = np.array([np.log(10.0),  np.log(20.0)])
        if np.any(log_nu_star < LOG_NU_MIN) or np.any(log_nu_star > LOG_NU_MAX):
            return self.nu, False
        nu_star = np.exp(log_nu_star).tolist()
 
        # --- Log acceptance ratio ---
        log_p_cur = self._log_posterior_nu(self.nu, f_Df, D_f_sample)
        try:
            log_p_star = self._log_posterior_nu(nu_star, f_Df, D_f_sample)
        except Exception:
            # Proposed nu invalid for OT -> reject
            return self.nu, False
 
        # Guard against -inf / nan
        if not np.isfinite(log_p_star):
            return self.nu, False
 
        # Jacobian correction for log-scale proposal : log|J| = sum(log nu_star - log nu_cur)
        log_jacobian = np.sum(log_nu_star - log_nu_cur)
        log_alpha = min(0.0, (log_p_star - log_p_cur) + log_jacobian)
 
        if np.log(float(ot.Uniform(0.0, 1.0).getRealization()[0])) < log_alpha:
            self.nu = ot.Point(nu_star)
            return self.nu, True
        else:
            return self.nu, False
    
    
    # ==============================================================================================
    # ------------------------------ Calibration & Initialisation ----------------------------------
    # ==============================================================================================
    
    def estimate_eps_mle(self, x, y):
        """Estimate the sub-domain log-intensities by maximum likelihood.
 
        For each zone S_j, the MLE of ε_j under a homogeneous Poisson model is:
 
            ε̂_j = log(N_j / (T |S_j|))
 
        where N_j is the observed count in sub-domain j. A floor of 1e-6 is applied
        to the rate to avoid log(0).
 
        Parameters
        ----------
        x, y : array_like, shape (N,)
            Spatial coordinates of observed events.
 
        Returns
        -------
        eps_mle : ndarray, shape (J,)
            Maximum likelihood estimates of the domain log-intensities.
        """
        zones = self._compute_event_domain_indices(x, y)
        counts = np.bincount(zones[zones >= 0], minlength=self.J)[:self.J].astype(float)
        eps_mle = np.zeros(self.J)
        for j in range(self.J):
            area_j = self.polygons[j].area
            rate_j = max(counts[j] / (self.T * area_j), 1e-6)
            eps_mle[j] = np.log(rate_j) 
        return eps_mle
    
    def calibrate_nu(self, x, y, verbose=True, method="openturns", plot_kde=False,
                     kde_cmap="viridis"):
        """Calibrate GP hyperparameters from a linearized sigmoid target.

        Parameters
        ----------
        x, y : array_like, shape (N,)
            Observed coordinates.
        verbose : bool, optional
            Print calibrated values.
        method : {"sklearn", "openturns"}, optional
            Gaussian-process fitter used for the calibration.
        plot_kde : bool, optional
            Display the OpenTURNS kernel-density estimate used to build the
            regression target.
        kde_cmap : str or Colormap, optional
            Colormap used by the calibration KDE plot.

        Returns
        -------
        v : float
            Calibrated GP marginal standard deviation.
        l_ot : float
            Calibrated OpenTURNS length scale.
        eps_mle : ndarray, shape (J,)
            Domainwise Poisson MLE used to initialize the chain.
        """
        method = str(method).lower()
        if method not in {"sklearn", "openturns"}:
            raise ValueError("method must be 'sklearn' or 'openturns'.")

        obs_pts = np.column_stack([
            np.asarray([float(v) for v in x]),
            np.asarray([float(v) for v in y]),
        ])
        sample_ot = ot.Sample(obs_pts.tolist())
        kde = ot.KernelSmoothing().build(sample_ot)
        p_hat = np.asarray(kde.computePDF(sample_ot), dtype=float).reshape(-1)

        if plot_kde:
            gx = np.linspace(self.X_bounds[0], self.X_bounds[1], 80)
            gy = np.linspace(self.Y_bounds[0], self.Y_bounds[1], 80)
            GX, GY = np.meshgrid(gx, gy)
            grid = ot.Sample(np.column_stack([GX.ravel(), GY.ravel()]).tolist())
            density = np.asarray(kde.computePDF(grid), dtype=float).reshape(GX.shape)
            fig, ax = plt.subplots(figsize=(6, 5))
            contour = ax.contourf(GX, GY, density, levels=20, cmap=kde_cmap)
            ax.scatter(obs_pts[:, 0], obs_pts[:, 1], s=10, c="white", edgecolors="black", linewidths=0.3)
            fig.colorbar(contour, ax=ax, label="KDE density")
            ax.set_title("OpenTURNS kernel-density estimate")
            ax.set_xlim(self.X_bounds); ax.set_ylim(self.Y_bounds)
            plt.tight_layout(); plt.show()

        eps_mle = self.estimate_eps_mle(x, y)
        if verbose:
            print(f"[calibrate_nu] eps_mle = {np.round(eps_mle, 4)}")
        domain_area = sum(poly.area for poly in self.polygons)
        target = 2.0 * domain_area * p_hat - 2.0

        if method == "sklearn":
            kernel = (
                C(0.1, (1e-3, 0.58 ** 2))
                * RBF(length_scale=0.3, length_scale_bounds=(1e-2, 5.0))
                + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-4, 1.0))
            )
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
            gp.fit(obs_pts, target)
            parameters = gp.kernel_.get_params()
            v_sq = float(parameters["k1__k1__constant_value"])
            l_ot = float(parameters["k1__k2__length_scale"]) * np.sqrt(2.0)
        else:
            basis = ot.ConstantBasisFactory(2).build()
            covariance = ot.SquaredExponential([0.3, 0.3], [0.3])
            fitter = otexp.GaussianProcessFitter(
                sample_ot, ot.Sample(target.reshape(-1, 1).tolist()), covariance, basis
            )
            fitter.setOptimizationBounds(ot.Interval([1e-2, 1e-2], [5.0, 5.0]))
            fitter.run()
            fitted_covariance = fitter.getResult().getCovarianceModel()
            l_ot = float(np.min(np.asarray(fitted_covariance.getScale(), dtype=float)))
            amplitude = float(fitted_covariance.getAmplitude()[0])
            v_sq = min(amplitude ** 2, 0.58 ** 2)

        v = np.sqrt(v_sq)
        self.nu = ot.Point([v_sq, l_ot])
        if verbose:
            print(f"[calibrate_nu:{method}] v = {v:.4f} ; l_ot = {l_ot:.4f}")
        return v, l_ot, eps_mle

    # def calibrate_nu(self, x, y, grid_size=50, verbose=True):
    #     """

    #     """
    #     xmin, xmax = self.X_bounds
    #     ymin, ymax = self.Y_bounds

    #     # Grid 
    #     gx = np.linspace(xmin, xmax, grid_size)
    #     gy = np.linspace(ymin, ymax, grid_size)
    #     GX, GY = np.meshgrid(gx, gy)
    #     grid_pts = np.column_stack([GX.ravel(), GY.ravel()])
    #     ot_grid = ot.Sample(grid_pts)

    #     # KDE -> p_hat 
    #     sample_ot = ot.Sample([[float(x[i]), float(y[i])] for i in range(len(x))])
    #     ks = ot.KernelSmoothing()
    #     kde = ks.build(sample_ot)
    #     p_hat = np.array(kde.computePDF(ot_grid)).flatten()

    #     # eps par MLE 
    #     eps_mle = self.estimate_eps_mle(x, y)

    #     if verbose:
    #         print(f"[calibrate_nu] eps_mle = {np.round(eps_mle, 4)}")

    #     # Target : z(x,y) = 2*N*|S_j|/N_j * p_hat - 2 


    # =================================================================================================
    # ----------------------------------------- Run du Gibbs ------------------------------------------
    # =================================================================================================

    def run(self, t, x, y, mala_step=0.05, n_iter=1000, learn_nu=False, t0_nu=50,
        step_nu_init=0.1, verbose=True, verbose_every=100, use_calibration=True,
        mu_star_func=None, grid_nx=30, grid_ny=30, thin=1,
        compute_emu=True, emu_every=10, calibration_method="sklearn",
        plot_calibration_kde=False, calibration_kde_cmap="viridis",
        gp_backend="exact", sparse_gp=None): 
        """Run the augmented SSGC Gibbs sampler.
        
        Parameters
        ----------
        t : array_like, shape (N,)
            Event times. They are accepted for API compatibility and are not otherwise
            used by the spatial SSGC updates.
        x, y : array_like, shape (N,)
            Event coordinates.
        mala_step : float, optional
            MALA step size for the zonal log-intensities.
        n_iter : int, optional
            Number of Gibbs iterations.
        learn_nu : bool, optional
            Update GP hyperparameters with adaptive Metropolis.
        t0_nu : int, optional
            Warm-up length for GP-hyperparameter adaptation.
        step_nu_init : float, optional
            Initial proposal variance for ``log(nu)``.
        verbose : bool, optional
            Print progress and acceptance rates.
        verbose_every : int, optional
            Number of iterations between progress messages.
        use_calibration : bool, optional
            Calibrate ``nu`` and initialize ``eps`` before sampling.
        mu_star_func : callable or None, optional
            Reference intensity ``mu_star(x_array, y_array)`` used only by the optional
            integrated squared-error diagnostic.
        grid_nx, grid_ny : int, optional
            Diagnostic-grid dimensions.
        thin : int, optional
            Store one state every ``thin`` iterations.
        compute_emu : bool, optional
            Compute ``E_mu`` when a reference intensity is provided.
        emu_every : int, optional
            Compute ``E_mu`` every ``emu_every`` iterations.
        calibration_method : {"sklearn", "openturns"}, optional
            GP fitter used by the optional pre-run calibration.
        plot_calibration_kde : bool, optional
            Display the KDE used by calibration.
        calibration_kde_cmap : str or Colormap, optional
            Colormap used for the calibration KDE plot.
        gp_backend : {"exact", "sparse"}, optional
            Exact GP values or injected sparse-basis coefficients.
        sparse_gp : object or None, optional
            Sparse basis object exposing ``m`` and ``regressorOT``. When omitted
            in sparse mode, :class:`SparseGP` is constructed from ``nu`` and the
            rectangular observation bounds.
        
        Returns
        -------
        results : dict
            Stored chains ``eps``, ``nPi``, ``f_data`` and ``nu``; the full-length
            ``E_mu`` diagnostic (NaN where not evaluated); acceptance rates; final
            state; covariance metadata; and storage settings."""

        N = len(t)
        Z = ot.Point([0.0] * N)
        N_j, _ = self._count_events_per_zone(x, y, Z, ot.Sample(0, 3))

        if use_calibration:
            if verbose:
                print("[Pre-run] Calibrating GP hyperparameters")
            _, _, eps_mle = self.calibrate_nu(
                x, y, verbose=verbose, method=calibration_method,
                plot_kde=plot_calibration_kde,
                kde_cmap=calibration_kde_cmap,
            )
        else:
            if verbose:
                print(f"[Pre-run] Using provided nu_init = {list(self.nu)}")
            eps_mle = self.estimate_eps_mle(x, y)

        gp_backend = str(gp_backend).lower()
        if gp_backend not in {"exact", "sparse"}:
            raise ValueError("gp_backend must be 'exact' or 'sparse'.")
        if gp_backend == "sparse":
            if sparse_gp is None:
                sparse_gp = SparseGP.from_bounds(
                    self.X_bounds, self.Y_bounds,
                    variance=float(self.nu[0]), length_scale=float(self.nu[1]),
                )
            self._validate_sparse_gp(sparse_gp)
            if learn_nu:
                raise ValueError("learn_nu=True is not supported with gp_backend='sparse'.")
            gp_coeffs = ot.Point(int(sparse_gp.m), 0.0)
            XY_observed = ot.Sample([[float(x[i]), float(y[i])] for i in range(N)])
            sparse_design_observed = np.asarray(sparse_gp.regressorOT(XY_observed), dtype=float)
        else:
            gp_coeffs = None
            sparse_design_observed = None

        if learn_nu and verbose:
           print("[Pre-run] nu will be updated at each iteration (Adaptive MH).")
        elif verbose:
           print(f"[Pre-run] nu fixed at : {np.round(np.array(self.nu), 4)} [v^2, l]")

        eps = ot.Point(eps_mle.tolist())
        f_data = ot.Point([0.0] * N)

        if verbose:
            if gp_backend == "sparse":
                print(f"[Initialisation] Sparse GP with {int(sparse_gp.m)} basis functions")
            print(f"[Initialisation] Using eps_mle as eps_init : {np.round(eps_mle, 4)}")
            print(f"[Initialisation] Initialise f to zero (zero-mean prior)")

        # ---------- Grille fixe pour le calcul optionnel de Eps_mu ----------
        compute_emu = bool(compute_emu and mu_star_func is not None)
        if compute_emu:
            xmin, xmax = self.X_bounds
            ymin, ymax = self.Y_bounds
            gx = np.linspace(xmin, xmax, grid_nx)
            gy = np.linspace(ymin, ymax, grid_ny)
            GX, GY = np.meshgrid(gx, gy)
            grid_x = GX.ravel()
            grid_y = GY.ravel()
            XY_grid = ot.Sample(np.column_stack([grid_x, grid_y]).tolist())
            M_grid = len(grid_x)
            domain_area = (xmax - xmin) * (ymax - ymin)
            mu_star_grid = mu_star_func(grid_x, grid_y)
        else:
            grid_x = grid_y = XY_grid = mu_star_grid = None
            M_grid = 0
            domain_area = 0.0

        # ---------- Stockage ----------
        n_store = (n_iter + thin - 1) // thin
        eps_chain = np.zeros((n_store, self.J))
        nPi_chain = np.zeros(n_store, dtype=int)
        fdata_chain = np.zeros((n_store, N))
        nu_chain = np.zeros((n_store, 2))
        gp_coeffs_chain = (
            np.zeros((n_store, int(sparse_gp.m))) if gp_backend == "sparse" else None
        )
        E_mu_chain = np.full(n_iter, np.nan)      # Je garde toutes les it pour E_mu
        store_idx = 0       # compteur de stockage
        acc_eps = 0
        acc_nu = 0
        history_log_nu = []         # used only when learn_nu=True

        if verbose:
            print("\n" + "=" * 100)
            print(
                "-" * 29
                + f" Démarrage Gibbs : {n_iter} itérations, N={N} "
                + "-" * 29
            )
            print("=" * 100)

        for it in range(n_iter):
            try:
                # Steps 1-3: Polya-Gamma marks, latent process, and GP update.
                if gp_backend == "sparse":
                    f_data_np = sparse_design_observed @ np.asarray(gp_coeffs, dtype=float)
                    f_data = ot.Point(f_data_np.tolist())
                    omega_D0 = ot.Point(random_polyagamma(1.0, f_data_np))
                    Pi_S = self.sample_Pi_S_sparse(eps, sparse_gp, gp_coeffs)
                    gp_coeffs = self.update_sparse_gp_coeffs(
                        x, y, Z, omega_D0, Pi_S, sparse_gp
                    )
                    f_data_np = sparse_design_observed @ np.asarray(gp_coeffs, dtype=float)
                    f_data = ot.Point(f_data_np.tolist())
                    f_Df = D_f_xy = None
                else:
                    f_data_np = np.array(f_data)
                    omega_D0 = ot.Point(random_polyagamma(1.0, f_data_np))
                    Pi_S = self.sample_Pi_S(x, y, f_data, eps)
                    f_D0, f_Df, D_f_xy, _ = self.update_f(x, y, Z, omega_D0, Pi_S)
                    f_data = f_D0

                # Step 4 : eps | f, pi_S (MALA) 
                # M_j changes at each iteration as pi_S is resampled
                _, M_j = self._count_events_per_zone(x, y, Z, Pi_S)
                eps_arr, accepted_eps = self.update_eps(eps, N_j, M_j, step=mala_step)
                eps = ot.Point(eps_arr.tolist())
                acc_eps += int(accepted_eps)

                # Step 5 (optional) : nu | f  (Adaptive MH)
                if learn_nu:
                    history_log_nu.append(np.log(np.array(self.nu)))
                    _, accepted_nu = self.update_nu(
                        f_Df, D_f_xy, history_log_nu, it, step_nu_init=step_nu_init, t0=t0_nu
                    )
                    acc_nu += int(accepted_nu)

                
                # ---------- Affichage ----------
                if verbose and (it % verbose_every == 0 or it == n_iter - 1):
                    acc_rate_eps = acc_eps / (it + 1) * 100
                    msg = (
                        f"[Iter {it}] "
                        f"pi_S = {Pi_S.getSize()} | "
                        f"acc_eps = {np.round(acc_rate_eps, 1)}%"
                    )
                    if learn_nu:
                        acc_rate_nu = acc_nu / (it + 1) * 100
                        msg += (f" | nu = {np.round(np.array(self.nu), 4)}"
                                f" | acc_nu = {np.round(acc_rate_nu, 1)}%")
                    print(msg)

                # ---------- Calcul de Eps_mu^(t) ----------
                # Calcul de Eps_mu toutes les X itérations seulement
                if compute_emu and (it % emu_every == 0):
                    if gp_backend == "sparse":
                        grid_design = np.asarray(sparse_gp.regressorOT(XY_grid), dtype=float)
                        f_draw_g = grid_design @ np.asarray(gp_coeffs, dtype=float)
                    else:
                        XY_data_ot = ot.Sample([[x[i], y[i]] for i in range(N)])
                        K_dd, K_gd, K_gg = self.compute_kernel(XY_data_ot, XY_grid)
                        K_dd_reg = ot.CovarianceMatrix(K_dd)
                        for ii in range(N):
                            K_dd_reg[ii, ii] += self.jitter
                        f_data_pt = ot.Point(list(f_data))
                        alpha = K_dd_reg.solveLinearSystem(f_data_pt)
                        mu_g = np.array(K_gd * alpha).flatten()
                        solved_cross = K_dd_reg.solveLinearSystem(
                            ot.Matrix(np.array(K_gd).T.tolist())
                        )
                        Sigma_g = np.array(K_gg) - np.array(K_gd * solved_cross)
                        Sigma_g = 0.5 * (Sigma_g + Sigma_g.T) + self.jitter * np.eye(M_grid)
                        Sigma_g_cov = ot.CovarianceMatrix(Sigma_g.tolist())
                        f_draw_g = np.array(
                            ot.Normal(ot.Point(mu_g.tolist()), Sigma_g_cov).getRealization()
                        )

                    mu_tilde_g = self.compute_mu_tilde(XY_grid, eps=eps_arr)
                    mu_draw_g = mu_tilde_g * (1.0 / (1.0 + np.exp(-f_draw_g)))

                    E_mu_chain[it] = (domain_area / M_grid) * np.sum((mu_draw_g - mu_star_grid) ** 2)

                # ---------- Stockage ----------
                if it % thin == 0:
                    eps_chain[store_idx, :] = eps_arr
                    nPi_chain[store_idx] = Pi_S.getSize()
                    fdata_chain[store_idx, :] = np.array(f_data)
                    nu_chain[store_idx, :] = np.array(self.nu)
                    if gp_coeffs_chain is not None:
                        gp_coeffs_chain[store_idx, :] = np.asarray(gp_coeffs, dtype=float)
                    store_idx += 1

            except Exception as e:
                print(f"\nError at iteration {it} : {e}")
                raise           
        
        if verbose:
            print("=" * 100)
            print("-" * 41 + " Gibbs terminé !! " + "-" * 41)
            print("=" * 100 + "\n")
            print(f"eps acceptance rate : {np.round(acc_eps / n_iter * 100, 1)}%"
                  f" (target ~57% -> {'increase' if acc_eps/n_iter > 0.57 else 'decrease'} mala_step)")
            if learn_nu:
                print(f"nu acceptance rate : {np.round(acc_nu  / n_iter * 100, 1)}%"
                      f" (target ~23% -> {'increase' if acc_nu/n_iter > 0.23 else 'decrease'} step_nu_init)")

        return {
            "eps"            : eps_chain[:store_idx],
            "nPi"            : nPi_chain[:store_idx],
            "f_data"         : fdata_chain[:store_idx],
            "nu"             : nu_chain[:store_idx],
            "E_mu"           : E_mu_chain,
            "acceptance_eps" : acc_eps / n_iter,
            "acceptance_nu"  : acc_nu / n_iter if learn_nu else None,
            "last_state"     : {"eps": eps_arr, "nu": list(self.nu), "delta": list(self.delta)},
            "Sigma_eps"      : self.Sigma_eps,
            "centroids"      : self.centroids_xy,
            "thin"           : thin,
            "n_iter"         : n_iter,
            "gp_backend"     : gp_backend,
            "gp_coeffs"      : gp_coeffs_chain[:store_idx] if gp_coeffs_chain is not None else None,
        }
    

    # ================================================================================================
    # ---------------------------------- Analyse posterior -------------------------------------------
    # ================================================================================================

    def posterior_summary(self, results, burn_in=0.3):
        """Compute posterior mean estimates from MCMC chain output.
 
        Discards the first ``burn_in`` fraction of the chain and averages the
        remaining samples for each parameter.
 
        Parameters
        ----------
        results : dict
            Output of :meth:`run`.
        burn_in : float, optional
            Fraction of the chain to discard as burn-in (default 0.3).
 
        Returns
        -------
        summary : dict
            - ``'eps_hat'``: ndarray (J,), posterior mean of ε.
            - ``'f_data_hat'``: ndarray (N,), posterior mean of f at data locations.
            - ``'nu_hat'``: ndarray (2,), posterior mean of ν = (v², ℓ).
        """
        eps_chain = np.asarray(results["eps"])
        f_chain = np.asarray(results["f_data"])
        nu_chain = np.asarray(results["nu"])
        burn = int(eps_chain.shape[0] * burn_in)
        return {
            "eps_hat" : eps_chain[burn:].mean(axis=0),
            "f_data_hat" : f_chain[burn:].mean(axis=0),
            "nu_hat" : nu_chain[burn:].mean(axis=0),
        }

    def posterior_gp(self, XY_data, f_data_hat, mesh, eps_hat):
        """Compute the GP predictive posterior at mesh vertices.
 
        Given the posterior mean of f at the data locations, computes the
        kriging mean and covariance at the mesh vertices using the standard
        GP conditional formulas:
 
            μ_* = K_{*d} K_{dd}⁻¹ f̂,
            Σ_* = K_{**} − K_{*d} K_{dd}⁻¹ K_{*d}ᵀ.
 
        Parameters
        ----------
        XY_data : ot.Sample, shape (N, 2)
            Conditioning locations (observed event positions).
        f_data_hat : ot.Point or array_like, shape (N,)
            Posterior mean of the GP at conditioning locations.
        mesh : ot.Mesh
            OpenTURNS mesh whose vertices define the prediction locations.
        eps_hat : ndarray, shape (J,)
            Posterior mean of the zonal log-intensities (not used in GP
            prediction but kept in the signature for API consistency).
 
        Returns
        -------
        mu_post : ot.Point, shape (M,)
            Kriging mean at mesh vertices.
        Sigma_post : ot.CovarianceMatrix, shape (M, M)
            Kriging covariance at mesh vertices.
        """
        XY_grid = mesh.getVertices()
        N = XY_data.getSize()
        M = XY_grid.getSize()

        K_dd, K_gd, K_gg = self.compute_kernel(XY_data, XY_grid)
        K_dd_reg = ot.CovarianceMatrix(K_dd)
        for i in range(N):
            K_dd_reg[i, i] += self.jitter
        f_hat_pt = f_data_hat if isinstance(f_data_hat, ot.Point) else ot.Point(list(f_data_hat))
        alpha = K_dd_reg.solveLinearSystem(f_hat_pt)
        mu_post = K_gd * alpha

        solved_cross = K_dd_reg.solveLinearSystem(ot.Matrix(np.array(K_gd).T.tolist()))
        Sigma_arr = np.array(K_gg) - np.array(K_gd * solved_cross)
        Sigma_arr = 0.5 * (Sigma_arr + Sigma_arr.T)
        Sigma_arr += self.jitter * np.eye(M)
        Sigma_post = ot.CovarianceMatrix(Sigma_arr.tolist())

        return mu_post, Sigma_post
    

    def posterior_intensity(self, x, y, t, results, nx=70, ny=70, burn_in=0.3,
                             cmap="viridis", event_cmap="plasma",
                             savefigure=False, title_savefig="posterior",
                             savefigure_Emu=False, title_savefig_Emu="Emu", color_Emu="steelblue",
                             mu_star_func=None, alpha_ecp=0.95):
        """Plot the posterior intensity estimate with uncertainty quantification.
 
        Generates Monte Carlo draws from the full GP posterior on a regular mesh,
        transforms them through μ̃ · σ(f) to obtain posterior intensity samples,
        and computes point-wise statistics (mean, std, credible intervals).
 
        If a ground-truth intensity ``mu_star_func`` is provided, also computes
        and displays RMSE, MAE, CRPS (via properscoring), and ECP metrics, along
        with a pointwise relative error map.
 
        Parameters
        ----------
        x, y : array_like, shape (N,)
            Spatial coordinates of observed events.
        t : array_like, shape (N,)
            Event times.
        results : dict
            Output of :meth:`run`.
        nx, ny : int, optional
            Mesh resolution for posterior evaluation (default 70×70).
        burn_in : float, optional
            Burn-in fraction (default 0.3).
        cmap : str or Colormap, optional
            Colormap used for intensity fields (default 'viridis').
        event_cmap : str or Colormap, optional
            Colormap used for event times in the observation panel.
        savefigure : bool, optional
            If True, save the main figure as PDF (default False).
        title_savefig : str, optional
            Filename stem for the main figure (default 'posterior').
        savefigure_Emu : bool, optional
            If True, save the E_μ trace plot (default False).
        title_savefig_Emu : str, optional
            Filename stem for the E_μ figure (default 'Emu').
        color_Emu : str, optional
            Line colour for the E_μ trace (default 'steelblue').
        mu_star_func : callable or None, optional
            Ground-truth intensity function. If None, only the estimate is plotted.
        alpha_ecp : float, optional
            Nominal coverage level for ECP computation (default 0.95).
 
        Returns
        -------
        output : dict
            Dictionary with keys including 'mu_hat', 'mu_star', 'diff',
            'var_mu_hat', 'std_mu_hat', 'lower_mu_hat', 'upper_mu_hat',
            'mu_hat_sims' (MC samples), 'mesh', 'eps_hat', 'f_data_hat',
            'E_mu_bar', 'rmse', 'mae', 'crps', 'ecp', and corresponding
            OpenTURNS Field objects.
        """

        post_sum = self.posterior_summary(results, burn_in)
        eps_hat = post_sum["eps_hat"]
        f_data_hat = post_sum["f_data_hat"]
        nu_hat = post_sum["nu_hat"]
        self.nu = ot.Point(nu_hat)

        N = len(t)
        XY_data = ot.Sample([[x[i], y[i]] for i in range(N)])

        xmin, xmax = self.X_bounds
        ymin, ymax = self.Y_bounds
        interval = ot.Interval([xmin, ymin], [xmax, ymax])
        mesher = ot.IntervalMesher([nx - 1, ny - 1])
        mesh = mesher.build(interval)

        XY_grid = mesh.getVertices()
        M = XY_grid.getSize()

        if M > 10000:
            raise ValueError(f"Mesh too large : {M} points")

        ### Évaluation des marginales (suppose que les évaluations du GP sont indépendantes)
        # mu_post_grid, Sigma_post_grid = self.posterior_gp(
        #     XY_data, ot.Point(list(f_data_hat)), mesh, eps_hat
        # )

        # means = np.array(mu_post_grid).flatten()
        # std_devs = np.sqrt(np.diagonal(np.array(Sigma_post_grid)))
        # # n_mc = 1000
        # n_mc = 500

        # f_sims = means[:, None] + std_devs[:, None] * noise
        # XY_grid = mesh.getVertices()
        # mu_tilde_grid = self.compute_mu_tilde(XY_grid, eps=eps_hat)
        # sig_sims = expit(f_sims)
        # mu_hat_sims = mu_tilde_grid[:, None] * sig_sims   # shape (M, n_mc)
        # mu_hat = mu_hat_sims.mean(axis=1)
        # squared_mu_hat = (mu_hat_sims ** 2).mean(axis=1)

        # mu_hat_sample = ot.Sample([[val] for val in mu_hat])
        # mu_hat_field  = ot.Field(mesh, mu_hat_sample)

        # ---------- Simulation posterior complète du champ GP ----------
        mu_post_grid, Sigma_post_grid = self.posterior_gp(
            XY_data, ot.Point(list(f_data_hat)), mesh, eps_hat
        )

        means = np.asarray(mu_post_grid).reshape(-1)
        Sigma = np.asarray(Sigma_post_grid)

        if means.shape[0] != M:
            raise ValueError(
                f"Inconsistent posterior mean size: means has size {means.shape[0]}, "
                f"but mesh has {M} vertices"
            )

        if Sigma.shape != (M, M):
            raise ValueError(
                f"Inconsistent posterior covariance shape: Sigma has shape {Sigma.shape}, "
                f"but expected {(M, M)}"
            )

        # n_mc = 1000
        n_mc = 500

        # Symétrisation numérique de la covariance
        Sigma = 0.5 * (Sigma + Sigma.T)

        # Cholesky robuste avec augmentation progressive du jitter
        base_jitter = getattr(self, "jitter", 1e-8)
        jitter_values = [
            base_jitter,
            1e-8,
            1e-7,
            1e-6,
            1e-5,
            1e-4
        ]

        f_sample = None
        last_error = None

        for jitter in jitter_values:
            try:
                Sigma_cov = ot.CovarianceMatrix((Sigma + jitter * np.eye(M)).tolist())
                f_sample = ot.Normal(ot.Point(means.tolist()), Sigma_cov).getSample(n_mc)
                break
            except Exception as e:
                last_error = e

        if f_sample is None:
            raise RuntimeError(
                f"OpenTURNS Normal sampling failed even with jitter up to {jitter_values[-1]}"
            ) from last_error

        # Simulation du processus GP complet : chaque colonne de f_sims est une réalisation spatiale corrélée du champ
        f_sims = np.asarray(f_sample).T

        # Intensité de base mu_tilde
        mu_tilde_grid = np.asarray(self.compute_mu_tilde(XY_grid, eps=eps_hat)).reshape(-1)

        if mu_tilde_grid.shape[0] != M:
            raise ValueError(
                f"Inconsistent mu_tilde size : mu_tilde_grid has size {mu_tilde_grid.shape[0]}, "
                f"but mesh has {M} vertices"
            )

        sig_sims = expit(f_sims)

        # Simulations de l'intensité posterior complète
        mu_hat_sims = mu_tilde_grid[:, None] * sig_sims  # shape (M, n_mc)

        # Moyenne, moment d'ordre 2, variance et intervalles crédibles point par point
        mu_hat = mu_hat_sims.mean(axis=1)
        squared_mu_hat = (mu_hat_sims ** 2).mean(axis=1)

        var_mu_hat = squared_mu_hat - mu_hat**2
        std_mu_hat = np.sqrt(np.maximum(var_mu_hat, 0.0))

        lower_mu_hat = np.quantile(mu_hat_sims, 0.025, axis=1)
        upper_mu_hat = np.quantile(mu_hat_sims, 0.975, axis=1)

        # Conversion OpenTURNS
        mu_hat_sample = ot.Sample([[val] for val in mu_hat])
        mu_hat_field = ot.Field(mesh, mu_hat_sample)

        std_mu_hat_sample = ot.Sample([[val] for val in std_mu_hat])
        std_mu_hat_field = ot.Field(mesh, std_mu_hat_sample)

        lower_mu_hat_sample = ot.Sample([[val] for val in lower_mu_hat])
        lower_mu_hat_field = ot.Field(mesh, lower_mu_hat_sample)

        upper_mu_hat_sample = ot.Sample([[val] for val in upper_mu_hat])
        upper_mu_hat_field = ot.Field(mesh, upper_mu_hat_sample)

        # ---------- Calcul de E_mu ----------
        domain_area = (xmax - xmin) * (ymax - ymin)
        E_mu_full = results["E_mu"]
        mask = ~np.isnan(E_mu_full)
        E_mu_post = E_mu_full[mask]
        iters_post = np.where(mask)[0]

        E_mu_bar = None
        if len(E_mu_post) > 0:
            E_mu_bar = E_mu_post.mean()

            fig_err, ax_err = plt.subplots(figsize=(9, 3))
            ax_err.plot(iters_post, E_mu_post, linewidth=0.8, color=color_Emu)
            ax_err.set_xlabel("Iteration")
            ax_err.set_ylabel(r"$\mathcal{E}_\mu^{(t)}$")
            ax_err.set_title(r"$L^2$ reconstruction error $\mathcal{E}_\mu^{(t)}$" + "\n")
            ax_err.grid(alpha=0.3)
            plt.tight_layout()

            if savefigure_Emu:
                try:
                    save_path = save_figure(fig_err, title_savefig_Emu)
                    print(f"Figure E_mu sauvegardée : {save_path}")
                except Exception as e:
                    print(f"Erreur lors de la sauvegarde E_mu : {e}")

            plt.show()

        # ---------- Vraie intensité + métriques (si fournie) ----------
        mu_star_grid = None
        rmse = mae = ecp = crps_bar = diff = None

        if mu_star_func is not None:
            import properscoring as ps

            grid_xy = np.array(XY_grid)
            mu_star_grid = mu_star_func(grid_xy[:, 0], grid_xy[:, 1])

            mu_star_sample = ot.Sample([[val] for val in mu_star_grid])
            mu_star_field = ot.Field(mesh, mu_star_sample)

            diff = np.abs(mu_hat - mu_star_grid) / (mu_star_grid + self.jitter)
            diff_sample = ot.Sample([[val] for val in diff])
            diff_field = ot.Field(mesh, diff_sample)

            # --- RMSE ---
            rmse = np.sqrt(np.mean((mu_hat - mu_star_grid) ** 2))

            # --- MAE ---
            mae = np.mean(np.abs(mu_hat - mu_star_grid))

            # --- CRPS ---
            # ps.crps_ensemble attend (observations, forecasts)
            # observations : shape (M,)
            # forecasts : shape (M, n_mc) 
            crps_pointwise = ps.crps_ensemble(mu_star_grid, mu_hat_sims)  # shape (M,)
            crps_bar = float(crps_pointwise.mean())

            # --- ECP(alpha) ---
            q_lo = np.quantile(mu_hat_sims, (1 - alpha_ecp) / 2, axis=1)
            q_hi = np.quantile(mu_hat_sims, 1 - (1 - alpha_ecp) / 2, axis=1)
            ecp = np.mean((mu_star_grid >= q_lo) & (mu_star_grid <= q_hi))

            print(f"\n{'='*45}")
            print(f"  Métriques (grille {nx}x{ny}, n_mc={n_mc})")
            print(f"{'='*45}")
            print(f"  RMSE          : {rmse:.4f}")
            print(f"  MAE           : {mae:.4f}")
            print(f"  CRPS          : {crps_bar:.4f}")
            print(f"  ECP({alpha_ecp:.2f})   : {ecp:.4f}  (cible : {alpha_ecp:.2f})")
            print(f"{'='*45}\n")

        # ---------- Figure ----------
        if mu_star_func is not None:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

            ax = axes[0]
            plot_field(mu_star_field, mode="subplot", ax=ax, cmap=cmap, add_colorbar=True)
            ax.set_title(r"Vraie intensité $\mu^\star(s)$" + "\n")
            ax.set_xlim(self.X_bounds)
            ax.set_ylim(self.Y_bounds)
            ax.grid(alpha=0.3, color="white", linewidth=0.5)

            ax = axes[1]
            plot_field(mu_hat_field, mode="subplot", ax=ax, cmap=cmap, add_colorbar=True)
            ax.set_title(r"Intensité estimée $\hat{\mu}(s)$" + "\n")
            ax.set_xlim(self.X_bounds)
            ax.set_ylim(self.Y_bounds)
            ax.grid(alpha=0.3, color="white", linewidth=0.5)

            ax = axes[2]
            plot_field(diff_field, mode="subplot", ax=ax, cmap=cmap, add_colorbar=True)
            ax.set_title(r"Erreur relative $\frac{|\hat{\mu}(s) - \mu^\star(s)|}{\mu^\star(s)}$" + "\n")
            ax.set_xlim(self.X_bounds)
            ax.set_ylim(self.Y_bounds)
            ax.grid(alpha=0.3, color="white", linewidth=0.5)

        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

            ax = axes[0]
            ax.scatter(x, y, c=t, s=12, alpha=0.7, edgecolors="black", cmap=event_cmap)
            ax.set_title(f"Données observées (N={N})" + "\n")
            ax.set_xlim(self.X_bounds)
            ax.set_ylim(self.Y_bounds)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.3)

            ax = axes[1]
            plot_field(mu_hat_field, mode="subplot", ax=ax, cmap=cmap, add_colorbar=True)
            ax.set_title(r"Intensité estimée $\hat{\mu}(s)$" + "\n")
            ax.set_xlim(self.X_bounds)
            ax.set_ylim(self.Y_bounds)
            ax.grid(alpha=0.3, color="white", linewidth=0.5)

        plt.tight_layout()

        if savefigure:
            try:
                save_path = save_figure(fig, title_savefig)
                print(f"Figure sauvegardée : {save_path}")
            except Exception as e:
                print(f"Erreur lors de la sauvegarde : {e}")

        plt.show()

        return {
            "mu_hat"              : mu_hat,
            "mu_star"             : mu_star_grid,
            "diff"                : diff,
            "squared_mu_hat"      : squared_mu_hat,
            "var_mu_hat"          : var_mu_hat,
            "std_mu_hat"          : std_mu_hat,
            "lower_mu_hat"        : lower_mu_hat,
            "upper_mu_hat"        : upper_mu_hat,
            "mu_hat_sims"         : mu_hat_sims,
            "mu_field"            : mu_hat_field,
            "std_mu_field"        : std_mu_hat_field,
            "lower_mu_field"      : lower_mu_hat_field,
            "upper_mu_field"      : upper_mu_hat_field,
            "mesh"                : mesh,
            "mu_post_gp"          : mu_post_grid,
            "Sigma_post_gp"       : Sigma_post_grid,
            "eps_hat"             : eps_hat,
            "f_data_hat"          : f_data_hat,
            "E_mu_bar"            : E_mu_bar,
            "E_mu_chain"          : E_mu_post,
            "rmse"                : rmse,
            "mae"                 : mae,
            "crps"                : crps_bar,
            "ecp"                 : ecp,
            "alpha_ecp"           : alpha_ecp,
        }
    
    
    def plot_chains(self, results, figsize=(9, 5), burn_in=0.3,
                    savefigure=False, title_savefig="traces_eps",
                    trace_color=None, hist_color="steelblue", burn_in_color="red"):
        """Plot full traces and post-burn-in histograms for ε and optionally ν.
 
        Produces a panel of (J + 2) × 2 subplots: for each component of ε and
        (if learned) ν, the left column shows the chain trace and the right
        column shows the marginal histogram.
 
        Parameters
        ----------
        results : dict
            Output of :meth:`run`.
        figsize : tuple of float, optional
            Figure size (width, height) in inches (default (9, 5)).
        burn_in : float, optional
            Fraction excluded from histograms; full traces remain visible.
        savefigure : bool, optional
            If True, save the figure as PDF (default False).
        title_savefig : str, optional
            Filename stem (default 'traces_eps').
        trace_color : color-like, optional
            Trace-line color. If None, Matplotlib chooses it.
        hist_color : color-like, optional
            Histogram fill color.
        burn_in_color : color-like, optional
            Color of the burn-in limit line.
        """
        eps_chain = np.asarray(results["eps"])
        nu_chain = np.asarray(results["nu"])
        thin = results.get("thin", 1)
        n_iter = results.get("n_iter", eps_chain.shape[0])
        n_store = eps_chain.shape[0]
        if not 0.0 <= burn_in < 1.0:
            raise ValueError("burn_in must be in [0, 1).")
        burn = int(n_store * burn_in)

        # Axe x en vraies itérations
        iters = np.arange(n_store) * thin

        J = eps_chain.shape[1]
        fig, axes = plt.subplots(J, 2, figsize=(figsize[0], 3 * J), squeeze=False)
        for j in range(J):
            axes[j, 0].plot(iters, eps_chain[:, j], linewidth=1, color=trace_color)
            axes[j, 0].axvline(burn * thin, color=burn_in_color, linestyle="--", alpha=0.5)
            axes[j, 0].set_title(rf"Trace $\epsilon_{j}$")
            axes[j, 0].set_xlabel(f"Iteration (thin={thin})")
            axes[j, 0].grid(alpha=0.3)
            axes[j, 1].hist(eps_chain[burn:, j], bins=30, density=True,
                            edgecolor="black", alpha=0.7, color=hist_color)
            axes[j, 1].set_title(rf"Posterior $\epsilon_{j}$")
            axes[j, 1].grid(alpha=0.3)
        plt.tight_layout()
        if savefigure:
            try:
                save_path = save_figure(fig, title_savefig)
                print(f"Figure sauvegardée : {save_path}")
            except Exception as e:
                print(f"Erreur lors de la sauvegarde : {e}")
        plt.show()

        if results["acceptance_nu"] is not None:
            fig, axes = plt.subplots(2, 2, figsize=(figsize[0], 6), squeeze=False)
            labels = [r"$v^2$", r"$\ell$"]
            for k in range(2):
                axes[k, 0].plot(iters, nu_chain[:, k], linewidth=1, color=trace_color)
                axes[k, 0].axvline(burn * thin, color=burn_in_color, linestyle="--", alpha=0.5)
                axes[k, 0].set_title(rf"Trace {labels[k]}")
                axes[k, 0].set_xlabel(f"Iteration (thin={thin})")
                axes[k, 0].grid(alpha=0.3)
                axes[k, 1].hist(nu_chain[burn:, k], bins=30, density=True,
                                edgecolor="black", alpha=0.7, color=hist_color)
                axes[k, 1].set_title(rf"Posterior {labels[k]}")
                axes[k, 1].grid(alpha=0.3)
            plt.tight_layout()
            if savefigure:
                try:
                    save_path = save_figure(fig, "traces_nu")
                    print(f"Figure sauvegardée : {save_path}")
                except Exception as e:
                    print(f"Erreur lors de la sauvegarde : {e}")
            plt.show()


    def plot_acf(self, results, burn_in=0.3, max_lag=50, figsize=(8, 6),
                 savefigure=False, title_savefig="trace_acf"):
        """Plot post-burn-in autocorrelation functions for SSGC chains.

        Parameters
        ----------
        results : dict
            Output from :meth:`run`.
        burn_in : float, optional
            Fraction of stored draws discarded.
        max_lag : int, optional
            Largest displayed lag.
        figsize : tuple of float, optional
            Base figure width and height.
        savefigure : bool, optional
            Save the figure as PDF.
        title_savefig : str, optional
            Output filename or stem.

        Returns
        -------
        fig : matplotlib.figure.Figure or None
            Figure, or ``None`` when too few post-burn-in draws are available.
        """
        eps_chain = np.asarray(results["eps"])
        burn = int(burn_in * eps_chain.shape[0])
        n_post = eps_chain.shape[0] - burn
        max_lag = min(int(max_lag), n_post - 1)
        if max_lag < 1:
            print(f"[plot_acf] Not enough post-burn-in draws ({n_post}).")
            return None

        plots = [(rf"$\epsilon_{j}$", eps_chain[burn:, j]) for j in range(eps_chain.shape[1])]
        if results.get("acceptance_nu") is not None:
            nu_chain = np.asarray(results["nu"])
            plots.extend([(r"$v^2$", nu_chain[burn:, 0]), (r"$\ell$", nu_chain[burn:, 1])])

        fig, axes = plt.subplots(len(plots), 1, figsize=(figsize[0], 3.0 * len(plots)), squeeze=False)
        lags = np.arange(max_lag + 1)
        thin = results.get("thin", 1)
        for ax, (label, chain) in zip(axes[:, 0], plots):
            values = self._acf(chain, max_lag)
            ax.plot(lags[:len(values)], values)
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set(xlim=(0, max_lag), ylim=(-1.0, 1.0), xlabel="Lag")
            ax.set_title(f"ACF - {label} (thin={thin})")
            ax.grid(alpha=0.3)
        plt.tight_layout()
        if savefigure:
            save_figure(fig, title_savefig)
        plt.show()
        return fig

    def compute_diagnostics_multichain(self, results_list, burn_in=0.3):
        """Compute rank-normalized R-hat and effective sample sizes for eps.

        Parameters
        ----------
        results_list : sequence of dict
            At least two independent outputs from :meth:`run`.
        burn_in : float, optional
            Fraction discarded independently from each stored chain.

        Returns
        -------
        r_hat, ess_bulk, ess_tail : tuple of ndarray
            One diagnostic value per domain log-intensity.
        """
        import arviz as az

        if len(results_list) < 2:
            raise ValueError("At least two independent chains are required.")
        post_chains = []
        for result in results_list:
            chain = np.asarray(result["eps"], dtype=float)
            burn = int(burn_in * chain.shape[0])
            post_chains.append(chain[burn:])
        draws = min(chain.shape[0] for chain in post_chains)
        if draws < 4:
            raise ValueError("At least four post-burn-in draws per chain are required.")
        if any(chain.shape[1] != self.J for chain in post_chains):
            raise ValueError("All chains must contain the same number of domains.")

        eps_array = np.stack([chain[:draws] for chain in post_chains], axis=0)
        r_hat = np.asarray(az.rhat(eps_array, method="rank"), dtype=float)
        ess_bulk = np.asarray(az.ess(eps_array, method="bulk"), dtype=float)
        ess_tail = np.asarray(
            az.ess(eps_array, method="tail", prob=0.05), dtype=float
        )
        return r_hat, ess_bulk, ess_tail


"""
SPIN_H_GibbsSampler

Extends SSGC_GibbsSampler with spatio-temporal self-excitation (ETAS),
latent branching structure, and optional magnitude modelling (Gutenberg-Richter).

Modes :
    use_etas=False            →  SSGC sampler inherited from the parent
    use_etas=True,  m=None    →  Hawkes ST       : θ_φ = {A, c, p, d, q}
    use_etas=True,  m=array   →  Hawkes marqué   : θ_φ = {A, α, c, p, d, q, γ}  ± β

All MH blocks use Adaptive Metropolis (Haario, Saksman & Tamminen, 2001).
"""

import numpy as np
import openturns as ot
import matplotlib.pyplot as plt
from shapely.geometry import Point as ShapelyPoint

# from .ssgc_gibbs_sampler import SSGC_GibbsSampler
