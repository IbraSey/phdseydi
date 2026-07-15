import math

import matplotlib.pyplot as plt
import numpy as np
import openturns as ot
import openturns.experimental as otexp
from polyagamma import random_polyagamma
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from scipy.special import expit
from shapely.geometry import Point as ShapelyPoint

from ..models import SSGCModel

from .backends import SparseGP

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
    model : SSGCModel
        Configured SSGC model defining the spatial domains, observation window,
        GP prior and epsilon prior.
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
        model,
        rng_seed=None,
    ):
        """Initialize the SSGC sampler; see the class docstring for parameters."""
        if not isinstance(model, SSGCModel):
            raise TypeError("model must be an SSGCModel instance.")

        self.model = model
        self.X_bounds = tuple(model.x_bounds)
        self.Y_bounds = tuple(model.y_bounds)
        self.T = float(model.duration)
        self.lambda_nu = float(model.nu_prior_rate)
        self.nu = ot.Point([model.gp_prior.variance, model.gp_prior.length_scale])
        self.delta = ot.Point([model.eps_prior_variance, model.eps_prior_length_scale])
        self.jitter = float(model.jitter)
        self.domain_partition = model.domains

        self.domains = list(self.domain_partition.polygons)
        self.prepared_domains = list(self.domain_partition.prepared_domains)
        self.domain_areas = self.domain_partition.areas
        self.eps_init = self.domain_partition.initial_log_intensities
        self.n_domains = len(self.domain_partition)

        # Compact internal aliases retained by the numerical implementation.
        self.polygons = self.domains
        self.areas = self.prepared_domains
        self.J = self.n_domains

        if rng_seed is not None:
            ot.RandomGenerator.SetSeed(int(rng_seed))
            self.rng_state = ot.RandomGenerator.GetState()

        self.centroids_xy, self.Sigma_eps = self.compute_Sigma_eps()
        Sigma_eps_reg = ot.CovarianceMatrix(
            (self.Sigma_eps + self.jitter * np.eye(self.J)).tolist()
        )
        self.Sigma_eps_cov = Sigma_eps_reg

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
        counts = np.zeros(self.J)
        for i in range(len(x)):
            pt = ShapelyPoint(float(x[i]), float(y[i]))
            for j, poly in enumerate(self.areas):
                if poly.covers(pt):
                    counts[j] += 1
                    break
        eps_mle = np.zeros(self.J)
        for j in range(self.J):
            area_j = self.polygons[j].area
            rate_j = max(counts[j] / (self.T * area_j), 1e-6)
            eps_mle[j] = np.log(rate_j) 
        return eps_mle
    
    def calibrate_nu(self, x, y, verbose=True, plot_kde=True, method=False, kde_cmap='viridis'):
        """Heuristic calibration of GP hyperparameters via linearised sigmoid inversion.
 
        Under the approximation σ(f) ≈ ½ + ¼f (valid for small v), the model
        can be inverted to yield a Gaussian regression target:
 
            z(x,y) = 2|D| p̂(x,y) − 2 + noise,
 
        where p̂ is a KDE estimate of the spatial density. The hyperparameters
        (v², ℓ) are then obtained by maximising the marginal likelihood of a
        GP regression model fitted to z, using scikit-learn's
        GaussianProcessRegressor.
 
        The variance v² is constrained to remain below 0.58² ≈ 0.34 to ensure
        the linearisation remains valid at the 99% confidence level (see
        Appendix B of the paper).
 
        Parameters
        ----------
        x, y : array_like, shape (N,)
            Spatial coordinates of observed events.
        verbose : bool, optional
            If True, print calibrated values (default True).
        plot_kde : bool, optional
            If True, plot KDE (default False).
        method : bool, optional
            default False
 
        Returns
        -------
        v : float
            Calibrated marginal standard deviation of the GP (√v²).
        l_ot : float
            Calibrated length-scale in the OpenTURNS convention (ℓ_OT = ℓ_sklearn · √2).
        eps_mle : ndarray, shape (J,)
            MLE of zonal log-intensities (computed as a by-product).
        """
        N_obs = len(x)
        obs_pts = np.array([[float(x[i]), float(y[i])] for i in range(N_obs)])

        # KDE -> p_hat aux points observés
        sample_ot = ot.Sample(obs_pts)
        ks = ot.KernelSmoothing()
        kde = ks.build(sample_ot)
        p_hat = np.array(kde.computePDF(sample_ot)).flatten()
        
        if plot_kde:
            graph = kde.drawPDF([self.X_bounds[0], self.Y_bounds[0]], [self.X_bounds[1], self.Y_bounds[1]]) 
            graph.add(ot.Cloud( np.vstack((x, y)).T ))
            view = View(graph)
            view.show()
            view.save("kde.png")


        # # KDE leave-one-out aux points observés
        # sample_ot = ot.Sample(obs_pts)
        # ks = ot.KernelSmoothing()
        # h = ks.computeSilvermanBandwidth(sample_ot)
        # kde_full = ks.build(sample_ot, h)
        # p_hat_full = np.array(kde_full.computePDF(sample_ot)).flatten()
        # K0 = 1.0 / (2.0 * np.pi * float(h[0]) * float(h[1]))
        # p_hat = (N_obs * p_hat_full - K0) / (N_obs - 1)
        # p_hat = np.maximum(p_hat, 1e-10)

        # eps MLE (conservé pour initialisation du sampler)
        eps_mle = self.estimate_eps_mle(x, y)
        if verbose:
            print(f"[calibrate_nu] eps_mle = {np.round(eps_mle, 4)}")

        # Aire totale du domaine
        D_area = sum(self.polygons[j].area for j in range(self.J))

        # Cible : z(x,y) = 2|D| p_hat(x,y) - 2
        z = 2.0 * D_area * p_hat - 2.0
        #print(z)

        # GP regression
        if method and False:
            print("using_scikit_learn")
            kernel = (
                C(0.1, (1e-3, 0.58 ** 2))
                #C(0.1, (1e-3, 2.0))
                * RBF(length_scale=0.3, length_scale_bounds=(1e-2, 5.0))
                + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-4, 1.0))
            )
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
            gp.fit(obs_pts, z)

            k_params = gp.kernel_.get_params()
            v_sq = float(k_params["k1__k1__constant_value"])
            l = float(k_params["k1__k2__length_scale"])
            v = np.sqrt(v_sq)
            l_ot = l * np.sqrt(2.0)
            self.nu = ot.Point([v_sq, l_ot])
        else:
            print("using_OT")
            dimension = 2
            basis = ot.ConstantBasisFactory(dimension).build()
            covarianceModel = ot.SquaredExponential( [1.]*dimension, [1.])
            # covarianceModel =  ot.IsotropicCovarianceModel(ot.SquaredExponential( [1.], [1.]), dimension)
            fitter_algo = otexp.GaussianProcessFitter(sample_ot, z.reshape(-1, 1),covarianceModel, basis)
            # fitter_algo.setOptimizationAlgorithm(ot.NLopt("LN_COBYLA"))
            fitter_algo.run()
            fitter_result = fitter_algo.getResult().getCovarianceModel()
            l_ot = min(fitter_result.getScale())
            v_sq = fitter_result.getAmplitude()[0]
            self.nu = ot.Point([v_sq, l_ot])
        
        if verbose:
            print(f"[calibrate_nu] v_sq = {np.round(v_sq, 4)} ; l_ot = {l_ot:.4f}")

        return v_sq, l_ot, eps_mle
    
    

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
        compute_emu=True, emu_every=10, calibration_method="openturns",
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
            "sparse_gp"      : sparse_gp if gp_backend == "sparse" else None,
        }
    
