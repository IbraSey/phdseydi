import math

import numpy as np
import openturns as ot
from polyagamma import random_polyagamma
from shapely.geometry import Point as ShapelyPoint

from ..config import ETASParameters
from ..models import SPINHModel

from .backends import SparseGP

from .ssgc_gibbs import SSGC_GibbsSampler

class SPIN_H_GibbsSampler(SSGC_GibbsSampler):
    """Gibbs sampler for the spatial SPIN-Hawkes model.
    
    This class extends :class:`SSGC_GibbsSampler` with a spatio-temporal ETAS
    triggering component. An event is either background (``Z[i] == 0``) or is
    assigned to an earlier parent ``j`` using the one-based label ``Z[i] == j+1``.
    When magnitudes are supplied, productivity and spatial scale depend on the
    parent magnitude and the truncated Gutenberg-Richter rate ``beta`` can be
    updated.
    
    Parameters
    ----------
    model : SPINHModel
        Configured SPIN-H model defining the SSGC background and ETAS settings.
    theta_phi_priors : dict or None, optional
        Shape/rate hyperparameters of Gamma priors. Priors for ``p`` and ``q``
        apply to ``p-1`` and ``q-1``.
    m : array_like or None, optional
        Event magnitudes in the same order as the observations. Passing ``None``
        selects the unmarked spatio-temporal model.
    beta_init : float, optional
        Initial Gutenberg-Richter rate.
    beta_priors : dict or None, optional
        Gamma shape/rate entries ``a_beta`` and ``b_beta``.
    sigma_MH_etas : float, optional
        Initial random-walk standard deviation for ETAS log-parameters.
    sigma_MH_beta : float, optional
        Initial random-walk standard deviation for ``log(beta)``.
    t0_etas : int, optional
        Number of warm-up iterations before adaptive Metropolis covariance is used.
    eps_mh_etas : float, optional
        Diagonal regularization of adaptive proposal covariances.
    rng_seed : int or None, optional
        OpenTURNS random seed forwarded to the parent sampler."""

    # ── Clipping bounds (log scale) pour stabilité numérique ──
    _LOG_BOUNDS = {
        "A":     (np.log(1e-6),  np.log(100.0)),
        "alpha": (np.log(1e-4),  np.log(20.0)),
        "c":     (np.log(1e-6),  np.log(10.0)),
        "p_m1":  (np.log(1e-4),  np.log(20.0)),
        "d":     (np.log(1e-8),  np.log(50.0)),
        "q_m1":  (np.log(1e-4),  np.log(20.0)),
        "gamma": (np.log(1e-4),  np.log(20.0)),
        "beta":  (np.log(0.1),   np.log(30.0)),
    }

    # ─────────────────────────────────────────────────────────
    #  __init__
    # ─────────────────────────────────────────────────────────

    def __init__(
        self,
        model,
        theta_phi_priors=None,
        m=None,
        beta_init=2.3, beta_priors=None,
        sigma_MH_etas=0.1, sigma_MH_beta=0.1,
        t0_etas=50, eps_mh_etas=1e-6,
        rng_seed=None,
    ):
        """Initialize the SPIN-Hawkes sampler; see the class docstring for parameters."""
        if not isinstance(model, SPINHModel):
            raise TypeError("model must be a SPINHModel instance.")
        super().__init__(model=model, rng_seed=rng_seed)
        self.use_etas = True
        self.t0_etas = t0_etas
        self.eps_mh_etas = eps_mh_etas
        self.sigma_MH_etas = sigma_MH_etas
        self.sigma_MH_beta = sigma_MH_beta

        self.m_c = model.magnitude_min
        if m is not None:
            self.m = np.asarray(m)
            self.use_magnitudes = True
            self.m_max = (
                model.magnitude_max
                if model.magnitude_max is not None
                else float(np.max(self.m)) + 1.0
            )
            self.beta = float(beta_init)
            self.beta_priors = {"a_beta": 2.0, "b_beta": 1.0, **(beta_priors or {})}
        else:
            self.m, self.use_magnitudes = None, False
            self.m_max, self.beta, self.beta_priors = None, None, None

        self.theta_phi = model.etas_parameters.as_dict()
        pr = {"a_A": 2.0, "b_A": 1.0, "a_c": 2.0, "b_c": 1.0,
              "a_p": 2.0, "b_p": 1.0, "a_d": 2.0, "b_d": 1.0,
              "a_q": 2.0, "b_q": 1.0}
        if self.use_magnitudes:
            pr.update({"a_alpha": 2.0, "b_alpha": 1.0,
                       "a_gamma": 2.0, "b_gamma": 1.0})
        self.theta_phi_priors = {**pr, **(theta_phi_priors or {})}

    def _etas_parameters(self, **updates):
        values = dict(self.theta_phi)
        values.update(updates)
        if not self.use_magnitudes:
            values.pop("alpha", None)
            values.pop("gamma", None)
        return ETASParameters(**values)

    def _compute_T_j(self, t, c=None, p=None):
        """Compute temporal truncation factors T_j for every event.

        T_j = 1 − (c / (T − t_j + c))^{p−1},   j = 1, …, N.

        Parameters
        ----------
        t : array_like, shape (N,)
            Event times.
        c : float or None
            Omori parameter. Uses self.theta_phi['c'] if None.
        p : float or None
            Omori exponent. Uses self.theta_phi['p'] if None.

        Returns
        -------
        T_j : ndarray, shape (N,)
            Values in (0, 1].
        """
        parameters = self._etas_parameters(
            c=self.theta_phi["c"] if c is None else float(c),
            p=self.theta_phi["p"] if p is None else float(p),
        )
        return self.model.temporal_compensator(
            np.asarray([float(value) for value in t]), parameters
        )

    def _spatial_integration_grid(self, n=40):
        """Build and cache a rectangular midpoint-like quadrature grid.
        
        Parameters
        ----------
        n : int, optional
            Number of grid coordinates along each axis.
        
        Returns
        -------
        xy : ndarray, shape (n*n, 2)
            Quadrature locations over the bounding rectangle.
        weight : float
            Common area weight assigned to each location."""
        cache_key = f"_spatial_quad_{n}"
        if hasattr(self, cache_key):
            return getattr(self, cache_key)

        xmin, xmax = self.X_bounds
        ymin, ymax = self.Y_bounds
        gx = np.linspace(xmin, xmax, n)
        gy = np.linspace(ymin, ymax, n)
        GX, GY = np.meshgrid(gx, gy)
        xy = np.column_stack([GX.ravel(), GY.ravel()])
        area = (xmax - xmin) * (ymax - ymin)
        quad = (xy, area / xy.shape[0])
        setattr(self, cache_key, quad)
        return quad

    def _spatial_truncation_factors(self, x, y, d=None, q=None, gamma=None, n_grid=40):
        """Approximate spatial ETAS mass retained inside the observation window.
        
        For each potential parent ``j``, this computes
        ``S_j = integral_W phi_s(u-s_j | m_j) du`` on a cached regular grid. The result
        corrects the ETAS compensator for spatial boundary loss.
        
        Parameters
        ----------
        x, y : array_like, shape (N,)
            Parent coordinates.
        d : float or None, optional
            Candidate spatial scale; uses the current value when omitted.
        q : float or None, optional
            Candidate spatial tail exponent; uses the current value when omitted.
        gamma : float or None, optional
            Candidate magnitude-scale coefficient.
        n_grid : int, optional
            Number of quadrature coordinates per axis.
        
        Returns
        -------
        S_j : ndarray, shape (N,)
            Approximate retained spatial masses, clipped to ``[1e-8, 1]``."""
        updates = {
            "d": self.theta_phi["d"] if d is None else float(d),
            "q": self.theta_phi["q"] if q is None else float(q),
        }
        if self.use_magnitudes:
            updates["gamma"] = (
                self.theta_phi["gamma"] if gamma is None else float(gamma)
            )
        parameters = self._etas_parameters(**updates)
        magnitudes = (
            self.m if self.use_magnitudes
            else np.full(len(x), self.m_c, dtype=float)
        )
        return self.model.spatial_compensator(
            np.asarray([float(value) for value in x]),
            np.asarray([float(value) for value in y]),
            magnitudes,
            parameters,
            n_grid=n_grid,
        )

    # ─────────────────────────────────────────────────────────
    #  Triggering kernel  φ_{ij}
    # ─────────────────────────────────────────────────────────

    def _phi_ij(self, i, j, t, x, y):
        """Evaluate the ETAS triggering density from event ``j`` to event ``i``.
        
        Parameters
        ----------
        i, j : int
            Zero-based child and candidate-parent indices.
        t, x, y : array_like, shape (N,)
            Event times and coordinates.
        
        Returns
        -------
        float
            ``A * phi_m * phi_t * phi_s`` when ``t[i] > t[j]``; otherwise zero."""
        delta_t = float(t[i]) - float(t[j])
        if delta_t <= 0:
            return 0.0
        magnitude = self.m[j] if self.use_magnitudes else self.m_c
        distance_squared = (
            (float(x[i]) - float(x[j])) ** 2
            + (float(y[i]) - float(y[j])) ** 2
        )
        return float(
            self.model.etas_kernel.pairwise(
                np.asarray(delta_t),
                np.asarray(distance_squared),
                np.asarray(magnitude),
                self._etas_parameters(),
                self.m_c,
            )
        )

    # ─────────────────────────────────────────────────────────
    #  Branching structure  Z
    # ─────────────────────────────────────────────────────────

    def update_Z(self, t, x, y, eps_arr, f_data):
        """Sample the branching labels conditional on the current intensities.
        
        For event ``i``, label zero has weight equal to its background intensity and
        label ``j+1`` has weight ``phi_ij`` for each earlier event ``j``.
        
        Parameters
        ----------
        t, x, y : array_like, shape (N,)
            Ordered event times and coordinates.
        eps_arr : array_like, shape (J,)
            Current zonal log-intensities.
        f_data : array_like, shape (N,)
            Current or kriged GP values at every observed event.
        
        Returns
        -------
        Z : ot.Point, shape (N,)
            One-based parent labels, with zero denoting a background event."""
        N = len(t)
        if not self.use_etas:
            return ot.Point([0.0] * N)

        t_arr = getattr(self, "_t_obs_arr", None)
        x_arr = getattr(self, "_x_obs_arr", None)
        y_arr = getattr(self, "_y_obs_arr", None)
        XY = getattr(self, "_XY_obs", None)
        if t_arr is None or len(t_arr) != N:
            t_arr = np.array([float(t[i]) for i in range(N)], dtype=float)
            x_arr = np.array([float(x[i]) for i in range(N)], dtype=float)
            y_arr = np.array([float(y[i]) for i in range(N)], dtype=float)
            XY = ot.Sample(np.column_stack([x_arr, y_arr]).tolist())
        mu_t = self.compute_mu_tilde(XY, eps=eps_arr)
        sig = 1.0 / (1.0 + np.exp(-np.array(f_data)))
        mu = mu_t * sig

        Z_new = np.zeros(N)
        for i in range(N):
            if i == 0:
                Z_new[i] = 0.0
                continue

            dt_mat = getattr(self, "_etas_dt_mat", None)
            valid_mat = getattr(self, "_etas_valid_mat", None)
            if dt_mat is not None and valid_mat is not None and dt_mat.shape == (N, N):
                dt = dt_mat[i, :i]
                valid = valid_mat[i, :i]
            else:
                dt = t_arr[i] - t_arr[:i]
                valid = dt > 0.0
            labels = np.concatenate(([0], np.arange(1, i + 1)[valid]))

            if np.any(valid):
                tp = self.theta_phi
                parent_idx = np.arange(i)[valid]
                prod = np.full(parent_idx.size, tp["A"], dtype=float)
                if self.use_magnitudes:
                    dm_all = getattr(self, "_etas_dm", None)
                    dm = dm_all[parent_idx] if dm_all is not None and len(dm_all) == N else self.m[parent_idx] - self.m_c
                    prod *= np.exp(tp["alpha"] * dm)
                    R = tp["d"] * np.exp(tp["gamma"] * dm)
                else:
                    R = np.full(parent_idx.size, tp["d"], dtype=float)

                dt_valid = dt[valid]
                phi_t = (
                    (tp["p"] - 1.0)
                    * tp["c"] ** (tp["p"] - 1.0)
                    * (dt_valid + tp["c"]) ** (-tp["p"])
                )
                r2_mat = getattr(self, "_etas_r2_mat", None)
                if r2_mat is not None and r2_mat.shape == (N, N):
                    r2 = r2_mat[i, parent_idx]
                else:
                    r2 = (x_arr[i] - x_arr[parent_idx]) ** 2 + (y_arr[i] - y_arr[parent_idx]) ** 2
                phi_s = (tp["q"] - 1.0) / (np.pi * R) * (1.0 + r2 / R) ** (-tp["q"])
                weights = np.concatenate(([mu[i]], prod * phi_t * phi_s))
            else:
                weights = np.array([mu[i]])

            total = weights.sum()
            Z_new[i] = np.random.choice(labels, p=weights / total) if total > 0 else 0
        return ot.Point(Z_new.tolist())

    # ─────────────────────────────────────────────────────────
    #  Block {A, α}  or  {A}
    #
    #  log p(A,α|·) ∝  (a_A−1+Σo_j) log A − b_A A
    #                 + (a_α−1) log α − b_α α + α Σ o_j(m_j−m_c)
    #                 − A Σ exp(α(m_j−m_c)) T_j
    #
    #  T_j uses the current (c, p), held fixed in this conditional.
    # ─────────────────────────────────────────────────────────

    def _log_posterior_A_alpha(self, A, alpha, t, x, y, Z):
        """Evaluate the unnormalized conditional log-posterior of productivity.
        
        Parameters
        ----------
        A : float
            Candidate baseline productivity.
        alpha : float
            Candidate magnitude-productivity coefficient; ignored in the unmarked model.
        t, x, y : array_like, shape (N,)
            Event times and coordinates used by the finite-window compensator.
        Z : array_like, shape (N,)
            Current branching labels.
        
        Returns
        -------
        float
            Log-posterior value, or ``-inf`` outside the support."""
        if A <= 0:
            return -np.inf
        if self.use_magnitudes and alpha <= 0:
            return -np.inf

        tp = self.theta_phi_priors
        Z_arr = np.asarray([int(float(Z[i])) for i in range(len(Z))], dtype=int)
        parent_idx = Z_arr[Z_arr > 0] - 1
        o_j = np.bincount(parent_idx, minlength=len(Z_arr)).astype(float)

        T_j = self._compute_T_j(t)  # uses current c, p
        S_j = self._spatial_truncation_factors(x, y)

        log_lik = o_j.sum() * np.log(A)
        if self.use_magnitudes:
            dm = self.m - self.m_c
            log_lik += alpha * np.sum(o_j * dm)
            log_lik -= A * np.sum(np.exp(alpha * dm) * T_j * S_j)
        else:
            log_lik -= A * np.sum(T_j * S_j)

        log_pr = (tp["a_A"] - 1) * np.log(A) - tp["b_A"] * A
        if self.use_magnitudes:
            log_pr += (tp["a_alpha"] - 1) * np.log(alpha) - tp["b_alpha"] * alpha
        return log_lik + log_pr

    def update_A_alpha(self, t, x, y, Z, history, it):
        """Update ``(A, alpha)`` with adaptive Metropolis on the log scale.
        
        Parameters
        ----------
        t, x, y : array_like, shape (N,)
            Event times and coordinates.
        Z : array_like, shape (N,)
            Current branching labels.
        history : list
            Previous log-parameter states; appended in place.
        it : int
            Current Gibbs iteration.
        
        Returns
        -------
        accepted : bool
            Whether the proposal was accepted. In the unmarked model only ``A`` is
            updated."""
        use_mag = self.use_magnitudes
        dim = 2 if use_mag else 1
        sd = 2.38 ** 2 / dim
        B = self._LOG_BOUNDS

        A_cur = self.theta_phi["A"]
        alpha_cur = self.theta_phi.get("alpha", 0.0)
        log_cur = (np.array([np.log(A_cur), np.log(alpha_cur)]) if use_mag
                   else np.array([np.log(A_cur)]))
        history.append(log_cur.copy())

        # Proposal
        if it > self.t0_etas and len(history) > self.t0_etas:
            h = np.array(history)
            cov = (sd * np.cov(h.T) + self.eps_mh_etas * np.eye(dim)) if dim > 1 \
                  else None
            std = np.sqrt(sd * np.var(h, ddof=1) + self.eps_mh_etas) if dim == 1 else None
        else:
            cov = self.sigma_MH_etas ** 2 * np.eye(dim) if dim > 1 else None
            std = self.sigma_MH_etas if dim == 1 else None

        if dim == 1:
            log_star = np.array([log_cur[0] + std * float(ot.Normal().getRealization()[0])])
            if not (B["A"][0] <= log_star[0] <= B["A"][1]):
                return False
            A_s, alpha_s = np.exp(log_star[0]), 0.0
        else:
            log_star = log_cur + np.array(ot.Normal(ot.Point(dim, 0.0), ot.CovarianceMatrix(cov.tolist())).getRealization())
            if not (B["A"][0] <= log_star[0] <= B["A"][1]):
                return False
            if not (B["alpha"][0] <= log_star[1] <= B["alpha"][1]):
                return False
            A_s, alpha_s = np.exp(log_star[0]), np.exp(log_star[1])

        lp_cur = self._log_posterior_A_alpha(A_cur, alpha_cur, t, x, y, Z)
        lp_star = self._log_posterior_A_alpha(A_s, alpha_s, t, x, y, Z)
        if not np.isfinite(lp_star):
            return False
        if np.log(float(ot.Uniform(0.0, 1.0).getRealization()[0])) < min(0.0, (lp_star - lp_cur) + np.sum(log_star - log_cur)):
            self.theta_phi["A"] = A_s
            if use_mag:
                self.theta_phi["alpha"] = alpha_s
            return True
        return False

    # ─────────────────────────────────────────────────────────
    #  Block {c, p}
    #
    #  log p(c,p|·) ∝  Σ_j Σ_{i∈O_j} [log(p−1) + (p−1)log c − p log(Δt+c)]
    #                 − A Σ exp(α(m_j−m_c)) T_j(c,p)
    #                 + prior terms
    #
    #  The integral term depends on the CANDIDATE (c,p) through T_j.
    # ─────────────────────────────────────────────────────────

    def _log_posterior_c_p(self, c, p, t, x, y, Z):
        """Evaluate the unnormalized conditional log-posterior of ``(c, p)``.
        
        Parameters
        ----------
        c : float
            Candidate Omori time offset.
        p : float
            Candidate Omori exponent, constrained to ``p > 1``.
        t, x, y : array_like, shape (N,)
            Event times and coordinates. Coordinates enter the spatial boundary
            correction held at the current spatial parameters.
        Z : array_like, shape (N,)
            Current branching labels.
        
        Returns
        -------
        float
            Log-posterior value, or ``-inf`` outside the support."""
        if c <= 0 or p <= 1:
            return -np.inf
        tp = self.theta_phi_priors

        # Product term over triggered pairs
        Z_arr = np.asarray([int(float(Z[i])) for i in range(len(Z))], dtype=int)
        child_idx = np.flatnonzero(Z_arr > 0)
        parent_idx = Z_arr[child_idx] - 1
        if child_idx.size:
            dt_mat = getattr(self, "_etas_dt_mat", None)
            if dt_mat is not None and dt_mat.shape[0] == len(Z_arr):
                dt = dt_mat[child_idx, parent_idx]
            else:
                t_arr = np.asarray([float(v) for v in t], dtype=float)
                dt = t_arr[child_idx] - t_arr[parent_idx]
            dt = dt[dt > 0.0]
            log_lik = np.sum(np.log(p - 1) + (p - 1) * np.log(c) - p * np.log(dt + c))
        else:
            log_lik = 0.0

        # Integral term:  − A Σ exp(α(m_j−m_c)) T_j(c, p)
        # T_j depends on the candidate c, p
        T_j = self._compute_T_j(t, c=c, p=p)
        S_j = self._spatial_truncation_factors(x, y)
        A_cur = self.theta_phi["A"]
        if self.use_magnitudes:
            alpha_cur = self.theta_phi["alpha"]
            log_lik -= A_cur * np.sum(np.exp(alpha_cur * (self.m - self.m_c)) * T_j * S_j)
        else:
            log_lik -= A_cur * np.sum(T_j * S_j)

        log_pr = (tp["a_c"] - 1) * np.log(c) - tp["b_c"] * c
        log_pr += (tp["a_p"] - 1) * np.log(p - 1) - tp["b_p"] * (p - 1)
        return log_lik + log_pr

    def update_c_p(self, t, x, y, Z, history, it):
        """Update ``(c, p)`` with adaptive Metropolis.
        
        The proposal is Gaussian in ``(log(c), log(p-1))``.
        
        Parameters
        ----------
        t, x, y : array_like, shape (N,)
            Event times and coordinates.
        Z : array_like, shape (N,)
            Current branching labels.
        history : list
            Previous transformed states; appended in place.
        it : int
            Current Gibbs iteration.
        
        Returns
        -------
        accepted : bool
            Whether the proposal was accepted."""
        dim, sd = 2, 2.38 ** 2 / 2
        B = self._LOG_BOUNDS
        log_cur = np.array([np.log(self.theta_phi["c"]),
                            np.log(self.theta_phi["p"] - 1.0)])
        history.append(log_cur.copy())

        if it > self.t0_etas and len(history) > self.t0_etas:
            cov = sd * np.cov(np.array(history).T) + self.eps_mh_etas * np.eye(dim)
        else:
            cov = self.sigma_MH_etas ** 2 * np.eye(dim)

        log_star = log_cur + np.array(ot.Normal(ot.Point(dim, 0.0), ot.CovarianceMatrix(cov.tolist())).getRealization())
        if not (B["c"][0] <= log_star[0] <= B["c"][1]):
            return False
        if not (B["p_m1"][0] <= log_star[1] <= B["p_m1"][1]):
            return False
        c_s, p_s = np.exp(log_star[0]), np.exp(log_star[1]) + 1.0

        lp_cur = self._log_posterior_c_p(self.theta_phi["c"], self.theta_phi["p"], t, x, y, Z)
        lp_star = self._log_posterior_c_p(c_s, p_s, t, x, y, Z)
        if not np.isfinite(lp_star):
            return False
        if np.log(float(ot.Uniform(0.0, 1.0).getRealization()[0])) < min(0.0, (lp_star - lp_cur) + np.sum(log_star - log_cur)):
            self.theta_phi["c"] = c_s
            self.theta_phi["p"] = p_s
            return True
        return False

    # ─────────────────────────────────────────────────────────
    #  Block {d, q, γ}  or  {d, q}
    #
    #  The spatial compensator is integrated over the observed window.
    # ─────────────────────────────────────────────────────────

    def _log_posterior_d_q_gamma(self, d, q, gamma, t, x, y, Z):
        """Evaluate the conditional log-posterior of the spatial ETAS parameters.
        
        Parameters
        ----------
        d : float
            Candidate spatial scale.
        q : float
            Candidate spatial tail exponent, constrained to ``q > 1``.
        gamma : float
            Candidate magnitude-scale coefficient; ignored in the unmarked model.
        t, x, y : array_like, shape (N,)
            Event times and coordinates.
        Z : array_like, shape (N,)
            Current branching labels.
        
        Returns
        -------
        float
            Unnormalized log-posterior including the finite-window spatial compensator,
            or ``-inf`` outside the support."""
        if d <= 0 or q <= 1:
            return -np.inf
        if self.use_magnitudes and gamma <= 0:
            return -np.inf
        tp = self.theta_phi_priors
        Z_arr = np.asarray([int(float(Z[i])) for i in range(len(Z))], dtype=int)
        child_idx = np.flatnonzero(Z_arr > 0)
        parent_idx = Z_arr[child_idx] - 1
        if child_idx.size:
            r2_mat = getattr(self, "_etas_r2_mat", None)
            if r2_mat is not None and r2_mat.shape[0] == len(Z_arr):
                r2 = r2_mat[child_idx, parent_idx]
            else:
                x_arr = np.asarray([float(v) for v in x], dtype=float)
                y_arr = np.asarray([float(v) for v in y], dtype=float)
                r2 = (x_arr[child_idx] - x_arr[parent_idx]) ** 2 + (y_arr[child_idx] - y_arr[parent_idx]) ** 2

            if self.use_magnitudes:
                dm_all = getattr(self, "_etas_dm", None)
                dm = dm_all[parent_idx] if dm_all is not None and len(dm_all) == len(Z_arr) else self.m[parent_idx] - self.m_c
                R = d * np.exp(gamma * dm)
            else:
                R = np.full(child_idx.size, d, dtype=float)
            log_lik = np.sum(np.log(q - 1) - np.log(np.pi * R) - q * np.log1p(r2 / R))
        else:
            log_lik = 0.0

        T_j = self._compute_T_j(t)
        S_j = self._spatial_truncation_factors(x, y, d=d, q=q, gamma=gamma)
        A_cur = self.theta_phi["A"]
        if self.use_magnitudes:
            alpha_cur = self.theta_phi["alpha"]
            log_lik -= A_cur * np.sum(np.exp(alpha_cur * (self.m - self.m_c)) * T_j * S_j)
        else:
            log_lik -= A_cur * np.sum(T_j * S_j)

        log_pr = (tp["a_d"] - 1) * np.log(d) - tp["b_d"] * d
        log_pr += (tp["a_q"] - 1) * np.log(q - 1) - tp["b_q"] * (q - 1)
        if self.use_magnitudes:
            log_pr += (tp["a_gamma"] - 1) * np.log(gamma) - tp["b_gamma"] * gamma
        return log_lik + log_pr

    def update_d_q_gamma(self, t, x, y, Z, history, it):
        """Update the spatial ETAS block with adaptive Metropolis.
        
        The proposal uses ``log(d)``, ``log(q-1)``, and, in the marked model,
        ``log(gamma)``.
        
        Parameters
        ----------
        t, x, y : array_like, shape (N,)
            Event times and coordinates.
        Z : array_like, shape (N,)
            Current branching labels.
        history : list
            Previous transformed states; appended in place.
        it : int
            Current Gibbs iteration.
        
        Returns
        -------
        accepted : bool
            Whether the proposal was accepted."""
        use_mag = self.use_magnitudes
        dim = 3 if use_mag else 2
        sd = 2.38 ** 2 / dim
        B = self._LOG_BOUNDS

        if use_mag:
            log_cur = np.array([np.log(self.theta_phi["d"]),
                                np.log(self.theta_phi["q"] - 1.0),
                                np.log(self.theta_phi["gamma"])])
        else:
            log_cur = np.array([np.log(self.theta_phi["d"]),
                                np.log(self.theta_phi["q"] - 1.0)])
        history.append(log_cur.copy())

        if it > self.t0_etas and len(history) > self.t0_etas:
            cov = sd * np.cov(np.array(history).T) + self.eps_mh_etas * np.eye(dim)
        else:
            cov = self.sigma_MH_etas ** 2 * np.eye(dim)

        log_star = log_cur + np.array(ot.Normal(ot.Point(dim, 0.0), ot.CovarianceMatrix(cov.tolist())).getRealization())
        if not (B["d"][0] <= log_star[0] <= B["d"][1]):
            return False
        if not (B["q_m1"][0] <= log_star[1] <= B["q_m1"][1]):
            return False
        d_s = np.exp(log_star[0])
        q_s = np.exp(log_star[1]) + 1.0
        gamma_s = 0.0
        if use_mag:
            if not (B["gamma"][0] <= log_star[2] <= B["gamma"][1]):
                return False
            gamma_s = np.exp(log_star[2])

        lp_cur = self._log_posterior_d_q_gamma(
            self.theta_phi["d"], self.theta_phi["q"],
            self.theta_phi.get("gamma", 0.0), t, x, y, Z)
        lp_star = self._log_posterior_d_q_gamma(d_s, q_s, gamma_s, t, x, y, Z)
        if not np.isfinite(lp_star):
            return False
        if np.log(float(ot.Uniform(0.0, 1.0).getRealization()[0])) < min(0.0, (lp_star - lp_cur) + np.sum(log_star - log_cur)):
            self.theta_phi["d"] = d_s
            self.theta_phi["q"] = q_s
            if use_mag:
                self.theta_phi["gamma"] = gamma_s
            return True
        return False

    # ─────────────────────────────────────────────────────────
    #  Block β  (d = 1)
    #
    #  log p(β|·) ∝  (a_β−1+N) log β − N log(1−e^{−β Δm})
    #              − β Σ(m_i−m_c) − b_β β
    # ─────────────────────────────────────────────────────────

    def _log_posterior_beta(self, beta):
        """Evaluate the truncated Gutenberg-Richter log-posterior.
        
        Parameters
        ----------
        beta : float
            Candidate positive rate on ``[m_c, m_max]``.
        
        Returns
        -------
        float
            Unnormalized log-posterior, or ``-inf`` outside the support."""
        if beta <= 0:
            return -np.inf
        N = len(self.m)
        bm = beta * (self.m_max - self.m_c)
        log_trunc = np.log(-np.expm1(-bm))  # = log(1 − e^{−bm}), stable
        if not np.isfinite(log_trunc):
            return -np.inf
        bp = self.beta_priors
        return ((bp["a_beta"] - 1 + N) * np.log(beta)
                - N * log_trunc
                - beta * np.sum(self.m - self.m_c)
                - bp["b_beta"] * beta)

    def update_beta(self, history, it):
        """Update the Gutenberg-Richter rate with adaptive Metropolis.
        
        Parameters
        ----------
        history : list
            Previous ``log(beta)`` states; appended in place.
        it : int
            Current Gibbs iteration.
        
        Returns
        -------
        accepted : bool
            Whether the log-scale proposal was accepted."""
        sd = 2.38 ** 2
        B = self._LOG_BOUNDS
        log_cur = np.log(self.beta)
        history.append(log_cur)

        if it > self.t0_etas and len(history) > self.t0_etas:
            std = np.sqrt(sd * np.var(history, ddof=1) + self.eps_mh_etas)
        else:
            std = self.sigma_MH_beta

        log_star = log_cur + std * float(ot.Normal().getRealization()[0])
        if not (B["beta"][0] <= log_star <= B["beta"][1]):
            return False
        beta_s = np.exp(log_star)

        lp_cur = self._log_posterior_beta(self.beta)
        lp_star = self._log_posterior_beta(beta_s)
        if not np.isfinite(lp_star):
            return False
        if np.log(float(ot.Uniform(0.0, 1.0).getRealization()[0])) < min(0.0, (lp_star - lp_cur) + (log_star - log_cur)):
            self.beta = beta_s
            return True
        return False

    # ─────────────────────────────────────────────────────────
    #  posterior_summary  (extends parent)
    # ─────────────────────────────────────────────────────────

    def posterior_summary(self, results, burn_in=0.3):
        """Compute posterior means and declustering probabilities.
        
        Parameters
        ----------
        results : dict
            Output from :meth:`run`.
        burn_in : float, optional
            Fraction of stored draws discarded from the beginning.
        
        Returns
        -------
        summary : dict
            Parent SSGC summaries plus ``theta_phi_hat`` and eventwise
            ``p_background`` when ETAS is active, and ``beta_hat`` when beta was stored."""
        summary = super().posterior_summary(results, burn_in)
        burn = int(np.asarray(results["eps"]).shape[0] * burn_in)

        if results.get("use_etas", False) and results.get("theta_phi") is not None:
            tp = np.asarray(results["theta_phi"])
            names = results.get("theta_phi_names", [])
            if not names:
                names = ["A", "alpha", "c", "p", "d", "q", "gamma"][:tp.shape[1]]
            summary["theta_phi_hat"] = {n: tp[burn:, i].mean() for i, n in enumerate(names)}
            summary["p_background"] = np.mean(np.asarray(results["Z"])[burn:] == 0, axis=0)

        if results.get("beta") is not None:
            summary["beta_hat"] = np.asarray(results["beta"])[burn:].mean()
        return summary

    # ─────────────────────────────────────────────────────────
    #  run
    # ─────────────────────────────────────────────────────────

    def initialize_Z_heuristic(self, t, x, y, m, mu_tilde_obs, theta_phi, m_c):
        """Initialize branching labels by deterministic maximum-weight assignment.
        
        Only candidate parents within the 99.5% Omori quantile are considered. The
        background weight is ``mu_tilde_obs`` and triggering weights use the supplied
        initial ETAS parameters.
        
        Parameters
        ----------
        t, x, y : array_like, shape (N,)
            Time-ordered event data.
        m : array_like, shape (N,)
            Magnitudes, or zeros for the unmarked model.
        mu_tilde_obs : array_like, shape (N,)
            Initial background weights at observed locations.
        theta_phi : dict
            Initial ETAS parameter values.
        m_c : float
            Magnitude of completeness.
        
        Returns
        -------
        Z : ot.Point, shape (N,)
            Initial one-based parent labels, with zero for background."""
        N = len(t)
        Z = np.zeros(N)
        tp = theta_phi
        dt_max = tp["c"] * ((1.0 - 0.995) ** (-1.0 / (tp["p"] - 1.0)) - 1.0)
        dt_max = min(float(dt_max), float(self.T))

        for i in range(N):
            w_bg = mu_tilde_obs[i]
            best_w, best_j = w_bg, 0

            for j in range(i):
                dt = t[i] - t[j]
                if dt <= 0 or dt > dt_max:
                    continue
                prod = tp["A"]
                if "alpha" in tp:
                    prod *= np.exp(tp["alpha"] * (m[j] - m_c))
                phi_t = (tp["p"]-1) * tp["c"]**(tp["p"]-1) * (dt+tp["c"])**(-tp["p"])
                R = tp["d"]
                if "gamma" in tp:
                    R *= np.exp(tp["gamma"] * (m[j] - m_c))
                r2 = (x[i]-x[j])**2 + (y[i]-y[j])**2
                phi_s = (tp["q"]-1) / (np.pi*R) * (1 + r2/R)**(-tp["q"])
                w = prod * phi_t * phi_s

                if w > best_w:
                    best_w = w
                    best_j = j + 1   # convention Z : j+1 = triggered by event j

            Z[i] = best_j

        n_bg = np.sum(Z == 0)
        print(f"  [init_Z] {n_bg} bg / {N} total ({n_bg/N:.1%})")
        return ot.Point(Z.tolist())
    
    def run(self, t, x, y, mala_step=0.05, n_iter=1000,
            learn_nu=False, learn_beta=False,
            t0_nu=50, step_nu_init=0.1,
            verbose=True, verbose_every=100, use_calibration=True,
            mu_star_func=None, grid_nx=30, grid_ny=30, thin=1,
            compute_emu=False, emu_every=10, calibration_method="sklearn",
            plot_calibration_kde=False, calibration_kde_cmap="viridis",
            gp_backend="exact", sparse_gp=None):
        """Run the SPIN-Hawkes Gibbs sampler.
        
        The update conditions the latent thinned process on current background events,
        kriges GP values to triggered events, samples branching labels, and updates the
        ETAS parameter blocks. With ``use_etas=False`` it delegates to the parent SSGC
        implementation.
        
        Parameters
        ----------
        t, x, y : array_like, shape (N,)
            Time-ordered event times and coordinates.
        mala_step : float, optional
            MALA step size for zonal log-intensities.
        n_iter : int, optional
            Number of Gibbs iterations.
        learn_nu : bool, optional
            Update GP hyperparameters.
        learn_beta : bool, optional
            Update the Gutenberg-Richter rate. Ignored in the unmarked model.
        t0_nu : int, optional
            Warm-up length for GP-hyperparameter adaptation.
        step_nu_init : float, optional
            Initial proposal variance for ``log(nu)``.
        verbose : bool, optional
            Print progress and acceptance rates.
        verbose_every : int, optional
            Iterations between progress messages.
        use_calibration : bool, optional
            Calibrate GP hyperparameters and initialize zonal intensities.
        mu_star_func : callable or None, optional
            Reference background intensity for the optional ``E_mu`` diagnostic.
        grid_nx, grid_ny : int, optional
            Diagnostic-grid dimensions.
        thin : int, optional
            Store one state every ``thin`` iterations.
        compute_emu : bool, optional
            Compute ``E_mu`` when ``mu_star_func`` is supplied. Disabled by default.
        emu_every : int, optional
            Iterations between ``E_mu`` evaluations.
        calibration_method : {"sklearn", "openturns"}, optional
            GP fitter used by pre-run calibration.
        plot_calibration_kde : bool, optional
            Display the KDE used for calibration.
        calibration_kde_cmap : str or Colormap, optional
            Colormap used for the calibration KDE plot.
        gp_backend : {"exact", "sparse"}, optional
            GP backend. In sparse mode, the latent field is represented by
            finite Fourier coefficients instead of exact GP values.
        sparse_gp : object or None, optional
            Optional sparse basis object. When omitted in sparse mode,
            :class:`SparseGP` is constructed from ``nu`` and the rectangular
            observation bounds.
        
        Returns
        -------
        results : dict
            SSGC chains plus ``Z``, ``theta_phi``, optional ``beta``, ETAS acceptance
            rates, model flags, transformed-state histories, and final parameter values."""
        if not self.use_etas:
            return super().run(
                t, x, y, mala_step=mala_step, n_iter=n_iter, learn_nu=learn_nu,
                t0_nu=t0_nu, step_nu_init=step_nu_init, verbose=verbose,
                verbose_every=verbose_every, use_calibration=use_calibration,
                mu_star_func=mu_star_func, grid_nx=grid_nx, grid_ny=grid_ny, thin=thin,
                compute_emu=compute_emu, emu_every=emu_every,
                calibration_method=calibration_method,
                plot_calibration_kde=plot_calibration_kde,
                calibration_kde_cmap=calibration_kde_cmap,
                gp_backend=gp_backend, sparse_gp=sparse_gp)

        gp_backend = str(gp_backend).lower()
        if gp_backend not in {"exact", "sparse"}:
            raise ValueError("gp_backend must be 'exact' or 'sparse'.")

        N = len(t)
        self._t_obs_arr = np.asarray([float(v) for v in t], dtype=float)
        self._x_obs_arr = np.asarray([float(v) for v in x], dtype=float)
        self._y_obs_arr = np.asarray([float(v) for v in y], dtype=float)
        self._XY_obs = ot.Sample(np.column_stack([self._x_obs_arr, self._y_obs_arr]).tolist())
        self._compute_event_domain_indices(self._x_obs_arr, self._y_obs_arr)
        self._etas_dt_mat = self._t_obs_arr[:, None] - self._t_obs_arr[None, :]
        self._etas_valid_mat = np.tril(self._etas_dt_mat > 0.0, k=-1)
        dx = self._x_obs_arr[:, None] - self._x_obs_arr[None, :]
        dy = self._y_obs_arr[:, None] - self._y_obs_arr[None, :]
        self._etas_r2_mat = dx * dx + dy * dy
        self._etas_dm = (self.m - self.m_c) if self.m is not None else np.zeros(N)
        if learn_beta and not self.use_magnitudes:
            if verbose:
                print("[Warning] learn_beta ignoré : pas de magnitudes")
            learn_beta = False

        tp_names = (["A", "alpha", "c", "p", "d", "q", "gamma"]
                    if self.use_magnitudes else ["A", "c", "p", "d", "q"])
        n_tp = len(tp_names)

        # ── Calibration GP ──────────────────────────────────────────────────
        if use_calibration:
            if verbose: print("[Pre-run] Calibrating GP hyperparameters")
            _, _, eps_mle_all = self.calibrate_nu(
                x, y, verbose=verbose, method=calibration_method,
                plot_kde=plot_calibration_kde,
                kde_cmap=calibration_kde_cmap,
            )
        else:
            if verbose: print(f"[Pre-run] nu_init = {list(self.nu)}")
            eps_mle_all = self.estimate_eps_mle(x, y)

        if gp_backend == "sparse":
            if sparse_gp is None:
                sparse_gp = SparseGP.from_bounds(
                    self.X_bounds,
                    self.Y_bounds,
                    variance=float(self.nu[0]),
                    length_scale=float(self.nu[1]),
                )
            self._validate_sparse_gp(sparse_gp)
            if learn_nu:
                raise ValueError("learn_nu=True is not supported with gp_backend='sparse'.")
            gp_coeffs = ot.Point(int(sparse_gp.m), 0.0)
            sparse_design_observed = np.asarray(
                sparse_gp.regressorOT(self._XY_obs), dtype=float
            )
        else:
            gp_coeffs = None
            sparse_design_observed = None

        f_data = ot.Point([0.0] * N)

        # ── Initialisation informée de Z ────────────────────────────────────
        XY_init    = self._XY_obs
        mu_t_init  = self.compute_mu_tilde(XY_init, eps=eps_mle_all)
        mu_init    = mu_t_init * 0.5          # σ(0) = 0.5 (f initialisé à 0)

        Z = self.initialize_Z_heuristic(
            np.array([float(t[i]) for i in range(N)]),
            np.array([float(x[i]) for i in range(N)]),
            np.array([float(y[i]) for i in range(N)]),
            self.m if self.m is not None else np.zeros(N),
            mu_init, self.theta_phi, self.m_c,
        )

        # ── Fix B3 : eps_mle recalculé sur les background events uniquement ─
        N_j_bg, _ = self._count_events_per_zone(x, y, Z, ot.Sample(0, 3))
        areas_j   = np.array([self.polygons[j].area for j in range(self.J)])
        eps_mle   = np.log(np.maximum(N_j_bg / (self.T * areas_j), 1e-6))
        eps       = ot.Point(eps_mle.tolist())

        if verbose:
            mode  = "Hawkes marqué" if self.use_magnitudes else "Hawkes ST"
            flags = [f for f in ["learn_ν" if learn_nu else "",
                                  "learn_β" if learn_beta else ""] if f]
            print(f"[Init] {mode} | θ_φ = {tp_names} | {', '.join(flags) or '—'}")
            if gp_backend == "sparse":
                print(f"[Init] Sparse GP with {int(sparse_gp.m)} basis functions")
            print(f"[Init] ε = {np.round(eps_mle, 4)} | θ_φ = {self.theta_phi}")
            if self.use_magnitudes:
                print(f"[Init] β = {self.beta} (learn={learn_beta}) | AM t0 = {self.t0_etas}")

        # ── Grille E_μ optionnelle ───────────────────────────────────────────
        compute_emu = bool(compute_emu and mu_star_func is not None)
        if compute_emu:
            xmin, xmax = self.X_bounds; ymin, ymax = self.Y_bounds
            GX, GY     = np.meshgrid(np.linspace(xmin, xmax, grid_nx),
                                      np.linspace(ymin, ymax, grid_ny))
            grid_x, grid_y = GX.ravel(), GY.ravel()
            Xg          = ot.Sample(np.column_stack([grid_x, grid_y]).tolist())
            M_grid      = len(grid_x)
            domain_area = (xmax - xmin) * (ymax - ymin)
            mu_star_grid = mu_star_func(grid_x, grid_y)
        else:
            grid_x = grid_y = Xg = mu_star_grid = None
            M_grid = 0
            domain_area = 0.0

        # ── Chaînes ──────────────────────────────────────────────────────────
        ns       = (n_iter + thin - 1) // thin
        eps_ch   = np.zeros((ns, self.J))
        nPi_ch   = np.zeros(ns, dtype=int)
        f_ch     = np.zeros((ns, N))
        nu_ch    = np.zeros((ns, 2))
        Z_ch     = np.zeros((ns, N))
        tp_ch    = np.zeros((ns, n_tp))
        beta_ch  = np.zeros(ns) if learn_beta else None
        gp_coeffs_ch = (
            np.zeros((ns, int(sparse_gp.m))) if gp_backend == "sparse" else None
        )
        Emu      = np.full(n_iter, np.nan)
        si       = 0

        ae, an, ab = 0, 0, 0
        acc        = {"A_alpha": 0, "c_p": 0, "d_q_gamma": 0}
        hnu, hAa, hcp, hdqg, hb = [], [], [], [], []

        if verbose:
            print("\n" + "=" * 100)
            print(f"{'':>32}Gibbs : {n_iter} iter, N={N}{'':>32}")
            print("=" * 100)

        for it in range(n_iter):
            try:
                # ── Steps 1-3 : ω, π_S, latent GP ─────────────────────────
                if gp_backend == "sparse":
                    f_data_np = sparse_design_observed @ np.asarray(gp_coeffs, dtype=float)
                    f_data = ot.Point(f_data_np.tolist())
                    omega = ot.Point(
                        __import__("polyagamma").random_polyagamma(1.0, f_data_np)
                    )
                    Pi_S = self.sample_Pi_S_sparse(eps, sparse_gp, gp_coeffs)
                    gp_coeffs = self.update_sparse_gp_coeffs(
                        x, y, Z, omega, Pi_S, sparse_gp
                    )
                    f_data_np = sparse_design_observed @ np.asarray(gp_coeffs, dtype=float)
                    f_data = ot.Point(f_data_np.tolist())
                    f_Df = D_f = None
                    idx_bg = [i for i in range(N) if float(Z[i]) == 0.0]
                else:
                    omega = ot.Point(
                        __import__("polyagamma").random_polyagamma(1.0, np.array(f_data))
                    )

                    # Condition the thinned PP only on background events so that
                    # f_data values at triggered locations (kept as GP predictions,
                    # not sampled) do not pollute the kriging mean inside sample_Pi_S.
                    idx_bg = [i for i in range(N) if float(Z[i]) == 0.0]
                    x_bg   = ot.Point([float(x[i]) for i in idx_bg])
                    y_bg   = ot.Point([float(y[i]) for i in idx_bg])
                    f_bg   = ot.Point([float(f_data[i]) for i in idx_bg])
                    Pi_S   = self.sample_Pi_S(x_bg, y_bg, f_bg, eps)

                    # update_f already filters Z internally (background only).
                    # After the draw, propagate f to triggered locations via the
                    # GP kriging mean conditioned on the background draw f_D0.
                    f_D0, f_Df, D_f, K_ff = self.update_f(x, y, Z, omega, Pi_S)

                    fa   = np.array(f_data)
                    idx0 = [i for i in range(N) if float(Z[i]) == 0.0]
                    for k, i in enumerate(idx0):
                        fa[i] = float(f_D0[k])

                    idx_trig = [i for i in range(N) if float(Z[i]) != 0.0]
                    if idx_trig and idx0:
                        XY_bg  = ot.Sample([[float(x[i]), float(y[i])] for i in idx0])
                        XY_tr  = ot.Sample([[float(x[i]), float(y[i])] for i in idx_trig])
                        K_bg   = self.compute_kernel(XY_bg)
                        for ii in range(len(idx0)):
                            K_bg[ii, ii] += self.jitter
                        _, K_tr_bg, _ = self.compute_kernel(XY_bg, XY_tr)
                        alpha_bg = K_bg.solveLinearSystem(f_D0)
                        f_pred = np.array(K_tr_bg * alpha_bg).flatten()
                        for k, i in enumerate(idx_trig):
                            fa[i] = float(f_pred[k])

                    f_data = ot.Point(fa.tolist())

                # ── Step 4 : ε | f, π_S  (MALA) ────────────────────────────
                N_j, M_j = self._count_events_per_zone(x, y, Z, Pi_S)
                ea, ok   = self.update_eps(eps, N_j, M_j, step=mala_step)
                eps      = ot.Point(ea.tolist()); ae += int(ok)

                # ── Step 5 : ν | f  (AM, optional) ─────────────────────────
                if learn_nu:
                    hnu.append(np.log(np.array(self.nu)))
                    _, ok = self.update_nu(f_Df, D_f, hnu, it,
                                           step_nu_init=step_nu_init, t0=t0_nu)
                    an += int(ok)

                # ── Step 6 : Z | f, ε, θ_φ ──────────────────────────────────
                Z = self.update_Z(t, x, y, ea, f_data)

                # ── Step 7 : θ_φ | Z, t, x, y  (AM) ────────────────────────
                acc["A_alpha"]   += int(self.update_A_alpha(t, x, y, Z, hAa, it))
                acc["c_p"]       += int(self.update_c_p(t, x, y, Z, hcp, it))
                acc["d_q_gamma"] += int(self.update_d_q_gamma(t, x, y, Z, hdqg, it))

                # ── Step 8 : β | m  (AM, optional) ─────────────────────────
                if learn_beta:
                    ab += int(self.update_beta(hb, it))

                # ── Verbose ───────────────────────────────────────────────────
                if verbose and (it % verbose_every == 0 or it == n_iter - 1):
                    denom = it + 1
                    nb = int(np.sum(np.array(Z) == 0))
                    acc_Aa = acc["A_alpha"] / denom * 100.0
                    acc_cp = acc["c_p"] / denom * 100.0
                    acc_dqg = acc["d_q_gamma"] / denom * 100.0
                    msg = (
                        f"[{it:>5d}] pi_S={Pi_S.getSize():<4d} "
                        f"acc_eps={ae/denom*100:.0f}% "
                        f"bg={nb}/{N} "
                        f"acc_Aα={acc_Aa:.0f}% "
                        f"acc_cp={acc_cp:.0f}% "
                        f"acc_dqγ={acc_dqg:.0f}%"
                    )
                    if learn_beta:
                        msg += f" β={self.beta:.3f} acc_β={ab/denom*100:.0f}%"
                    if learn_nu:
                        msg += f" acc_ν={an/denom*100:.0f}%"
                    if it > self.t0_etas: msg += " [AM]"
                    print(msg)

                # ── E_μ (diagnostic, background GP only) ────────────────────
                if compute_emu and it % emu_every == 0:
                    if gp_backend == "sparse":
                        grid_design = np.asarray(sparse_gp.regressorOT(Xg), dtype=float)
                        fg = grid_design @ np.asarray(gp_coeffs, dtype=float)
                    else:
                        XY_bg_g = ot.Sample(np.column_stack([self._x_obs_arr[idx_bg], self._y_obs_arr[idx_bg]]).tolist())
                        f_bg_pt = ot.Point([float(f_data[i]) for i in idx_bg])
                        Kd, Kg, Kgg = self.compute_kernel(XY_bg_g, Xg)
                        Kr = ot.CovarianceMatrix(Kd)
                        for ii in range(len(idx_bg)): Kr[ii, ii] += self.jitter
                        alpha = Kr.solveLinearSystem(f_bg_pt)
                        mg   = np.array(Kg * alpha).flatten()
                        solved_cross = Kr.solveLinearSystem(ot.Matrix(np.array(Kg).T.tolist()))
                        S    = np.array(Kgg) - np.array(Kg * solved_cross)
                        S    = .5*(S+S.T) + self.jitter*np.eye(M_grid)
                        S_cov = ot.CovarianceMatrix(S.tolist())
                        fg   = np.array(ot.Normal(ot.Point(mg.tolist()), S_cov).getRealization())
                    mt   = self.compute_mu_tilde(Xg, eps=ea)
                    Emu[it] = (domain_area/M_grid) * np.sum(
                        (mt / (1 + np.exp(-fg)) - mu_star_grid) ** 2
                    )

                # ── Stockage ─────────────────────────────────────────────────
                if it % thin == 0:
                    eps_ch[si]  = ea
                    nPi_ch[si]  = Pi_S.getSize()
                    f_ch[si]    = np.array(f_data)
                    nu_ch[si]   = np.array(self.nu)
                    Z_ch[si]    = np.array(Z)
                    tp_ch[si]   = [self.theta_phi[k] for k in tp_names]
                    if beta_ch is not None: beta_ch[si] = self.beta
                    if gp_coeffs_ch is not None:
                        gp_coeffs_ch[si] = np.asarray(gp_coeffs, dtype=float)
                    si += 1

            except Exception as e:
                print(f"\n[ERROR] iter {it}: {e}"); raise

        if verbose:
            print("=" * 100 + "\n")
            print(f"  ε (MALA)       : {ae/n_iter*100:.1f}%  (target ≈57%)")
            if learn_nu: print(f"  ν (AM)         : {an/n_iter*100:.1f}%")
            bl = {"A_alpha": "{A,α}" if self.use_magnitudes else "{A}",
                  "c_p":     "{c,p}",
                  "d_q_gamma": "{d,q,γ}" if self.use_magnitudes else "{d,q}"}
            for k, v in acc.items():
                print(f"  {bl[k]:14s}  : {v/n_iter*100:.1f}%")
            if learn_beta: print(f"  {'β':14s}  : {ab/n_iter*100:.1f}%")

        return {
            "eps": eps_ch[:si], "nPi": nPi_ch[:si], "f_data": f_ch[:si],
            "nu": nu_ch[:si], "Z": Z_ch[:si],
            "theta_phi": tp_ch[:si], "theta_phi_names": tp_names,
            "beta": beta_ch[:si] if beta_ch is not None else None,
            "E_mu": Emu,
            "acceptance_eps":   ae/n_iter,
            "acceptance_nu":    an/n_iter if learn_nu else None,
            "acceptance_beta":  ab/n_iter if learn_beta else None,
            "acceptance_etas":  {k: v/n_iter for k, v in acc.items()},
            "last_state": {
                "eps": ea, "nu": list(self.nu), "delta": list(self.delta),
                "theta_phi": dict(self.theta_phi), "beta": self.beta,
            },
            "Sigma_eps": self.Sigma_eps, "centroids": self.centroids_xy,
            "thin": thin, "n_iter": n_iter,
            "gp_backend": gp_backend,
            "gp_coeffs": gp_coeffs_ch[:si] if gp_coeffs_ch is not None else None,
            "use_etas": True, "use_magnitudes": self.use_magnitudes,
            "learn_beta": learn_beta, "learn_nu": learn_nu,
            "am_history": {
                "A_alpha":   np.array(hAa) if hAa else None,
                "c_p":       np.array(hcp) if hcp else None,
                "d_q_gamma": np.array(hdqg) if hdqg else None,
                "beta":      np.array(hb)  if hb  else None,
                "nu":        np.array(hnu) if hnu else None,
            },
        }
