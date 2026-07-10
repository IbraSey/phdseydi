"""Simple hybrid CAVI inference for the SPIN-H model.

The implementation follows the Polya-Gamma augmented updates used by SPIN-H,
but keeps the first usable version intentionally modest:

- q(Z) is categorical for each event;
- q(omega) is represented through its Polya-Gamma expectation;
- q(pi_S) is represented by deterministic quadrature expected counts;
- q(epsilon) is updated by a Laplace/Newton step;
- q(f) is a Gaussian approximation at observed locations;
- ETAS and beta are updated as weighted MAP point estimates with SciPy.

This is therefore a practical CAVI/MAP hybrid rather than a fully conjugate
mean-field implementation for every ETAS parameter. Parameters can be fixed with
``fixed_etas`` and ``fixed_beta`` in the same spirit as the Gibbs sampler.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logsumexp

from package.config import ETASParameters
from data.catalog import EventCatalog
from ..models.spinh import SPINHModel
from .backends import SparseGP
from .results import SPINHVIResults


ETAS_PARAMETER_NAMES = ("A", "alpha", "c", "p", "d", "q", "gamma")


@dataclass(frozen=True)
class SPINHVIConfig:
    """Configuration for the simple SPIN-H CAVI/MAP hybrid routine."""

    n_iter: int = 200
    tolerance: float = 1e-4
    verbose: bool = True
    verbose_every: int = 10
    random_seed: int | None = None

    update_z: bool = True
    update_polya_gamma: bool = True
    update_latent_poisson: bool = True
    update_gp: bool = True
    update_eps: bool = True
    update_etas: bool = True
    learn_beta: bool = True

    fixed_etas: dict[str, float] = field(default_factory=dict)
    fixed_beta: float | None = None
    beta_init: float = 2.3
    beta_prior: dict[str, float] = field(
        default_factory=lambda: {"a_beta": 2.0, "b_beta": 1.0}
    )
    theta_priors: dict[str, float] = field(default_factory=dict)

    quadrature_nx: int = 30
    quadrature_ny: int = 30
    eps_newton_steps: int = 8
    eps_damping: float = 1.0
    eps_bounds: tuple[float, float] = (-20.0, 8.0)
    f_bounds: tuple[float, float] = (-15.0, 15.0)
    latent_poisson_damping: float = 0.5
    latent_poisson_max_multiplier: float | None = 1.5
    etas_update_start: int = 10
    parameter_damping: float = 0.6
    max_optimizer_iter: int = 80
    full_gp_max_events: int = 800
    gp_backend: str = "exact"
    sparse_gp: object | None = None
    spatial_compensator_grid: int = 0
    jitter: float = 1e-6

    def __post_init__(self):
        if self.n_iter <= 0:
            raise ValueError("n_iter must be positive.")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive.")
        if self.verbose_every <= 0:
            raise ValueError("verbose_every must be positive.")
        if self.quadrature_nx <= 1 or self.quadrature_ny <= 1:
            raise ValueError("quadrature grid sizes must be greater than one.")
        if self.eps_newton_steps <= 0:
            raise ValueError("eps_newton_steps must be positive.")
        if not 0 < self.eps_damping <= 1:
            raise ValueError("eps_damping must be in (0, 1].")
        if self.eps_bounds[0] >= self.eps_bounds[1]:
            raise ValueError("eps_bounds must be increasing.")
        if self.f_bounds[0] >= self.f_bounds[1]:
            raise ValueError("f_bounds must be increasing.")
        if not 0 < self.latent_poisson_damping <= 1:
            raise ValueError("latent_poisson_damping must be in (0, 1].")
        if (
            self.latent_poisson_max_multiplier is not None
            and self.latent_poisson_max_multiplier <= 0
        ):
            raise ValueError("latent_poisson_max_multiplier must be positive or None.")
        if self.etas_update_start < 0:
            raise ValueError("etas_update_start must be non-negative.")
        if not 0 < self.parameter_damping <= 1:
            raise ValueError("parameter_damping must be in (0, 1].")
        if self.max_optimizer_iter <= 0:
            raise ValueError("max_optimizer_iter must be positive.")
        if self.full_gp_max_events <= 0:
            raise ValueError("full_gp_max_events must be positive.")
        if str(self.gp_backend).lower() not in {"exact", "sparse"}:
            raise ValueError("gp_backend must be 'exact' or 'sparse'.")
        if self.sparse_gp is not None and str(self.gp_backend).lower() != "sparse":
            raise ValueError("sparse_gp requires gp_backend='sparse'.")
        if self.spatial_compensator_grid < 0:
            raise ValueError("spatial_compensator_grid must be non-negative.")
        if self.jitter <= 0:
            raise ValueError("jitter must be positive.")
        if self.beta_init <= 0:
            raise ValueError("beta_init must be positive.")
        if self.fixed_beta is not None and self.fixed_beta <= 0:
            raise ValueError("fixed_beta must be positive.")
        unknown = set(self.fixed_etas).difference(ETAS_PARAMETER_NAMES)
        if unknown:
            raise ValueError(f"Unknown fixed ETAS parameters: {sorted(unknown)}")
        for name, value in self.fixed_etas.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"fixed_etas[{name!r}] must be numeric.")


@dataclass
class BranchingFactor:
    """Categorical variational factor over event parents."""

    probabilities: np.ndarray

    @classmethod
    def background_initialization(cls, n_events: int):
        probabilities = np.zeros((n_events, n_events + 1), dtype=float)
        probabilities[:, 0] = 1.0
        return cls(probabilities)

    @property
    def p_background(self) -> np.ndarray:
        return self.probabilities[:, 0]


@dataclass
class PolyaGammaFactor:
    """Polya-Gamma variational factor at observed and quadrature locations."""

    observed_mean: np.ndarray
    observed_tilt: np.ndarray
    grid_mean: np.ndarray | None = None
    grid_tilt: np.ndarray | None = None


@dataclass
class LatentPoissonFactor:
    """Quadrature representation of q(pi_S)."""

    grid_xy: np.ndarray
    grid_domain_index: np.ndarray
    grid_intensity: np.ndarray
    cell_area: float
    expected_counts_by_domain: np.ndarray

    @property
    def expected_count(self) -> float:
        return float(np.sum(self.expected_counts_by_domain))


@dataclass
class GPFactor:
    """Gaussian approximation for the latent GP at data and quadrature points."""

    f_data_mean: np.ndarray
    f_data_var: np.ndarray
    f_grid_mean: np.ndarray
    f_grid_var: np.ndarray
    covariance: np.ndarray | None = None
    coefficients_mean: np.ndarray | None = None
    coefficients_covariance: np.ndarray | None = None


@dataclass
class EpsilonFactor:
    """Gaussian Laplace approximation for domain log-intensities."""

    mean: np.ndarray
    covariance: np.ndarray


@dataclass
class ETASFactor:
    """Point-estimate variational block for ETAS and magnitude parameters."""

    parameters_mean: ETASParameters
    beta_mean: float
    fixed_etas: dict[str, float] = field(default_factory=dict)


@dataclass
class SPINHVIState:
    """Current variational state."""

    branching: BranchingFactor
    polya_gamma: PolyaGammaFactor
    latent_poisson: LatentPoissonFactor
    gp: GPFactor
    eps: EpsilonFactor
    etas: ETASFactor


class SPINHVI:
    """Simple coordinate-ascent variational inference for SPIN-H."""

    def __init__(self, model, catalog, config=None):
        if not isinstance(model, SPINHModel):
            raise TypeError("model must be a SPINHModel instance.")
        if not isinstance(catalog, EventCatalog):
            raise TypeError("catalog must be an EventCatalog instance.")
        self.model = model
        self.catalog = catalog
        self.config = SPINHVIConfig() if config is None else config
        if not isinstance(self.config, SPINHVIConfig):
            raise TypeError("config must be a SPINHVIConfig instance.")
        self.domain_index = model.validate_catalog(catalog)
        self.gp_backend = str(self.config.gp_backend).lower()
        self.sparse_gp = self._make_sparse_gp() if self.gp_backend == "sparse" else None
        self.rng = np.random.default_rng(self.config.random_seed)
        self.priors = self._default_theta_priors() | dict(self.config.theta_priors)
        self.beta_prior = {"a_beta": 2.0, "b_beta": 1.0} | dict(
            self.config.beta_prior
        )
        self._validate_fixed_parameters()
        self.state = self.initialize_state()



    # ===================================================================================================
    # ============================================ OUTILLAGE ============================================
    # ===================================================================================================

    # **************************************** OUTILS BACKGROUND ****************************************
    def _validate_fixed_parameters(self):
        if not self.model.etas_parameters.marked:
            marked_fixed = {"alpha", "gamma"}.intersection(self.config.fixed_etas)
            if marked_fixed:
                raise ValueError(
                    f"Cannot fix marked parameters for an unmarked model: {sorted(marked_fixed)}"
                )
        for name, value in self.config.fixed_etas.items():
            if name == "A" and value < 0:
                raise ValueError("A must be non-negative.")
            if name in {"alpha", "gamma"} and value < 0:
                raise ValueError(f"{name} must be non-negative.")
            if name in {"c", "d"} and value <= 0:
                raise ValueError(f"{name} must be positive.")
            if name in {"p", "q"} and value <= 1:
                raise ValueError(f"{name} must be greater than one.")

    def _default_theta_priors(self) -> dict[str, float]:
        return {
            "a_A": 2.0, "b_A": 2.0,
            "a_alpha": 2.0, "b_alpha": 2.0,
            "a_c": 2.0, "b_c": 20.0,
            "a_p": 2.0, "b_p": 2.0,
            "a_d": 2.0, "b_d": 20.0,
            "a_q": 2.0, "b_q": 2.0,
            "a_gamma": 2.0, "b_gamma": 2.0,
        }

    def _parameters_with_fixed(self, parameters: ETASParameters) -> ETASParameters:
        values = parameters.as_dict()
        values.update(self.config.fixed_etas)
        if not parameters.marked:
            values.pop("alpha", None)
            values.pop("gamma", None)
        else:
            values.setdefault("alpha", 0.0 if parameters.alpha is None else parameters.alpha)
            values.setdefault("gamma", 0.0 if parameters.gamma is None else parameters.gamma)
        return ETASParameters(**values)
    
    def _make_sparse_gp(self):
        if self.config.sparse_gp is not None:
            return self.config.sparse_gp
        return SparseGP.from_bounds(
            self.model.x_bounds,
            self.model.y_bounds,
            self.model.gp_prior.variance,
            self.model.gp_prior.length_scale,
        )

    def _sparse_design(self, xy: np.ndarray) -> np.ndarray:
        if self.sparse_gp is None:
            raise RuntimeError("Sparse GP backend is not initialized.")
        xy = np.asarray(xy, dtype=float)
        if xy.size == 0:
            return np.zeros((0, int(self.sparse_gp.m)), dtype=float)
        x = xy[:, 0:1]
        y = xy[:, 1:2]
        phi_x = np.sin(
            np.pi * self.sparse_gp.S[0][None, :]
            * (x - self.sparse_gp.c1 + self.sparse_gp.L1)
            / (2.0 * self.sparse_gp.L1)
        ) / np.sqrt(self.sparse_gp.L1)
        phi_y = np.sin(
            np.pi * self.sparse_gp.S[1][None, :]
            * (y - self.sparse_gp.c2 + self.sparse_gp.L2)
            / (2.0 * self.sparse_gp.L2)
        ) / np.sqrt(self.sparse_gp.L2)
        return phi_x * phi_y * self.sparse_gp.sqrt_Delta[None, :]

    def _make_quadrature_grid(self) -> tuple[np.ndarray, np.ndarray, float]:
        nx = self.config.quadrature_nx
        ny = self.config.quadrature_ny
        x_edges = np.linspace(self.model.x_bounds[0], self.model.x_bounds[1], nx + 1)
        y_edges = np.linspace(self.model.y_bounds[0], self.model.y_bounds[1], ny + 1)
        x_mid = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_mid = 0.5 * (y_edges[:-1] + y_edges[1:])
        X, Y = np.meshgrid(x_mid, y_mid)
        grid_xy = np.column_stack([X.ravel(), Y.ravel()])
        domain_index = self.model.domains.locate(grid_xy[:, 0], grid_xy[:, 1])
        inside = domain_index >= 0
        cell_area = float((x_edges[1] - x_edges[0]) * (y_edges[1] - y_edges[0]))
        return grid_xy[inside], domain_index[inside], cell_area

    def _magnitudes(self) -> np.ndarray:
        if self.catalog.magnitudes is None:
            return np.full(len(self.catalog), self.model.magnitude_min, dtype=float)
        return self.catalog.magnitudes
    
    @staticmethod
    def _expected_log_sigmoid(mean, variance, sign=1.0, n_nodes=20):
        mean = np.asarray(mean, dtype=float)
        variance = np.maximum(np.asarray(variance, dtype=float), 0.0)
        nodes, weights = np.polynomial.hermite.hermgauss(n_nodes)
        values = sign * (mean[..., None] + np.sqrt(2.0 * variance[..., None]) * nodes)
        log_sigmoid = -np.logaddexp(0.0, -values)
        return np.sum(weights * log_sigmoid, axis=-1) / np.sqrt(np.pi)
    
    def _pair_log_etas(self, child_idx, parent_idx, params: ETASParameters) -> np.ndarray:
        child_idx = np.atleast_1d(np.asarray(child_idx, dtype=int))
        parent_idx = np.atleast_1d(np.asarray(parent_idx, dtype=int))
        if child_idx.size == 1 and parent_idx.size > 1:
            child_idx = np.full(parent_idx.shape, child_idx.item(), dtype=int)
        elif parent_idx.size == 1 and child_idx.size > 1:
            parent_idx = np.full(child_idx.shape, parent_idx.item(), dtype=int)
        elif child_idx.size != parent_idx.size:
            raise ValueError("child_idx and parent_idx must be broadcastable.")
        dt = self.catalog.t[child_idx] - self.catalog.t[parent_idx]
        valid = dt > 0
        out = np.full(dt.shape, -np.inf, dtype=float)
        if not np.any(valid):
            return out
        dx = self.catalog.x[child_idx[valid]] - self.catalog.x[parent_idx[valid]]
        dy = self.catalog.y[child_idx[valid]] - self.catalog.y[parent_idx[valid]]
        r2 = dx**2 + dy**2
        magnitudes = self._magnitudes()
        dm = magnitudes[parent_idx[valid]] - self.model.magnitude_min
        alpha = 0.0 if params.alpha is None else params.alpha
        gamma = 0.0 if params.gamma is None else params.gamma
        R = params.d * np.exp(gamma * dm)
        out[valid] = (
            np.log(max(params.A, self.config.jitter))
            + alpha * dm
            + np.log(params.p - 1.0)
            + (params.p - 1.0) * np.log(params.c)
            - params.p * np.log(dt[valid] + params.c)
            + np.log(params.q - 1.0)
            - np.log(np.pi)
            - np.log(R)
            - params.q * np.log1p(r2 / R)
        )
        return out
    
    def _stabilize_latent_poisson_intensity(self, intensity: np.ndarray) -> np.ndarray:
        intensity = np.asarray(intensity, dtype=float)
        intensity = np.where(np.isfinite(intensity), intensity, 0.0)
        intensity = np.maximum(intensity, 0.0)
        old_intensity = self.state.latent_poisson.grid_intensity
        if old_intensity.shape == intensity.shape and np.any(old_intensity > 0):
            damping = self.config.latent_poisson_damping
            intensity = (1.0 - damping) * old_intensity + damping * intensity
        if self.config.latent_poisson_max_multiplier is not None:
            cap = self.config.latent_poisson_max_multiplier * max(1, len(self.catalog))
            expected_count = float(np.sum(intensity) * self.state.latent_poisson.cell_area)
            if expected_count > cap:
                intensity = intensity * (cap / expected_count)
        return intensity
    
    def _rbf_kernel(self, xy1: np.ndarray, xy2: np.ndarray) -> np.ndarray:
        diff = xy1[:, None, :] - xy2[None, :, :]
        dist2 = np.sum(diff**2, axis=2)
        return self.model.gp_prior.variance * np.exp(
            -dist2 / (2.0 * self.model.gp_prior.length_scale**2)
        )
    
    @staticmethod
    def _pg_mean(c: np.ndarray) -> np.ndarray:
        c = np.asarray(c, dtype=float)
        out = np.empty_like(c)
        small = c < 1e-8
        out[small] = 0.25
        out[~small] = np.tanh(c[~small] / 2.0) / (2.0 * c[~small])
        return out

    @staticmethod
    def _log_cosh(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        abs_x = np.abs(x)
        return abs_x + np.log1p(np.exp(-2.0 * abs_x)) - np.log(2.0)

    def _pg_tilting_entropy_correction(
        self,
        c: np.ndarray,
        omega_mean: np.ndarray,
    ) -> np.ndarray:
        return 0.5 * c**2 * omega_mean - self._log_cosh(c / 2.0)

    def _augmented_log_sigmoid_expectation(
        self,
        mean: np.ndarray,
        variance: np.ndarray,
        omega_mean: np.ndarray,
        sign: float,
        omega_tilt: np.ndarray | None = None,
    ) -> np.ndarray:
        second_moment = mean**2 + variance
        if omega_tilt is None:
            omega_tilt = np.sqrt(np.maximum(second_moment, 0.0))
        return (
            0.5 * sign * mean
            - 0.5 * omega_mean * second_moment
            - np.log(2.0)
            + self._pg_tilting_entropy_correction(omega_tilt, omega_mean)
        )
    
    @staticmethod
    def _gaussian_entropy_from_covariance(covariance, jitter=1e-8):
        covariance = np.asarray(covariance, dtype=float)
        n_dim = covariance.shape[0]
        sign, logdet = np.linalg.slogdet(
            covariance + jitter * np.eye(n_dim)
        )
        if sign <= 0:
            return 0.0
        return 0.5 * (n_dim * (1.0 + np.log(2.0 * np.pi)) + logdet)

    @staticmethod
    def _independent_gaussian_entropy(variance, jitter=1e-8):
        variance = np.maximum(np.asarray(variance, dtype=float), jitter)
        return 0.5 * float(np.sum(1.0 + np.log(2.0 * np.pi * variance)))
    

    # ************************************************ OUTILS TRIGGERING ************************************************
    def _free_etas_names(self) -> list[str]:
        names = ["A", "c", "p", "d", "q"]
        if self.model.etas_parameters.marked:
            names.extend(["alpha", "gamma"])
        return [name for name in names if name not in self.config.fixed_etas]

    def _etas_optimizer_bounds(self, names: list[str]) -> list[tuple[float, float]]:
        raw_bounds = {
            "A": (1e-4, 5.0),
            "alpha": (1e-6, 3.0),
            "c": (1e-4, 1.0),
            "p": (0.05, 4.0),
            "d": (1e-4, 2.0),
            "q": (0.05, 5.0),
            "gamma": (1e-6, 3.0),
        }
        return [(np.log(raw_bounds[name][0]), np.log(raw_bounds[name][1])) for name in names]

    def _params_to_vector(self, params: ETASParameters, names: list[str]) -> np.ndarray:
        values = []
        data = params.as_dict()
        for name in names:
            value = max(float(data[name]), self.config.jitter)
            if name in {"p", "q"}:
                values.append(np.log(max(value - 1.0, self.config.jitter)))
            else:
                values.append(np.log(value))
        return np.asarray(values, dtype=float)

    def _vector_to_params(self, vector: np.ndarray, names: list[str]) -> ETASParameters:
        values = self.state.etas.parameters_mean.as_dict()
        for name, raw in zip(names, vector):
            value = float(np.exp(np.clip(raw, -30.0, 30.0)))
            values[name] = 1.0 + value if name in {"p", "q"} else value
        values.update(self.config.fixed_etas)
        if not self.model.etas_parameters.marked:
            values.pop("alpha", None)
            values.pop("gamma", None)
        return ETASParameters(**values)

    def _damped_parameters(
        self, current: ETASParameters, proposed: ETASParameters
    ) -> ETASParameters:
        damping = self.config.parameter_damping
        current_values = current.as_dict()
        proposed_values = proposed.as_dict()
        values = {}
        for name, value in proposed_values.items():
            if name in self.config.fixed_etas:
                values[name] = self.config.fixed_etas[name]
            else:
                values[name] = (1.0 - damping) * current_values[name] + damping * value
        return ETASParameters(**values)
    
    def _triggering_compensator(self, params: ETASParameters) -> float:
        magnitudes = self._magnitudes()
        dm = magnitudes - self.model.magnitude_min
        alpha = 0.0 if params.alpha is None else params.alpha
        productivity = params.A * np.exp(alpha * dm)
        temporal = self.model.temporal_compensator(self.catalog.t, params)
        if self.config.spatial_compensator_grid > 0:
            spatial = self.model.spatial_compensator(
                self.catalog.x,
                self.catalog.y,
                magnitudes,
                params,
                n_grid=self.config.spatial_compensator_grid,
            )
        else:
            spatial = np.ones(len(self.catalog), dtype=float)
        return float(np.sum(productivity * temporal * spatial))
    
    def _etas_log_prior(self, params: ETASParameters) -> float:
        values = params.as_dict()
        total = 0.0
        for name, value in values.items():
            shifted = value - 1.0 if name in {"p", "q"} else value
            if shifted <= 0:
                return -np.inf
            a = self.priors.get(f"a_{name}", 1.0)
            b = self.priors.get(f"b_{name}", 0.0)
            total += (a - 1.0) * np.log(shifted) - b * shifted
        return float(total)

    def _etas_log_posterior(self, params: ETASParameters) -> float:
        values = params.as_dict()
        if any(value <= 0 for name, value in values.items() if name != "A"):
            return -np.inf
        if values["A"] <= 0 or values["p"] <= 1 or values["q"] <= 1:
            return -np.inf
        q_parent = np.tril(self.state.branching.probabilities[:, 1:], k=-1)
        child_idx, parent_idx = np.nonzero(q_parent > 0)
        weights = q_parent[child_idx, parent_idx]
        if weights.size:
            pair_ll = float(
                np.sum(weights * self._pair_log_etas(child_idx, parent_idx, params))
            )
        else:
            pair_ll = 0.0
        compensator = self._triggering_compensator(params)
        return pair_ll - compensator + self._etas_log_prior(params)
    

    # ********************************************* OUTILS MARK *********************************************
    def _beta_log_posterior(self):
        if self.catalog.magnitudes is None:
            return 0.0
        beta = float(self.state.etas.beta_mean)
        if beta <= 0:
            return -np.inf
        magnitudes = self.catalog.magnitudes
        lower = self.model.magnitude_min
        upper = self.model.magnitude_max
        excess = magnitudes - lower
        a = self.beta_prior.get("a_beta", 2.0)
        b = self.beta_prior.get("b_beta", 1.0)
        value = (a - 1.0 + len(magnitudes)) * np.log(beta) - beta * (
            b + np.sum(excess)
        )
        if upper is not None:
            width = max(upper - lower, self.config.jitter)
            normalizer = max(1.0 - np.exp(-beta * width), self.config.jitter)
            value -= len(magnitudes) * np.log(normalizer)
        return float(value)



    # ===================================================================================================
    # --------------------------------------------- UPDATES ---------------------------------------------
    # ===================================================================================================

    def _update_branching(self):
        n_events = len(self.catalog)
        probabilities = np.zeros((n_events, n_events + 1), dtype=float)
        eps = self.state.eps.mean
        f_mean = self.state.gp.f_data_mean
        f_second = f_mean**2 + self.state.gp.f_data_var
        omega = self.state.polya_gamma.observed_mean
        bg_log = (
            eps[self.domain_index]
            + 0.5 * f_mean
            - 0.5 * omega * f_second
            - np.log(2.0)
        )
        params = self.state.etas.parameters_mean
        for i in range(n_events):
            weights = [bg_log[i]]
            if i > 0:
                weights.extend(self._pair_log_etas(i, np.arange(i), params).tolist())
            probabilities[i, : i + 1] = np.exp(weights - logsumexp(weights))
        self.state.branching.probabilities = probabilities

    def _update_polya_gamma(self):
        second_moment = self.state.gp.f_data_mean**2 + self.state.gp.f_data_var
        c = np.sqrt(np.maximum(second_moment, 0.0))
        self.state.polya_gamma.observed_tilt = c
        self.state.polya_gamma.observed_mean = self._pg_mean(c)
        if self.state.latent_poisson.grid_xy.size:
            m_grid = self.state.gp.f_grid_mean
            v_grid = self.state.gp.f_grid_var
            c_grid = np.sqrt(np.maximum(m_grid**2 + v_grid, 0.0))
            self.state.polya_gamma.grid_tilt = c_grid
            self.state.polya_gamma.grid_mean = self._pg_mean(c_grid)

    def _update_latent_poisson(self):
        grid = self.state.latent_poisson.grid_xy
        if grid.size == 0:
            return
        eps_mean = self.state.eps.mean[self.state.latent_poisson.grid_domain_index]
        f_mean = self.state.gp.f_grid_mean
        f_var = self.state.gp.f_grid_var
        if self.state.polya_gamma.grid_mean is None:
            omega_tilt = np.sqrt(np.maximum(f_mean**2 + f_var, 0.0))
            omega_mean = self._pg_mean(omega_tilt)
        else:
            omega_mean = self.state.polya_gamma.grid_mean
            omega_tilt = self.state.polya_gamma.grid_tilt
        log_rejected = self._augmented_log_sigmoid_expectation(
            f_mean,
            f_var,
            omega_mean,
            sign=-1.0,
            omega_tilt=omega_tilt,
        )
        log_intensity = np.log(self.model.duration) + eps_mean + log_rejected
        intensity = np.exp(np.clip(log_intensity, -60.0, 60.0))
        intensity = self._stabilize_latent_poisson_intensity(intensity)
        counts = np.bincount(
            self.state.latent_poisson.grid_domain_index,
            weights=intensity * self.state.latent_poisson.cell_area,
            minlength=self.model.n_domains,
        )
        self.state.latent_poisson.grid_intensity = intensity
        self.state.latent_poisson.expected_counts_by_domain = counts

    def _update_eps(self):
        p_bg = self.state.branching.p_background
        counts = np.bincount(
            self.domain_index,
            weights=p_bg,
            minlength=self.model.n_domains,
        ).astype(float)
        if self.config.update_latent_poisson:
            counts = counts + self.state.latent_poisson.expected_counts_by_domain
        exposure = self.model.duration * np.asarray(self.model.domains.areas, dtype=float)
        prior_cov = self.model.epsilon_prior_covariance()
        prior_precision = np.linalg.inv(
            prior_cov + self.config.jitter * np.eye(self.model.n_domains)
        )
        eps = np.clip(self.state.eps.mean.copy(), *self.config.eps_bounds)
        for _ in range(self.config.eps_newton_steps):
            mu = np.exp(np.clip(eps, *self.config.eps_bounds))
            gradient = counts - exposure * mu - prior_precision @ eps
            precision = prior_precision + np.diag(exposure * mu + self.config.jitter)
            step = np.linalg.solve(precision, gradient)
            eps = np.clip(eps + self.config.eps_damping * step, *self.config.eps_bounds)
            if np.linalg.norm(step) < 1e-6:
                break
        mu = np.exp(np.clip(eps, *self.config.eps_bounds))
        precision = prior_precision + np.diag(exposure * mu + self.config.jitter)
        covariance = np.linalg.inv(precision)
        self.state.eps.mean = eps
        self.state.eps.covariance = covariance

    def _update_gp(self):
        n_events = len(self.catalog)
        grid_xy = self.state.latent_poisson.grid_xy
        n_grid = grid_xy.shape[0]
        p_bg = self.state.branching.p_background
        omega_data = np.maximum(
            self.state.polya_gamma.observed_mean * p_bg,
            self.config.jitter,
        )
        grid_counts = self.state.latent_poisson.grid_intensity * self.state.latent_poisson.cell_area
        if self.state.polya_gamma.grid_mean is None:
            omega_grid = np.full(n_grid, 0.25, dtype=float)
        else:
            omega_grid = self.state.polya_gamma.grid_mean
        omega_grid = np.maximum(grid_counts * omega_grid, self.config.jitter)
        natural = np.concatenate([0.5 * p_bg, -0.5 * grid_counts])
        likelihood_precision = np.concatenate([omega_data, omega_grid])
        xy = np.vstack([self.catalog.xy, grid_xy])
        total_points = n_events + n_grid

        if self.gp_backend == "sparse":
            design = self._sparse_design(xy)
            precision = np.eye(design.shape[1]) + (
                design.T * likelihood_precision
            ) @ design
            covariance = np.linalg.inv(
                precision + self.config.jitter * np.eye(design.shape[1])
            )
            coefficients_mean = covariance @ (design.T @ natural)
            mean = design @ coefficients_mean
            variance = np.maximum(
                np.sum((design @ covariance) * design, axis=1),
                self.config.jitter,
            )
            self.state.gp.f_data_mean = np.clip(mean[:n_events], *self.config.f_bounds)
            self.state.gp.f_data_var = variance[:n_events]
            self.state.gp.f_grid_mean = np.clip(mean[n_events:], *self.config.f_bounds)
            self.state.gp.f_grid_var = variance[n_events:]
            self.state.gp.covariance = None
            self.state.gp.coefficients_mean = coefficients_mean
            self.state.gp.coefficients_covariance = covariance
            return

        if total_points > self.config.full_gp_max_events:
            prior_precision = 1.0 / self.model.gp_prior.variance
            data_precision = prior_precision + omega_data
            grid_precision = prior_precision + omega_grid
            self.state.gp.f_data_var = 1.0 / data_precision
            self.state.gp.f_data_mean = np.clip(
                self.state.gp.f_data_var * natural[:n_events],
                *self.config.f_bounds,
            )
            self.state.gp.f_grid_var = 1.0 / grid_precision
            self.state.gp.f_grid_mean = np.clip(
                self.state.gp.f_grid_var * natural[n_events:],
                *self.config.f_bounds,
            )
            self.state.gp.covariance = None
            self.state.gp.coefficients_mean = None
            self.state.gp.coefficients_covariance = None
            return

        K = self._rbf_kernel(xy, xy)
        K.flat[:: total_points + 1] += self.config.jitter
        K_inv = np.linalg.inv(K)
        precision = K_inv + np.diag(likelihood_precision)
        covariance = np.linalg.inv(precision)
        mean = covariance @ natural
        variance = np.maximum(np.diag(covariance), self.config.jitter)
        self.state.gp.f_data_mean = np.clip(mean[:n_events], *self.config.f_bounds)
        self.state.gp.f_data_var = variance[:n_events]
        self.state.gp.f_grid_mean = np.clip(mean[n_events:], *self.config.f_bounds)
        self.state.gp.f_grid_var = variance[n_events:]
        self.state.gp.covariance = covariance
        self.state.gp.coefficients_mean = None
        self.state.gp.coefficients_covariance = None

    def _update_etas(self):
        free_names = self._free_etas_names()
        if not free_names:
            return
        start = self._params_to_vector(self.state.etas.parameters_mean, free_names)

        def negative_elbo_block(vector):
            params = self._vector_to_params(vector, free_names)
            value = self._etas_log_posterior(params)
            if not np.isfinite(value):
                return 1e100
            return -value

        result = minimize(
            negative_elbo_block,
            start,
            method="L-BFGS-B",
            bounds=self._etas_optimizer_bounds(free_names),
            options={"maxiter": self.config.max_optimizer_iter, "ftol": 1e-6},
        )
        if not result.success and not np.isfinite(result.fun):
            return
        proposed = self._vector_to_params(result.x, free_names)
        current = self.state.etas.parameters_mean
        damped = self._damped_parameters(current, proposed)
        self.state.etas.parameters_mean = self._parameters_with_fixed(damped)

    def _update_beta(self):
        if self.catalog.magnitudes is None:
            return
        magnitudes = self.catalog.magnitudes
        lower = self.model.magnitude_min
        upper = self.model.magnitude_max
        excess = magnitudes - lower
        a = self.beta_prior.get("a_beta", 2.0)
        b = self.beta_prior.get("b_beta", 1.0)
        width = None if upper is None else max(upper - lower, self.config.jitter)

        def negative_elbo_block(log_beta):
            beta = float(np.exp(log_beta))
            value = (a - 1.0 + len(magnitudes)) * np.log(beta) - beta * (
                b + np.sum(excess)
            )
            if width is not None:
                normalizer = max(1.0 - np.exp(-beta * width), self.config.jitter)
                value -= len(magnitudes) * np.log(normalizer)
            return -value

        result = minimize_scalar(
            negative_elbo_block,
            bounds=(np.log(1e-3), np.log(50.0)),
            method="bounded",
            options={"maxiter": self.config.max_optimizer_iter},
        )
        if result.success:
            beta = float(np.exp(result.x))
            damping = self.config.parameter_damping
            self.state.etas.beta_mean = (1.0 - damping) * self.state.etas.beta_mean + damping * beta
    


    # ===================================================================================================
    # ----------------------------------------------- FIT -----------------------------------------------
    # ===================================================================================================

    def initialize_state(self) -> SPINHVIState:
        n_events = len(self.catalog)
        n_domains = self.model.n_domains
        counts = np.bincount(self.domain_index, minlength=n_domains).astype(float)
        exposure = self.model.duration * np.asarray(self.model.domains.areas, dtype=float)
        eps_mean = np.log((counts + 0.5) / np.maximum(exposure, self.config.jitter))
        eps_cov = self.model.epsilon_prior_covariance()
        f_mean = np.zeros(n_events, dtype=float)
        f_var = np.full(n_events, self.model.gp_prior.variance, dtype=float)
        grid_xy, grid_domains, cell_area = self._make_quadrature_grid()
        latent = LatentPoissonFactor(
            grid_xy=grid_xy,
            grid_domain_index=grid_domains,
            grid_intensity=np.zeros(grid_xy.shape[0], dtype=float),
            cell_area=cell_area,
            expected_counts_by_domain=np.zeros(n_domains, dtype=float),
        )
        f_grid_mean = np.zeros(grid_xy.shape[0], dtype=float)
        f_grid_var = np.full(grid_xy.shape[0], self.model.gp_prior.variance, dtype=float)
        params = self._parameters_with_fixed(self.model.etas_parameters)
        beta = (
            float(self.config.fixed_beta)
            if self.config.fixed_beta is not None
            else float(self.config.beta_init)
        )
        return SPINHVIState(
            branching=BranchingFactor.background_initialization(n_events),
            polya_gamma=PolyaGammaFactor(
                observed_mean=np.full(n_events, 0.25, dtype=float),
                observed_tilt=np.zeros(n_events, dtype=float),
                grid_mean=np.full(grid_xy.shape[0], 0.25, dtype=float),
                grid_tilt=np.zeros(grid_xy.shape[0], dtype=float),
            ),
            latent_poisson=latent,
            gp=GPFactor(
                f_data_mean=f_mean,
                f_data_var=f_var,
                f_grid_mean=f_grid_mean,
                f_grid_var=f_grid_var,
                covariance=None,
                coefficients_mean=(
                    np.zeros(int(self.sparse_gp.m), dtype=float)
                    if self.sparse_gp is not None else None
                ),
                coefficients_covariance=(
                    np.eye(int(self.sparse_gp.m), dtype=float)
                    if self.sparse_gp is not None else None
                ),
            ),
            eps=EpsilonFactor(mean=eps_mean, covariance=eps_cov),
            etas=ETASFactor(
                parameters_mean=params,
                beta_mean=beta,
                fixed_etas=dict(self.config.fixed_etas),
            ),
        )

    def _print_progress(self, iteration: int, elbo: float):
        params = self.state.etas.parameters_mean
        print(
            f"[VI {iteration:04d}] elbo={elbo:.3f} "
            f"p_bg={self.state.branching.p_background.mean():.3f} "
            f"E[pi_S]={self.state.latent_poisson.expected_count:.2f} "
            f"A={params.A:.3f} alpha={params.alpha:.3f} c={params.c:.4f} p={params.p:.3f} "
            f"d={params.d:.4f} q={params.q:.3f} gamma={params.gamma:.3f} beta={self.state.etas.beta_mean:.3f}"
        )
    
    def fit(self) -> SPINHVIResults:
        elbo_trace: list[float] = []
        previous = -np.inf
        converged = False
        last_elbo_terms: dict[str, float] = {}
        for iteration in range(self.config.n_iter):
            if self.config.update_polya_gamma:
                self._update_polya_gamma()
            if self.config.update_z:
                self._update_branching()
            if self.config.update_latent_poisson:
                self._update_latent_poisson()
            if self.config.update_eps:
                self._update_eps()
            if self.config.update_gp:
                self._update_gp()
            if self.config.update_etas and iteration >= self.config.etas_update_start:
                self._update_etas()
            if self.config.learn_beta and self.config.fixed_beta is None:
                self._update_beta()

            elbo, last_elbo_terms = self._elbo()
            elbo_trace.append(elbo)
            if self.config.verbose and iteration % self.config.verbose_every == 0:
                self._print_progress(iteration, elbo)
            if np.isfinite(previous) and np.isfinite(elbo):
                scale = max(1.0, abs(previous))
                if abs(elbo - previous) / scale < self.config.tolerance:
                    converged = True
                    break
            previous = elbo
        diagnostics = {
            "converged": converged,
            "n_iter_run": len(elbo_trace),
            "expected_latent_poisson_count": self.state.latent_poisson.expected_count,
            "elbo_terms": last_elbo_terms,
        }
        return SPINHVIResults(
            self.state,
            self.model,
            self.catalog,
            self.config,
            elbo_trace,
            diagnostics,
        )



    # ===================================================================================================
    # ----------------------------------------------- ELBO -----------------------------------------------
    # ===================================================================================================
    
    def _background_observation_elbo(self):
        probabilities = self.state.branching.p_background
        eps_mean = self.state.eps.mean[self.domain_index]
        log_sigmoid = self._augmented_log_sigmoid_expectation(
            self.state.gp.f_data_mean,
            self.state.gp.f_data_var,
            self.state.polya_gamma.observed_mean,
            sign=1.0,
            omega_tilt=self.state.polya_gamma.observed_tilt,
        )
        return float(np.sum(probabilities * (eps_mean + log_sigmoid)))
    
    def _branching_entropy(self):
        probabilities = np.maximum(
            self.state.branching.probabilities,
            self.config.jitter,
        )
        return -float(np.sum(probabilities * np.log(probabilities)))

    def _latent_poisson_elbo(self):
        latent = self.state.latent_poisson
        if latent.grid_xy.size == 0:
            return 0.0
        intensity = np.maximum(latent.grid_intensity, self.config.jitter)
        cell_area = latent.cell_area
        domain_index = latent.grid_domain_index
        eps_mean = self.state.eps.mean[domain_index]
        if self.state.polya_gamma.grid_mean is None:
            omega_tilt = np.sqrt(
                np.maximum(
                    self.state.gp.f_grid_mean**2 + self.state.gp.f_grid_var,
                    0.0,
                )
            )
            omega_grid = self._pg_mean(omega_tilt)
        else:
            omega_grid = self.state.polya_gamma.grid_mean
            omega_tilt = self.state.polya_gamma.grid_tilt
        log_sigmoid_rejected = self._augmented_log_sigmoid_expectation(
            self.state.gp.f_grid_mean,
            self.state.gp.f_grid_var,
            omega_grid,
            sign=-1.0,
            omega_tilt=omega_tilt,
        )
        log_model_intensity = np.log(self.model.duration) + eps_mean + log_sigmoid_rejected
        return float(
            np.sum(intensity * cell_area * (log_model_intensity - np.log(intensity) + 1.0))
        )
    
    def _poisson_envelope_compensator_elbo(self):
        eps_mean = self.state.eps.mean
        eps_var = np.diag(self.state.eps.covariance)
        expected_baseline = np.exp(np.clip(eps_mean + 0.5 * eps_var, -60.0, 60.0))
        exposure = self.model.duration * np.asarray(self.model.domains.areas, dtype=float)
        return -float(np.sum(exposure * expected_baseline))
    
    def _epsilon_prior_elbo(self):
        mean = self.state.eps.mean
        covariance = self.state.eps.covariance
        n_dim = mean.size
        prior_covariance = self.model.epsilon_prior_covariance()
        prior_covariance = prior_covariance + self.config.jitter * np.eye(n_dim)
        sign, logdet = np.linalg.slogdet(prior_covariance)
        if sign <= 0:
            return -np.inf
        prior_precision = np.linalg.inv(prior_covariance)
        quadratic = float(mean @ prior_precision @ mean)
        trace = float(np.trace(prior_precision @ covariance))
        expected_log_prior = -0.5 * (
            quadratic + trace + logdet + n_dim * np.log(2.0 * np.pi)
        )
        entropy = self._gaussian_entropy_from_covariance(
            covariance,
            self.config.jitter,
        )
        return expected_log_prior + entropy

    def _gp_prior_elbo(self):
        if self.gp_backend == "sparse":
            mean = self.state.gp.coefficients_mean
            covariance = self.state.gp.coefficients_covariance
            if mean is None or covariance is None:
                return 0.0
            n_dim = mean.size
            expected_log_prior = -0.5 * float(
                mean @ mean
                + np.trace(covariance)
                + n_dim * np.log(2.0 * np.pi)
            )
            entropy = self._gaussian_entropy_from_covariance(
                covariance,
                self.config.jitter,
            )
            return expected_log_prior + entropy

        mean = np.concatenate([
            self.state.gp.f_data_mean,
            self.state.gp.f_grid_mean,
        ])
        variance = np.concatenate([
            self.state.gp.f_data_var,
            self.state.gp.f_grid_var,
        ])
        n_dim = mean.size
        if n_dim == 0:
            return 0.0
        xy = np.vstack([self.catalog.xy, self.state.latent_poisson.grid_xy])
        q_covariance = self.state.gp.covariance
        if q_covariance is None or q_covariance.shape != (n_dim, n_dim):
            entropy = self._independent_gaussian_entropy(variance, self.config.jitter)
            prior_variance = float(self.model.gp_prior.variance)
            expected_log_prior = -0.5 * float(
                np.sum((mean**2 + variance) / prior_variance)
                + n_dim * np.log(2.0 * np.pi * prior_variance)
            )
            return expected_log_prior + entropy
        covariance = self._rbf_kernel(xy, xy)
        covariance.flat[:: n_dim + 1] += self.config.jitter
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0:
            return -np.inf
        precision = np.linalg.inv(covariance)
        quadratic = float(mean @ precision @ mean)
        trace = float(np.trace(precision @ q_covariance))
        expected_log_prior = -0.5 * (
            quadratic + trace + logdet + n_dim * np.log(2.0 * np.pi)
        )
        entropy = self._gaussian_entropy_from_covariance(
            q_covariance,
            self.config.jitter,
        )
        return expected_log_prior + entropy

    def _elbo(self) -> tuple[float, dict[str, float]]:
        etas = self._etas_log_posterior(self.state.etas.parameters_mean)
        beta = self._beta_log_posterior()
        terms = {
            "background_observed_augmented": self._background_observation_elbo(),
            "branching_entropy": self._branching_entropy(),
            "latent_poisson_augmented": self._latent_poisson_elbo(),
            "poisson_envelope_compensator": self._poisson_envelope_compensator_elbo(),
            "epsilon_prior_entropy": self._epsilon_prior_elbo(),
            "gp_prior_entropy": self._gp_prior_elbo(),
            "etas_map": etas if np.isfinite(etas) else -1e50,
            "beta_map": beta if np.isfinite(beta) else -1e50,
        }
        return float(sum(terms.values())), terms



