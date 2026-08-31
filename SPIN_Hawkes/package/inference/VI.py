"""Mean-field CAVI inference for SSGC and SPIN-H models.

The implementation follows the Polya-Gamma augmented updates used by SPIN-H,
but keeps the first usable version intentionally modest:

- q(Z) is categorical for each event;
- q(Z) and the observed q(omega) factors are mean-field independent;
- q(omega) is represented through its Polya-Gamma expectation;
- q(pi_S) is represented by deterministic quadrature expected counts;
- q(epsilon) is updated by a Laplace/Newton step;
- q(f) is a Gaussian approximation at observed locations;
- positive ETAS parameters and beta use Gamma variational factors.

The ETAS block is fully Bayesian at the factor level: q(A), q(c), q(d),
q(alpha), q(gamma), q(p - 1), q(q - 1) and q(beta) are Gamma factors when the
corresponding parameter is learned. Parameters can still be fixed with
``fixed_etas`` and ``fixed_beta`` in the same spirit as the Gibbs sampler.
"""

import sys
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, issparse
from scipy.special import digamma, gammaln, logsumexp, roots_genlaguerre
from scipy.stats import gamma as gamma_distribution
from tqdm.auto import tqdm

from package.config import ETASParameters, SPINHVIConfig, SSGCVIConfig
from data.catalog import EventCatalog
from ..models.ssgc import SSGCModel
from ..models.spinh import SPINHModel
from .backends import SparseGP
from .branching import TemporalCandidateGraph
from .results import VIResults


_ETAS_PARAMETER_TO_FACTOR = {
    "p": "p_minus_1",
    "q": "q_minus_1",
}
_ETAS_FACTOR_TO_PARAMETER = {
    factor_name: parameter_name
    for parameter_name, factor_name in _ETAS_PARAMETER_TO_FACTOR.items()
}


@dataclass
class BranchingFactor:
    """Categorical variational factor over event parents."""

    probabilities: np.ndarray

    @classmethod
    def background_initialization(
        cls,
        n_events: int,
        include_parents: bool = True,
        sparse: bool = False,
    ):
        n_columns = max(1, n_events) if include_parents else 1
        if sparse:
            rows = np.arange(n_events, dtype=int)
            columns = np.zeros(n_events, dtype=int)
            data = np.ones(n_events, dtype=float)
            return cls(
                csr_matrix(
                    (data, (rows, columns)),
                    shape=(n_events, n_columns),
                )
            )
        probabilities = np.zeros((n_events, n_columns), dtype=float)
        probabilities[:, 0] = 1.0
        return cls(probabilities)

    @property
    def p_background(self) -> np.ndarray:
        if issparse(self.probabilities):
            return self.probabilities.getcol(0).toarray().reshape(-1)
        return self.probabilities[:, 0]

    @property
    def is_sparse(self) -> bool:
        return bool(issparse(self.probabilities))

    @property
    def expected_offspring(self) -> float:
        if self.probabilities.shape[1] <= 1:
            return 0.0
        return float(self.probabilities[:, 1:].sum())

    @property
    def memory_bytes(self) -> int:
        if self.is_sparse:
            matrix = self.probabilities
            return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)
        return int(self.probabilities.nbytes)

    def parent_entries(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return child indices, parent indices and non-zero parent weights."""
        if self.is_sparse:
            entries = self.probabilities.tocoo(copy=False)
            keep = entries.col > 0
            return (
                entries.row[keep].astype(int, copy=False),
                (entries.col[keep] - 1).astype(int, copy=False),
                entries.data[keep],
            )
        child, parent = np.nonzero(self.probabilities[:, 1:] > 0.0)
        valid = parent < child
        child = child[valid]
        parent = parent[valid]
        return child, parent, self.probabilities[child, parent + 1]

    def entropy(self) -> float:
        values = (
            self.probabilities.data
            if self.is_sparse
            else self.probabilities[self.probabilities > 0.0]
        )
        positive = values > 0.0
        return -float(np.sum(values[positive] * np.log(values[positive])))

    def most_likely_parent(self, child_index: int) -> int:
        child_index = int(child_index)
        if self.is_sparse:
            row = self.probabilities.getrow(child_index)
            parent_entries = row.indices > 0
            if not np.any(parent_entries):
                return -1
            columns = row.indices[parent_entries]
            values = row.data[parent_entries]
            return int(columns[np.argmax(values)] - 1)
        if child_index <= 0:
            return -1
        values = self.probabilities[child_index, 1 : child_index + 1]
        return -1 if values.size == 0 else int(np.argmax(values))

    def to_dense(self) -> np.ndarray:
        if self.is_sparse:
            return self.probabilities.toarray()
        return np.asarray(self.probabilities)


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
    quadrature_weights: np.ndarray
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
class GammaFactor:
    """Gamma(shape, rate) variational factor for one positive scalar."""

    shape: float
    rate: float

    def __post_init__(self):
        self.shape = float(self.shape)
        self.rate = float(self.rate)
        if (
            not np.isfinite(self.shape)
            or not np.isfinite(self.rate)
            or self.shape <= 0
            or self.rate <= 0
        ):
            raise ValueError("GammaFactor shape and rate must be finite and positive.")

    @property
    def mean(self) -> float:
        return self.shape / self.rate

    @property
    def expected_log(self) -> float:
        return float(digamma(self.shape) - np.log(self.rate))

    @property
    def entropy(self) -> float:
        return float(
            self.shape
            - np.log(self.rate)
            + gammaln(self.shape)
            + (1.0 - self.shape) * digamma(self.shape)
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "shape": float(self.shape),
            "rate": float(self.rate),
            "mean": float(self.mean),
            "expected_log": float(self.expected_log),
        }


@dataclass
class ETASFactor:
    """Gamma variational block for ETAS and magnitude parameters."""

    parameters_mean: ETASParameters | None
    beta_mean: float | None
    gamma_factors: dict[str, GammaFactor] = field(default_factory=dict)
    beta_gamma: GammaFactor | None = None
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
    """Shared generalized CAVI engine for SSGC and SPIN-H."""

    def __init__(self, model, catalog, config=None):
        if not isinstance(model, SSGCModel):
            raise TypeError("model must be an SSGCModel or SPINHModel instance.")
        if not isinstance(catalog, EventCatalog):
            raise TypeError("catalog must be an EventCatalog instance.")
        self.model = model
        self.catalog = catalog
        self.use_etas = isinstance(model, SPINHModel)
        config_type = SPINHVIConfig if self.use_etas else SSGCVIConfig
        self.config = config_type() if config is None else config
        config_is_valid = isinstance(self.config, config_type)
        if not self.use_etas and isinstance(self.config, SPINHVIConfig):
            config_is_valid = False
        if not config_is_valid:
            raise TypeError(f"config must be a {config_type.__name__} instance.")
        self.domain_index = model.validate_catalog(catalog)
        self.gp_backend = str(self.config.gp_backend).lower()
        self.sparse_gp = self._make_sparse_gp() if self.gp_backend == "sparse" else None
        self._spatial_compensator_geometry = None
        self.priors = (
            self._default_theta_priors() | dict(self.config.theta_priors)
            if self.use_etas
            else {}
        )
        self.beta_prior = {"a_beta": 2.0, "b_beta": 1.0} | dict(
            self.config.beta_prior
        )
        self._validate_fixed_parameters()
        parent_time_window = (
            self.config.parent_time_window if self.use_etas else None
        )
        self.parent_candidate_graph = (
            None
            if parent_time_window is None
            else TemporalCandidateGraph.from_times(
                self.catalog.t,
                parent_time_window,
            )
        )
        self.state = self.initialize_state()



    # ===================================================================================================
    # ============================================ OUTILLAGE ============================================
    # ===================================================================================================

    # **************************************** OUTILS BACKGROUND ****************************************
    def _validate_fixed_parameters(self):
        if not self.use_etas:
            return
        if not self.model.etas_parameters.marked:
            marked_fixed = {"alpha", "gamma"}.intersection(self.config.fixed_etas)
            if marked_fixed:
                raise ValueError(
                    f"Cannot fix marked parameters for an unmarked model: {sorted(marked_fixed)}"
                )
            marked_initial = {"alpha", "gamma"}.intersection(
                self.config.initial_gamma_factors
            )
            if marked_initial:
                raise ValueError(
                    "Cannot initialize marked Gamma factors for an unmarked model: "
                    f"{sorted(marked_initial)}"
                )
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

    def _make_quadrature_grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        grid_xy = grid_xy[inside]
        domain_index = domain_index[inside]
        points_per_domain = np.bincount(
            domain_index,
            minlength=self.model.n_domains,
        )
        missing_domains = np.flatnonzero(points_per_domain == 0)
        if missing_domains.size:
            representative_points = []
            for domain in missing_domains:
                point = self.model.domains.polygons[domain].representative_point()
                representative_points.append((float(point.x), float(point.y)))
            grid_xy = np.vstack([grid_xy, representative_points])
            domain_index = np.concatenate([domain_index, missing_domains])
            points_per_domain = np.bincount(
                domain_index,
                minlength=self.model.n_domains,
            )
        domain_areas = np.asarray(self.model.domains.areas, dtype=float)
        quadrature_weights = (
            domain_areas[domain_index] / points_per_domain[domain_index]
        )
        return grid_xy, domain_index, quadrature_weights

    def _magnitudes(self) -> np.ndarray:
        if self.catalog.magnitudes is None:
            return np.full(len(self.catalog), self.model.magnitude_min, dtype=float)
        return self.catalog.magnitudes
    
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
    def _observed_log_sigmoid_expectation(
        mean: np.ndarray,
        variance: np.ndarray,
        omega_mean: np.ndarray,
    ) -> np.ndarray:
        second_moment = mean**2 + variance
        return 0.5 * mean - 0.5 * omega_mean * second_moment - np.log(2.0)
    
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

    # ************************************************ OUTILS TRIGGERING ************************************************
    def _free_etas_factor_names(self, include_A: bool = True) -> list[str]:
        if not self.use_etas:
            return []
        names = ["A", "c", "p", "d", "q"]
        if self.model.etas_parameters.marked:
            names.extend(["alpha", "gamma"])
        names = [
            _ETAS_PARAMETER_TO_FACTOR.get(name, name)
            for name in names
            if name not in self.config.fixed_etas
        ]
        if not include_A:
            names = [name for name in names if name != "A"]
        return names

    def _alpha_rate_floor(self) -> float:
        """Smallest numerically safe rate for a finite productivity moment."""
        if not self.use_etas or not self.model.etas_parameters.marked:
            return 0.0
        excess = self._magnitudes() - self.model.magnitude_min
        max_excess = float(np.max(excess)) if excess.size else 0.0
        margin = max(self.config.jitter, 1e-8 * max(1.0, max_excess))
        return max_excess + margin

    def _fixed_or_factor_value(self, name: str, moment: str = "mean") -> float:
        parameter_name = _ETAS_FACTOR_TO_PARAMETER.get(name, name)
        if parameter_name in self.config.fixed_etas:
            value = float(self.config.fixed_etas[parameter_name])
            if moment == "log":
                return -np.inf if value == 0.0 else np.log(value)
            if name in {"p_minus_1", "q_minus_1"}:
                shifted = max(value - 1.0, self.config.jitter)
                return np.log(shifted) if moment == "log_shifted" else shifted
            return np.log(max(value - 1.0, self.config.jitter)) if moment == "log_shifted" else value
        factor_name = _ETAS_PARAMETER_TO_FACTOR.get(parameter_name, parameter_name)
        factor = self.state.etas.gamma_factors[factor_name]
        if moment == "log":
            return factor.expected_log
        if moment == "log_shifted":
            return factor.expected_log
        return factor.mean

    def _etas_mean_from_factors(
        self,
        factors: dict[str, GammaFactor] | None = None,
    ) -> ETASParameters:
        factors = self.state.etas.gamma_factors if factors is None else factors
        current = self.state.etas.parameters_mean.as_dict()
        values = {}
        for name in ["A", "c", "d"]:
            values[name] = (
                float(self.config.fixed_etas[name])
                if name in self.config.fixed_etas
                else factors[name].mean
            )
        for name, factor_name in [("p", "p_minus_1"), ("q", "q_minus_1")]:
            values[name] = (
                float(self.config.fixed_etas[name])
                if name in self.config.fixed_etas
                else 1.0 + factors[factor_name].mean
            )
        if self.model.etas_parameters.marked:
            for name in ["alpha", "gamma"]:
                values[name] = (
                    float(self.config.fixed_etas[name])
                    if name in self.config.fixed_etas
                    else factors[name].mean
                )
        else:
            current.pop("alpha", None)
            current.pop("gamma", None)
        current.update(values)
        return ETASParameters(**current)

    def _sync_etas_means(self):
        if self.use_etas:
            self.state.etas.parameters_mean = self._parameters_with_fixed(
                self._etas_mean_from_factors()
            )
        if self.state.etas.beta_gamma is not None:
            self.state.etas.beta_mean = self.state.etas.beta_gamma.mean

    @staticmethod
    def _gamma_prior_entropy_term(factor: GammaFactor, a: float, b: float) -> float:
        expected_log_prior = (
            a * np.log(b)
            - gammaln(a)
            + (a - 1.0) * factor.expected_log
            - b * factor.mean
        )
        return float(expected_log_prior + factor.entropy)

    def _gamma_factor_quadrature(
        self,
        factor: GammaFactor,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_nodes = self.config.gamma_quadrature_nodes
        if self.use_etas and self.config.etas_quadrature_nodes is not None:
            n_nodes = self.config.etas_quadrature_nodes
        if factor.shape < 160.0:
            nodes, weights = roots_genlaguerre(n_nodes, factor.shape - 1.0)
            weight_sum = float(np.sum(weights))
            if (
                np.all(np.isfinite(nodes))
                and np.all(np.isfinite(weights))
                and np.isfinite(weight_sum)
                and weight_sum > 0.0
            ):
                return nodes / factor.rate, weights / weight_sum

        # Generalized-Laguerre weights can overflow when shape is very large.
        # Integrating over Gamma quantiles is less exact but remains stable.
        quantile_nodes, quantile_weights = np.polynomial.legendre.leggauss(n_nodes)
        probabilities = 0.5 * (quantile_nodes + 1.0)
        values = gamma_distribution.ppf(
            probabilities,
            a=factor.shape,
            scale=1.0 / factor.rate,
        )
        return values, 0.5 * quantile_weights

    def _gamma_quadrature(self, factor_name: str) -> tuple[np.ndarray, np.ndarray]:
        parameter_name = _ETAS_FACTOR_TO_PARAMETER.get(factor_name, factor_name)
        if parameter_name in self.config.fixed_etas:
            value = float(self.config.fixed_etas[parameter_name])
            if factor_name in {"p_minus_1", "q_minus_1"}:
                value -= 1.0
            return np.asarray([value], dtype=float), np.asarray([1.0], dtype=float)
        factor = self.state.etas.gamma_factors[factor_name]
        return self._gamma_factor_quadrature(factor)

    def _expected_log_dt_plus_c(self, dt: np.ndarray) -> np.ndarray:
        if "c" in self.config.fixed_etas:
            return np.log(dt + float(self.config.fixed_etas["c"]))
        c_nodes, c_weights = self._gamma_quadrature("c")
        values = np.zeros_like(dt, dtype=float)
        for c_value, weight in zip(c_nodes, c_weights):
            values += weight * np.log(dt + c_value)
        return values

    def _expected_temporal_compensator(self) -> np.ndarray:
        remaining = np.maximum(self.model.duration - self.catalog.t, 0.0)
        c_nodes, c_weights = self._gamma_quadrature("c")
        p_nodes, p_weights = self._gamma_quadrature("p_minus_1")
        expectation = np.zeros_like(remaining, dtype=float)
        for c_value, c_weight in zip(c_nodes, c_weights):
            log_ratio = np.log(c_value) - np.log(remaining + c_value)
            for p_minus_1, p_weight in zip(p_nodes, p_weights):
                expectation += c_weight * p_weight * (
                    -np.expm1(p_minus_1 * log_ratio)
                )
        return expectation

    def _expected_exp_alpha_dm(self, dm: np.ndarray) -> np.ndarray:
        if not self.model.etas_parameters.marked:
            return np.ones_like(dm, dtype=float)
        if "alpha" in self.config.fixed_etas:
            return np.exp(float(self.config.fixed_etas["alpha"]) * dm)
        factor = self.state.etas.gamma_factors["alpha"]
        denominator = factor.rate - dm
        if np.any(denominator <= 0.0):
            return np.full_like(dm, np.inf, dtype=float)
        log_values = factor.shape * (
            np.log(factor.rate) - np.log(denominator)
        )
        return np.exp(log_values)

    def _expected_log_spatial_scale_plus_distance(
        self,
        r2: np.ndarray,
        dm: np.ndarray,
    ) -> np.ndarray:
        d_nodes, d_weights = self._gamma_quadrature("d")
        gamma_nodes = np.asarray([0.0], dtype=float)
        gamma_weights = np.asarray([1.0], dtype=float)
        if self.model.etas_parameters.marked:
            gamma_nodes, gamma_weights = self._gamma_quadrature("gamma")
        values = np.zeros_like(r2, dtype=float)
        log_r2 = np.full_like(r2, -np.inf, dtype=float)
        positive_distance = r2 > 0.0
        log_r2[positive_distance] = np.log(r2[positive_distance])
        for d_value, d_weight in zip(d_nodes, d_weights):
            for gamma_value, gamma_weight in zip(gamma_nodes, gamma_weights):
                log_scale = np.log(d_value) + gamma_value * dm
                values += (
                    d_weight
                    * gamma_weight
                    * np.logaddexp(log_scale, log_r2)
                )
        if np.any(~positive_distance):
            gamma_mean = (
                0.0
                if not self.model.etas_parameters.marked
                else self._fixed_or_factor_value("gamma")
            )
            values[~positive_distance] = (
                self._fixed_or_factor_value("d", "log")
                + gamma_mean * dm[~positive_distance]
            )
        return values

    def _pair_expected_log_etas(self, child_idx, parent_idx) -> np.ndarray:
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
        dm = self._magnitudes()[parent_idx[valid]] - self.model.magnitude_min
        alpha_mean = (
            0.0
            if not self.model.etas_parameters.marked
            else self._fixed_or_factor_value("alpha")
        )
        gamma_mean = (
            0.0
            if not self.model.etas_parameters.marked
            else self._fixed_or_factor_value("gamma")
        )
        p_minus_1_mean = self._fixed_or_factor_value("p_minus_1")
        q_minus_1_mean = self._fixed_or_factor_value("q_minus_1")
        expected_log_scale = (
            self._fixed_or_factor_value("d", "log")
            + gamma_mean * dm
        )
        expected_log_scale_plus_r2 = (
            self._expected_log_spatial_scale_plus_distance(r2, dm)
        )
        temporal = (
            self._fixed_or_factor_value("p_minus_1", "log_shifted")
            + p_minus_1_mean * self._fixed_or_factor_value("c", "log")
            - (1.0 + p_minus_1_mean) * self._expected_log_dt_plus_c(dt[valid])
        )
        spatial = (
            self._fixed_or_factor_value("q_minus_1", "log_shifted")
            - np.log(np.pi)
            + q_minus_1_mean * expected_log_scale
            - (1.0 + q_minus_1_mean) * expected_log_scale_plus_r2
        )
        out[valid] = (
            self._fixed_or_factor_value("A", "log")
            + alpha_mean * dm
            + temporal
            + spatial
        )
        return out
    
    def _expected_spatial_compensator(self) -> np.ndarray:
        if self.config.spatial_compensator_grid <= 0:
            return np.ones(len(self.catalog), dtype=float)
        n_grid = self.config.spatial_compensator_grid
        cached = self._spatial_compensator_geometry
        if cached is None or cached[0] != n_grid:
            xmin, xmax = self.model.x_bounds
            ymin, ymax = self.model.y_bounds
            dx = (xmax - xmin) / n_grid
            dy = (ymax - ymin) / n_grid
            grid_x, grid_y = np.meshgrid(
                xmin + (np.arange(n_grid) + 0.5) * dx,
                ymin + (np.arange(n_grid) + 0.5) * dy,
            )
            grid_xy = np.column_stack([grid_x.ravel(), grid_y.ravel()])
            inside = self.model.domains.locate(grid_xy[:, 0], grid_xy[:, 1]) >= 0
            grid_xy = grid_xy[inside]
            distance_squared = (
                (self.catalog.x[:, None] - grid_xy[None, :, 0]) ** 2
                + (self.catalog.y[:, None] - grid_xy[None, :, 1]) ** 2
            )
            cached = (n_grid, distance_squared, float(dx * dy))
            self._spatial_compensator_geometry = cached

        _, distance_squared, cell_area = cached
        dm = self._magnitudes() - self.model.magnitude_min
        d_nodes, d_weights = self._gamma_quadrature("d")
        if self.model.etas_parameters.marked:
            gamma_nodes, gamma_weights = self._gamma_quadrature("gamma")
        else:
            gamma_nodes = np.asarray([0.0])
            gamma_weights = np.asarray([1.0])

        q_is_fixed = "q" in self.config.fixed_etas
        if q_is_fixed:
            q_minus_1 = float(self.config.fixed_etas["q"]) - 1.0
        else:
            q_factor = self.state.etas.gamma_factors["q_minus_1"]

        expectation = np.zeros(len(self.catalog), dtype=float)
        for d_value, d_weight in zip(d_nodes, d_weights):
            for gamma_value, gamma_weight in zip(gamma_nodes, gamma_weights):
                scale = d_value * np.exp(gamma_value * dm)
                log_tail = np.log1p(distance_squared / scale[:, None])
                if q_is_fixed:
                    density = (
                        q_minus_1
                        / (np.pi * scale[:, None])
                        * np.exp(-(1.0 + q_minus_1) * log_tail)
                    )
                else:
                    log_laplace_moment = (
                        np.log(q_factor.shape)
                        + q_factor.shape * np.log(q_factor.rate)
                        - (q_factor.shape + 1.0)
                        * np.log(q_factor.rate + log_tail)
                    )
                    density = np.exp(
                        -np.log(np.pi * scale[:, None])
                        - log_tail
                        + log_laplace_moment
                    )
                expectation += (
                    d_weight
                    * gamma_weight
                    * cell_area
                    * density.sum(axis=1)
                )
        # Midpoint quadrature can overshoot the normalized spatial mass.
        return np.minimum(expectation, 1.0)

    def _triggering_compensator_without_A(self) -> float:
        magnitudes = self._magnitudes()
        dm = magnitudes - self.model.magnitude_min
        productivity = self._expected_exp_alpha_dm(dm)
        temporal = self._expected_temporal_compensator()
        spatial = self._expected_spatial_compensator()
        return float(np.sum(productivity * temporal * spatial))

    def _etas_expected_log_likelihood_terms(self) -> tuple[float, float]:
        child_idx, parent_idx, weights = self.state.branching.parent_entries()
        if weights.size:
            pair_ll = float(
                np.sum(weights * self._pair_expected_log_etas(child_idx, parent_idx))
            )
        else:
            pair_ll = 0.0
        A_mean = self._fixed_or_factor_value("A")
        compensator_without_A = self._triggering_compensator_without_A()
        return pair_ll, -float(A_mean * compensator_without_A)

    def _etas_prior_entropy_elbo(self) -> float:
        total = 0.0
        for name in self.state.etas.parameters_mean.as_dict():
            factor_name = _ETAS_PARAMETER_TO_FACTOR.get(name, name)
            a = self.priors.get(f"a_{name}", 1.0)
            b = self.priors.get(f"b_{name}", 1.0)
            if name in self.config.fixed_etas:
                continue
            total += self._gamma_prior_entropy_term(
                self.state.etas.gamma_factors[factor_name],
                a,
                b,
            )
        return float(total)

    def _etas_elbo(self) -> float:
        parent_term, compensator_term = self._etas_expected_log_likelihood_terms()
        return parent_term + compensator_term + self._etas_prior_entropy_elbo()
    

    # ********************************************* OUTILS MARK *********************************************
    def _expected_log_truncated_beta_normalizer(self, factor: GammaFactor, width: float) -> float:
        nodes, weights = self._gamma_factor_quadrature(factor)
        values = np.log(np.maximum(1.0 - np.exp(-nodes * width), self.config.jitter))
        return float(np.sum(weights * values))

    def _beta_elbo(self):
        if self.catalog.magnitudes is None:
            return 0.0
        magnitudes = self.catalog.magnitudes
        lower = self.model.magnitude_min
        upper = self.model.magnitude_max
        excess = magnitudes - lower
        beta_factor = self.state.etas.beta_gamma
        if beta_factor is None:
            beta = float(self.state.etas.beta_mean)
            value = len(magnitudes) * np.log(beta) - beta * np.sum(excess)
            if upper is not None:
                width = max(upper - lower, self.config.jitter)
                value -= len(magnitudes) * np.log(
                    max(1.0 - np.exp(-beta * width), self.config.jitter)
                )
            return float(value)
        a = self.beta_prior.get("a_beta", 2.0)
        b = self.beta_prior.get("b_beta", 1.0)
        value = len(magnitudes) * beta_factor.expected_log - beta_factor.mean * np.sum(excess)
        if upper is not None:
            width = max(upper - lower, self.config.jitter)
            value -= len(magnitudes) * self._expected_log_truncated_beta_normalizer(
                beta_factor,
                width,
            )
        value += self._gamma_prior_entropy_term(beta_factor, a, b)
        return float(value)



    # ===================================================================================================
    # --------------------------------------------- UPDATES ---------------------------------------------
    # ===================================================================================================

    def _update_branching(self):
        n_events = len(self.catalog)
        eps = self.state.eps.mean
        bg_log = eps[self.domain_index] + self._observed_log_sigmoid_expectation(
            self.state.gp.f_data_mean,
            self.state.gp.f_data_var,
            self.state.polya_gamma.observed_mean,
        )

        graph = self.parent_candidate_graph
        if graph is not None:
            pair_log_weights = self._pair_expected_log_etas(
                graph.child_indices,
                graph.parent_indices,
            )
            probability_indptr = graph.indptr + np.arange(n_events + 1)
            probability_indices = np.empty(n_events + graph.n_edges, dtype=int)
            probability_values = np.empty(n_events + graph.n_edges, dtype=float)
            for child in range(n_events):
                edge_slice = graph.row_slice(child)
                output_start = child + edge_slice.start
                output_stop = child + edge_slice.stop + 1
                weights = np.concatenate(
                    ([bg_log[child]], pair_log_weights[edge_slice])
                )
                row_probabilities = np.exp(weights - logsumexp(weights))
                probability_indices[output_start] = 0
                probability_indices[output_start + 1 : output_stop] = (
                    graph.parent_indices[edge_slice] + 1
                )
                probability_values[output_start:output_stop] = row_probabilities
            self.state.branching.probabilities = csr_matrix(
                (probability_values, probability_indices, probability_indptr),
                shape=(n_events, max(1, n_events)),
            )
            return

        probabilities = np.zeros((n_events, max(1, n_events)), dtype=float)
        for i in range(n_events):
            weights = [bg_log[i]]
            if i > 0:
                weights.extend(self._pair_expected_log_etas(i, np.arange(i)).tolist())
            probabilities[i, : i + 1] = np.exp(weights - logsumexp(weights))
        self.state.branching.probabilities = probabilities

    def _update_polya_gamma(self):
        second_moment = self.state.gp.f_data_mean**2 + self.state.gp.f_data_var
        c = np.sqrt(self.state.branching.p_background * second_moment)
        self.state.polya_gamma.observed_tilt = c
        self.state.polya_gamma.observed_mean = self._pg_mean(c)
        if self.state.latent_poisson.grid_xy.size:
            m_grid = self.state.gp.f_grid_mean
            v_grid = self.state.gp.f_grid_var
            c_grid = np.sqrt(m_grid**2 + v_grid)
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
        intensity = np.exp(log_intensity)
        counts = np.bincount(
            self.state.latent_poisson.grid_domain_index,
            weights=intensity * self.state.latent_poisson.quadrature_weights,
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
        eps = self.state.eps.mean.copy()
        for _ in range(self.config.eps_newton_steps):
            mu = np.exp(eps)
            gradient = counts - exposure * mu - prior_precision @ eps
            precision = prior_precision + np.diag(exposure * mu + self.config.jitter)
            step = np.linalg.solve(precision, gradient)
            eps = eps + step
            if np.linalg.norm(step) < 1e-6:
                break
        mu = np.exp(eps)
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
        grid_counts = (
            self.state.latent_poisson.grid_intensity
            * self.state.latent_poisson.quadrature_weights
        )
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
            self.state.gp.f_data_mean = mean[:n_events]
            self.state.gp.f_data_var = variance[:n_events]
            self.state.gp.f_grid_mean = mean[n_events:]
            self.state.gp.f_grid_var = variance[n_events:]
            self.state.gp.covariance = None
            self.state.gp.coefficients_mean = coefficients_mean
            self.state.gp.coefficients_covariance = covariance
            return

        K = self._rbf_kernel(xy, xy)
        K.flat[:: total_points + 1] += self.config.jitter
        K_inv = np.linalg.inv(K)
        precision = K_inv + np.diag(likelihood_precision)
        covariance = np.linalg.inv(precision)
        mean = covariance @ natural
        variance = np.maximum(np.diag(covariance), self.config.jitter)
        self.state.gp.f_data_mean = mean[:n_events]
        self.state.gp.f_data_var = variance[:n_events]
        self.state.gp.f_grid_mean = mean[n_events:]
        self.state.gp.f_grid_var = variance[n_events:]
        self.state.gp.covariance = covariance
        self.state.gp.coefficients_mean = None
        self.state.gp.coefficients_covariance = None

    def _update_etas(self, optimize_nonconjugate: bool = True):
        if "A" not in self.config.fixed_etas and "A" in self.state.etas.gamma_factors:
            expected_offspring = self.state.branching.expected_offspring
            a = self.priors.get("a_A", 1.0)
            b = self.priors.get("b_A", 1.0)
            proposed = GammaFactor(
                shape=max(a + expected_offspring, self.config.jitter),
                rate=max(b + self._triggering_compensator_without_A(), self.config.jitter),
            )
            self.state.etas.gamma_factors["A"] = proposed
            self._sync_etas_means()

        free_names = self._free_etas_factor_names(include_A=False)
        if not free_names or not optimize_nonconjugate:
            self._sync_etas_means()
            return

        start = []
        for name in free_names:
            factor = self.state.etas.gamma_factors[name]
            start.extend([np.log(factor.shape), np.log(factor.rate)])
        start = np.asarray(start, dtype=float)

        def factors_from_vector(vector):
            proposed = dict(self.state.etas.gamma_factors)
            for index, name in enumerate(free_names):
                shape = float(np.exp(vector[2 * index]))
                rate = float(np.exp(vector[2 * index + 1]))
                proposed[name] = GammaFactor(shape=max(shape, self.config.jitter), rate=max(rate, self.config.jitter))
            return proposed

        current_factors = self.state.etas.gamma_factors
        current_params = self.state.etas.parameters_mean

        def negative_elbo_block(vector):
            self.state.etas.gamma_factors = factors_from_vector(vector)
            self._sync_etas_means()
            value = self._etas_elbo()
            if not np.isfinite(value):
                return 1e100
            return -value

        initial_objective = negative_elbo_block(start)
        bounds = []
        for name in free_names:
            rate_floor = (
                max(1e-3, self._alpha_rate_floor())
                if name == "alpha"
                else 1e-3
            )
            rate_upper = max(1e5, 100.0 * rate_floor)
            bounds.extend(
                [
                    (np.log(1e-3), np.log(1e3)),
                    (np.log(rate_floor), np.log(rate_upper)),
                ]
            )
        result = minimize(
            negative_elbo_block,
            start,
            method="COBYQA",
            bounds=bounds,
            options={
                "maxiter": self.config.max_optimizer_iter,
                "maxfev": max(40, 3 * self.config.max_optimizer_iter),
                "initial_tr_radius": 1.0,
                "final_tr_radius": 1e-4,
                "scale": False,
            },
        )
        self.state.etas.gamma_factors = current_factors
        self.state.etas.parameters_mean = current_params
        objective_tolerance = 1e-10 * max(1.0, abs(initial_objective))
        if (
            not np.isfinite(result.fun)
            or result.fun > initial_objective + objective_tolerance
        ):
            self._sync_etas_means()
            return
        self.state.etas.gamma_factors = factors_from_vector(result.x)
        self._sync_etas_means()
        applied_objective = -self._etas_elbo()
        if (
            not np.isfinite(applied_objective)
            or applied_objective > initial_objective + objective_tolerance
        ):
            self.state.etas.gamma_factors = current_factors
            self.state.etas.parameters_mean = current_params
            self._sync_etas_means()

    def _update_beta(self):
        beta_factor = self.state.etas.beta_gamma
        if self.catalog.magnitudes is None or beta_factor is None:
            return
        start = np.asarray([np.log(beta_factor.shape), np.log(beta_factor.rate)], dtype=float)
        current = self.state.etas.beta_gamma
        current_mean = self.state.etas.beta_mean

        def negative_elbo_block(vector):
            shape = float(np.exp(vector[0]))
            rate = float(np.exp(vector[1]))
            self.state.etas.beta_gamma = GammaFactor(shape=max(shape, self.config.jitter), rate=max(rate, self.config.jitter))
            self._sync_etas_means()
            value = self._beta_elbo()
            if not np.isfinite(value):
                return 1e100
            return -value

        initial_objective = negative_elbo_block(start)
        beta_maxiter = max(50, self.config.max_optimizer_iter)
        result = minimize(
            negative_elbo_block,
            start,
            method="COBYQA",
            bounds=[(np.log(1e-3), np.log(1e3)), (np.log(1e-3), np.log(1e5))],
            options={
                "maxiter": beta_maxiter,
                "maxfev": max(100, 2 * beta_maxiter),
                "initial_tr_radius": 1.0,
                "final_tr_radius": 1e-4,
                "scale": False,
            },
        )
        self.state.etas.beta_gamma = current
        self.state.etas.beta_mean = current_mean
        objective_tolerance = 1e-10 * max(1.0, abs(initial_objective))
        if (
            not np.isfinite(result.fun)
            or result.fun > initial_objective + objective_tolerance
        ):
            self._sync_etas_means()
            return
        self.state.etas.beta_gamma = GammaFactor(
            shape=float(np.exp(result.x[0])),
            rate=float(np.exp(result.x[1])),
        )
        self._sync_etas_means()
    


    # ===================================================================================================
    # ----------------------------------------------- ELBO -----------------------------------------------
    # ===================================================================================================
    
    def _background_observation_elbo(self):
        probabilities = self.state.branching.p_background
        eps_mean = self.state.eps.mean[self.domain_index]
        log_sigmoid = self._observed_log_sigmoid_expectation(
            self.state.gp.f_data_mean,
            self.state.gp.f_data_var,
            self.state.polya_gamma.observed_mean,
        )
        pg_correction = self._pg_tilting_entropy_correction(
            self.state.polya_gamma.observed_tilt,
            self.state.polya_gamma.observed_mean,
        )
        return float(
            np.sum(probabilities * (eps_mean + log_sigmoid))
            + np.sum(pg_correction)
        )
    
    def _branching_entropy(self):
        return self.state.branching.entropy()

    def _latent_poisson_elbo(self):
        latent = self.state.latent_poisson
        if latent.grid_xy.size == 0:
            return 0.0
        intensity = np.maximum(latent.grid_intensity, self.config.jitter)
        quadrature_weights = latent.quadrature_weights
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
            np.sum(
                intensity
                * quadrature_weights
                * (log_model_intensity - np.log(intensity) + 1.0)
            )
        )
    
    def _poisson_envelope_compensator_elbo(self):
        eps_mean = self.state.eps.mean
        eps_var = np.diag(self.state.eps.covariance)
        expected_baseline = np.exp(eps_mean + 0.5 * eps_var)
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
            variance = np.maximum(variance, self.config.jitter)
            entropy = 0.5 * float(
                np.sum(1.0 + np.log(2.0 * np.pi * variance))
            )
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
        terms = {
            "background_observed_augmented": self._background_observation_elbo(),
            "latent_poisson_augmented": self._latent_poisson_elbo(),
            "poisson_envelope_compensator": self._poisson_envelope_compensator_elbo(),
            "epsilon_prior_entropy": self._epsilon_prior_elbo(),
            "gp_prior_entropy": self._gp_prior_elbo(),
            "beta_expected_likelihood_prior_entropy": self._beta_elbo(),
        }
        if self.use_etas:
            etas_parent_term, etas_compensator_term = (
                self._etas_expected_log_likelihood_terms()
            )
            terms.update(
                {
                    "branching_entropy": self._branching_entropy(),
                    "etas_parent_log_likelihood": etas_parent_term,
                    "etas_triggering_compensator": etas_compensator_term,
                    "etas_prior_entropy": self._etas_prior_entropy_elbo(),
                }
            )
        non_finite = {
            name: value for name, value in terms.items() if not np.isfinite(value)
        }
        if non_finite:
            details = ", ".join(
                f"{name}={value!r}" for name, value in non_finite.items()
            )
            raise FloatingPointError(f"Non-finite ELBO term(s): {details}")

        total = float(sum(terms.values()))
        if not np.isfinite(total):
            raise FloatingPointError(f"Non-finite ELBO total: {total!r}")
        if self.use_etas:
            terms["etas_expected_complete_likelihood_prior_entropy"] = (
                terms["etas_parent_log_likelihood"]
                + terms["etas_triggering_compensator"]
                + terms["etas_prior_entropy"]
            )
        return total, terms



    # ===================================================================================================
    # ----------------------------------------------- FIT -----------------------------------------------
    # ===================================================================================================

    def _initial_gamma_factor(
        self,
        factor_name: str,
    ) -> GammaFactor:
        configured = self.config.initial_gamma_factors.get(factor_name)
        if configured is not None:
            shape, rate = configured
            if factor_name == "alpha" and rate < self._alpha_rate_floor():
                raise ValueError(
                    "The initial q(alpha) rate must be greater than every "
                    "observed magnitude excess m_i - magnitude_min so that "
                    "E[exp(alpha * (m_i - magnitude_min))] is finite."
                )
            return GammaFactor(shape=shape, rate=rate)
        if factor_name == "beta":
            return GammaFactor(
                shape=self.beta_prior["a_beta"],
                rate=self.beta_prior["b_beta"],
            )
        parameter_name = _ETAS_FACTOR_TO_PARAMETER.get(factor_name, factor_name)
        rate = self.priors[f"b_{parameter_name}"]
        if factor_name == "alpha":
            rate_floor = self._alpha_rate_floor()
            rate = max(
                rate,
                rate_floor + max(1.0, 0.25 * rate_floor),
            )
        return GammaFactor(
            shape=self.priors[f"a_{parameter_name}"],
            rate=rate,
        )

    def initialize_state(self) -> SPINHVIState:
        n_events = len(self.catalog)
        n_domains = self.model.n_domains
        counts = np.bincount(self.domain_index, minlength=n_domains).astype(float)
        exposure = self.model.duration * np.asarray(self.model.domains.areas, dtype=float)
        eps_mean = np.log(2.0 * (counts + 0.5) / np.maximum(exposure, self.config.jitter))
        eps_cov = self.model.epsilon_prior_covariance()
        f_mean = np.zeros(n_events, dtype=float)
        f_var = np.full(n_events, self.model.gp_prior.variance, dtype=float)
        grid_xy, grid_domains, quadrature_weights = self._make_quadrature_grid()
        latent = LatentPoissonFactor(
            grid_xy=grid_xy,
            grid_domain_index=grid_domains,
            grid_intensity=np.zeros(grid_xy.shape[0], dtype=float),
            quadrature_weights=quadrature_weights,
            expected_counts_by_domain=np.zeros(n_domains, dtype=float),
        )
        f_grid_mean = np.zeros(grid_xy.shape[0], dtype=float)
        f_grid_var = np.full(grid_xy.shape[0], self.model.gp_prior.variance, dtype=float)
        sparse_dimension = None
        if self.sparse_gp is not None:
            probe_xy = grid_xy[:1] if grid_xy.size else self.catalog.xy[:1]
            sparse_dimension = self._sparse_design(probe_xy).shape[1]
        gamma_factors: dict[str, GammaFactor] = {}
        params = None
        if self.use_etas:
            params = self._parameters_with_fixed(self.model.etas_parameters)
            for factor_name in self._free_etas_factor_names():
                gamma_factors[factor_name] = self._initial_gamma_factor(factor_name)
            initial_values = params.as_dict()
            for factor_name, factor in gamma_factors.items():
                parameter_name = _ETAS_FACTOR_TO_PARAMETER.get(factor_name, factor_name)
                initial_values[parameter_name] = factor.mean + (
                    1.0 if parameter_name in {"p", "q"} else 0.0
                )
            params = self._parameters_with_fixed(ETASParameters(**initial_values))
        beta_gamma = None
        if self.catalog.magnitudes is None:
            beta = None
        elif self.config.fixed_beta is None:
            beta_gamma = self._initial_gamma_factor("beta")
            beta = beta_gamma.mean
        else:
            beta = float(self.config.fixed_beta)
        return SPINHVIState(
            branching=BranchingFactor.background_initialization(
                n_events,
                include_parents=self.use_etas,
                sparse=self.parent_candidate_graph is not None,
            ),
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
                    np.zeros(sparse_dimension, dtype=float)
                    if sparse_dimension is not None else None
                ),
                coefficients_covariance=(
                    np.eye(sparse_dimension, dtype=float)
                    if sparse_dimension is not None else None
                ),
            ),
            eps=EpsilonFactor(mean=eps_mean, covariance=eps_cov),
            etas=ETASFactor(
                parameters_mean=params,
                beta_mean=beta,
                gamma_factors=gamma_factors,
                beta_gamma=beta_gamma,
                fixed_etas=(dict(self.config.fixed_etas) if self.use_etas else {}),
            ),
        )

    def _progress_message(self, iteration: int, elbo: float) -> str:
        if not self.use_etas:
            beta_text = (
                ""
                if self.state.etas.beta_mean is None
                else f" beta={self.state.etas.beta_mean:.3f}"
            )
            return (
                f"[VI {iteration:04d}] elbo={elbo:.3f} "
                f"E[pi_S]={self.state.latent_poisson.expected_count:.2f}"
                f"{beta_text}"
            )
        params = self.state.etas.parameters_mean
        beta_text = (
            ""
            if self.state.etas.beta_mean is None
            else f" beta={self.state.etas.beta_mean:.3f}"
        )
        p_background = self.state.branching.p_background
        p_background_text = (
            f"{p_background.mean():.3f}" if p_background.size else "n/a"
        )
        marked_text = ""
        if params.alpha is not None and params.gamma is not None:
            marked_text = (
                f" alpha={params.alpha:.3f} gamma={params.gamma:.3f}"
            )
        return (
            f"[VI {iteration:04d}] elbo={elbo:.3f} "
            f"p_bg={p_background_text} "
            f"E[pi_S]={self.state.latent_poisson.expected_count:.2f} "
            f"A={params.A:.3f} c={params.c:.4f} p={params.p:.3f} "
            f"d={params.d:.4f} q={params.q:.3f}"
            f"{marked_text}"
            f"{beta_text}"
        )

    def fit(self) -> VIResults:
        elbo_trace: list[float] = []
        elbo_iterations: list[int] = []
        convergence_iterations: list[int] = []
        previous_checkpoint = -np.inf
        converged = False
        iteration = -1
        has_free_etas = self.use_etas and bool(
            self._free_etas_factor_names(include_A=True)
        )
        has_nonconjugate_etas = self.use_etas and bool(
            self._free_etas_factor_names(include_A=False)
        )
        progress = tqdm(
            range(self.config.n_iter),
            desc="SPIN-H MF-VI" if self.use_etas else "SSGC MF-VI",
            unit="iter",
            disable=not self.config.verbose,
            dynamic_ncols=True,
            miniters=self.config.verbose_every,
            file=sys.stdout,
        )
        for iteration in progress:
            updated_etas = False
            optimized_nonconjugate_etas = False
            if self.config.update_polya_gamma:
                self._update_polya_gamma()
            if self.use_etas and self.config.update_z:
                self._update_branching()
            if self.config.update_latent_poisson:
                self._update_latent_poisson()
            if self.config.update_eps:
                self._update_eps()
            if self.config.update_gp:
                self._update_gp()
            if (
                self.use_etas
                and self.config.update_etas
                and iteration >= self.config.etas_update_start
            ):
                updated_etas = True
                optimized_nonconjugate_etas = (
                    has_nonconjugate_etas
                    and (iteration - self.config.etas_update_start)
                    % self.config.etas_update_every
                    == 0
                )
                self._update_etas(
                    optimize_nonconjugate=optimized_nonconjugate_etas
                )
            # The magnitude likelihood is independent of all other VI blocks.
            if (
                iteration == 0
                and self.config.fixed_beta is None
            ):
                self._update_beta()

            evaluate_elbo = (
                iteration % self.config.elbo_every == 0
                or optimized_nonconjugate_etas
                or iteration == self.config.n_iter - 1
            )
            if not evaluate_elbo:
                continue
            elbo, _ = self._elbo()
            elbo_trace.append(elbo)
            elbo_iterations.append(iteration)
            if self.config.verbose and iteration % self.config.verbose_every == 0:
                progress.set_postfix_str(
                    self._progress_message(iteration, elbo),
                    refresh=False,
                )
            convergence_checkpoint = (
                not self.use_etas
                or not self.config.update_etas
                or not has_free_etas
                or (
                    updated_etas
                    and (
                        not has_nonconjugate_etas
                        or optimized_nonconjugate_etas
                    )
                )
            )
            if not convergence_checkpoint:
                continue
            convergence_iterations.append(iteration)
            if np.isfinite(previous_checkpoint) and np.isfinite(elbo):
                scale = max(1.0, abs(previous_checkpoint))
                if (
                    abs(elbo - previous_checkpoint) / scale
                    < self.config.tolerance
                ):
                    converged = True
                    break
            previous_checkpoint = elbo
        progress.close()
        diagnostics = {
            "converged": converged,
            "n_iter_run": iteration + 1,
            "n_elbo_evaluations": len(elbo_trace),
            "elbo_iterations": np.asarray(elbo_iterations, dtype=int),
            "convergence_iterations": np.asarray(
                convergence_iterations,
                dtype=int,
            ),
            "expected_latent_poisson_count": self.state.latent_poisson.expected_count,
            "use_etas": self.use_etas,
            "model_type": "spinh" if self.use_etas else "ssgc",
            "use_calibration": self.config.use_calibration,
            "gp_prior_variance": self.model.gp_prior.variance,
            "gp_prior_length_scale": self.model.gp_prior.length_scale,
        }
        if self.parent_candidate_graph is not None:
            diagnostics["branching_truncation"] = {
                **self.parent_candidate_graph.diagnostics(),
                "branching_probability_memory_bytes": (
                    self.state.branching.memory_bytes
                ),
            }
        return VIResults(
            self.state,
            self.model,
            self.catalog,
            self.config,
            elbo_trace,
            diagnostics,
        )
