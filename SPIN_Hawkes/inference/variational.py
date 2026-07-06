"""Skeleton for hybrid CAVI
"""

from dataclasses import dataclass, field
import numpy as np

from ..config import ETASParameters
from ..data.catalog import EventCatalog
from ..models.spinh import SPINHModel
from .backends import SparseGP


@dataclass(frozen=True)
class HybridCAVIConfig:
    """
    """
    n_iter: int = 1000
    tolerance: float = 1e-4
    verbose: bool = True
    verbose_every: int = 25
    gp_backend: str = "sparse"
    max_inner_iter: int = 50
    use_latent_poisson: bool = True
    update_z: bool = True
    update_polya_gamma: bool = True
    update_poisson: bool = True
    update_gp: bool = True
    update_eps: bool = True
    update_etas: bool = True
    update_beta: bool = True
    elbo_mc_samples: int = 0
    random_seed: int | None = None

    def __post_init__(self):
        if self.n_iter <= 0:
            raise ValueError("n_iter must be positive.")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive.")
        if self.verbose_every <= 0:
            raise ValueError("verbose_every must be positive.")
        if self.gp_backend not in {"exact", "sparse"}:
            raise ValueError("gp_backend must be 'exact' or 'sparse'.")
        if self.max_inner_iter <= 0:
            raise ValueError("max_inner_iter must be positive.")
        if self.elbo_mc_samples < 0:
            raise ValueError("elbo_mc_samples must be non-negative.")


@dataclass
class BranchingFactor:
    """
    """
    probabilities: np.ndarray

    @classmethod
    def background_initialization(cls, n_events: int):
        probabilities = np.zeros((n_events, n_events + 1), dtype=float)
        probabilities[:, 0] = 1.0
        return cls(probabilities)


@dataclass
class PolyaGammaFactor:
    """
    """
    observed_mean: np.ndarray
    latent_mean: np.ndarray | None = None


@dataclass
class LatentPoissonFactor:
    """
    """
    expected_count: float = 0.0
    reference_intensity: float = 1.0
    locations: np.ndarray | None = None
    omega_mean: np.ndarray | None = None


@dataclass
class GPFactor:
    """
    """
    f_data_mean: np.ndarray
    f_data_cov: np.ndarray | None = None
    coeff_mean: np.ndarray | None = None
    coeff_cov: np.ndarray | None = None


@dataclass
class EpsilonFactor:
    """
    """
    mean: np.ndarray
    covariance: np.ndarray


@dataclass
class ETASFactor:
    """
    """
    parameters_mean: ETASParameters
    beta_shape: float | None = None
    beta_rate: float | None = None
    block_covariances: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class HybridCAVIState:
    """
    """
    branching: BranchingFactor
    polya_gamma: PolyaGammaFactor
    latent_poisson: LatentPoissonFactor
    gp: GPFactor
    eps: EpsilonFactor
    etas: ETASFactor


@dataclass
class HybridCAVIResults:
    """
    """
    state: HybridCAVIState
    model: SPINHModel
    catalog: EventCatalog
    config: HybridCAVIConfig
    elbo_trace: list[float]
    diagnostics: dict = field(default_factory=dict)

    def summary(self):
        """
        """
        return {
            "eps_mean": self.state.eps.mean,
            "f_data_mean": self.state.gp.f_data_mean,
            "etas_mean": self.state.etas.parameters_mean.as_dict(),
            "p_background": self.state.branching.probabilities[:, 0],
            "elbo_trace": np.asarray(self.elbo_trace, dtype=float),
        }


class SPINHHybridCAVI:
    """
    """

    def __init__(self, model, catalog, config=None, sparse_gp=None):
        if not isinstance(model, SPINHModel):
            raise TypeError("model must be a SPINHModel instance.")
        if not isinstance(catalog, EventCatalog):
            raise TypeError("catalog must be an EventCatalog instance.")
        self.model = model
        self.catalog = catalog
        self.config = HybridCAVIConfig() if config is None else config
        if not isinstance(self.config, HybridCAVIConfig):
            raise TypeError("config must be a HybridCAVIConfig instance.")
        model.validate_catalog(catalog)
        self.rng = np.random.default_rng(self.config.random_seed)
        self.sparse_gp = sparse_gp
        if self.config.gp_backend == "sparse" and sparse_gp is None:
            self.sparse_gp = SparseGP.from_bounds(
                model.x_bounds,
                model.y_bounds,
                model.gp_prior.variance,
                model.gp_prior.length_scale,
            )

    def initialize_state(self):
        """
        """
        n_events = len(self.catalog)
        n_domains = self.model.n_domains
        eps_mean = np.asarray(self.model.domains.initial_log_intensities, dtype=float)
        eps_cov = self.model.epsilon_prior_covariance()
        f_mean = np.zeros(n_events, dtype=float)
        f_cov = np.eye(n_events)
        beta_shape = None
        
        return HybridCAVIState()

    def fit(self):
        """
        """
        state = self.initialize_state()
        elbo_trace = []
        for iteration in range(self.config.n_iter):
            state = self.update_branching(state)
            state = self.update_polya_gamma(state)
            state = self.update_latent_poisson(state)
            state = self.update_gp(state)
            state = self.update_eps(state)
            state = self.update_etas_blocks(state)
            state = self.update_beta(state)
            elbo = self.compute_elbo(state)
            elbo_trace.append(elbo)
            if self.config.verbose and iteration % self.config.verbose_every == 0:
                self._print_progress(iteration, elbo)
            if self._has_converged(elbo_trace):
                break
        diagnostics = {
            "implemented_updates": self.implemented_updates(),
            "n_iterations": len(elbo_trace),
        }
        return HybridCAVIResults(state, self.model, self.catalog, self.config, elbo_trace, diagnostics)

    def implemented_updates(self):
        """Report which blocks are still placeholders."""
        return {
            "branching": False,
            "polya_gamma": False,
            "latent_poisson": False,
            "gp": False,
            "eps": False,
            "etas_blocks": False,
            "beta": False,
            "elbo": False,
        }

    def update_branching(self, state):
        """
        """
        if not self.config.update_z:
            return state
        return state

    def update_polya_gamma(self, state):
        """
        """
        if not self.config.update_polya_gamma:
            return state
        return state

    def update_latent_poisson(self, state):
        """
        """
        if not self.config.use_latent_poisson or not self.config.update_poisson:
            return state
        return state

    def update_gp(self, state):
        """
        """
        if not self.config.update_gp:
            return state
        return state

    def update_eps(self, state):
        """
        """
        if not self.config.update_eps:
            return state
        return state

    def update_etas_blocks(self, state):
        """
        """
        if not self.config.update_etas:
            return state
        return state

    def update_beta(self, state):
        """
        """
        if not self.config.update_beta or not self.model.etas_parameters.marked:
            return state
        return state

    def compute_elbo(self, state):
        """
        """
        return np.nan

    def _has_converged(self, elbo_trace):
        if len(elbo_trace) < 2:
            return False
        current = elbo_trace[-1]
        previous = elbo_trace[-2]
        if not np.isfinite(current) or not np.isfinite(previous):
            return False
        return abs(current - previous) <= self.config.tolerance * (1.0 + abs(previous))

