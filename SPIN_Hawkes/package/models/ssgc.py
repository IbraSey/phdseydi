"""Spatially structured sigmoidal Gaussian Cox-process model."""

from dataclasses import dataclass, field

import numpy as np
from scipy.special import expit

from ..config import GPParameters, GibbsConfig
from data.catalog import EventCatalog
from spatial.domain import DomainPartition
from .base import PointProcessModel


@dataclass
class SSGCModel(PointProcessModel):
    """Spatially structured sigmoidal Gaussian Cox-process model."""

    domains: DomainPartition
    duration: float
    x_bounds: tuple[float, float]
    y_bounds: tuple[float, float]
    gp_prior: GPParameters = field(default_factory=GPParameters)
    eps_prior_variance: float = 1.0
    eps_prior_length_scale: float = 0.5
    nu_prior_rate: float = 0.5
    jitter: float = 1e-5
    magnitude_min: float = 0.0
    magnitude_max: float | None = None

    def __post_init__(self):
        if self.duration <= 0:
            raise ValueError("duration must be positive.")
        if self.x_bounds[0] >= self.x_bounds[1]:
            raise ValueError("x_bounds must be increasing.")
        if self.y_bounds[0] >= self.y_bounds[1]:
            raise ValueError("y_bounds must be increasing.")
        if self.eps_prior_variance <= 0 or self.eps_prior_length_scale <= 0:
            raise ValueError("Epsilon-prior parameters must be positive.")
        self.magnitude_min = float(self.magnitude_min)
        if not np.isfinite(self.magnitude_min):
            raise ValueError("magnitude_min must be finite.")
        if self.magnitude_max is not None:
            self.magnitude_max = float(self.magnitude_max)
            if not np.isfinite(self.magnitude_max):
                raise ValueError("magnitude_max must be finite.")
            if self.magnitude_max < self.magnitude_min:
                raise ValueError("magnitude_max must be >= magnitude_min.")

    @classmethod
    def from_polygons(
        cls,
        polygons,
        duration: float,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        initial_log_intensities=0.0,
        **kwargs,
    ) -> "SSGCModel":
        return cls(
            domains=DomainPartition.from_polygons(
                polygons, initial_log_intensities
            ),
            duration=duration,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            **kwargs,
        )

    @property
    def n_domains(self) -> int:
        return len(self.domains)

    def validate_catalog(self, catalog: EventCatalog) -> np.ndarray:
        if catalog.t[-1] > self.duration + 1e-12:
            raise ValueError("The catalog contains events after the observation duration.")
        if catalog.magnitudes is not None:
            if np.any(catalog.magnitudes < self.magnitude_min):
                raise ValueError("Catalog magnitudes must be >= magnitude_min.")
            if self.magnitude_max is not None and np.any(
                catalog.magnitudes > self.magnitude_max
            ):
                raise ValueError("Catalog magnitudes exceed magnitude_max.")
        return self.domains.validate_points(catalog.x, catalog.y)

    def baseline_intensity(self, x, y, eps) -> np.ndarray:
        eps = np.asarray(eps, dtype=float).reshape(-1)
        if eps.size != self.n_domains:
            raise ValueError("eps must contain one value per domain.")
        domain_index = self.domains.locate(x, y)
        values = np.zeros(domain_index.size, dtype=float)
        inside = domain_index >= 0
        values[inside] = np.exp(eps[domain_index[inside]])
        return values

    def background_intensity(self, x, y, eps, latent_gp) -> np.ndarray:
        latent_gp = np.asarray(latent_gp, dtype=float).reshape(-1)
        baseline = self.baseline_intensity(x, y, eps)
        if latent_gp.size != baseline.size:
            raise ValueError("latent_gp must have one value per evaluation point.")
        return baseline * expit(latent_gp)

    def epsilon_prior_covariance(self) -> np.ndarray:
        centroids = self.domains.centroids
        differences = centroids[:, None, :] - centroids[None, :, :]
        squared_distance = np.sum(differences**2, axis=2)
        return self.eps_prior_variance * np.exp(
            -squared_distance / (2.0 * self.eps_prior_length_scale**2)
        )

    def calibrate_gp_prior(
        self,
        catalog: EventCatalog,
        rng_seed: int | None = None,
        verbose: bool = True,
    ) -> GPParameters:
        """Calibrate GP hyperparameters with the Gibbs calibration routine."""
        self.validate_catalog(catalog)
        from ..inference.ssgc_gibbs import SSGC_GibbsSampler

        sampler = SSGC_GibbsSampler(model=self, rng_seed=rng_seed)
        variance, length_scale, _ = sampler.calibrate_nu(
            catalog.x,
            catalog.y,
            verbose=verbose,
        )
        return GPParameters(variance=variance, length_scale=length_scale)

    def gibbs(
        self,
        catalog,
        config=None,
        gp_backend="sparse",
        sparse_gp=None,
        rng_seed=None,
        reference_intensity=None,
    ):
        """Estimate this SSGC model with its Gibbs sampler.

        The model remains unchanged. Posterior chains, predictions and
        diagnostics are returned in a GibbsResults object.
        """
        config, gp_backend = self._prepare_gibbs(
            catalog, config, gp_backend, GibbsConfig
        )
        from ..inference.ssgc_gibbs import SSGC_GibbsSampler

        sampler = SSGC_GibbsSampler(
            model=self,
            m=catalog.magnitudes,
            beta_init=(
                config.beta_init
                if config.fixed_beta is None
                else config.fixed_beta
            ),
            beta_priors=config.beta_prior,
            sigma_MH_beta=config.sigma_mh_beta,
            t0_beta=config.adaptation_start,
            eps_mh_beta=config.proposal_jitter,
            rng_seed=rng_seed,
        )
        return self._run_gibbs(
            sampler,
            catalog,
            config,
            gp_backend,
            sparse_gp=sparse_gp,
            reference_intensity=reference_intensity,
            fixed_beta=config.fixed_beta,
        )
