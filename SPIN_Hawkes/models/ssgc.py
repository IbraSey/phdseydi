"""Spatially structured sigmoidal Gaussian Cox-process model."""

from dataclasses import dataclass, field

import numpy as np
from scipy.special import expit

from ..config import GPParameters
from ..data.catalog import EventCatalog
from ..spatial.domain import DomainPartition
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

    def __post_init__(self):
        if self.duration <= 0:
            raise ValueError("duration must be positive.")
        if self.x_bounds[0] >= self.x_bounds[1]:
            raise ValueError("x_bounds must be increasing.")
        if self.y_bounds[0] >= self.y_bounds[1]:
            raise ValueError("y_bounds must be increasing.")
        if self.eps_prior_variance <= 0 or self.eps_prior_length_scale <= 0:
            raise ValueError("Epsilon-prior parameters must be positive.")

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

    def _build_gibbs_sampler(self, rng_seed=None):
        from ..inference.ssgc_gibbs import SSGC_GibbsSampler

        return SSGC_GibbsSampler(
            model=self,
            rng_seed=rng_seed,
        )

    def gibbs(
        self,
        catalog,
        config=None,
        gp_backend="sparse",
        rng_seed=None,
        reference_intensity=None,
    ):
        """Estimate this SSGC model with its Gibbs sampler.

        The model remains unchanged. Posterior chains, predictions and
        diagnostics are returned in a GibbsResults object.
        """
        config, gp_backend = self._prepare_gibbs(
            catalog, config, gp_backend
        )
        sampler = self._build_gibbs_sampler(rng_seed)
        return self._run_gibbs(
            sampler,
            catalog,
            config,
            gp_backend,
            reference_intensity,
        )
