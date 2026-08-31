"""Spatially structured sigmoidal Gaussian Cox-process model."""

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.special import expit
from shapely.geometry import box

from ..config import GPParameters, GibbsConfig, SPINHVIConfig, SSGCVIConfig
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
        if not isinstance(self.domains, DomainPartition):
            raise TypeError("domains must be a DomainPartition instance.")
        self.duration = float(self.duration)
        if not np.isfinite(self.duration) or self.duration <= 0.0:
            raise ValueError("duration must be finite and positive.")
        validated_bounds = []
        for name, bounds in (("x_bounds", self.x_bounds), ("y_bounds", self.y_bounds)):
            try:
                lower, upper = map(float, bounds)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{name} must contain two finite bounds.") from error
            if not np.isfinite([lower, upper]).all() or lower >= upper:
                raise ValueError(f"{name} must contain two finite increasing bounds.")
            validated_bounds.append((lower, upper))
        self.x_bounds, self.y_bounds = validated_bounds
        if not isinstance(self.gp_prior, GPParameters):
            raise TypeError("gp_prior must be a GPParameters instance.")
        for name in (
            "eps_prior_variance",
            "eps_prior_length_scale",
            "nu_prior_rate",
            "jitter",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
            setattr(self, name, value)

        observation_window = box(
            self.x_bounds[0],
            self.y_bounds[0],
            self.x_bounds[1],
            self.y_bounds[1],
        )
        if not observation_window.buffer(1e-12).covers(
            self.domains.observation_geometry
        ):
            raise ValueError("Every spatial domain must lie inside x_bounds/y_bounds.")
        self.magnitude_min = float(self.magnitude_min)
        if not np.isfinite(self.magnitude_min):
            raise ValueError("magnitude_min must be finite.")
        if self.magnitude_max is not None:
            self.magnitude_max = float(self.magnitude_max)
            if not np.isfinite(self.magnitude_max):
                raise ValueError("magnitude_max must be finite.")
            if self.magnitude_max <= self.magnitude_min:
                raise ValueError("magnitude_max must be greater than magnitude_min.")

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
        if not isinstance(catalog, EventCatalog):
            raise TypeError("catalog must be an EventCatalog instance.")
        if np.any(catalog.t < -1e-12):
            raise ValueError("The catalog contains events before time zero.")
        if np.any(catalog.t > self.duration + 1e-12):
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
        if not np.all(np.isfinite(eps)):
            raise ValueError("eps must contain only finite values.")
        x_values, y_values = np.broadcast_arrays(
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
        )
        output_shape = x_values.shape
        domain_index = self.domains.locate(x_values, y_values)
        values = np.zeros(domain_index.size, dtype=float)
        inside = domain_index >= 0
        values[inside] = np.exp(eps[domain_index[inside]])
        return values.reshape(output_shape)

    def background_intensity(self, x, y, eps, latent_gp) -> np.ndarray:
        latent_gp = np.asarray(latent_gp, dtype=float).reshape(-1)
        baseline = self.baseline_intensity(x, y, eps)
        if latent_gp.size != baseline.size:
            raise ValueError("latent_gp must have one value per evaluation point.")
        if not np.all(np.isfinite(latent_gp)):
            raise ValueError("latent_gp must contain only finite values.")
        return baseline * expit(latent_gp.reshape(baseline.shape))

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
        if len(catalog) < 2:
            raise ValueError("GP calibration requires at least two observed events.")
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

    def vi(self, catalog, config=None, rng_seed=None):
        """Estimate this SSGC model with mean-field variational inference.

        All observed events belong to the background process. The shared VI
        engine therefore omits branching and ETAS factors automatically.
        """
        from ..inference.VI import SPINHVI

        if config is None:
            config = SSGCVIConfig(random_seed=rng_seed)
        elif not isinstance(config, SSGCVIConfig) or isinstance(
            config,
            SPINHVIConfig,
        ):
            raise TypeError("config must be an SSGCVIConfig instance.")
        elif rng_seed is not None:
            if config.random_seed is not None and config.random_seed != rng_seed:
                raise ValueError(
                    "rng_seed conflicts with config.random_seed; provide only one seed."
                )
            config = replace(config, random_seed=rng_seed)
        inference_model = self
        if config.use_calibration:
            calibrated_prior = self.calibrate_gp_prior(
                catalog,
                rng_seed=config.random_seed,
                verbose=config.verbose,
            )
            inference_model = replace(self, gp_prior=calibrated_prior)
        engine = SPINHVI(inference_model, catalog, config=config)
        return engine.fit()
