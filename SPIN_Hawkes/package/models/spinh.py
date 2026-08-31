"""SPIN-H model: SSGC background plus marked ETAS triggering."""

from dataclasses import dataclass, field, replace

import numpy as np

from package.config import ETASParameters, SPINHGibbsConfig, SPINHVIConfig
from data.catalog import EventCatalog
from .kernels import ETASKernel
from .ssgc import SSGCModel


@dataclass
class SPINHModel(SSGCModel):
    """SPIN-H model: SSGC background plus optional marked ETAS triggering."""

    etas_parameters: ETASParameters = field(default_factory=ETASParameters)
    etas_kernel: ETASKernel = field(default_factory=ETASKernel)

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.etas_parameters, ETASParameters):
            raise TypeError("etas_parameters must be an ETASParameters instance.")
        if not isinstance(self.etas_kernel, ETASKernel):
            raise TypeError("etas_kernel must be an ETASKernel instance.")

    def validate_catalog(self, catalog: EventCatalog) -> np.ndarray:
        indices = super().validate_catalog(catalog)
        if self.etas_parameters.marked and catalog.magnitudes is None:
            raise ValueError("The marked ETAS model requires event magnitudes.")
        return indices

    def gibbs(
        self,
        catalog,
        config=None,
        gp_backend="sparse",
        sparse_gp=None,
        rng_seed=None,
        reference_intensity=None,
    ):
        """Estimate this SPIN-H model with its Gibbs sampler.

        ``etas_parameters`` supplies the initial ETAS state. The model is not
        mutated; posterior chains and analyses are returned in GibbsResults.
        """
        config, gp_backend = self._prepare_gibbs(
            catalog, config, gp_backend, SPINHGibbsConfig
        )
        from ..inference.spinh_gibbs import SPIN_H_GibbsSampler

        sampler = SPIN_H_GibbsSampler(
            model=self,
            theta_phi_priors=config.theta_priors,
            m=catalog.magnitudes if self.etas_parameters.marked else None,
            beta_init=(
                config.beta_init
                if config.fixed_beta is None
                else config.fixed_beta
            ),
            beta_priors=config.beta_prior,
            sigma_MH_etas=config.sigma_mh_etas,
            sigma_MH_beta=config.sigma_mh_beta,
            t0_etas=config.adaptation_start,
            eps_mh_etas=config.proposal_jitter,
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
            sample_z=config.sample_z,
            known_z=config.known_z,
            fixed_etas=config.fixed_etas,
            parent_time_window=config.parent_time_window,
        )

    def vi(self, catalog, config=None, rng_seed=None):
        """Estimate this SPIN-H model with simple hybrid CAVI/VI.

        The model remains unchanged. The returned VIResults object contains
        variational posterior summaries for Z, epsilon, the GP and ETAS blocks.
        """
        from ..inference.VI import SPINHVI

        if config is None:
            config = SPINHVIConfig(random_seed=rng_seed)
        elif not isinstance(config, SPINHVIConfig):
            raise TypeError("config must be a SPINHVIConfig instance.")
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

    def parent_time_window_from_kernel(
        self,
        relative_density: float,
        parameters: ETASParameters | None = None,
    ) -> float:
        """Map a relative spatio-temporal kernel height to a parent time window.

        For fixed parent magnitude and distance, the relative decay of the full
        ETAS kernel is the relative decay of its temporal Omori component.
        """
        parameters = self.etas_parameters if parameters is None else parameters
        if not isinstance(parameters, ETASParameters):
            raise TypeError("parameters must be an ETASParameters instance.")
        return self.etas_kernel.temporal.lag_at_relative_density(
            relative_density,
            parameters,
        )

    def triggering_intensity(
        self,
        t_eval,
        x_eval,
        y_eval,
        history: EventCatalog,
        parameters: ETASParameters | None = None,
    ) -> np.ndarray:
        parameters = self.etas_parameters if parameters is None else parameters
        if not isinstance(parameters, ETASParameters):
            raise TypeError("parameters must be an ETASParameters instance.")
        if not isinstance(history, EventCatalog):
            raise TypeError("history must be an EventCatalog instance.")
        t_eval, x_eval, y_eval = np.broadcast_arrays(
            np.asarray(t_eval, dtype=float),
            np.asarray(x_eval, dtype=float),
            np.asarray(y_eval, dtype=float),
        )
        shape = t_eval.shape
        if not np.all(np.isfinite([t_eval, x_eval, y_eval])):
            raise ValueError("Evaluation times and coordinates must be finite.")
        t_eval = t_eval.reshape(-1)
        x_eval = x_eval.reshape(-1)
        y_eval = y_eval.reshape(-1)
        if len(history) == 0:
            return np.zeros(shape, dtype=float)
        if parameters.marked and history.magnitudes is None:
            raise ValueError("Marked triggering intensity requires history magnitudes.")
        magnitudes = (
            history.magnitudes
            if history.magnitudes is not None
            else np.full(len(history), self.magnitude_min)
        )
        delta_t = t_eval[:, None] - history.t[None, :]
        distance_squared = (
            (x_eval[:, None] - history.x[None, :]) ** 2
            + (y_eval[:, None] - history.y[None, :]) ** 2
        )
        values = self.etas_kernel.pairwise(
            delta_t,
            distance_squared,
            magnitudes[None, :],
            parameters,
            self.magnitude_min,
        )
        return values.sum(axis=1).reshape(shape)

    def conditional_intensity(
        self,
        t_eval,
        x_eval,
        y_eval,
        history: EventCatalog,
        eps,
        latent_gp,
        parameters: ETASParameters | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        background = self.background_intensity(x_eval, y_eval, eps, latent_gp)
        triggering = self.triggering_intensity(
            t_eval, x_eval, y_eval, history, parameters
        )
        return background, triggering, background + triggering

    def temporal_compensator(
        self,
        parent_times,
        parameters: ETASParameters | None = None,
    ) -> np.ndarray:
        parameters = self.etas_parameters if parameters is None else parameters
        if not isinstance(parameters, ETASParameters):
            raise TypeError("parameters must be an ETASParameters instance.")
        parent_times = np.asarray(parent_times, dtype=float)
        if not np.all(np.isfinite(parent_times)):
            raise ValueError("parent_times must contain only finite values.")
        return self.etas_kernel.temporal.integral_until(
            parent_times, self.duration, parameters
        )

    def spatial_compensator(
        self,
        parent_x,
        parent_y,
        parent_magnitudes=None,
        parameters: ETASParameters | None = None,
        n_grid: int = 40,
        observation_domain=None,
    ) -> np.ndarray:
        parameters = self.etas_parameters if parameters is None else parameters
        if not isinstance(parameters, ETASParameters):
            raise TypeError("parameters must be an ETASParameters instance.")
        parent_x = np.asarray(parent_x, dtype=float).reshape(-1)
        parent_y = np.asarray(parent_y, dtype=float).reshape(-1)
        if parent_magnitudes is None:
            parent_magnitudes = np.full(
                parent_x.size, self.magnitude_min
            )
        parent_magnitudes = np.asarray(parent_magnitudes, dtype=float).reshape(-1)
        if not (
            parent_x.size == parent_y.size == parent_magnitudes.size
        ):
            raise ValueError(
                "Parent coordinates and magnitudes must have matching lengths."
            )
        if not np.all(
            np.isfinite(np.r_[parent_x, parent_y, parent_magnitudes])
        ):
            raise ValueError(
                "Parent coordinates and magnitudes must contain only finite values."
            )
        return self.etas_kernel.spatial.retained_mass(
            parent_x,
            parent_y,
            parent_magnitudes,
            parameters,
            self.magnitude_min,
            self.x_bounds,
            self.y_bounds,
            n_grid=n_grid,
            observation_domain=(
                self.domains.observation_geometry
                if observation_domain is None
                else observation_domain
            ),
        )

    def triggering_compensator(
        self,
        catalog: EventCatalog,
        parameters: ETASParameters | None = None,
        n_grid: int = 40,
    ) -> float:
        parameters = self.etas_parameters if parameters is None else parameters
        if not isinstance(parameters, ETASParameters):
            raise TypeError("parameters must be an ETASParameters instance.")
        self.validate_catalog(catalog)
        magnitudes = (
            catalog.magnitudes
            if catalog.magnitudes is not None
            else np.full(len(catalog), self.magnitude_min)
        )
        productivity = self.etas_kernel.productivity.evaluate(
            magnitudes, parameters, self.magnitude_min
        )
        return float(
            np.sum(
                productivity
                * self.temporal_compensator(catalog.t, parameters)
                * self.spatial_compensator(
                    catalog.x,
                    catalog.y,
                    magnitudes,
                    parameters,
                    n_grid=n_grid,
                )
            )
        )
