"""SPIN-H model: SSGC background plus marked ETAS triggering."""

from dataclasses import dataclass, field

import numpy as np

from package.config import ETASParameters, SPINHGibbsConfig
from data.catalog import EventCatalog
from .kernels import ETASKernel
from .ssgc import SSGCModel


@dataclass
class SPINHModel(SSGCModel):
    """SPIN-H model: SSGC background plus optional marked ETAS triggering."""

    etas_parameters: ETASParameters = field(default_factory=ETASParameters)
    etas_kernel: ETASKernel = field(default_factory=ETASKernel)
    magnitude_min: float = 0.0
    magnitude_max: float | None = None

    def __post_init__(self):
        super().__post_init__()
        self.magnitude_min = float(self.magnitude_min)
        if not np.isfinite(self.magnitude_min):
            raise ValueError("magnitude_min must be finite.")
        if self.magnitude_max is not None:
            self.magnitude_max = float(self.magnitude_max)
            if not np.isfinite(self.magnitude_max):
                raise ValueError("magnitude_max must be finite.")
            if self.magnitude_max < self.magnitude_min:
                raise ValueError("magnitude_max must be >= magnitude_min.")

    def validate_catalog(self, catalog: EventCatalog) -> np.ndarray:
        indices = super().validate_catalog(catalog)
        if self.etas_parameters.marked and catalog.magnitudes is None:
            raise ValueError("The marked ETAS model requires event magnitudes.")
        if catalog.magnitudes is not None:
            if np.any(catalog.magnitudes < self.magnitude_min):
                raise ValueError("Catalog magnitudes must be >= magnitude_min.")
            if self.magnitude_max is not None and np.any(
                catalog.magnitudes > self.magnitude_max
            ):
                raise ValueError("Catalog magnitudes exceed magnitude_max.")
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
            beta_init=config.beta_init,
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
            learn_beta=config.learn_beta,
            sample_z=config.sample_z,
            known_z=config.known_z,
            sample_etas=config.sample_etas,
            fixed_etas=config.fixed_etas,
        )

    def vi(self, catalog, config=None, rng_seed=None):
        """Estimate this SPIN-H model with simple hybrid CAVI/VI.

        The model remains unchanged. The returned SPINHVIResults object contains
        variational posterior summaries for Z, epsilon, the GP and ETAS blocks.
        """
        from ..inference.VI import SPINHVI, SPINHVIConfig

        if config is None:
            config = SPINHVIConfig(random_seed=rng_seed)
        elif not isinstance(config, SPINHVIConfig):
            raise TypeError("config must be a SPINHVIConfig instance.")
        engine = SPINHVI(self, catalog, config=config)
        return engine.fit()

    def triggering_intensity(
        self,
        t_eval,
        x_eval,
        y_eval,
        history: EventCatalog,
        parameters: ETASParameters | None = None,
    ) -> np.ndarray:
        parameters = parameters or self.etas_parameters
        t_eval, x_eval, y_eval = np.broadcast_arrays(
            np.asarray(t_eval, dtype=float),
            np.asarray(x_eval, dtype=float),
            np.asarray(y_eval, dtype=float),
        )
        shape = t_eval.shape
        t_eval = t_eval.reshape(-1)
        x_eval = x_eval.reshape(-1)
        y_eval = y_eval.reshape(-1)
        if len(history) == 0:
            return np.zeros(shape, dtype=float)
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
        ).reshape(-1)
        return background, triggering, background + triggering

    def temporal_compensator(
        self,
        parent_times,
        parameters: ETASParameters | None = None,
    ) -> np.ndarray:
        return self.etas_kernel.temporal.integral_until(
            parent_times, self.duration, parameters or self.etas_parameters
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
        parameters = parameters or self.etas_parameters
        parent_x = np.asarray(parent_x, dtype=float)
        if parent_magnitudes is None:
            parent_magnitudes = np.full(
                parent_x.size, self.magnitude_min
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
        parameters = parameters or self.etas_parameters
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
