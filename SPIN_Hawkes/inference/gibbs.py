from dataclasses import dataclass, field
import openturns as ot
from ..data.catalog import EventCatalog
from ..config import ETASInferenceConfig, MCMCConfig
from ..models import SPINHModel, SSGCModel
from .backends import ExactGPBackend, GPBackend, FourierSparseGPBackend
from .base import InferenceMethod
from .results import GibbsResults
from .ssgc_gibbs import SSGC_GibbsSampler
from .spinh_gibbs import SPIN_H_GibbsSampler


# High-level Gibbs inference facades.
@dataclass
class SSGCGibbsInference(InferenceMethod):
    """Gibbs inference facade for :class:`SSGCModel`."""

    model: SSGCModel
    config: MCMCConfig = field(default_factory=MCMCConfig)
    gp_backend: GPBackend = field(default_factory=ExactGPBackend)
    rng_seed: int | None = None

    def _build_sampler(self) -> SSGC_GibbsSampler:
        areas, polygons = self.model.sampler_geometry()
        return SSGC_GibbsSampler(
            X_bounds=self.model.x_bounds,
            Y_bounds=self.model.y_bounds,
            T=self.model.duration,
            Areas=areas,
            lambda_nu=self.model.nu_prior_rate,
            nu=[
                self.model.gp_prior.variance,
                self.model.gp_prior.length_scale,
            ],
            delta=[
                self.model.eps_prior_variance,
                self.model.eps_prior_length_scale,
            ],
            polygons=polygons,
            jitter=self.model.jitter,
            rng_seed=self.rng_seed,
        )

    def fit(
        self,
        catalog: EventCatalog,
        reference_intensity=None,
    ) -> GibbsResults:
        self.model.validate_catalog(catalog)
        sampler = self._build_sampler()
        backend_options = self.gp_backend.sampler_options(self.model)
        cfg = self.config
        raw = sampler.run(
            t=ot.Point(catalog.t.tolist()),
            x=ot.Point(catalog.x.tolist()),
            y=ot.Point(catalog.y.tolist()),
            mala_step=cfg.mala_step,
            n_iter=cfg.n_iter,
            learn_nu=cfg.learn_nu,
            t0_nu=cfg.t0_nu,
            step_nu_init=cfg.step_nu_init,
            verbose=cfg.verbose,
            verbose_every=cfg.verbose_every,
            use_calibration=cfg.use_calibration,
            mu_star_func=reference_intensity,
            grid_nx=cfg.grid_nx,
            grid_ny=cfg.grid_ny,
            thin=cfg.thin,
            compute_emu=cfg.compute_emu,
            emu_every=cfg.emu_every,
            calibration_method=cfg.calibration_method,
            **backend_options,
        )
        return GibbsResults(raw, self.model, catalog, sampler)


@dataclass
class SPINHGibbsInference(InferenceMethod):
    """Gibbs inference facade for :class:`SPINHModel`."""

    model: SPINHModel
    config: MCMCConfig = field(default_factory=MCMCConfig)
    etas_config: ETASInferenceConfig = field(
        default_factory=ETASInferenceConfig
    )
    gp_backend: GPBackend = field(default_factory=FourierSparseGPBackend)
    rng_seed: int | None = None

    def _build_sampler(self, catalog: EventCatalog) -> SPIN_H_GibbsSampler:
        areas, polygons = self.model.sampler_geometry()
        marked_magnitudes = (
            catalog.magnitudes if self.model.etas_parameters.marked else None
        )
        return SPIN_H_GibbsSampler(
            X_bounds=self.model.x_bounds,
            Y_bounds=self.model.y_bounds,
            T=self.model.duration,
            Areas=areas,
            lambda_nu=self.model.nu_prior_rate,
            nu=[
                self.model.gp_prior.variance,
                self.model.gp_prior.length_scale,
            ],
            delta=[
                self.model.eps_prior_variance,
                self.model.eps_prior_length_scale,
            ],
            polygons=polygons,
            use_etas=True,
            theta_phi_init=self.model.etas_parameters.as_dict(),
            theta_phi_priors=self.etas_config.theta_priors,
            m=marked_magnitudes,
            m_c=self.model.magnitude_min,
            m_max=self.model.magnitude_max,
            beta_init=self.etas_config.beta_init,
            beta_priors=self.etas_config.beta_prior,
            sigma_MH_etas=self.etas_config.sigma_mh_etas,
            sigma_MH_beta=self.etas_config.sigma_mh_beta,
            t0_etas=self.etas_config.adaptation_start,
            eps_mh_etas=self.etas_config.proposal_jitter,
            jitter=self.model.jitter,
            rng_seed=self.rng_seed,
        )

    def fit(
        self,
        catalog: EventCatalog,
        reference_intensity=None,
    ) -> GibbsResults:
        self.model.validate_catalog(catalog)
        sampler = self._build_sampler(catalog)
        backend_options = self.gp_backend.sampler_options(self.model)
        cfg = self.config
        raw = sampler.run(
            t=ot.Point(catalog.t.tolist()),
            x=ot.Point(catalog.x.tolist()),
            y=ot.Point(catalog.y.tolist()),
            mala_step=cfg.mala_step,
            n_iter=cfg.n_iter,
            learn_nu=cfg.learn_nu,
            learn_beta=self.etas_config.learn_beta,
            t0_nu=cfg.t0_nu,
            step_nu_init=cfg.step_nu_init,
            verbose=cfg.verbose,
            verbose_every=cfg.verbose_every,
            use_calibration=cfg.use_calibration,
            mu_star_func=reference_intensity,
            grid_nx=cfg.grid_nx,
            grid_ny=cfg.grid_ny,
            thin=cfg.thin,
            compute_emu=cfg.compute_emu,
            emu_every=cfg.emu_every,
            calibration_method=cfg.calibration_method,
            **backend_options,
        )
        return GibbsResults(raw, self.model, catalog, sampler)

