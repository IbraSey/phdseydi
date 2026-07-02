"""Common public interface for point-process models."""

from abc import ABC, abstractmethod

import openturns as ot

from ..config import MCMCConfig
from ..data.catalog import EventCatalog


class PointProcessModel(ABC):
    """Base class for models that can be estimated with Gibbs sampling."""

    @staticmethod
    def _resolve_gp_backend(gp_backend):
        from ..inference.backends import (
            ExactGPBackend,
            FourierSparseGPBackend,
            GPBackend,
        )

        if gp_backend is None or gp_backend == "exact":
            return ExactGPBackend()
        if gp_backend == "sparse":
            return FourierSparseGPBackend()
        if isinstance(gp_backend, GPBackend):
            return gp_backend
        raise ValueError(
            "gp_backend must be 'exact', 'sparse', or a GPBackend instance."
        )

    def _prepare_gibbs(self, catalog, config, gp_backend):
        if not isinstance(catalog, EventCatalog):
            raise TypeError("catalog must be an EventCatalog instance.")
        if config is None:
            config = MCMCConfig()
        elif not isinstance(config, MCMCConfig):
            raise TypeError("config must be a MCMCConfig instance.")
        self.validate_catalog(catalog)
        return config, self._resolve_gp_backend(gp_backend)

    def _run_gibbs(
        self,
        sampler,
        catalog,
        config,
        gp_backend,
        reference_intensity=None,
        **run_overrides,
    ):
        """Run a configured sampler and return its user-facing result."""
        from ..inference.results import GibbsResults

        run_options = {
            "t": ot.Point(catalog.t.tolist()),
            "x": ot.Point(catalog.x.tolist()),
            "y": ot.Point(catalog.y.tolist()),
            "mala_step": config.mala_step,
            "n_iter": config.n_iter,
            "learn_nu": config.learn_nu,
            "t0_nu": config.t0_nu,
            "step_nu_init": config.step_nu_init,
            "verbose": config.verbose,
            "verbose_every": config.verbose_every,
            "use_calibration": config.use_calibration,
            "mu_star_func": reference_intensity,
            "grid_nx": config.grid_nx,
            "grid_ny": config.grid_ny,
            "thin": config.thin,
            "compute_emu": config.compute_emu,
            "emu_every": config.emu_every,
            "calibration_method": config.calibration_method,
        }
        run_options.update(gp_backend.sampler_options(self))
        run_options.update(run_overrides)
        raw = sampler.run(**run_options)
        return GibbsResults(raw, self, catalog)

    @abstractmethod
    def validate_catalog(self, catalog):
        """Validate that a catalog belongs to the model observation window."""

    @abstractmethod
    def gibbs(self, catalog, config=None, **kwargs):
        """Estimate the model and return posterior Gibbs results."""
