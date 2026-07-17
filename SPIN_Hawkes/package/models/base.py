"""Common public interface for point-process models."""

from abc import ABC, abstractmethod

import openturns as ot

from data.catalog import EventCatalog


class PointProcessModel(ABC):
    """Base class for models that can be estimated with Gibbs sampling."""

    @staticmethod
    def _resolve_gp_backend(gp_backend):
        name = "exact" if gp_backend is None else str(gp_backend).lower()
        if name not in {"exact", "sparse"}:
            raise ValueError("gp_backend must be 'exact' or 'sparse'.")
        return name

    def _prepare_gibbs(self, catalog, config, gp_backend, config_type):
        if not isinstance(catalog, EventCatalog):
            raise TypeError("catalog must be an EventCatalog instance.")
        if config is None:
            config = config_type()
        elif not isinstance(config, config_type):
            raise TypeError(
                f"config must be a {config_type.__name__} instance."
            )
        self.validate_catalog(catalog)
        return config, self._resolve_gp_backend(gp_backend)

    def _run_gibbs(
        self,
        sampler,
        catalog,
        config,
        gp_backend,
        sparse_gp=None,
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
        }
        if sparse_gp is not None and gp_backend != "sparse":
            raise ValueError("sparse_gp requires gp_backend='sparse'.")
        run_options.update({"gp_backend": gp_backend, "sparse_gp": sparse_gp})
        run_options.update(run_overrides)
        raw = sampler.run(**run_options)
        return GibbsResults(raw, self, catalog)

    @abstractmethod
    def validate_catalog(self, catalog):
        """Validate that a catalog belongs to the model observation window."""

    @abstractmethod
    def gibbs(self, catalog, config=None, **kwargs):
        """Estimate the model and return posterior Gibbs results."""
