"""Public API for SSGC and SPIN-H modelling, inference and simulation.

Exports are loaded lazily so importing a low-level module such as
``simulation`` does not create a cycle through this convenience namespace.
"""

from importlib import import_module


_EXPORTS = {
    "EventCatalog": ("data", "EventCatalog"),
    "HawkesProcessSimulation": ("simulation", "HawkesProcessSimulation"),
    "SimulationGrid": ("simulation", "SimulationGrid"),
    "SpatialProcessSimulation": ("simulation", "SpatialProcessSimulation"),
    "generate_voronoi_cells": ("simulation", "generate_voronoi_cells"),
    "simulate_hawkes_process": ("simulation", "simulate_hawkes_process"),
    "simulate_spatial_process": ("simulation", "simulate_spatial_process"),
    "DomainPartition": ("spatial", "DomainPartition"),
    "SpatialDomain": ("spatial", "SpatialDomain"),
    "DEFAULT_FIGURES_DIR": ("visualization", "DEFAULT_FIGURES_DIR"),
    "FIGURE_DPI": ("visualization", "FIGURE_DPI"),
    "RASTER_FIGURE_DPI": ("visualization", "RASTER_FIGURE_DPI"),
    "RASTER_FIGURE_FORMAT": ("visualization", "RASTER_FIGURE_FORMAT"),
    "VECTOR_FIGURE_FORMAT": ("visualization", "VECTOR_FIGURE_FORMAT"),
    "plot_field": ("visualization", "plot_field"),
    "plot_process_dashboard": ("visualization", "plot_process_dashboard"),
    "plot_voronoi_cells": ("visualization", "plot_voronoi_cells"),
    "save_figure": ("visualization", "save_figure"),
    "ETASParameters": ("package.config", "ETASParameters"),
    "GPParameters": ("package.config", "GPParameters"),
    "GibbsConfig": ("package.config", "GibbsConfig"),
    "SPINHGibbsConfig": ("package.config", "SPINHGibbsConfig"),
    "SPINHVIConfig": ("package.config", "SPINHVIConfig"),
    "SSGCVIConfig": ("package.config", "SSGCVIConfig"),
    "GibbsResults": ("package.inference", "GibbsResults"),
    "SPINHVI": ("package.inference", "SPINHVI"),
    "SPINHVIResults": ("package.inference", "SPINHVIResults"),
    "SPINHVIState": ("package.inference", "SPINHVIState"),
    "SparseGP": ("package.inference", "SparseGP"),
    "TemporalCandidateGraph": ("package.inference", "TemporalCandidateGraph"),
    "VIResults": ("package.inference", "VIResults"),
    "ETASKernel": ("package.models", "ETASKernel"),
    "OmoriKernel": ("package.models", "OmoriKernel"),
    "PointProcessModel": ("package.models", "PointProcessModel"),
    "ProductivityKernel": ("package.models", "ProductivityKernel"),
    "SPINHModel": ("package.models", "SPINHModel"),
    "SSGCModel": ("package.models", "SSGCModel"),
    "SpatialPowerLawKernel": ("package.models", "SpatialPowerLawKernel"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    """Load one public object on first access and cache it in this module."""
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
