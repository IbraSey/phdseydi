"""Public API for SSGC and SPIN-H modelling, inference and simulation."""

from .config import ETASInferenceConfig, ETASParameters, GPParameters, MCMCConfig
from .data import EventCatalog
from .inference import (
    ExactGPBackend, FourierSparseGPBackend, GPBackend, GibbsResults,
    SparseGP,
)
from .models import (
    ETASKernel, OmoriKernel, PointProcessModel, ProductivityKernel, SPINHModel,
    SSGCModel, SpatialPowerLawKernel,
)
from .simulation import (
    HawkesProcessSimulation, SimulationGrid, SpatialProcessSimulation,
    generate_voronoi_cells,
    simulate_hawkes_process, simulate_process, simulate_spatial_process,
)
from .spatial import DomainPartition, SpatialDomain
from .visualization import (
    FIGURE_DPI, plot_field, plot_process_dashboard, plot_voronoi_cells, save_figure,
)

__all__ = [
    "DomainPartition", "ETASInferenceConfig", "ETASKernel", "ETASParameters",
    "EventCatalog", "ExactGPBackend", "FIGURE_DPI", "FourierSparseGPBackend",
    "GPBackend", "GPParameters", "GibbsResults",
    "MCMCConfig", "OmoriKernel", "PointProcessModel",
    "ProductivityKernel", "SPINHModel", "SSGCModel",
    "HawkesProcessSimulation", "SimulationGrid",
    "SparseGP", "SpatialDomain", "SpatialPowerLawKernel",
    "SpatialProcessSimulation",
    "generate_voronoi_cells", "plot_field", "plot_process_dashboard",
    "plot_voronoi_cells", "save_figure", "simulate_hawkes_process",
    "simulate_process",
    "simulate_spatial_process",
]
