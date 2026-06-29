"""Simulation of spatial sigmoidal Cox and marked Hawkes processes."""

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import openturns as ot
from scipy.special import expit
from shapely.geometry import box

from ..config import ETASParameters
from ..data.catalog import EventCatalog
from ..models.kernels import ETASKernel
from ..spatial.domain import DomainPartition

LatentField = Callable[..., np.ndarray | float]


@dataclass(frozen=True)
class SimulationGrid:
    """Regular-grid values used to inspect a simulated spatial process."""

    x: np.ndarray
    y: np.ndarray
    baseline: np.ndarray
    latent: np.ndarray
    sigmoid: np.ndarray
    intensity: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        """Return a dictionary representation for experiment scripts."""
        return {
            "GX": self.x,
            "GY": self.y,
            "mu_tilde": self.baseline,
            "f_star": self.latent,
            "sig_f": self.sigmoid,
            "mu_star": self.intensity,
        }


@dataclass(frozen=True)
class SpatialProcessSimulation:
    """A simulated catalog together with its generating spatial intensity."""

    catalog: EventCatalog
    domains: DomainPartition
    baseline_intensities: np.ndarray
    x_bounds: tuple[float, float]
    y_bounds: tuple[float, float]
    duration: float
    grid: SimulationGrid
    _component_evaluator: Callable = field(repr=False, compare=False)

    @property
    def sample(self) -> ot.Sample:
        """Events as an OpenTURNS sample with columns ``x, y, t``."""
        values = np.column_stack([self.catalog.x, self.catalog.y, self.catalog.t])
        return ot.Sample(values.tolist()) if len(values) else ot.Sample(0, 3)

    def spatial_components(self, x, y):
        """Evaluate ``mu_tilde``, ``f``, ``sigmoid(f)`` and ``mu``."""
        return self._component_evaluator(x, y)

    def as_mapping(self):
        """Return the ``(simulation_data, grid_data)`` pair used by experiment scripts."""
        sim_data = {
            "X": self.sample,
            "domains": list(self.domains.polygons),
            "mus_vec": self.baseline_intensities.tolist(),
            "bounds": (self.x_bounds, self.y_bounds, self.duration),
            "mu_tilde_func": lambda x, y: self.spatial_components(x, y)[0],
            "f_star_func": lambda x, y: self.spatial_components(x, y)[1],
            "mu_star_func": lambda x, y: self.spatial_components(x, y)[3],
        }
        return sim_data, self.grid.as_dict()


@dataclass(frozen=True)
class HawkesProcessSimulation:
    """Marked Hawkes catalog and its known branching structure.

    ``parent_indices[i]`` is ``-1`` for a background event and otherwise the
    zero-based index of the event that triggered catalog event ``i``.  The
    catalog is sorted in time, so every parent precedes its offspring.
    """

    catalog: EventCatalog
    parent_indices: np.ndarray
    generations: np.ndarray
    background_simulation: SpatialProcessSimulation

    def __post_init__(self):
        parent_indices = np.asarray(self.parent_indices, dtype=int).reshape(-1)
        generations = np.asarray(self.generations, dtype=int).reshape(-1)
        n_events = len(self.catalog)
        if parent_indices.size != n_events or generations.size != n_events:
            raise ValueError("One parent index and generation are required per event.")
        event_indices = np.arange(n_events)
        if np.any(parent_indices >= event_indices):
            raise ValueError("Every Hawkes parent must precede its offspring.")
        if np.any(parent_indices < -1):
            raise ValueError("Parent indices must be -1 or valid event indices.")
        if np.any(generations < 0):
            raise ValueError("Generations must be non-negative.")
        object.__setattr__(self, "parent_indices", parent_indices)
        object.__setattr__(self, "generations", generations)

    @property
    def is_background(self) -> np.ndarray:
        """Boolean mask identifying immigrant/background events."""
        return self.parent_indices < 0

    @property
    def branching_labels(self) -> np.ndarray:
        """One-based parent labels with zero reserved for background events."""
        return np.where(self.is_background, 0, self.parent_indices + 1)

    @property
    def n_background(self) -> int:
        """Number of background events in the observed catalog."""
        return int(self.is_background.sum())

    @property
    def n_triggered(self) -> int:
        """Number of triggered events in the observed catalog."""
        return int((~self.is_background).sum())


def _sample_truncated_exponential(rng, beta, magnitude_min, magnitude_max, size):
    """Sample magnitudes from an exponential law truncated to finite bounds."""
    upper_tail = np.exp(-beta * (magnitude_max - magnitude_min))
    uniforms = rng.uniform(size=int(size))
    return magnitude_min - np.log1p(-uniforms * (1.0 - upper_tail)) / beta


def simulate_hawkes_process(
    X_bounds: tuple[float, float] = (0.0, 2.0),
    Y_bounds: tuple[float, float] = (0.0, 2.0),
    T: float = 20.0,
    polygons: Sequence[object] | None = None,
    n_cols: int = 2,
    n_rows: int = 2,
    mus: Sequence[float] | float = 5.0,
    f: LatentField | Sequence[LatentField] | None = None,
    etas_parameters: ETASParameters = ETASParameters(),
    beta: float = 2.3,
    magnitude_min: float = 2.0,
    magnitude_max: float = 6.0,
    rng_seed: int | None = 0,
    grid_res: int = 100,
    max_events: int = 100_000,
    verbose: bool = False,
    **f_kwargs,
) -> HawkesProcessSimulation:
    """Simulate a marked SPIN-H catalog by its branching representation.

    Background events are first drawn from the spatial sigmoidal Cox process.
    Each observed event then produces a Poisson number of offspring with mean
    given by the ETAS productivity kernel.  Offspring are drawn from the
    normalized Omori-Utsu and spatial power-law kernels, then discarded when
    they fall beyond ``T`` or outside the union of the spatial domains.  The
    returned parent indices are the simulation ground truth for declustering.

    The construction starts from background events inside the observation
    window; it therefore does not include clusters whose immigrant lies outside
    that window.  This is the same finite-window convention used by the Gibbs
    smoke tests.
    """
    if not isinstance(etas_parameters, ETASParameters):
        raise TypeError("etas_parameters must be an ETASParameters instance.")
    beta = float(beta)
    magnitude_min = float(magnitude_min)
    magnitude_max = float(magnitude_max)
    if beta <= 0:
        raise ValueError("beta must be positive.")
    if not magnitude_min < magnitude_max:
        raise ValueError("magnitude_min must be smaller than magnitude_max.")
    if max_events <= 0:
        raise ValueError("max_events must be positive.")

    background = simulate_spatial_process(
        X_bounds=X_bounds,
        Y_bounds=Y_bounds,
        T=T,
        polygons=polygons,
        n_cols=n_cols,
        n_rows=n_rows,
        mus=mus,
        f=f,
        rng_seed=rng_seed,
        grid_res=grid_res,
        verbose=False,
        **f_kwargs,
    )
    rng = np.random.default_rng(rng_seed)
    background_catalog = background.catalog
    n_background = len(background_catalog)
    background_magnitudes = _sample_truncated_exponential(
        rng, beta, magnitude_min, magnitude_max, n_background
    )

    events = [
        [
            float(background_catalog.t[index]),
            float(background_catalog.x[index]),
            float(background_catalog.y[index]),
            float(background_magnitudes[index]),
            -1,
            0,
        ]
        for index in range(n_background)
    ]
    kernel = ETASKernel()
    observation_domains = background.domains
    duration = float(T)
    parent_index = 0

    while parent_index < len(events):
        parent_time, parent_x, parent_y, parent_magnitude, _, parent_generation = (
            events[parent_index]
        )
        offspring_mean = float(
            kernel.productivity.evaluate(
                [parent_magnitude], etas_parameters, magnitude_min
            )[0]
        )
        offspring_count = int(rng.poisson(offspring_mean))
        spatial_scale = float(
            kernel.spatial.scale(
                [parent_magnitude], etas_parameters, magnitude_min
            )[0]
        )

        for _ in range(offspring_count):
            delta_t = etas_parameters.c * (
                (1.0 - rng.uniform()) ** (-1.0 / (etas_parameters.p - 1.0)) - 1.0
            )
            child_time = parent_time + delta_t
            if child_time >= duration:
                continue

            radius_squared = spatial_scale * (
                (1.0 - rng.uniform()) ** (-1.0 / (etas_parameters.q - 1.0)) - 1.0
            )
            angle = rng.uniform(0.0, 2.0 * np.pi)
            child_x = parent_x + np.sqrt(radius_squared) * np.cos(angle)
            child_y = parent_y + np.sqrt(radius_squared) * np.sin(angle)
            if observation_domains.locate([child_x], [child_y])[0] < 0:
                continue
            if len(events) >= max_events:
                raise RuntimeError(
                    "The Hawkes simulation reached max_events; choose a subcritical "
                    "configuration or increase max_events."
                )

            child_magnitude = float(
                _sample_truncated_exponential(
                    rng, beta, magnitude_min, magnitude_max, 1
                )[0]
            )
            events.append(
                [
                    child_time,
                    child_x,
                    child_y,
                    child_magnitude,
                    parent_index,
                    parent_generation + 1,
                ]
            )
        parent_index += 1

    if events:
        values = np.asarray(events, dtype=float)
        order = np.argsort(values[:, 0], kind="stable")
        inverse_order = np.empty(len(order), dtype=int)
        inverse_order[order] = np.arange(len(order))
        ordered = values[order]
        original_parents = ordered[:, 4].astype(int)
        parent_indices = np.where(
            original_parents < 0, -1, inverse_order[original_parents]
        )
        catalog = EventCatalog(
            ordered[:, 0], ordered[:, 1], ordered[:, 2], ordered[:, 3]
        )
        generations = ordered[:, 5].astype(int)
    else:
        catalog = EventCatalog([], [], [], [])
        parent_indices = np.empty(0, dtype=int)
        generations = np.empty(0, dtype=int)

    simulation = HawkesProcessSimulation(
        catalog=catalog,
        parent_indices=parent_indices,
        generations=generations,
        background_simulation=background,
    )
    if verbose:
        print(
            f"Simulated {len(catalog)} Hawkes events "
            f"({simulation.n_background} background, {simulation.n_triggered} triggered)."
        )
    return simulation


def _rectangular_domains(x_bounds, y_bounds, n_cols: int, n_rows: int):
    if n_cols <= 0 or n_rows <= 0:
        raise ValueError("n_cols and n_rows must be positive.")
    xmin, xmax = x_bounds
    ymin, ymax = y_bounds
    dx = (xmax - xmin) / n_cols
    dy = (ymax - ymin) / n_rows
    return [
        box(xmin + col * dx, ymin + row * dy,
            xmin + (col + 1) * dx, ymin + (row + 1) * dy)
        for row in range(n_rows)
        for col in range(n_cols)
    ]


def simulate_spatial_process(
    X_bounds: tuple[float, float] = (0.0, 2.0),
    Y_bounds: tuple[float, float] = (0.0, 2.0),
    T: float = 20.0,
    polygons: Sequence[object] | None = None,
    n_cols: int = 2,
    n_rows: int = 2,
    mus: Sequence[float] | float = 5.0,
    f: LatentField | Sequence[LatentField] | None = None,
    rng_seed: int | None = 0,
    grid_res: int = 100,
    verbose: bool = False,
    **f_kwargs,
) -> SpatialProcessSimulation:
    """Simulate a spatial sigmoidal Cox process by exact Poisson thinning.

    The proposal intensity is ``max(mus)``. This is a valid homogeneous upper
    bound because the logistic link is at most one, unlike a maximum estimated
    on a finite plotting grid.
    """
    xmin, xmax = map(float, X_bounds)
    ymin, ymax = map(float, Y_bounds)
    duration = float(T)
    if not xmin < xmax or not ymin < ymax or duration <= 0:
        raise ValueError("Bounds must increase and T must be positive.")
    if grid_res < 2:
        raise ValueError("grid_res must be at least 2.")
    if rng_seed is not None:
        ot.RandomGenerator.SetSeed(int(rng_seed))

    domain_polygons = (
        list(polygons)
        if polygons is not None
        else _rectangular_domains((xmin, xmax), (ymin, ymax), n_cols, n_rows)
    )
    n_domains = len(domain_polygons)
    baseline = (
        np.full(n_domains, float(mus))
        if np.isscalar(mus)
        else np.asarray(mus, dtype=float).reshape(-1)
    )
    if baseline.size != n_domains:
        raise ValueError("One baseline intensity is required per spatial domain.")
    if np.any(~np.isfinite(baseline)) or np.any(baseline <= 0):
        raise ValueError("Baseline intensities must be finite and positive.")

    domains = DomainPartition.from_polygons(domain_polygons, np.log(baseline))
    global_field = f is None or callable(f)
    if f is None:
        latent_field = lambda x, y, **kwargs: np.zeros_like(x, dtype=float)
    elif global_field:
        latent_field = f
    else:
        latent_fields = tuple(f)
        if len(latent_fields) != n_domains or not all(callable(fn) for fn in latent_fields):
            raise ValueError("One callable latent field is required per domain.")

    def evaluate_components(x, y):
        x_values, y_values = np.broadcast_arrays(
            np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        )
        original_shape = x_values.shape
        x_flat = x_values.reshape(-1)
        y_flat = y_values.reshape(-1)
        domain_indices = domains.locate(x_flat, y_flat)
        mu_tilde = np.zeros(x_flat.size, dtype=float)
        latent = np.zeros(x_flat.size, dtype=float)

        inside = domain_indices >= 0
        mu_tilde[inside] = baseline[domain_indices[inside]]
        if global_field:
            values = np.asarray(latent_field(x_flat, y_flat, **f_kwargs), dtype=float)
            latent[:] = np.broadcast_to(values, x_flat.shape)
        else:
            for index, field_function in enumerate(latent_fields):
                mask = domain_indices == index
                if mask.any():
                    values = np.asarray(
                        field_function(x_flat[mask], y_flat[mask], **f_kwargs),
                        dtype=float,
                    )
                    latent[mask] = np.broadcast_to(values, (mask.sum(),))

        sigmoid = expit(latent)
        intensity = mu_tilde * sigmoid
        return tuple(
            value.reshape(original_shape)
            for value in (mu_tilde, latent, sigmoid, intensity)
        )

    grid_x, grid_y = np.meshgrid(
        np.linspace(xmin, xmax, grid_res),
        np.linspace(ymin, ymax, grid_res),
    )
    grid_values = evaluate_components(grid_x, grid_y)
    grid = SimulationGrid(grid_x, grid_y, *grid_values)

    upper_intensity = float(baseline.max())
    volume = (xmax - xmin) * (ymax - ymin) * duration
    n_candidates = int(ot.Poisson(upper_intensity * volume).getRealization()[0])
    if n_candidates:
        proposal = ot.JointDistribution(
            [ot.Uniform(xmin, xmax), ot.Uniform(ymin, ymax), ot.Uniform(0.0, duration)]
        )
        candidates = np.asarray(proposal.getSample(n_candidates), dtype=float)
        candidate_intensity = evaluate_components(candidates[:, 0], candidates[:, 1])[3]
        uniforms = np.asarray(ot.Uniform(0.0, 1.0).getSample(n_candidates), dtype=float).reshape(-1)
        accepted = candidates[uniforms < candidate_intensity / upper_intensity]
        accepted = accepted[np.argsort(accepted[:, 2])]
    else:
        accepted = np.empty((0, 3), dtype=float)

    catalog = EventCatalog(accepted[:, 2], accepted[:, 0], accepted[:, 1])
    simulation = SpatialProcessSimulation(
        catalog=catalog,
        domains=domains,
        baseline_intensities=baseline,
        x_bounds=(xmin, xmax),
        y_bounds=(ymin, ymax),
        duration=duration,
        grid=grid,
        _component_evaluator=evaluate_components,
    )
    if verbose:
        print(
            f"Simulated {len(catalog)} events in {n_domains} domains "
            f"over T={duration:g}."
        )
    return simulation


def simulate_process(*args, **kwargs):
    """Simulate a process and return dictionary payloads for experiment scripts."""
    return simulate_spatial_process(*args, **kwargs).as_mapping()
