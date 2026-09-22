"""Reusable spatial quadrature rules."""

from dataclasses import dataclass
from hashlib import sha256
from numbers import Integral

import numpy as np
from shapely import contains_xy
from shapely.geometry import box

from .domain import DomainPartition


@dataclass(frozen=True, eq=False)
class SpatialQuadrature:
    """Spatial integration nodes and their positive integration weights."""

    points: np.ndarray
    weights: np.ndarray

    def __post_init__(self):
        points = np.asarray(self.points, dtype=float)
        weights = np.asarray(self.weights, dtype=float).reshape(-1)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must have shape (n_points, 2).")
        if points.shape[0] == 0 or points.shape[0] != weights.size:
            raise ValueError("points and weights must have the same non-zero length.")
        if not np.all(np.isfinite(points)):
            raise ValueError("quadrature points must be finite.")
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("quadrature weights must be finite and positive.")
        points = np.array(points, copy=True)
        weights = np.array(weights, copy=True)
        points.setflags(write=False)
        weights.setflags(write=False)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "weights", weights)

    def integrate(self, values, axis=-1):
        """Integrate arrays whose selected axis indexes the quadrature nodes."""
        values = np.asarray(values, dtype=float)
        if isinstance(axis, bool) or not isinstance(axis, Integral):
            raise TypeError("axis must be an integer.")
        axis = int(axis)
        if axis < 0:
            axis += values.ndim
        if axis < 0 or axis >= values.ndim:
            raise ValueError("axis is outside the value-array dimensions.")
        if values.shape[axis] != self.weights.size:
            raise ValueError("The integration axis must match the number of nodes.")
        return np.tensordot(values, self.weights, axes=(axis, 0))

    def fingerprint(self):
        """Return a stable identifier suitable for experiment checkpoints."""
        digest = sha256()
        digest.update(np.ascontiguousarray(self.points).tobytes())
        digest.update(np.ascontiguousarray(self.weights).tobytes())
        return digest.hexdigest()


def _validate_grid_size(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def midpoint_quadrature(
    x_bounds,
    y_bounds,
    nx,
    ny=None,
    *,
    observation_domain=None,
):
    """Build a midpoint rule on a rectangle or polygonal observation domain."""
    nx = _validate_grid_size(nx, "nx")
    ny = nx if ny is None else _validate_grid_size(ny, "ny")
    xmin, xmax = map(float, x_bounds)
    ymin, ymax = map(float, y_bounds)
    if not np.all(np.isfinite([xmin, xmax, ymin, ymax])) or not (
        xmin < xmax and ymin < ymax
    ):
        raise ValueError("Spatial bounds must be finite and strictly increasing.")
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    grid_x, grid_y = np.meshgrid(
        xmin + (np.arange(nx) + 0.5) * dx,
        ymin + (np.arange(ny) + 0.5) * dy,
    )
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    geometry = (
        box(xmin, ymin, xmax, ymax)
        if observation_domain is None
        else observation_domain
    )
    if geometry.is_empty or not geometry.is_valid:
        raise ValueError("observation_domain must be a non-empty valid geometry.")
    inside = contains_xy(geometry, points[:, 0], points[:, 1])
    if not np.any(inside):
        raise ValueError("The observation domain contains no quadrature nodes.")
    points = points[inside]
    return SpatialQuadrature(points, np.full(points.shape[0], dx * dy))


def partition_midpoint_quadrature(partition, x_bounds, y_bounds, nx, ny=None):
    """Build a midpoint rule that integrates each partition area exactly."""
    if not isinstance(partition, DomainPartition):
        raise TypeError("partition must be a DomainPartition instance.")
    rule = midpoint_quadrature(
        x_bounds,
        y_bounds,
        nx,
        ny,
    )
    # Start from the full rectangle: even when no midpoint hits the partition,
    # representative points below must still supply every small domain.
    inside = contains_xy(
        partition.observation_geometry, rule.points[:, 0], rule.points[:, 1]
    )
    points = np.array(rule.points[inside], copy=True)
    domain_index = partition.locate(points[:, 0], points[:, 1])
    points_per_domain = np.bincount(domain_index, minlength=len(partition))
    missing = np.flatnonzero(points_per_domain == 0)
    if missing.size:
        representatives = [
            partition.polygons[index].representative_point() for index in missing
        ]
        points = np.vstack(
            [points, [(float(point.x), float(point.y)) for point in representatives]]
        )
        domain_index = np.concatenate([domain_index, missing])
        points_per_domain = np.bincount(domain_index, minlength=len(partition))
    weights = partition.areas[domain_index] / points_per_domain[domain_index]
    return SpatialQuadrature(points, weights)
