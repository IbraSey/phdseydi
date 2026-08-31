"""Deterministic ETAS kernel components."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from shapely import contains_xy
from shapely.geometry import box

from package.config import ETASParameters


@dataclass(frozen=True)
class ProductivityKernel:
    """Magnitude-dependent expected offspring multiplier."""

    def evaluate(
        self,
        parent_magnitudes,
        parameters: ETASParameters,
        magnitude_min: float,
    ) -> np.ndarray:
        magnitudes = np.asarray(parent_magnitudes, dtype=float)
        values = np.full(magnitudes.shape, parameters.A, dtype=float)
        if parameters.alpha is not None:
            values *= np.exp(
                parameters.alpha * (magnitudes - float(magnitude_min))
            )
        return values


@dataclass(frozen=True)
class OmoriKernel:
    """Normalized temporal Omori-Utsu density."""

    def evaluate(self, delta_t, parameters: ETASParameters) -> np.ndarray:
        delta_t = np.asarray(delta_t, dtype=float)
        values = np.zeros(delta_t.shape, dtype=float)
        valid = delta_t > 0
        values[valid] = (
            (parameters.p - 1.0)
            * parameters.c ** (parameters.p - 1.0)
            * (delta_t[valid] + parameters.c) ** (-parameters.p)
        )
        return values

    def integral_until(
        self,
        parent_times,
        end_time: float,
        parameters: ETASParameters,
    ) -> np.ndarray:
        remaining = np.maximum(
            float(end_time) - np.asarray(parent_times, dtype=float), 0.0
        )
        return 1.0 - (
            parameters.c / (remaining + parameters.c)
        ) ** (parameters.p - 1.0)

    def relative_density(self, delta_t, parameters: ETASParameters) -> np.ndarray:
        """Return ``phi_t(delta_t) / phi_t(0+)`` for non-negative lags."""
        delta_t = np.asarray(delta_t, dtype=float)
        if np.any(~np.isfinite(delta_t)) or np.any(delta_t < 0.0):
            raise ValueError("delta_t must contain finite non-negative values.")
        return (parameters.c / (delta_t + parameters.c)) ** parameters.p

    def tail_mass(self, delta_t, parameters: ETASParameters) -> np.ndarray:
        """Return the normalized temporal mass after each non-negative lag."""
        delta_t = np.asarray(delta_t, dtype=float)
        if np.any(~np.isfinite(delta_t)) or np.any(delta_t < 0.0):
            raise ValueError("delta_t must contain finite non-negative values.")
        return (
            parameters.c / (delta_t + parameters.c)
        ) ** (parameters.p - 1.0)

    def lag_at_relative_density(
        self,
        relative_density: float,
        parameters: ETASParameters,
    ) -> float:
        """Convert a relative kernel-height threshold into a time window."""
        try:
            relative_density = float(relative_density)
        except (TypeError, ValueError) as error:
            raise ValueError("relative_density must lie strictly between 0 and 1.") from error
        if (
            not np.isfinite(relative_density)
            or not 0.0 < relative_density < 1.0
        ):
            raise ValueError("relative_density must lie strictly between 0 and 1.")
        return float(
            parameters.c
            * (relative_density ** (-1.0 / parameters.p) - 1.0)
        )


@dataclass(frozen=True)
class SpatialPowerLawKernel:
    """Normalized isotropic spatial power-law density."""

    def scale(
        self,
        parent_magnitudes,
        parameters: ETASParameters,
        magnitude_min: float,
    ) -> np.ndarray:
        magnitudes = np.asarray(parent_magnitudes, dtype=float)
        scale = np.full(magnitudes.shape, parameters.d, dtype=float)
        if parameters.gamma is not None:
            scale *= np.exp(
                parameters.gamma * (magnitudes - float(magnitude_min))
            )
        return scale

    def evaluate(
        self,
        distance_squared,
        parent_magnitudes,
        parameters: ETASParameters,
        magnitude_min: float,
    ) -> np.ndarray:
        distance_squared = np.asarray(distance_squared, dtype=float)
        scale = self.scale(parent_magnitudes, parameters, magnitude_min)
        return (
            (parameters.q - 1.0)
            / (np.pi * scale)
            * (1.0 + distance_squared / scale) ** (-parameters.q)
        )

    def retained_mass(
        self,
        parent_x,
        parent_y,
        parent_magnitudes,
        parameters: ETASParameters,
        magnitude_min: float,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        n_grid: int = 40,
        observation_domain=None,
    ) -> np.ndarray:
        """Approximate ETAS mass retained by a Shapely observation domain.

        Midpoint Riemann quadrature is evaluated on the rectangular bounding
        window.  ``observation_domain`` may be any Shapely polygonal geometry:
        Polygon, non-convex Polygon, Polygon with holes, or MultiPolygon.  The
        rectangular window is used only when no geometry is supplied.
        """
        if isinstance(n_grid, bool) or not isinstance(n_grid, Integral):
            raise ValueError("n_grid must be an integer.")
        n_grid = int(n_grid)
        if n_grid < 1:
            raise ValueError("n_grid must be positive.")
        xmin, xmax = map(float, x_bounds)
        ymin, ymax = map(float, y_bounds)
        if not xmin < xmax or not ymin < ymax:
            raise ValueError("Spatial bounds must be strictly increasing.")

        dx = (xmax - xmin) / n_grid
        dy = (ymax - ymin) / n_grid
        grid_x, grid_y = np.meshgrid(
            xmin + (np.arange(n_grid) + 0.5) * dx,
            ymin + (np.arange(n_grid) + 0.5) * dy,
        )
        grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        geometry = box(xmin, ymin, xmax, ymax) if observation_domain is None else observation_domain
        if geometry.is_empty or not geometry.is_valid:
            raise ValueError("observation_domain must be a non-empty valid Shapely geometry.")
        mask = contains_xy(geometry, grid_points[:, 0], grid_points[:, 1])
        if not mask.any():
            raise ValueError("The observation domain contains no quadrature cell centres; increase n_grid.")

        points_x = grid_points[mask, 0][None, :]
        points_y = grid_points[mask, 1][None, :]
        parent_x = np.asarray(parent_x, dtype=float).reshape(-1, 1)
        parent_y = np.asarray(parent_y, dtype=float).reshape(-1, 1)
        magnitudes = np.asarray(parent_magnitudes, dtype=float).reshape(-1)
        if not (parent_x.shape[0] == parent_y.shape[0] == magnitudes.size):
            raise ValueError("Parent coordinates and magnitudes must have matching lengths.")

        distance_squared = ((points_x - parent_x) ** 2 + (points_y - parent_y) ** 2)
        density = self.evaluate(
            distance_squared,
            magnitudes[:, None],
            parameters,
            magnitude_min,
        )
        return np.clip(dx * dy * density.sum(axis=1), 1e-8, 1.0)


@dataclass(frozen=True)
class ETASKernel:
    """Composition of productivity, temporal and spatial ETAS kernels."""

    productivity: ProductivityKernel = ProductivityKernel()
    temporal: OmoriKernel = OmoriKernel()
    spatial: SpatialPowerLawKernel = SpatialPowerLawKernel()

    def pairwise(
        self,
        delta_t,
        distance_squared,
        parent_magnitudes,
        parameters: ETASParameters,
        magnitude_min: float,
    ) -> np.ndarray:
        return (
            self.productivity.evaluate(parent_magnitudes, parameters, magnitude_min)
            * self.temporal.evaluate(delta_t, parameters)
            * self.spatial.evaluate(
                distance_squared,
                parent_magnitudes,
                parameters,
                magnitude_min,
            )
        )
