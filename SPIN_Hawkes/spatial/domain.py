"""Spatial-domain objects shared by SPIN-H models and inference engines."""

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.prepared import prep


@dataclass(frozen=True)
class SpatialDomain:
    """One polygonal domain and its initial log-intensity."""

    polygon: object
    initial_log_intensity: float = 0.0

    def __post_init__(self):
        if self.polygon.is_empty or not self.polygon.is_valid:
            raise ValueError("Spatial-domain polygons must be non-empty and valid.")
        if self.polygon.area <= 0:
            raise ValueError("Spatial-domain polygons must have positive area.")
        object.__setattr__(self, "initial_log_intensity", float(self.initial_log_intensity))
        object.__setattr__(self, "prepared_polygon", prep(self.polygon))

    @property
    def area(self) -> float:
        return float(self.polygon.area)

    @property
    def centroid(self) -> tuple[float, float]:
        return float(self.polygon.centroid.x), float(self.polygon.centroid.y)

    def covers(self, x: float, y: float) -> bool:
        return bool(self.prepared_polygon.covers(Point(float(x), float(y))))


class DomainPartition:
    """Validated collection of non-overlapping spatial domains."""

    def __init__(self, domains: Sequence[SpatialDomain]):
        self.domains = tuple(domains)
        if not self.domains:
            raise ValueError("At least one spatial domain is required.")
        for i, left in enumerate(self.domains):
            for right in self.domains[i + 1 :]:
                if left.polygon.intersection(right.polygon).area > 1e-12:
                    raise ValueError("Spatial domains must not overlap in their interiors.")
        self._observation_geometry = unary_union([domain.polygon for domain in self.domains])

    @classmethod
    def from_polygons(
        cls,
        polygons: Iterable[object],
        initial_log_intensities: Sequence[float] | float = 0.0,
    ) -> "DomainPartition":
        polygons = tuple(polygons)
        if np.isscalar(initial_log_intensities):
            eps = np.full(len(polygons), float(initial_log_intensities))
        else:
            eps = np.asarray(initial_log_intensities, dtype=float)
        if len(polygons) != eps.size:
            raise ValueError("One initial log-intensity is required per polygon.")
        return cls(
            [SpatialDomain(polygon, value) for polygon, value in zip(polygons, eps)]
        )

    def __len__(self) -> int:
        return len(self.domains)

    @property
    def polygons(self) -> tuple[object, ...]:
        return tuple(domain.polygon for domain in self.domains)

    @property
    def observation_geometry(self):
        """Shapely union of every domain in the partition."""
        return self._observation_geometry

    @property
    def prepared_domains(self) -> tuple[object, ...]:
        return tuple(domain.prepared_polygon for domain in self.domains)

    @property
    def areas(self) -> np.ndarray:
        return np.asarray([domain.area for domain in self.domains], dtype=float)

    @property
    def centroids(self) -> np.ndarray:
        return np.asarray([domain.centroid for domain in self.domains], dtype=float)

    @property
    def initial_log_intensities(self) -> np.ndarray:
        return np.asarray([domain.initial_log_intensity for domain in self.domains], dtype=float)

    def locate(self, x, y) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        if x.size != y.size:
            raise ValueError("x and y must have the same length.")
        indices = np.full(x.size, -1, dtype=int)
        for i, (xi, yi) in enumerate(zip(x, y)):
            for j, domain in enumerate(self.domains):
                if domain.covers(xi, yi):
                    indices[i] = j
                    break
        return indices

    def validate_points(self, x, y) -> np.ndarray:
        indices = self.locate(x, y)
        outside = np.flatnonzero(indices < 0)
        if outside.size:
            preview = ", ".join(map(str, outside[:5]))
            raise ValueError(f"Events outside every spatial domain at indices: {preview}.")
        return indices

