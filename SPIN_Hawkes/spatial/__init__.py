"""Spatial domains and partitions."""

from .domain import DomainPartition, SpatialDomain
from .quadrature import (
    SpatialQuadrature,
    midpoint_quadrature,
    partition_midpoint_quadrature,
)

__all__ = [
    "DomainPartition",
    "SpatialDomain",
    "SpatialQuadrature",
    "midpoint_quadrature",
    "partition_midpoint_quadrature",
]
