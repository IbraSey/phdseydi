"""Inference algorithms and numerical GP backends."""

from .backends import SparseGP
from .results import GibbsResults

__all__ = [
    "GibbsResults",
    "SparseGP",
]
