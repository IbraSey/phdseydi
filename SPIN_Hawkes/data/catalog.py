"""Validated event-catalog representation."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class EventCatalog:
    """Time-ordered spatial event catalog with optional magnitudes."""

    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    magnitudes: np.ndarray | None = None

    def __post_init__(self):
        t = np.asarray(self.t, dtype=float).reshape(-1)
        x = np.asarray(self.x, dtype=float).reshape(-1)
        y = np.asarray(self.y, dtype=float).reshape(-1)
        if not (t.size == x.size == y.size):
            raise ValueError("t, x, and y must have the same length.")
        if not np.all(np.isfinite(np.column_stack([t, x, y]))):
            raise ValueError("Event times and coordinates must be finite.")
        if np.any(np.diff(t) < 0):
            raise ValueError("Events must be sorted by non-decreasing time.")
        magnitudes = self.magnitudes
        if magnitudes is not None:
            magnitudes = np.asarray(magnitudes, dtype=float).reshape(-1)
            if magnitudes.size != t.size:
                raise ValueError("One magnitude is required per event.")
            if not np.all(np.isfinite(magnitudes)):
                raise ValueError("Magnitudes must be finite.")
        object.__setattr__(self, "t", t)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "magnitudes", magnitudes)

    def __len__(self) -> int:
        return self.t.size

    @property
    def xy(self) -> np.ndarray:
        return np.column_stack([self.x, self.y])

    def history_before(self, time: float) -> "EventCatalog":
        mask = self.t < float(time)
        magnitudes = None if self.magnitudes is None else self.magnitudes[mask]
        return EventCatalog(self.t[mask], self.x[mask], self.y[mask], magnitudes)

