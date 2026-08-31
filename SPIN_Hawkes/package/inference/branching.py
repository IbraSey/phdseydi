"""Sparse candidate-parent graphs for Hawkes branching variables."""

from dataclasses import dataclass

import numpy as np


def _validate_time_window(value) -> float:
    if isinstance(value, bool):
        raise ValueError("parent_time_window must be a finite positive number.")
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("parent_time_window must be a finite positive number.") from error
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("parent_time_window must be a finite positive number.")
    return value


@dataclass(frozen=True)
class TemporalCandidateGraph:
    """Ragged graph of earlier events retained by a maximum time lag.

    Row ``i`` contains the zero-based indices of events ``j`` satisfying
    ``0 < t[i] - t[j] <= max_lag``. The row structure is stored in CSR form.
    """

    indptr: np.ndarray
    parent_indices: np.ndarray
    child_indices: np.ndarray
    time_lags: np.ndarray
    max_lag: float
    dense_candidate_count: int

    @classmethod
    def from_times(cls, event_times, max_lag) -> "TemporalCandidateGraph":
        times = np.asarray(event_times, dtype=float).reshape(-1)
        if np.any(~np.isfinite(times)):
            raise ValueError("event_times must contain only finite values.")
        if np.any(np.diff(times) < 0.0):
            raise ValueError("event_times must be sorted in non-decreasing order.")
        max_lag = _validate_time_window(max_lag)

        # Equal-time events cannot trigger each other. ``right`` therefore uses
        # the first occurrence of each child time rather than the child index.
        right = np.searchsorted(times, times, side="left")
        lower_time = np.nextafter(times - max_lag, -np.inf)
        left = np.searchsorted(times, lower_time, side="left")
        counts = np.maximum(right - left, 0).astype(np.int64, copy=False)
        indptr = np.empty(times.size + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(counts, out=indptr[1:])

        parent_indices = np.empty(int(indptr[-1]), dtype=np.int64)
        for child in range(times.size):
            start, stop = int(indptr[child]), int(indptr[child + 1])
            if stop > start:
                parent_indices[start:stop] = np.arange(
                    left[child], right[child], dtype=np.int64
                )
        child_indices = np.repeat(np.arange(times.size, dtype=np.int64), counts)
        time_lags = times[child_indices] - times[parent_indices]

        for values in (indptr, parent_indices, child_indices, time_lags):
            values.setflags(write=False)
        return cls(
            indptr=indptr,
            parent_indices=parent_indices,
            child_indices=child_indices,
            time_lags=time_lags,
            max_lag=max_lag,
            dense_candidate_count=int(np.sum(right, dtype=np.int64)),
        )

    @property
    def n_events(self) -> int:
        return int(self.indptr.size - 1)

    @property
    def n_edges(self) -> int:
        return int(self.parent_indices.size)

    @property
    def retained_fraction(self) -> float:
        if self.dense_candidate_count == 0:
            return 1.0
        return self.n_edges / self.dense_candidate_count

    @property
    def memory_bytes(self) -> int:
        return int(
            self.indptr.nbytes
            + self.parent_indices.nbytes
            + self.child_indices.nbytes
            + self.time_lags.nbytes
        )

    def row_slice(self, child_index: int) -> slice:
        child_index = int(child_index)
        if child_index < 0 or child_index >= self.n_events:
            raise IndexError("child_index is outside the candidate graph.")
        return slice(
            int(self.indptr[child_index]),
            int(self.indptr[child_index + 1]),
        )

    def parents_of(self, child_index: int) -> np.ndarray:
        return self.parent_indices[self.row_slice(child_index)]

    def diagnostics(self) -> dict[str, float | int]:
        return {
            "parent_time_window": float(self.max_lag),
            "candidate_parent_count": self.n_edges,
            "dense_candidate_count": self.dense_candidate_count,
            "retained_candidate_fraction": float(self.retained_fraction),
            "candidate_graph_memory_bytes": self.memory_bytes,
        }
