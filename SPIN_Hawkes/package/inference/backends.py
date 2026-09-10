"""Fourier-basis approximation of a two-dimensional squared-exponential GP."""

from functools import partial

import numpy as np
import openturns as ot


def fourier_mode(x, index, radius):
    """Evaluate a normalized sine basis mode on ``[-radius, radius]``."""
    return np.sin(np.pi * index * (x + radius) / (2.0 * radius)) / np.sqrt(radius)


def _basis_values(points, *, center, radii, modes, spectral_scale):
    """Evaluate points or samples without retaining the owning SparseGP object."""
    points = np.asarray(points, dtype=float)
    phi_x = fourier_mode(points[..., 0, None] - center[0], modes[0], radii[0])
    phi_y = fourier_mode(points[..., 1, None] - center[1], modes[1], radii[1])
    return phi_x * phi_y * spectral_scale


class SparseGP:
    """Finite Fourier representation of a 2D squared-exponential GP.

    Parameters
    ----------
    hypers : sequence of float, shape (7,)
        ``(ell_x, ell_y, center_x, center_y, radius_x, radius_y, variance)``.
    """

    def __init__(self, hypers):
        values = np.asarray(hypers, dtype=float).reshape(-1)
        if values.size != 7:
            raise ValueError("hypers must contain exactly seven values.")
        if not np.all(np.isfinite(values)):
            raise ValueError("Sparse-GP hyperparameters must be finite.")
        ell_x, ell_y, center_x, center_y, radius_x, radius_y, variance = values
        if ell_x <= 0 or ell_y <= 0 or radius_x <= 0 or radius_y <= 0 or variance <= 0:
            raise ValueError("Length scales, radii, and variance must be strictly positive.")

        self.l1, self.l2 = ell_x, ell_y
        self.c1, self.c2 = center_x, center_y
        self.L1 = float(np.ceil(max(3.2 * ell_x, 1.2 * radius_x)))
        self.L2 = float(np.ceil(max(3.2 * ell_y, 1.2 * radius_y)))
        self.m1 = max(1, int(np.ceil(1.75 * self.L1 / ell_x)))
        self.m2 = max(1, int(np.ceil(1.75 * self.L2 / ell_y)))
        self.m = self.m1 * self.m2

        # Sine modes are indexed from one; index zero would be identically null.
        indices_x = np.arange(1, self.m1 + 1)
        indices_y = np.arange(1, self.m2 + 1)
        self.S = np.vstack([
            np.repeat(indices_x, self.m2),
            np.tile(indices_y, self.m1),
        ])
        spectral_variance = (
            2.0 * np.pi * variance * ell_x * ell_y
            * np.exp(
                -0.125 * np.pi ** 2
                * ((self.S[0] * ell_x / self.L1) ** 2
                   + (self.S[1] * ell_y / self.L2) ** 2)
            )
        )
        self.sqrt_Delta = np.sqrt(spectral_variance)
        # A bound callback creates a Python/C++ ownership cycle that GC cannot
        # reclaim. This callback owns only the small, immutable basis arrays.
        evaluator = partial(
            _basis_values, center=(self.c1, self.c2), radii=(self.L1, self.L2),
            modes=self.S, spectral_scale=self.sqrt_Delta,
        )
        self.regressorOT = ot.MemoizeFunction(
            ot.PythonFunction(2, self.m, func=evaluator, func_sample=evaluator)
        )
        # Observed designs are stored by inference; changing latent locations
        # must not fill a separate high-dimensional cache on every chain.
        self.regressorOT.disableCache()
        self.regressorOT.disableHistory()

    @classmethod
    def from_bounds(cls, x_bounds, y_bounds, variance, length_scale):
        """Construct the basis from rectangular observation bounds."""
        x_bounds = np.asarray(x_bounds, dtype=float).reshape(-1)
        y_bounds = np.asarray(y_bounds, dtype=float).reshape(-1)
        if x_bounds.size != 2 or y_bounds.size != 2:
            raise ValueError("x_bounds and y_bounds must each contain two values.")
        if not np.all(np.isfinite(np.r_[x_bounds, y_bounds])):
            raise ValueError("Sparse-GP bounds must be finite.")
        center_x = 0.5 * (x_bounds[0] + x_bounds[1])
        center_y = 0.5 * (y_bounds[0] + y_bounds[1])
        radius_x = 0.5 * (x_bounds[1] - x_bounds[0])
        radius_y = 0.5 * (y_bounds[1] - y_bounds[0])
        return cls((length_scale, length_scale, center_x, center_y,
                    radius_x, radius_y, variance))

    def _regressor_point(self, point):
        return _basis_values(
            point, center=(self.c1, self.c2), radii=(self.L1, self.L2),
            modes=self.S, spectral_scale=self.sqrt_Delta,
        ).tolist()
