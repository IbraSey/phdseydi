"""Numerical GP representations used by inference engines."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import openturns as ot
from ..data.catalog import EventCatalog
from ..models.ssgc import SSGCModel


"""Fourier-basis approximation of a two-dimensional squared-exponential GP."""

def fourier_mode(x, index, radius):
    """Evaluate a normalized sine basis mode on ``[-radius, radius]``."""
    return np.sin(np.pi * index * (x + radius) / (2.0 * radius)) / np.sqrt(radius)


class SparseGP:
    """Finite Fourier representation of a 2D squared-exponential GP.

    Parameters
    ----------
    hypers : sequence of float, shape (7,)
        ``(ell_x, ell_y, center_x, center_y, radius_x, radius_y, variance)``.
    """

    def __init__(self, hypers):
        ell_x, ell_y, center_x, center_y, radius_x, radius_y, variance = map(float, hypers)
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
        self.regressorOT = ot.MemoizeFunction(
            ot.PythonFunction(2, self.m, self._regressor_point)
        )

    @classmethod
    def from_bounds(cls, x_bounds, y_bounds, variance, length_scale):
        """Construct the basis from rectangular observation bounds."""
        center_x = 0.5 * (x_bounds[0] + x_bounds[1])
        center_y = 0.5 * (y_bounds[0] + y_bounds[1])
        radius_x = 0.5 * (x_bounds[1] - x_bounds[0])
        radius_y = 0.5 * (y_bounds[1] - y_bounds[0])
        return cls((length_scale, length_scale, center_x, center_y,
                    radius_x, radius_y, variance))

    def _regressor_point(self, point):
        phi_x = fourier_mode(point[0] - self.c1, self.S[0], self.L1)
        phi_y = fourier_mode(point[1] - self.c2, self.S[1], self.L2)
        return (phi_x * phi_y * self.sqrt_Delta).tolist()

    def design_matrix(self, points):
        """Return the basis design matrix at one or more 2D points."""
        sample = points if isinstance(points, ot.Sample) else ot.Sample(np.asarray(points).tolist())
        return np.asarray(self.regressorOT(sample), dtype=float)

    def evaluate(self, coeffs, points):
        """Evaluate ``f(points) = Phi(points) @ coeffs``."""
        coeffs = np.asarray(coeffs, dtype=float)
        if coeffs.shape != (self.m,):
            raise ValueError(f"Expected {self.m} coefficients, got shape {coeffs.shape}.")
        return self.design_matrix(points) @ coeffs

    def estimate(self, X, Y):
        """Return the least-squares coefficient estimate."""
        design = self.design_matrix(X)
        target = np.asarray(Y, dtype=float).reshape(-1)
        return np.linalg.lstsq(design, target, rcond=None)[0]




class GPBackend(ABC):
    """Abstract numerical representation of the model's GP prior."""

    name: str

    @abstractmethod
    def initialize(self, model: SSGCModel, catalog: EventCatalog):
        """Create an inference-specific latent GP state."""

    @abstractmethod
    def evaluate(
        self,
        model: SSGCModel,
        catalog: EventCatalog,
        latent_state,
        x,
        y,
    ) -> np.ndarray:
        """Evaluate the represented latent field at query locations."""

    def sampler_options(self, model: SSGCModel) -> dict:
        """Options forwarded to the Gibbs sampler."""
        return {"gp_backend": self.name, "sparse_gp": None}


@dataclass(frozen=True)
class ExactGPBackend(GPBackend):
    """Exact GP values at observed locations with kriging prediction."""

    name: str = "exact"

    def initialize(self, model: SSGCModel, catalog: EventCatalog):
        return np.zeros(len(catalog), dtype=float)

    def evaluate(self, model, catalog, latent_state, x, y) -> np.ndarray:
        latent_state = np.asarray(latent_state, dtype=float).reshape(-1)
        if latent_state.size != len(catalog):
            raise ValueError("Exact GP state must have one value per event.")
        query = np.column_stack(
            np.broadcast_arrays(
                np.asarray(x, dtype=float), np.asarray(y, dtype=float)
            )
        ).reshape(-1, 2)
        train = catalog.xy
        variance = model.gp_prior.variance
        length = model.gp_prior.length_scale

        def covariance(left, right):
            squared_distance = np.sum(
                (left[:, None, :] - right[None, :, :]) ** 2, axis=2
            )
            return variance * np.exp( -squared_distance / (2.0 * length**2) )

        K = covariance(train, train) + model.jitter * np.eye(len(catalog))
        K_query = covariance(query, train)
        return K_query @ np.linalg.solve(K, latent_state)


@dataclass
class FourierSparseGPBackend(GPBackend):
    """Finite Fourier-basis representation of the GP."""

    sparse_gp: SparseGP | None = None
    name: str = "sparse"

    def _basis(self, model: SSGCModel) -> SparseGP:
        if self.sparse_gp is None:
            self.sparse_gp = SparseGP.from_bounds(
                model.x_bounds,
                model.y_bounds,
                variance=model.gp_prior.variance,
                length_scale=model.gp_prior.length_scale,
            )
        return self.sparse_gp

    def initialize(self, model: SSGCModel, catalog: EventCatalog):
        return np.zeros(self._basis(model).m, dtype=float)

    def evaluate(self, model, catalog, latent_state, x, y) -> np.ndarray:
        points = np.column_stack(
            np.broadcast_arrays(
                np.asarray(x, dtype=float), np.asarray(y, dtype=float)
            )
        ).reshape(-1, 2)
        return self._basis(model).evaluate(latent_state, points)

    def sampler_options(self, model: SSGCModel) -> dict:
        return {"gp_backend": self.name, "sparse_gp": self._basis(model)}

