"""User-facing posterior analysis and inference result objects."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

import numpy as np
import openturns as ot

from ..config import ETASParameters
from ..data.catalog import EventCatalog
from ..models.ssgc import SSGCModel


@dataclass
class PosteriorAnalysis:
    """Posterior summaries and predictions shared by Gibbs result objects."""

    raw: dict
    model: SSGCModel
    catalog: EventCatalog
    default_burn_in: float = 0.3

    def _burn_index(self, n_samples: int, burn_in: float | None = None) -> int:
        burn_in = self.default_burn_in if burn_in is None else burn_in
        if not 0.0 <= burn_in < 1.0:
            raise ValueError("burn_in must be in [0, 1).")
        burn = int(n_samples * burn_in)
        if burn >= n_samples:
            raise ValueError("burn_in leaves no posterior samples.")
        return burn

    def summary(self, burn_in: float | None = None) -> dict:
        """Compute posterior mean estimates from stored Gibbs chains."""
        eps_chain = np.asarray(self.raw["eps"])
        f_chain = np.asarray(self.raw["f_data"])
        nu_chain = np.asarray(self.raw["nu"])
        burn = self._burn_index(eps_chain.shape[0], burn_in)
        summary = {
            "eps_hat": eps_chain[burn:].mean(axis=0),
            "f_data_hat": f_chain[burn:].mean(axis=0),
            "nu_hat": nu_chain[burn:].mean(axis=0),
        }

        if self.raw.get("use_etas", False) and self.raw.get("theta_phi") is not None:
            theta_chain = np.asarray(self.raw["theta_phi"])
            names = self.raw.get("theta_phi_names", [])
            if not names:
                names = ["A", "alpha", "c", "p", "d", "q", "gamma"][:theta_chain.shape[1]]
            summary["theta_phi_hat"] = {
                name: theta_chain[burn:, index].mean()
                for index, name in enumerate(names)
            }
            summary["p_background"] = self.background_probabilities(burn_in)

        if self.raw.get("beta") is not None:
            summary["beta_hat"] = np.asarray(self.raw["beta"])[burn:].mean()
        return summary

    def background_probabilities(self, burn_in: float | None = None) -> np.ndarray:
        """Eventwise posterior probability of being background."""
        branching_chain = self.raw.get("Z")
        if branching_chain is None:
            return np.ones(len(self.catalog))
        branching_chain = np.asarray(branching_chain)
        burn = self._burn_index(branching_chain.shape[0], burn_in)
        return np.mean(branching_chain[burn:] == 0, axis=0)

    def _gp_conditional_mean(self, x_eval, y_eval, f_data_hat, nu_hat) -> np.ndarray:
        """Kriging mean of the latent GP at evaluation coordinates."""
        x_eval = np.asarray(x_eval, dtype=float).reshape(-1)
        y_eval = np.asarray(y_eval, dtype=float).reshape(-1)
        f_data_hat = np.asarray(f_data_hat, dtype=float).reshape(-1)
        n_eval = x_eval.size
        n_obs = len(self.catalog)
        if f_data_hat.size != n_obs:
            raise ValueError("f_data_hat must have one value per observed event.")
        if n_obs == 0:
            return np.zeros(n_eval, dtype=float)

        nu0, nu1 = map(float, np.asarray(nu_hat, dtype=float).reshape(-1))
        if nu0 <= 0.0 or nu1 <= 0.0:
            raise ValueError("nu_hat must contain positive GP variance and length scale.")
        sigma_amp = np.sqrt(nu0)
        kernel = ot.SquaredExponential([nu1, nu1], [sigma_amp])

        observed = ot.Sample(np.column_stack([self.catalog.x, self.catalog.y]).tolist())
        evaluation = ot.Sample(np.column_stack([x_eval, y_eval]).tolist())
        all_points = ot.Sample(n_obs + n_eval, 2)
        observed_array = np.asarray(observed)
        evaluation_array = np.asarray(evaluation)
        for index in range(n_obs):
            all_points[index, 0] = float(observed_array[index, 0])
            all_points[index, 1] = float(observed_array[index, 1])
        for index in range(n_eval):
            all_points[n_obs + index, 0] = float(evaluation_array[index, 0])
            all_points[n_obs + index, 1] = float(evaluation_array[index, 1])

        covariance = np.asarray(kernel.discretize(all_points), dtype=float)
        K_obs = ot.CovarianceMatrix(covariance[:n_obs, :n_obs].tolist())
        for index in range(n_obs):
            K_obs[index, index] += self.model.jitter
        K_eval_obs = ot.Matrix(covariance[n_obs:, :n_obs].tolist())
        alpha_gp = K_obs.solveLinearSystem(ot.Point(f_data_hat.tolist()))
        return np.asarray(K_eval_obs * alpha_gp, dtype=float).reshape(-1)

    def background_intensity(self, x, y, burn_in: float | None = None) -> np.ndarray:
        """Posterior-mean background intensity at spatial coordinates."""
        summary = self.summary(burn_in)
        x_eval, y_eval = np.broadcast_arrays(
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
        )
        x_flat = x_eval.reshape(-1)
        y_flat = y_eval.reshape(-1)
        f_eval = self._gp_conditional_mean(
            x_flat,
            y_flat,
            summary["f_data_hat"],
            summary["nu_hat"],
        )
        return self.model.background_intensity(
            x_flat,
            y_flat,
            summary["eps_hat"],
            f_eval,
        ).reshape(x_eval.shape)

    def conditional_intensity(self, t, x, y, burn_in: float | None = None):
        """Posterior-mean conditional SPIN-H intensity at evaluation points."""
        if not hasattr(self.model, "triggering_intensity"):
            raise TypeError("Conditional ETAS intensity is unavailable for this model.")
        summary = self.summary(burn_in)
        if "theta_phi_hat" not in summary:
            raise TypeError("Conditional ETAS intensity requires theta_phi samples.")

        t_eval, x_eval, y_eval = np.broadcast_arrays(
            np.asarray(t, dtype=float),
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
        )
        shape = t_eval.shape
        t_flat = t_eval.reshape(-1)
        x_flat = x_eval.reshape(-1)
        y_flat = y_eval.reshape(-1)

        f_eval = self._gp_conditional_mean(
            x_flat,
            y_flat,
            summary["f_data_hat"],
            summary["nu_hat"],
        )
        background = self.model.background_intensity(
            x_flat,
            y_flat,
            summary["eps_hat"],
            f_eval,
        ).reshape(-1)

        parameters = ETASParameters(**summary["theta_phi_hat"])
        triggering = self.model.triggering_intensity(
            t_flat,
            x_flat,
            y_flat,
            self.catalog,
            parameters,
        ).reshape(-1)
        total = background + triggering
        return background.reshape(shape), triggering.reshape(shape), total.reshape(shape)


@dataclass
class GibbsResults(Mapping):
    """Typed view over the Gibbs output dictionary."""

    raw: dict
    model: SSGCModel
    catalog: EventCatalog
    sampler: object = field(repr=False)
    default_burn_in: float = 0.3
    _analysis: PosteriorAnalysis | None = field(default=None, init=False, repr=False)

    @property
    def analysis(self) -> PosteriorAnalysis:
        if self._analysis is None:
            self._analysis = PosteriorAnalysis(
                self.raw,
                self.model,
                self.catalog,
                self.default_burn_in,
            )
        return self._analysis

    def __getitem__(self, key):
        return self.raw[key]

    def __iter__(self) -> Iterator:
        return iter(self.raw)

    def __len__(self) -> int:
        return len(self.raw)

    @property
    def eps_chain(self) -> np.ndarray:
        return np.asarray(self.raw["eps"])

    @property
    def latent_point_counts(self) -> np.ndarray:
        return np.asarray(self.raw["nPi"])

    @property
    def branching_chain(self) -> np.ndarray | None:
        values = self.raw.get("Z")
        return None if values is None else np.asarray(values)

    @property
    def etas_chain(self) -> np.ndarray | None:
        values = self.raw.get("theta_phi")
        return None if values is None else np.asarray(values)

    @property
    def acceptance_rates(self) -> dict:
        rates = {"eps": self.raw.get("acceptance_eps")}
        if self.raw.get("acceptance_nu") is not None:
            rates["nu"] = self.raw["acceptance_nu"]
        if self.raw.get("acceptance_beta") is not None:
            rates["beta"] = self.raw["acceptance_beta"]
        rates.update(self.raw.get("acceptance_etas") or {})
        return rates

    def summary(self, burn_in: float | None = None) -> dict:
        return self.analysis.summary(burn_in)

    def background_probabilities(
        self, burn_in: float | None = None
    ) -> np.ndarray:
        return self.analysis.background_probabilities(burn_in)

    def background_intensity(self, x, y, burn_in: float | None = None) -> np.ndarray:
        return self.analysis.background_intensity(x, y, burn_in)

    def conditional_intensity(
        self,
        t,
        x,
        y,
        burn_in: float | None = None,
    ):
        return self.analysis.conditional_intensity(t, x, y, burn_in)

    def plot_traces(self, burn_in: float | None = None, **kwargs):
        burn_in = self.default_burn_in if burn_in is None else burn_in
        if self.etas_chain is not None:
            return self.sampler.plot_etas_chains(
                self.raw, burn_in=burn_in, **kwargs
            )
        return self.sampler.plot_chains(
            self.raw, burn_in=burn_in, **kwargs
        )

    def plot_declustering(self, burn_in: float | None = None, **kwargs):
        if self.branching_chain is None:
            raise TypeError("Declustering is unavailable for an SSGC-only model.")
        burn_in = self.default_burn_in if burn_in is None else burn_in
        kwargs.setdefault("magnitudes", self.catalog.magnitudes)
        return self.sampler.plot_declustering(
            self.catalog.x,
            self.catalog.y,
            self.catalog.t,
            self.raw,
            burn_in=burn_in,
            **kwargs,
        )
