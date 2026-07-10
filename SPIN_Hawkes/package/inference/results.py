"""User-facing Gibbs posterior result object."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import openturns as ot
from scipy.special import expit

from package.config import ETASParameters
from data.catalog import EventCatalog
from ..models.ssgc import SSGCModel
from visualization import plot_field, save_figure


@dataclass
class SPINHVIResults:
    """User-facing result object returned by SPINHVI.fit()."""

    state: Any
    model: Any
    catalog: EventCatalog
    config: Any
    elbo_trace: list[float]
    diagnostics: dict = field(default_factory=dict)

    def beta_mean(self) -> float:
        return float(self.state.etas.beta_mean)

    def etas_mean(self) -> ETASParameters:
        return self.state.etas.parameters_mean

    def summary(self) -> dict:
        """Return posterior means and diagnostics in a compact dictionary."""
        return {
            "eps_mean": self.state.eps.mean.copy(),
            "eps_covariance": self.state.eps.covariance.copy(),
            "f_data_mean": self.state.gp.f_data_mean.copy(),
            "f_data_var": self.state.gp.f_data_var.copy(),
            "f_grid_mean": self.state.gp.f_grid_mean.copy(),
            "f_grid_var": self.state.gp.f_grid_var.copy(),
            "f_covariance": None if self.state.gp.covariance is None else self.state.gp.covariance.copy(),
            "gp_backend": getattr(self.config, "gp_backend", "exact"),
            "gp_coefficients_mean": (
                None if self.state.gp.coefficients_mean is None
                else self.state.gp.coefficients_mean.copy()
            ),
            "gp_coefficients_covariance": (
                None if self.state.gp.coefficients_covariance is None
                else self.state.gp.coefficients_covariance.copy()
            ),
            "p_background": self.state.branching.p_background.copy(),
            "parent_probabilities": self.state.branching.probabilities.copy(),
            "theta_phi_hat": self.state.etas.parameters_mean.as_dict(),
            "beta_hat": self.beta_mean(),
            "latent_poisson_expected_counts": (
                self.state.latent_poisson.expected_counts_by_domain.copy()
            ),
            "elbo_trace": np.asarray(self.elbo_trace, dtype=float),
            "diagnostics": dict(self.diagnostics),
        }

    def declustering(self, background_threshold: float = 0.5) -> dict:
        """Return a two-stage declustering decision from q(Z)."""
        probabilities = self.state.branching.probabilities
        p_background = probabilities[:, 0]
        labels = np.zeros(probabilities.shape[0], dtype=int)
        parent = np.full(probabilities.shape[0], -1, dtype=int)
        triggered = p_background < background_threshold
        for i in np.where(triggered)[0]:
            if i == 0:
                labels[i] = 0
                continue
            j = int(np.argmax(probabilities[i, 1 : i + 1]))
            labels[i] = 1
            parent[i] = j
        return {
            "p_background": p_background.copy(),
            "is_background": ~triggered,
            "parent": parent,
            "labels": labels,
        }


@dataclass
class GibbsResults(Mapping):
    """Gibbs chains, posterior summaries, predictions and diagnostics."""

    raw: dict
    model: SSGCModel
    catalog: EventCatalog
    default_burn_in: float = 0.3

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
        gp_coeffs = self.raw.get("gp_coeffs")
        if gp_coeffs is not None:
            gp_coeffs = np.asarray(gp_coeffs, dtype=float)
            if gp_coeffs.size:
                summary["gp_coeffs_hat"] = gp_coeffs[burn:].mean(axis=0)

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

    def _gp_conditional_mean_from_points(
        self, x_eval, y_eval, x_obs, y_obs, f_obs, nu_hat
    ) -> np.ndarray:
        """Kriging mean of the latent GP from selected conditioning points."""
        x_eval = np.asarray(x_eval, dtype=float).reshape(-1)
        y_eval = np.asarray(y_eval, dtype=float).reshape(-1)
        x_obs = np.asarray(x_obs, dtype=float).reshape(-1)
        y_obs = np.asarray(y_obs, dtype=float).reshape(-1)
        f_obs = np.asarray(f_obs, dtype=float).reshape(-1)
        n_eval = x_eval.size
        n_obs = x_obs.size
        if not (n_obs == y_obs.size == f_obs.size):
            raise ValueError("x_obs, y_obs and f_obs must have matching lengths.")
        if n_obs == 0:
            return np.zeros(n_eval, dtype=float)

        nu0, nu1 = map(float, np.asarray(nu_hat, dtype=float).reshape(-1))
        if nu0 <= 0.0 or nu1 <= 0.0:
            raise ValueError("nu_hat must contain positive GP variance and length scale.")
        kernel = ot.SquaredExponential([nu1, nu1], [np.sqrt(nu0)])

        observed_array = np.column_stack([x_obs, y_obs])
        evaluation_array = np.column_stack([x_eval, y_eval])
        all_points = ot.Sample(np.vstack([observed_array, evaluation_array]).tolist())
        covariance = np.asarray(kernel.discretize(all_points), dtype=float)
        K_obs = ot.CovarianceMatrix(covariance[:n_obs, :n_obs].tolist())
        for index in range(n_obs):
            K_obs[index, index] += self.model.jitter
        K_eval_obs = ot.Matrix(covariance[n_obs:, :n_obs].tolist())
        alpha_gp = K_obs.solveLinearSystem(ot.Point(f_obs.tolist()))
        return np.asarray(K_eval_obs * alpha_gp, dtype=float).reshape(-1)

    def _gp_conditional_mean(self, x_eval, y_eval, f_data_hat, nu_hat) -> np.ndarray:
        """Kriging mean of the latent GP at evaluation coordinates."""
        return self._gp_conditional_mean_from_points(
            x_eval, y_eval, self.catalog.x, self.catalog.y, f_data_hat, nu_hat
        )

    def background_intensity(self, x, y, burn_in: float | None = None) -> np.ndarray:
        """Posterior-mean background intensity at spatial coordinates."""
        summary = self.summary(burn_in)
        x_eval, y_eval = np.broadcast_arrays(
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
        )
        x_flat = x_eval.reshape(-1)
        y_flat = y_eval.reshape(-1)
        return self._posterior_background_flat(
            x_flat, y_flat, summary, burn_in
        ).reshape(x_eval.shape)

    def background_intensity_samples(
        self,
        x,
        y,
        burn_in: float | None = None,
        n_samples: int = 500,
    ) -> np.ndarray:
        """Draw background-intensity fields at arbitrary spatial coordinates."""
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")

        summary = self.summary(burn_in)
        x_eval, y_eval = np.broadcast_arrays(
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
        )
        x_flat = x_eval.reshape(-1)
        y_flat = y_eval.reshape(-1)

        eps_chain = np.asarray(self.raw.get("eps", []), dtype=float)
        sparse_gp = self.raw.get("sparse_gp")
        gp_coeffs = self.raw.get("gp_coeffs")
        if sparse_gp is not None and gp_coeffs is not None and eps_chain.ndim == 2:
            gp_coeffs = np.asarray(gp_coeffs, dtype=float)
            if gp_coeffs.ndim == 2 and gp_coeffs.shape[0] >= eps_chain.shape[0]:
                burn = self._burn_index(eps_chain.shape[0], burn_in)
                available_indices = np.arange(burn, eps_chain.shape[0])
                positions = np.linspace(
                    0, available_indices.size - 1, int(n_samples)
                ).round().astype(int)
                draw_indices = available_indices[positions]
                points = ot.Sample(np.column_stack([x_flat, y_flat]).tolist())
                design = np.asarray(sparse_gp.regressorOT(points), dtype=float)
                samples = np.empty((x_flat.size, draw_indices.size), dtype=float)
                for column, index in enumerate(draw_indices):
                    latent_gp = design @ gp_coeffs[index]
                    samples[:, column] = self.model.background_intensity(
                        x_flat, y_flat, eps_chain[index], latent_gp
                    ).reshape(-1)
                return samples

        evaluation_xy = np.column_stack([x_flat, y_flat])

        nu0, nu1 = map(float, np.asarray(summary["nu_hat"]).reshape(-1))
        kernel = ot.SquaredExponential([nu1, nu1], [np.sqrt(nu0)])
        all_points = ot.Sample(
            np.vstack([self.catalog.xy, evaluation_xy]).tolist()
        )
        covariance = np.asarray(kernel.discretize(all_points), dtype=float)
        n_obs = len(self.catalog)
        K_obs = ot.CovarianceMatrix(covariance[:n_obs, :n_obs].tolist())
        for index in range(n_obs):
            K_obs[index, index] += self.model.jitter
        K_eval_obs = ot.Matrix(covariance[n_obs:, :n_obs].tolist())
        K_eval = covariance[n_obs:, n_obs:]

        f_data_hat = ot.Point(np.asarray(summary["f_data_hat"]).tolist())
        mean = np.asarray(
            K_eval_obs * K_obs.solveLinearSystem(f_data_hat), dtype=float
        ).reshape(-1)
        solved_cross = K_obs.solveLinearSystem(
            ot.Matrix(np.asarray(K_eval_obs).T.tolist())
        )
        conditional_covariance = K_eval - np.asarray(K_eval_obs * solved_cross)
        conditional_covariance = 0.5 * (
            conditional_covariance + conditional_covariance.T
        )

        latent_samples = None
        last_error = None
        for jitter in (self.model.jitter, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4):
            try:
                covariance_ot = ot.CovarianceMatrix(
                    (conditional_covariance + jitter * np.eye(x_flat.size)).tolist()
                )
                latent_samples = np.asarray(
                    ot.Normal(ot.Point(mean.tolist()), covariance_ot).getSample(n_samples)
                ).T
                break
            except Exception as error:
                last_error = error
        if latent_samples is None:
            raise RuntimeError(
                "OpenTURNS could not sample the posterior GP."
            ) from last_error

        baseline = self.model.baseline_intensity(
            x_flat, y_flat, summary["eps_hat"]
        )
        return baseline[:, None] * expit(latent_samples)

    @staticmethod
    def _thin_indices(indices, max_draws=200):
        indices = np.asarray(indices, dtype=int).reshape(-1)
        if indices.size <= max_draws:
            return indices
        selected = np.linspace(0, indices.size - 1, int(max_draws)).round().astype(int)
        return np.unique(indices[selected])

    def _posterior_background_flat(
        self, x_flat, y_flat, summary, burn_in: float | None = None, max_draws=200
    ) -> np.ndarray:
        eps_chain = np.asarray(self.raw.get("eps", []), dtype=float)
        if eps_chain.ndim != 2 or eps_chain.shape[0] == 0:
            f_eval = self._gp_conditional_mean(
                x_flat, y_flat, summary["f_data_hat"], summary["nu_hat"]
            )
            return self.model.background_intensity(
                x_flat, y_flat, summary["eps_hat"], f_eval
            ).reshape(-1)

        burn = self._burn_index(eps_chain.shape[0], burn_in)
        draw_indices = self._thin_indices(np.arange(burn, eps_chain.shape[0]), max_draws)
        sparse_gp = self.raw.get("sparse_gp")
        gp_coeffs = self.raw.get("gp_coeffs")
        if sparse_gp is not None and gp_coeffs is not None:
            gp_coeffs = np.asarray(gp_coeffs, dtype=float)
            if gp_coeffs.size:
                points = ot.Sample(np.column_stack([x_flat, y_flat]).tolist())
                design = np.asarray(sparse_gp.regressorOT(points), dtype=float)
                mu_sum = np.zeros(np.asarray(x_flat).size, dtype=float)
                for index in draw_indices:
                    f_eval = design @ gp_coeffs[index]
                    mu_sum += self.model.background_intensity(
                        x_flat, y_flat, eps_chain[index], f_eval
                    ).reshape(-1)
                return mu_sum / draw_indices.size

        branching_chain = self.branching_chain
        f_chain = np.asarray(self.raw.get("f_data", []), dtype=float)
        nu_chain = np.asarray(self.raw.get("nu", []), dtype=float)
        if (
            branching_chain is not None
            and f_chain.ndim == 2
            and nu_chain.ndim == 2
            and f_chain.shape[0] >= eps_chain.shape[0]
            and nu_chain.shape[0] >= eps_chain.shape[0]
        ):
            mu_sum = np.zeros(np.asarray(x_flat).size, dtype=float)
            used = 0
            for index in draw_indices:
                mask = np.asarray(branching_chain[index], dtype=int) == 0
                if not np.any(mask):
                    continue
                f_eval = self._gp_conditional_mean_from_points(
                    x_flat, y_flat,
                    self.catalog.x[mask], self.catalog.y[mask],
                    f_chain[index, mask], nu_chain[index],
                )
                mu_sum += self.model.background_intensity(
                    x_flat, y_flat, eps_chain[index], f_eval
                ).reshape(-1)
                used += 1
            if used:
                return mu_sum / used

        f_eval = self._gp_conditional_mean(
            x_flat, y_flat, summary["f_data_hat"], summary["nu_hat"]
        )
        return self.model.background_intensity(
            x_flat, y_flat, summary["eps_hat"], f_eval
        ).reshape(-1)

    def _posterior_etas_parameters(self, summary) -> ETASParameters:
        if "theta_phi_hat" not in summary:
            raise TypeError("Conditional ETAS intensity requires theta_phi samples.")
        return ETASParameters(**summary["theta_phi_hat"])

    def conditional_intensity(self, t, x, y, burn_in: float | None = None):
        """Posterior-mean conditional SPIN-H intensity at evaluation points."""
        if not hasattr(self.model, "triggering_intensity"):
            raise TypeError("Conditional ETAS intensity is unavailable for this model.")
        summary = self.summary(burn_in)
        parameters = self._posterior_etas_parameters(summary)

        t_eval, x_eval, y_eval = np.broadcast_arrays(
            np.asarray(t, dtype=float),
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
        )
        shape = t_eval.shape
        t_flat = t_eval.reshape(-1)
        x_flat = x_eval.reshape(-1)
        y_flat = y_eval.reshape(-1)

        background = self._posterior_background_flat(x_flat, y_flat, summary, burn_in)
        triggering = self.model.triggering_intensity(
            t_flat,
            x_flat,
            y_flat,
            self.catalog,
            parameters,
        ).reshape(-1)
        total = background + triggering
        return background.reshape(shape), triggering.reshape(shape), total.reshape(shape)

    def _conditional_intensity_grid(self, times, burn_in, nx, ny):
        if not hasattr(self.model, "triggering_intensity"):
            raise TypeError("Conditional intensity plots require a SPIN-H model.")

        summary = self.summary(burn_in)
        parameters = self._posterior_etas_parameters(summary)
        times = np.asarray(times, dtype=float).reshape(-1)
        if times.size == 0:
            raise ValueError("times must contain at least one value.")
        if not np.all(np.isfinite(times)):
            raise ValueError("times must contain only finite values.")
        if int(nx) < 2 or int(ny) < 2:
            raise ValueError("nx and ny must be at least 2.")

        x_grid = np.linspace(self.model.x_bounds[0], self.model.x_bounds[1], int(nx))
        y_grid = np.linspace(self.model.y_bounds[0], self.model.y_bounds[1], int(ny))
        X, Y = np.meshgrid(x_grid, y_grid)
        x_flat = X.reshape(-1)
        y_flat = Y.reshape(-1)
        background = self._posterior_background_flat(
            x_flat, y_flat, summary, burn_in
        ).reshape(Y.shape)

        triggering_frames = []
        total_frames = []
        for time in times:
            triggering = self.model.triggering_intensity(
                np.full(x_flat.size, time),
                x_flat,
                y_flat,
                self.catalog,
                parameters,
            ).reshape(Y.shape)
            triggering_frames.append(triggering)
            total_frames.append(background + triggering)

        return {
            "times": times,
            "x_grid": X,
            "y_grid": Y,
            "background": background,
            "triggering": np.asarray(triggering_frames),
            "total": np.asarray(total_frames),
        }

    @staticmethod
    def _as_spatial_grid(name, values, shape):
        if values is None:
            return None
        array = np.asarray(values, dtype=float)
        if array.shape == shape:
            return array
        if array.size == int(np.prod(shape)):
            return array.reshape(shape)
        expected_size = int(np.prod(shape))
        raise ValueError(
            f"{name} must have shape {shape} "
            f"or be flat with {expected_size} values."
        )

    @staticmethod
    def _as_time_grids(name, values, shape):
        if values is None:
            return None
        array = np.asarray(values, dtype=float)
        if array.shape == shape:
            return array
        if array.ndim == 2 and shape[0] == 1 and array.shape == shape[1:]:
            return array[None, :, :]
        if array.size == int(np.prod(shape)):
            return array.reshape(shape)
        expected_size = int(np.prod(shape))
        raise ValueError(
            f"{name} must have shape {shape} "
            f"or be flat with {expected_size} values."
        )

    @staticmethod
    def _color_upper(values, color_quantile):
        value = float(np.nanquantile(values, color_quantile))
        if value <= 0.0:
            value = float(np.nanmax(values))
        return value if value > 0.0 else 1.0

    def plot_conditional_intensity_snapshots(
        self,
        times=None,
        burn_in: float | None = None,
        nx=50,
        ny=50,
        cmap_background="viridis",
        cmap_triggering="magma",
        cmap_total="inferno",
        color_quantile=0.98,
        true_background=None,
        true_triggering=None,
        true_total=None,
        figsize=None,
        show=True,
        savefigure=False,
        title_savefig="spinh_conditional_intensity_snapshots",
    ):
        """Plot static SPIN-H intensity snapshots for interactive sessions.

        Optional ``true_background``, ``true_triggering`` and ``true_total`` arrays
        can be supplied when simulated ground truth is available. They must be
        evaluated on the same grid and times as the posterior snapshots.
        """
        if times is None:
            times = np.linspace(
                0.2 * float(self.model.duration),
                float(self.model.duration),
                4,
            )
        grids = self._conditional_intensity_grid(times, burn_in, nx, ny)
        times = grids["times"]
        X = grids["x_grid"]
        Y = grids["y_grid"]
        background = grids["background"]
        triggering = grids["triggering"]
        total = grids["total"]

        true_background = self._as_spatial_grid(
            "true_background", true_background, background.shape
        )
        true_triggering = self._as_time_grids(
            "true_triggering", true_triggering, triggering.shape
        )
        true_total = self._as_time_grids("true_total", true_total, total.shape)
        has_truth = any(
            value is not None
            for value in (true_background, true_triggering, true_total)
        )

        n_times = times.size
        if not 0.0 < color_quantile <= 1.0:
            raise ValueError("color_quantile must be in (0, 1].")

        bg_values = (
            background
            if true_background is None
            else np.r_[background.ravel(), true_background.ravel()]
        )
        trig_values = (
            triggering
            if true_triggering is None
            else np.r_[triggering.ravel(), true_triggering.ravel()]
        )
        total_values = (
            total
            if true_total is None
            else np.r_[total.ravel(), true_total.ravel()]
        )
        bg_vmax = self._color_upper(bg_values, color_quantile)
        trig_vmax = self._color_upper(trig_values, color_quantile)
        total_vmax = self._color_upper(total_values, color_quantile)

        def setup_axis(ax):
            ax.set_xlim(self.model.x_bounds)
            ax.set_ylim(self.model.y_bounds)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.grid(alpha=0.2)

        def add_panel(ax, values, cmap, vmax, title, label):
            mesh = ax.pcolormesh(
                X, Y, values, shading="auto", cmap=cmap, vmin=0.0, vmax=vmax
            )
            fig.colorbar(mesh, ax=ax, label=label)
            ax.set_title(title)
            setup_axis(ax)
            return mesh

        if not has_truth:
            if figsize is None:
                figsize = (3.6 * (n_times + 1), 6.8)
            fig = plt.figure(figsize=figsize, constrained_layout=True)
            grid = fig.add_gridspec(2, n_times + 1)

            ax_bg = fig.add_subplot(grid[:, 0])
            mesh = ax_bg.pcolormesh(
                X, Y, background, shading="auto", cmap=cmap_background
            )
            fig.colorbar(mesh, ax=ax_bg, label=r"$\mu(x, y)$")
            ax_bg.set_title("Background")

            axes_triggering = []
            axes_total = []
            for index, time in enumerate(times):
                ax_trig = fig.add_subplot(grid[0, index + 1])
                add_panel(
                    ax_trig,
                    triggering[index],
                    cmap_triggering,
                    trig_vmax,
                    f"Triggering | t={time:.3g}",
                    r"$g(t, x, y)$",
                )
                axes_triggering.append(ax_trig)

                ax_total = fig.add_subplot(grid[1, index + 1])
                add_panel(
                    ax_total,
                    total[index],
                    cmap_total,
                    total_vmax,
                    f"Total | t={time:.3g}",
                    r"$\lambda(t, x, y)$",
                )
                axes_total.append(ax_total)

            setup_axis(ax_bg)
            axes = {
                "background": ax_bg,
                "triggering": axes_triggering,
                "total": axes_total,
            }
        else:
            if figsize is None:
                figsize = (3.6 * (n_times + 1), 12.2)
            fig = plt.figure(figsize=figsize, constrained_layout=True)
            grid = fig.add_gridspec(4, n_times + 1)

            axes_triggering_est = []
            axes_triggering_true = []
            axes_total_est = []
            axes_total_true = []

            ax_bg_est = fig.add_subplot(grid[0, 0])
            add_panel(
                ax_bg_est,
                background,
                cmap_background,
                bg_vmax,
                "Estimated background",
                r"$\mu(x, y)$",
            )

            ax_bg_true = fig.add_subplot(grid[1, 0])
            if true_background is None:
                ax_bg_true.axis("off")
                ax_bg_true.set_title("True background not provided")
            else:
                add_panel(
                    ax_bg_true,
                    true_background,
                    cmap_background,
                    bg_vmax,
                    "True background",
                    r"$\mu(x, y)$",
                )

            for row in (2, 3):
                ax_empty = fig.add_subplot(grid[row, 0])
                ax_empty.axis("off")

            for index, time in enumerate(times):
                ax_trig_est = fig.add_subplot(grid[0, index + 1])
                add_panel(
                    ax_trig_est,
                    triggering[index],
                    cmap_triggering,
                    trig_vmax,
                    f"Estimated triggering | t={time:.3g}",
                    r"$g(t, x, y)$",
                )
                axes_triggering_est.append(ax_trig_est)

                ax_trig_true = fig.add_subplot(grid[1, index + 1])
                if true_triggering is None:
                    ax_trig_true.axis("off")
                    ax_trig_true.set_title(
                        f"True triggering not provided | t={time:.3g}"
                    )
                else:
                    add_panel(
                        ax_trig_true,
                        true_triggering[index],
                        cmap_triggering,
                        trig_vmax,
                        f"True triggering | t={time:.3g}",
                        r"$g(t, x, y)$",
                    )
                axes_triggering_true.append(ax_trig_true)

                ax_total_est = fig.add_subplot(grid[2, index + 1])
                add_panel(
                    ax_total_est,
                    total[index],
                    cmap_total,
                    total_vmax,
                    f"Estimated total | t={time:.3g}",
                    r"$\lambda(t, x, y)$",
                )
                axes_total_est.append(ax_total_est)

                ax_total_true = fig.add_subplot(grid[3, index + 1])
                if true_total is None:
                    ax_total_true.axis("off")
                    ax_total_true.set_title(
                        f"True total not provided | t={time:.3g}"
                    )
                else:
                    add_panel(
                        ax_total_true,
                        true_total[index],
                        cmap_total,
                        total_vmax,
                        f"True total | t={time:.3g}",
                        r"$\lambda(t, x, y)$",
                    )
                axes_total_true.append(ax_total_true)

            axes = {
                "background": ax_bg_est,
                "true_background": ax_bg_true,
                "triggering": axes_triggering_est,
                "true_triggering": axes_triggering_true,
                "total": axes_total_est,
                "true_total": axes_total_true,
            }

        if savefigure:
            save_figure(fig, title_savefig, figure_type="raster")
        if show:
            plt.show()

        output = {
            "fig": fig,
            "axes": axes,
            **grids,
        }
        if true_background is not None:
            output["true_background"] = true_background
        if true_triggering is not None:
            output["true_triggering"] = true_triggering
        if true_total is not None:
            output["true_total"] = true_total
        return output

    def _make_mesh(self, nx, ny):
        xmin, xmax = self.model.x_bounds
        ymin, ymax = self.model.y_bounds
        interval = ot.Interval([xmin, ymin], [xmax, ymax])
        mesh = ot.IntervalMesher([nx - 1, ny - 1]).build(interval)
        vertices = np.asarray(mesh.getVertices(), dtype=float)
        return mesh, vertices

    def posterior_intensity(
        self,
        burn_in: float | None = None,
        nx=70,
        ny=70,
        cmap="viridis",
        event_cmap="plasma",
        savefigure=False,
        title_savefig="posterior",
        savefigure_Emu=False,
        title_savefig_Emu="Emu",
        color_Emu="steelblue",
        mu_star_func=None,
        n_mc=500,
    ):
        """Plot and summarize the posterior SSGC background intensity."""
        burn_in = self.default_burn_in if burn_in is None else burn_in
        mesh, vertices = self._make_mesh(nx, ny)
        if vertices.shape[0] > 22500:
            raise ValueError(f"Mesh too large: {vertices.shape[0]} points")

        x_grid = vertices[:, 0]
        y_grid = vertices[:, 1]
        mu_hat_sims = self.background_intensity_samples(
            x_grid,
            y_grid,
            burn_in=burn_in,
            n_samples=n_mc,
        )
        mu_hat = mu_hat_sims.mean(axis=1)
        squared_mu_hat = (mu_hat_sims ** 2).mean(axis=1)
        var_mu_hat = squared_mu_hat - mu_hat**2
        std_mu_hat = np.sqrt(np.maximum(var_mu_hat, 0.0))
        lower_mu_hat = np.quantile(mu_hat_sims, 0.025, axis=1)
        upper_mu_hat = np.quantile(mu_hat_sims, 0.975, axis=1)

        mu_hat_field = ot.Field(mesh, ot.Sample([[val] for val in mu_hat]))
        std_mu_hat_field = ot.Field(mesh, ot.Sample([[val] for val in std_mu_hat]))
        lower_mu_hat_field = ot.Field(mesh, ot.Sample([[val] for val in lower_mu_hat]))
        upper_mu_hat_field = ot.Field(mesh, ot.Sample([[val] for val in upper_mu_hat]))

        E_mu_full = np.asarray(self.raw.get("E_mu", []), dtype=float)
        E_mu_bar = None
        E_mu_post = np.array([], dtype=float)
        if E_mu_full.size:
            mask = np.isfinite(E_mu_full)
            E_mu_post = E_mu_full[mask]
            iters_post = np.where(mask)[0]
            if E_mu_post.size:
                E_mu_bar = float(E_mu_post.mean())
                fig_err, ax_err = plt.subplots(figsize=(9, 3))
                ax_err.plot(iters_post, E_mu_post, linewidth=0.8, color=color_Emu)
                ax_err.set_xlabel("Iteration")
                ax_err.set_ylabel(r"$\mathcal{E}_\mu^{(t)}$")
                ax_err.set_title(r"$L^2$ reconstruction error $\mathcal{E}_\mu^{(t)}$")
                ax_err.grid(alpha=0.3)
                plt.tight_layout()
                if savefigure_Emu:
                    save_figure(fig_err, title_savefig_Emu)
                plt.show()

        mu_star_grid = None
        diff = None
        rmse = mae = crps_bar = None
        if mu_star_func is not None:
            mu_star_grid = np.asarray(mu_star_func(x_grid, y_grid), dtype=float).reshape(-1)
            if mu_star_grid.size != mu_hat.size:
                raise ValueError("mu_star_func must return one value per grid point.")
            mu_star_field = ot.Field(mesh, ot.Sample([[val] for val in mu_star_grid]))
            diff = np.abs(mu_hat - mu_star_grid) / (mu_star_grid + self.model.jitter)
            diff_field = ot.Field(mesh, ot.Sample([[val] for val in diff]))
            rmse = float(np.sqrt(np.mean((mu_hat - mu_star_grid) ** 2)))
            mae = float(np.mean(np.abs(mu_hat - mu_star_grid)))
            try:
                import properscoring as ps

                crps_bar = float(ps.crps_ensemble(mu_star_grid, mu_hat_sims).mean())
            except Exception:
                crps_bar = None
            print(f"\n{'='*45}")
            print(f"  Metrics (grid {nx}x{ny}, n_mc={n_mc})")
            print(f"{'='*45}")
            print(f"  RMSE          : {rmse:.4f}")
            print(f"  MAE           : {mae:.4f}")
            if crps_bar is not None:
                print(f"  CRPS          : {crps_bar:.4f}")
            print(f"{'='*45}\n")

            fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
            plot_specs = [
                (mu_star_field, r"True intensity $\mu^\star(s)$"),
                (mu_hat_field, r"Estimated intensity $\hat{\mu}(s)$"),
                (diff_field, r"Relative error"),
            ]
            for ax, (field, title) in zip(axes, plot_specs):
                plot_field(field, mode="subplot", ax=ax, cmap=cmap, add_colorbar=True)
                ax.set_title(title)
                ax.set_xlim(self.model.x_bounds)
                ax.set_ylim(self.model.y_bounds)
                ax.grid(alpha=0.3, color="white", linewidth=0.5)
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
            ax = axes[0]
            ax.scatter(
                self.catalog.x,
                self.catalog.y,
                c=self.catalog.t,
                s=12,
                alpha=0.7,
                edgecolors="black",
                cmap=event_cmap,
            )
            ax.set_title(f"Observed data (N={len(self.catalog)})")
            ax.set_xlim(self.model.x_bounds)
            ax.set_ylim(self.model.y_bounds)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.3)

            ax = axes[1]
            plot_field(mu_hat_field, mode="subplot", ax=ax, cmap=cmap, add_colorbar=True)
            ax.set_title(r"Estimated intensity $\hat{\mu}(s)$")
            ax.set_xlim(self.model.x_bounds)
            ax.set_ylim(self.model.y_bounds)
            ax.grid(alpha=0.3, color="white", linewidth=0.5)

        plt.tight_layout()
        if savefigure:
            save_figure(fig, title_savefig, figure_type="raster")
        plt.show()

        summary = self.summary(burn_in)
        return {
            "mu_hat": mu_hat,
            "mu_star": mu_star_grid,
            "diff": diff,
            "squared_mu_hat": squared_mu_hat,
            "var_mu_hat": var_mu_hat,
            "std_mu_hat": std_mu_hat,
            "lower_mu_hat": lower_mu_hat,
            "upper_mu_hat": upper_mu_hat,
            "mu_hat_sims": mu_hat_sims,
            "mu_field": mu_hat_field,
            "std_mu_field": std_mu_hat_field,
            "lower_mu_field": lower_mu_hat_field,
            "upper_mu_field": upper_mu_hat_field,
            "mesh": mesh,
            "mu_post_gp": None,
            "Sigma_post_gp": None,
            "eps_hat": summary["eps_hat"],
            "f_data_hat": summary["f_data_hat"],
            "E_mu_bar": E_mu_bar,
            "E_mu_chain": E_mu_post,
            "rmse": rmse,
            "mae": mae,
            "crps": crps_bar,
        }

    def plot_traces(
        self,
        burn_in: float | None = None,
        figsize=None,
        savefigure=False,
        title_savefig=None,
        trace_color=None,
        hist_color="steelblue",
        burn_in_color="red",
    ):
        """Plot full MCMC traces and post-burn-in posterior histograms."""
        burn_in = self.default_burn_in if burn_in is None else burn_in
        if not 0.0 <= burn_in < 1.0:
            raise ValueError("burn_in must be in [0, 1).")

        if self.etas_chain is not None:
            chain = self.etas_chain
            names = self.raw.get("theta_phi_names", [])
            chains = [(name, chain[:, index]) for index, name in enumerate(names)]
            if self.raw.get("beta") is not None:
                chains.append(("beta", np.asarray(self.raw["beta"])))
            tex = {
                "A": r"$A$", "alpha": r"$\alpha$", "c": r"$c$", "p": r"$p$",
                "d": r"$d$", "q": r"$q$", "gamma": r"$\gamma$", "beta": r"$\beta$",
            }
            title_savefig = title_savefig or "traces_etas"
        else:
            eps_chain = self.eps_chain
            chains = [(rf"$\epsilon_{j}$", eps_chain[:, j]) for j in range(eps_chain.shape[1])]
            tex = {}
            title_savefig = title_savefig or "traces_eps"

        thin = self.raw.get("thin", 1)
        n_store = len(chains[0][1]) if chains else 0
        burn = int(n_store * burn_in)
        iters = np.arange(n_store) * thin
        if figsize is None:
            figsize = (10, max(2.0, 1.8 * len(chains)))
        fig, axes = plt.subplots(len(chains), 2, figsize=figsize, squeeze=False)
        for index, (name, values) in enumerate(chains):
            label = tex.get(name, name)
            axes[index, 0].plot(iters, values, lw=0.8, alpha=0.85, color=trace_color)
            axes[index, 0].axvline(burn * thin, c=burn_in_color, ls="--", alpha=0.45)
            axes[index, 0].set_title(f"Trace {label}")
            axes[index, 0].set_xlabel(f"Iteration (thin={thin})")
            axes[index, 0].grid(alpha=0.3)
            axes[index, 1].hist(
                values[burn:], bins=35, density=True, ec="k", alpha=0.7, color=hist_color
            )
            axes[index, 1].set_title(f"Posterior {label}")
            axes[index, 1].grid(alpha=0.3)
        plt.tight_layout()
        if savefigure:
            save_figure(fig, title_savefig)
        plt.show()

        nu_fig = None
        if self.etas_chain is None and self.raw.get("acceptance_nu") is not None:
            nu_chain = np.asarray(self.raw["nu"])
            labels = [r"$v^2$", r"$\ell$"]
            nu_fig, nu_axes = plt.subplots(2, 2, figsize=(figsize[0], 6), squeeze=False)
            for index, label in enumerate(labels):
                values = nu_chain[:, index]
                nu_axes[index, 0].plot(iters, values, linewidth=1, color=trace_color)
                nu_axes[index, 0].axvline(burn * thin, color=burn_in_color, linestyle="--", alpha=0.5)
                nu_axes[index, 0].set_title(f"Trace {label}")
                nu_axes[index, 0].set_xlabel(f"Iteration (thin={thin})")
                nu_axes[index, 0].grid(alpha=0.3)
                nu_axes[index, 1].hist(
                    values[burn:], bins=30, density=True, edgecolor="black", alpha=0.7, color=hist_color
                )
                nu_axes[index, 1].set_title(f"Posterior {label}")
                nu_axes[index, 1].grid(alpha=0.3)
            plt.tight_layout()
            if savefigure:
                save_figure(nu_fig, "traces_nu")
            plt.show()
        return (fig, nu_fig) if nu_fig is not None else fig

    @staticmethod
    def _acf(values, max_lag):
        values = np.asarray(values, dtype=float)
        values = values - values.mean()
        denom = np.dot(values, values)
        if denom <= 0.0:
            return np.ones(max_lag + 1)
        return np.array([
            np.dot(values[: values.size - lag], values[lag:]) / denom
            for lag in range(max_lag + 1)
        ])

    def plot_acf(
        self,
        burn_in: float | None = None,
        max_lag=50,
        figsize=(8, 6),
        savefigure=False,
        title_savefig="trace_acf",
    ):
        """Plot post-burn-in autocorrelations for the stored Gibbs chains."""
        burn_in = self.default_burn_in if burn_in is None else burn_in
        if not 0.0 <= burn_in < 1.0:
            raise ValueError("burn_in must be in [0, 1).")
        eps_chain = self.eps_chain
        burn = int(burn_in * eps_chain.shape[0])
        n_post = eps_chain.shape[0] - burn
        max_lag = min(int(max_lag), n_post - 1)
        if max_lag < 1:
            print(f"[plot_acf] Not enough post-burn-in draws ({n_post}).")
            return None

        plots = [(rf"$\epsilon_{j}$", eps_chain[burn:, j]) for j in range(eps_chain.shape[1])]
        if self.raw.get("acceptance_nu") is not None:
            nu_chain = np.asarray(self.raw["nu"])
            plots.extend([(r"$v^2$", nu_chain[burn:, 0]), (r"$\ell$", nu_chain[burn:, 1])])
        if self.etas_chain is not None:
            names = self.raw.get("theta_phi_names", [])
            for index, name in enumerate(names):
                plots.append((name, self.etas_chain[burn:, index]))
            if self.raw.get("beta") is not None:
                plots.append((r"$\beta$", np.asarray(self.raw["beta"])[burn:]))

        fig, axes = plt.subplots(len(plots), 1, figsize=(figsize[0], 3.0 * len(plots)), squeeze=False)
        lags = np.arange(max_lag + 1)
        thin = self.raw.get("thin", 1)
        for ax, (label, chain) in zip(axes[:, 0], plots):
            values = self._acf(chain, max_lag)
            ax.plot(lags[:len(values)], values)
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set(xlim=(0, max_lag), ylim=(-1.0, 1.0), xlabel="Lag")
            ax.set_title(f"ACF - {label} (thin={thin})")
            ax.grid(alpha=0.3)
        plt.tight_layout()
        if savefigure:
            save_figure(fig, title_savefig)
        plt.show()
        return fig

    def plot_declustering(
        self,
        burn_in: float | None = None,
        savefigure=False,
        title_savefig="declustering",
        probability_cmap="RdYlBu",
        branching_cmap="plasma",
        point_edge_color="black",
        link_color="black",
        magnitudes=None,
        true_parent=None,
        true_parent_convention="auto",
        background_threshold=0.5,
    ):
        """Plot background probabilities and the two-stage branching tree."""
        if self.branching_chain is None:
            raise TypeError("Declustering is unavailable for an SSGC-only model.")
        if not 0.0 <= background_threshold <= 1.0:
            raise ValueError("background_threshold must be in [0, 1].")

        burn_in = self.default_burn_in if burn_in is None else burn_in
        Z_chain = self.branching_chain.astype(int)
        burn = int(Z_chain.shape[0] * burn_in)
        Z_post = Z_chain[burn:]
        if Z_post.shape[0] == 0:
            raise ValueError("burn_in leaves no posterior declustering draws.")
        N = Z_post.shape[1]

        ta = np.asarray(self.catalog.t, dtype=float).reshape(-1)
        xa = np.asarray(self.catalog.x, dtype=float).reshape(-1)
        ya = np.asarray(self.catalog.y, dtype=float).reshape(-1)
        if ta.size != N or xa.size != N or ya.size != N:
            raise ValueError("catalog and results['Z'] must describe the same events.")

        if magnitudes is None:
            magnitudes = self.catalog.magnitudes
        if magnitudes is None:
            branching_colors = ta
            branching_color_label = "t"
        else:
            branching_colors = np.asarray(magnitudes, dtype=float).reshape(-1)
            if branching_colors.size != N:
                raise ValueError("magnitudes must have one value per event.")
            if not np.all(np.isfinite(branching_colors)):
                raise ValueError("magnitudes must contain only finite values.")
            branching_color_label = "magnitude"

        p_bg = np.mean(Z_post == 0, axis=0)
        background_mode = p_bg >= background_threshold
        parent_mode = np.zeros(N, dtype=int)
        parent_probability = p_bg.copy()

        for child in np.flatnonzero(~background_mode):
            positive_labels = Z_post[:, child]
            positive_labels = positive_labels[positive_labels > 0]
            if positive_labels.size == 0:
                background_mode[child] = True
                continue
            labels, counts = np.unique(positive_labels, return_counts=True)
            best = int(np.argmax(counts))
            parent_mode[child] = int(labels[best])
            parent_probability[child] = counts[best] / positive_labels.size

        generation = np.zeros(N, dtype=int)
        valid_link = np.zeros(N, dtype=bool)
        for child in np.flatnonzero(~background_mode):
            parent = parent_mode[child] - 1
            if 0 <= parent < child:
                generation[child] = generation[parent] + 1
                valid_link[child] = True
            else:
                background_mode[child] = True
                parent_mode[child] = 0
                parent_probability[child] = p_bg[child]
        linked = valid_link
        max_generation = int(generation.max()) if N else 0

        true_labels = None
        true_background = None
        parent_truth_probability = None
        parent_correct = None
        classification_report_text = None
        supervised = true_parent is not None

        if supervised:
            truth = np.asarray(true_parent, dtype=int).reshape(-1)
            if truth.size != N:
                raise ValueError("true_parent must have one label per event.")
            convention = str(true_parent_convention).lower()
            if convention not in {"auto", "branching", "indices"}:
                raise ValueError("true_parent_convention must be 'auto', 'branching', or 'indices'.")
            if convention == "auto":
                convention = "indices" if np.any(truth < 0) else "branching"
            true_labels = (
                np.where(truth < 0, 0, truth + 1).astype(int)
                if convention == "indices"
                else truth.astype(int)
            )
            if np.any(true_labels < 0):
                raise ValueError("true_parent contains invalid negative labels.")
            for child, label in enumerate(true_labels):
                if label > 0 and label - 1 >= child:
                    raise ValueError("true_parent must only reference earlier events as parents.")

            true_background = true_labels == 0
            parent_truth_probability = np.mean(Z_post == true_labels.reshape(1, -1), axis=0)
            parent_correct = parent_mode == true_labels

            from sklearn.metrics import classification_report

            classification_report_text = classification_report(
                true_background.astype(int),
                background_mode.astype(int),
                labels=[1, 0],
                target_names=["background", "triggered"],
                zero_division=0,
            )

        fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5))
        ax = axes[0]
        sc = ax.scatter(
            xa, ya, c=p_bg, cmap=probability_cmap, s=20,
            edgecolors=point_edge_color, linewidths=0.3, vmin=0, vmax=1,
        )
        if supervised:
            ax.scatter(
                xa[true_background], ya[true_background],
                facecolors="none", edgecolors="black", s=90,
                linewidths=1.0, label="True background", zorder=3,
            )
            ax.legend(loc="upper right")
        plt.colorbar(sc, ax=ax, label=r"$P(z_i = 0 \mid \mathcal{D})$")
        ax.set_title(
            f"Probability background (threshold={background_threshold:g})\n"
            f"({background_mode.sum()} background, {linked.sum()} triggered)"
        )
        ax.set_xlim(self.model.x_bounds)
        ax.set_ylim(self.model.y_bounds)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)

        ax = axes[1]
        for child in np.flatnonzero(linked):
            parent = parent_mode[child] - 1
            alpha = 0.15 + 0.75 * parent_probability[child]
            ax.plot(
                [ta[parent], ta[child]],
                [generation[parent], generation[child]],
                color=link_color, lw=0.8, alpha=alpha, zorder=1,
            )
        scatter = ax.scatter(
            ta,
            generation,
            c=branching_colors,
            cmap=branching_cmap,
            s=26 + 42 * parent_probability,
            edgecolors=point_edge_color,
            linewidths=0.35,
            alpha=0.9,
            zorder=2,
        )
        ax.set_xlabel("t")
        ax.set_ylabel("generation")
        ax.set_yticks(np.arange(max_generation + 1))
        ax.set_ylim(-0.5, max_generation + 0.7)
        ax.set_title("Two-stage branching tree")
        ax.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax, label=branching_color_label)

        plt.tight_layout()
        if savefigure:
            save_figure(fig, title_savefig)
        plt.show()

        diagnostics = {
            "p_bg": p_bg,
            "background_threshold": float(background_threshold),
            "predicted_background": background_mode,
            "parent_mode": parent_mode,
            "parent_probability": parent_probability,
            "generation": generation,
            "branching_color_values": branching_colors,
            "branching_color_label": branching_color_label,
        }

        if supervised:
            print("\nDeclustering classification report")
            print(classification_report_text)
            triggered_truth = true_labels > 0
            parent_accuracy_triggered = np.nan
            if np.any(triggered_truth):
                parent_accuracy_triggered = float(
                    np.mean(parent_mode[triggered_truth] == true_labels[triggered_truth])
                )
            diagnostics.update({
                "true_parent": true_labels,
                "true_background": true_background,
                "parent_correct": parent_correct,
                "parent_truth_probability": parent_truth_probability,
                "parent_accuracy": float(np.mean(parent_correct)),
                "parent_accuracy_triggered": parent_accuracy_triggered,
                "classification_report": classification_report_text,
            })

        return diagnostics
