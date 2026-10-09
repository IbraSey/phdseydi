"""Simulate, fit and visualize a small marked SPIN-H example."""

# %% Imports and user settings
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from simulation import simulate_hawkes_process
from package import (
    ETASParameters,
    GPParameters,
    SPINHGibbsConfig,
    SPINHModel,
    SPINHVIConfig,
    plot_spinh_parameter_marginals,
)
from visualization import save_figure


# =============================================================================
# USER SETTINGS
# =============================================================================
# Simulation
RNG_SEED = 42
X_BOUNDS = (0.0, 1.0)
Y_BOUNDS = (0.0, 1.0)
SIMULATION_END_TIME = 200.0
N_DOMAIN_COLUMNS = 2
N_DOMAIN_ROWS = 2
TRUE_BASELINE_INTENSITIES = (6.0, 2.0, 4.0, 7.0)
LATENT_FIELD_AMPLITUDE = 0.8
TRUE_BETA = 2.3
MAGNITUDE_MIN = 2.0
MAGNITUDE_MAX = 6.0
TRUE_ETAS = ETASParameters(
    A=0.50,
    alpha=0.50,
    c=0.03,
    p=1.30,
    d=0.05,
    q=1.70,
    gamma=0.20,
)

# Model and inference
INITIAL_ETAS = ETASParameters(
    A=0.45,
    alpha=0.50,
    c=0.04,
    p=1.25,
    d=0.06,
    q=1.60,
    gamma=0.20,
)
INITIAL_BETA = 2.0
GP_PRIOR = GPParameters(variance=0.6, length_scale=0.25)
INFERENCE_METHOD = "vi"  # "vi" or "gibbs"; Gibbs requires MALA_STEP
GP_BACKEND = "sparse"  # "sparse" (HSGP) or "exact"

# Gamma priors use shape/rate parameters. Priors for p and q apply to p-1
# and q-1. These broad values are shared with the numerical experiments.
ETAS_PRIORS = {
    "a_A": 2.0,
    "b_A": 5.0,
    "a_alpha": 2.0,
    "b_alpha": 2.0 / 0.6,
    "a_c": 2.0,
    "b_c": 50.0,
    "a_p": 2.0,
    "b_p": 5.0,
    "a_d": 2.0,
    "b_d": 20.0,
    "a_q": 2.0,
    "b_q": 3.0,
    "a_gamma": 1.0,
    "b_gamma": 1.0 / 0.3,
}
VI_INITIAL_GAMMA_FACTORS = {
    "A": (10.0 * INITIAL_ETAS.A, 10.0),
    "alpha": (10.0 * INITIAL_ETAS.alpha, 10.0),
    "c": (100.0 * INITIAL_ETAS.c, 100.0),
    "p_minus_1": (10.0 * (INITIAL_ETAS.p - 1.0), 10.0),
    "d": (50.0 * INITIAL_ETAS.d, 50.0),
    "q_minus_1": (10.0 * (INITIAL_ETAS.q - 1.0), 10.0),
    "gamma": (10.0 * INITIAL_ETAS.gamma, 10.0),
    "beta": (10.0 * INITIAL_BETA, 10.0),
}

# Candidate parents: False keeps every earlier event. If True, either set a
# time window directly or derive one from the initial temporal kernel.
USE_SPARSE_CANDIDATES = False
CANDIDATE_TIME_WINDOW = None
CANDIDATE_RELATIVE_DENSITY = 1e-3

# Fixed for this example from three 200-iteration sparse-GP pilot chains
# (seeds 42--44, N=671--691): epsilon acceptance was 58--68%.
# Recalibrate after changing the catalogue, model, or GP backend.
MALA_STEP = 0.08

VI_CONFIG = SPINHVIConfig(
    n_iter=400,
    gp_backend=GP_BACKEND,
    use_calibration=True,
    quadrature_nx=15,
    quadrature_ny=15,
    spatial_compensator_grid=20,
    etas_update_start=5,
    etas_update_every=10,
    max_optimizer_iter=10,
    gamma_quadrature_nodes=4,
    theta_priors=ETAS_PRIORS,
    initial_gamma_factors=VI_INITIAL_GAMMA_FACTORS,
    random_seed=RNG_SEED,
    verbose=True,
)
GIBBS_CONFIG = SPINHGibbsConfig(
    n_iter=4000,
    thin=5,
    mala_step=MALA_STEP,
    theta_priors=ETAS_PRIORS,
    use_calibration=True,
    beta_init=INITIAL_BETA,
    grid_nx=12,
    grid_ny=12,
    adaptation_start=200,
    etas_adaptation_end=2000,
    etas_target_acceptance=0.234,
    verbose=True,
)
GIBBS_BURN_IN = 0.5

# Output and posterior evaluation
REPOSITORY_ROOT = (
    Path(__file__).resolve().parents[1]
    if "__file__" in globals()
    else Path.cwd()
)
RESULTS_DIR = REPOSITORY_ROOT / "results" / "quickstart_spinh"
PLOT_GRID_SIZE = 60
POSTERIOR_FIELD_DRAWS = 200
POSTERIOR_PARAMETER_DRAWS = 1000
SNAPSHOT_TIME_FRACTION = 0.75
# Plotting only: None uses the full range; 0.99 saturates the highest 1% so
# localized ETAS peaks do not flatten the rest of each map.
INTENSITY_CLIP_QUANTILE = 0.99
INTENSITY_COLORMAPS = ("viridis", "magma", "inferno")
SHOW_FIGURES = True
# =============================================================================


# %% Model fitting and visualization
def latent_field(x, y):
    """Smooth residual field used to generate the toy background."""
    return LATENT_FIELD_AMPLITUDE * np.sin(2.0 * np.pi * x) * np.cos(
        2.0 * np.pi * y
    )


def candidate_time_window(model):
    """Resolve the dense or temporally truncated parent support."""
    if not USE_SPARSE_CANDIDATES:
        if CANDIDATE_TIME_WINDOW is not None:
            raise ValueError(
                "Set USE_SPARSE_CANDIDATES=True to use CANDIDATE_TIME_WINDOW."
            )
        return None
    if CANDIDATE_TIME_WINDOW is not None:
        window = float(CANDIDATE_TIME_WINDOW)
        if not np.isfinite(window) or window <= 0.0:
            raise ValueError("CANDIDATE_TIME_WINDOW must be finite and positive.")
        return window
    return model.parent_time_window_from_kernel(
        relative_density=CANDIDATE_RELATIVE_DENSITY,
        parameters=INITIAL_ETAS,
    )


def fit_model(model, catalog, parent_time_window):
    """Run the inference method selected in USER SETTINGS."""
    method = INFERENCE_METHOD.lower()
    if method == "vi":
        config = replace(VI_CONFIG, parent_time_window=parent_time_window)
        return model.vi(catalog, config=config)
    if method == "gibbs":
        config = replace(GIBBS_CONFIG, parent_time_window=parent_time_window)
        return model.gibbs(
            catalog,
            config=config,
            gp_backend=GP_BACKEND,
            rng_seed=RNG_SEED,
        )
    raise ValueError("INFERENCE_METHOD must be 'vi' or 'gibbs'.")


def posterior_background(fit, grid_x, grid_y):
    """Evaluate the posterior mean background intensity."""
    if INFERENCE_METHOD.lower() == "vi":
        return fit.background_intensity(
            grid_x,
            grid_y,
            n_samples=POSTERIOR_FIELD_DRAWS,
            rng_seed=RNG_SEED + 1,
        )
    return fit.background_intensity(grid_x, grid_y, burn_in=GIBBS_BURN_IN)


def posterior_etas(fit):
    """Return posterior-mean ETAS parameters for either inference method."""
    if INFERENCE_METHOD.lower() == "vi":
        return fit.etas_mean()
    estimates = fit.summary(burn_in=GIBBS_BURN_IN)["theta_phi_hat"]
    return ETASParameters(
        **{name: float(value) for name, value in estimates.items()}
    )


def posterior_background_probabilities(fit):
    """Return each event's posterior probability of being background."""
    if INFERENCE_METHOD.lower() == "vi":
        return np.asarray(fit.summary()["p_background"], dtype=float)
    return fit.background_probabilities(burn_in=GIBBS_BURN_IN)


def finish_figure(fig, filename, *, raster):
    """Save a quickstart figure and display it when requested."""
    destination = save_figure(
        fig,
        filename,
        output_dir=RESULTS_DIR,
        figure_type="raster" if raster else "vector",
    )
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)
    print("Saved figure:", destination)


def intensity_color_limit(values):
    """Return a shared display limit and whether high values are saturated."""
    values = np.asarray(values, dtype=float)
    maximum = float(np.max(values))
    if INTENSITY_CLIP_QUANTILE is None:
        return max(maximum, 1.0), False
    quantile = float(INTENSITY_CLIP_QUANTILE)
    if not np.isfinite(quantile) or not 0.0 < quantile <= 1.0:
        raise ValueError("INTENSITY_CLIP_QUANTILE must be in (0, 1] or None.")
    limit = float(np.quantile(values, quantile))
    if not np.isfinite(limit) or limit <= 0.0:
        limit = max(maximum, 1.0)
    return limit, bool(maximum > limit)


def plot_conditional_components(simulation, model, fit, grid_x, grid_y):
    """Compare true and estimated SPIN-H components at one time."""
    catalog = simulation.catalog
    snapshot_time = SNAPSHOT_TIME_FRACTION * model.duration
    evaluation_times = np.full(grid_x.shape, snapshot_time)
    true_background = simulation.background_simulation.spatial_components(
        grid_x,
        grid_y,
    )[3]
    estimated_background = posterior_background(fit, grid_x, grid_y)
    true_triggering = model.triggering_intensity(
        evaluation_times,
        grid_x,
        grid_y,
        catalog,
        parameters=TRUE_ETAS,
    )
    estimated_triggering = model.triggering_intensity(
        evaluation_times,
        grid_x,
        grid_y,
        catalog,
        parameters=posterior_etas(fit),
    )
    component_pairs = (
        ("Background", r"$\mu(x,y)$", true_background, estimated_background),
        (
            "Triggering",
            r"$\lambda_{\mathrm{trig}}(t,x,y)$",
            true_triggering,
            estimated_triggering,
        ),
        (
            "Total",
            r"$\lambda(t,x,y)$",
            true_background + true_triggering,
            estimated_background + estimated_triggering,
        ),
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(14, 8),
        layout="constrained",
        sharex=True,
        sharey=True,
    )
    history = catalog.t < snapshot_time
    for column, (component, colorbar_label, truth, estimate) in enumerate(
        component_pairs
    ):
        combined = np.concatenate([truth.reshape(-1), estimate.reshape(-1)])
        vmax, clipped = intensity_color_limit(combined)
        rows = (("True", truth), ("Estimated", estimate))
        for row, (label, values) in enumerate(rows):
            ax = axes[row, column]
            image = ax.pcolormesh(
                grid_x,
                grid_y,
                values,
                shading="auto",
                cmap=INTENSITY_COLORMAPS[column],
                vmin=0.0,
                vmax=vmax,
                rasterized=True,
            )
            ax.scatter(
                catalog.x[history],
                catalog.y[history],
                s=6,
                c="white",
                edgecolors="black",
                linewidths=0.15,
            )
            ax.set(
                xlim=X_BOUNDS,
                ylim=Y_BOUNDS,
                xlabel="x",
                ylabel="y",
                title=f"{label} {component.lower()}",
                aspect="equal",
            )
        fig.colorbar(
            image,
            ax=axes[:, column].tolist(),
            label=colorbar_label,
            extend="max" if clipped else "neither",
        )
    fig.suptitle(f"Conditional intensity components at t={snapshot_time:.2f}")
    return fig


def plot_declustering(simulation, fit):
    """Compare true event types with posterior background probabilities."""
    catalog = simulation.catalog
    probabilities = posterior_background_probabilities(fit)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.5, 4.5),
        layout="constrained",
    )
    truth = axes[0].scatter(
        catalog.x,
        catalog.y,
        c=simulation.is_background.astype(float),
        cmap="coolwarm",
        vmin=0.0,
        vmax=1.0,
        s=18,
        edgecolors="black",
        linewidths=0.2,
    )
    estimate = axes[1].scatter(
        catalog.x,
        catalog.y,
        c=probabilities,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        s=18,
        edgecolors="black",
        linewidths=0.2,
    )
    axes[0].set_title("True event type")
    axes[1].set_title("Posterior background probability")
    fig.colorbar(
        truth,
        ax=axes[0],
        ticks=(0.0, 1.0),
        label="0 triggered, 1 background",
    )
    fig.colorbar(
        estimate,
        ax=axes[1],
        label=r"$P(Z_i=0\mid\mathcal{D})$",
    )
    for ax in axes:
        ax.set(
            xlim=X_BOUNDS,
            ylim=Y_BOUNDS,
            xlabel="x",
            ylabel="y",
            aspect="equal",
        )
    return fig


def main():
    simulation = simulate_hawkes_process(
        X_bounds=X_BOUNDS,
        Y_bounds=Y_BOUNDS,
        T=SIMULATION_END_TIME,
        n_cols=N_DOMAIN_COLUMNS,
        n_rows=N_DOMAIN_ROWS,
        mus=TRUE_BASELINE_INTENSITIES,
        f=latent_field,
        etas_parameters=TRUE_ETAS,
        beta=TRUE_BETA,
        magnitude_min=MAGNITUDE_MIN,
        magnitude_max=MAGNITUDE_MAX,
        rng_seed=RNG_SEED,
    )
    catalog = simulation.catalog
    if len(catalog) == 0:
        raise RuntimeError("The simulated catalogue is empty; choose another seed.")

    model_duration = float(np.max(catalog.t))
    model = SPINHModel.from_polygons(
        polygons=simulation.background_simulation.domains.polygons,
        duration=model_duration,
        x_bounds=X_BOUNDS,
        y_bounds=Y_BOUNDS,
        gp_prior=GP_PRIOR,
        etas_parameters=INITIAL_ETAS,
        magnitude_min=MAGNITUDE_MIN,
        magnitude_max=MAGNITUDE_MAX,
    )
    parent_time_window = candidate_time_window(model)
    fit = fit_model(model, catalog, parent_time_window)
    summary = (
        fit.summary()
        if INFERENCE_METHOD.lower() == "vi"
        else fit.summary(burn_in=GIBBS_BURN_IN)
    )

    print(
        f"Simulated N={len(catalog)}: {simulation.n_background} background, "
        f"{simulation.n_triggered} triggered"
    )
    print(f"Model duration: {model_duration:.3f}")
    print("Inference method:", INFERENCE_METHOD.upper(), f"({GP_BACKEND} GP)")
    if parent_time_window is None:
        print("Candidate parents: all earlier events (dense)")
    else:
        print(f"Candidate parents: temporal window {parent_time_window:.4g}")
        truncation = (
            fit.diagnostics["branching_truncation"]
            if INFERENCE_METHOD.lower() == "vi"
            else fit.raw["branching_truncation"]
        )
        print(
            "Retained candidate pairs:",
            f"{truncation['candidate_parent_count']}/"
            f"{truncation['dense_candidate_count']}",
        )
        triggered = np.flatnonzero(simulation.parent_indices >= 0)
        if triggered.size:
            true_lags = (
                catalog.t[triggered]
                - catalog.t[simulation.parent_indices[triggered]]
            )
            print(
                "Simulation-only true-parent recall:",
                f"{np.mean(true_lags <= parent_time_window):.3f}",
            )
    print("Posterior mean beta:", round(float(summary["beta_hat"]), 3))
    print("Posterior mean ETAS parameters:", summary["theta_phi_hat"])
    print(
        "Background fraction (true, posterior mean):",
        f"{simulation.n_background / len(catalog):.3f},",
        f"{posterior_background_probabilities(fit).mean():.3f}",
    )
    if INFERENCE_METHOD.lower() == "gibbs":
        print("Gibbs proposal steps:", fit.raw["proposal_steps"])
        print("Gibbs acceptance rates:", fit.acceptance_rates)
    elif not fit.diagnostics["converged"]:
        print(
            "Warning: MF-VI reached its iteration limit. Increase VI_CONFIG.n_iter "
            "before treating the fit as converged."
        )
    if INTENSITY_CLIP_QUANTILE is not None:
        print(
            "Figure colors only: values above quantile "
            f"{INTENSITY_CLIP_QUANTILE:g} are saturated (colorbar arrow)."
        )

    grid_x, grid_y = np.meshgrid(
        np.linspace(*X_BOUNDS, PLOT_GRID_SIZE),
        np.linspace(*Y_BOUNDS, PLOT_GRID_SIZE),
    )
    conditional_figure = plot_conditional_components(
        simulation,
        model,
        fit,
        grid_x,
        grid_y,
    )
    finish_figure(
        conditional_figure,
        "conditional_intensity_components",
        raster=True,
    )

    declustering_figure = plot_declustering(simulation, fit)
    finish_figure(declustering_figure, "declustering", raster=False)

    marginal_result = plot_spinh_parameter_marginals(
        {f"{INFERENCE_METHOD.upper()} {GP_BACKEND} GP": fit},
        true_parameters=TRUE_ETAS,
        true_beta=TRUE_BETA,
        n_samples=POSTERIOR_PARAMETER_DRAWS,
        burn_in=GIBBS_BURN_IN,
        rng_seed=RNG_SEED + 2,
        savefigure=True,
        title_savefig="parameter_marginals",
        output_dir=RESULTS_DIR,
        show=SHOW_FIGURES,
    )
    if not SHOW_FIGURES:
        plt.close(marginal_result["figure"])
    print("Saved figure:", marginal_result["saved_path"])


if __name__ == "__main__":
    main()

# %%
