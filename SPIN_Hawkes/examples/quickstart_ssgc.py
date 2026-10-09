"""Simulate, fit and visualize a small SSGC background-only example."""

# %% Imports and user settings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from simulation import simulate_spatial_process
from package import GibbsConfig, GPParameters, SSGCModel, SSGCVIConfig
from visualization import save_figure


# =============================================================================
# USER SETTINGS
# =============================================================================
# Simulation
RNG_SEED = 42
X_BOUNDS = (0.0, 1.0)
Y_BOUNDS = (0.0, 1.0)
SIMULATION_END_TIME = 100.0
N_DOMAIN_COLUMNS = 2
N_DOMAIN_ROWS = 2
TRUE_BASELINE_INTENSITIES = (6.0, 2.0, 4.0, 7.0)
LATENT_FIELD_AMPLITUDE = 0.8

# Model and inference
GP_PRIOR = GPParameters(variance=0.6, length_scale=0.25)
INFERENCE_METHOD = "vi"  # "vi" or "gibbs"
GP_BACKEND = "sparse"  # "sparse" (HSGP) or "exact"
# Fixed for this example from three 200-iteration sparse-GP pilot chains
# (seeds 42--44, N=232--250): epsilon acceptance was 51.5--58%.
# Recalibrate after changing the catalogue, model, or GP backend.
MALA_STEP = 0.115
VI_CONFIG = SSGCVIConfig(
    n_iter=40,
    gp_backend=GP_BACKEND,
    use_calibration=False,
    quadrature_nx=8,
    quadrature_ny=8,
    random_seed=RNG_SEED,
    verbose=False,
)
GIBBS_CONFIG = GibbsConfig(
    n_iter=500,
    thin=2,
    mala_step=MALA_STEP,
    use_calibration=False,
    grid_nx=12,
    grid_ny=12,
    verbose=False,
)

# Output and posterior evaluation
REPOSITORY_ROOT = (
    Path(__file__).resolve().parents[1]
    if "__file__" in globals()
    else Path.cwd()
)
RESULTS_DIR = REPOSITORY_ROOT / "results" / "quickstart_ssgc"
PLOT_GRID_SIZE = 60
POSTERIOR_FIELD_DRAWS = 200
FIGURE_SIZE_PER_PANEL = (5.5, 4.5)
INTENSITY_COLORMAP = "viridis"
SHOW_FIGURES = True
# =============================================================================


# %% Model fitting and visualization
def latent_field(x, y):
    """Smooth residual field used to generate the toy background."""
    return LATENT_FIELD_AMPLITUDE * np.sin(2.0 * np.pi * x) * np.cos(
        2.0 * np.pi * y
    )


def fit_model(model, catalog):
    """Run the inference method selected in USER SETTINGS."""
    method = INFERENCE_METHOD.lower()
    if method == "vi":
        return model.vi(catalog, config=VI_CONFIG)
    if method == "gibbs":
        return model.gibbs(
            catalog,
            config=GIBBS_CONFIG,
            gp_backend=GP_BACKEND,
            rng_seed=RNG_SEED,
        )
    raise ValueError("INFERENCE_METHOD must be 'vi' or 'gibbs'.")


def posterior_background(fit, grid_x, grid_y):
    """Evaluate the posterior mean with the selected result API."""
    if INFERENCE_METHOD.lower() == "vi":
        return fit.background_intensity(
            grid_x,
            grid_y,
            n_samples=POSTERIOR_FIELD_DRAWS,
            rng_seed=RNG_SEED + 1,
        )
    return fit.background_intensity(grid_x, grid_y)


def plot_background_comparison(
    catalog,
    grid_x,
    grid_y,
    estimated_intensity,
    true_intensity=None,
):
    """Plot the truth beside the estimate when a generating field is known."""
    panels = [("Posterior mean background intensity", estimated_intensity)]
    if true_intensity is not None:
        panels.insert(0, ("True background intensity", true_intensity))

    combined = np.concatenate([values.reshape(-1) for _, values in panels])
    vmin, vmax = float(combined.min()), float(combined.max())
    width, height = FIGURE_SIZE_PER_PANEL
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(width * len(panels), height),
        squeeze=False,
        layout="constrained",
        sharex=True,
        sharey=True,
    )
    for ax, (title, values) in zip(axes[0], panels):
        image = ax.pcolormesh(
            grid_x,
            grid_y,
            values,
            shading="auto",
            cmap=INTENSITY_COLORMAP,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )
        ax.scatter(
            catalog.x,
            catalog.y,
            s=8,
            c="white",
            edgecolors="black",
            linewidths=0.2,
        )
        ax.set(
            xlim=X_BOUNDS,
            ylim=Y_BOUNDS,
            xlabel="x",
            ylabel="y",
            title=title,
            aspect="equal",
        )
    fig.colorbar(image, ax=axes[0].tolist(), label=r"$\mu(x,y)$")
    return fig


def main():
    simulation = simulate_spatial_process(
        X_bounds=X_BOUNDS,
        Y_bounds=Y_BOUNDS,
        T=SIMULATION_END_TIME,
        n_cols=N_DOMAIN_COLUMNS,
        n_rows=N_DOMAIN_ROWS,
        mus=TRUE_BASELINE_INTENSITIES,
        f=latent_field,
        rng_seed=RNG_SEED,
    )
    catalog = simulation.catalog
    if len(catalog) == 0:
        raise RuntimeError("The simulated catalogue is empty; choose another seed.")

    model_duration = float(np.max(catalog.t))
    model = SSGCModel.from_polygons(
        polygons=simulation.domains.polygons,
        duration=model_duration,
        x_bounds=X_BOUNDS,
        y_bounds=Y_BOUNDS,
        gp_prior=GP_PRIOR,
    )
    fit = fit_model(model, catalog)
    summary = fit.summary()
    eps_mean = (
        summary["eps_mean"]
        if INFERENCE_METHOD.lower() == "vi"
        else summary["eps_hat"]
    )

    print(f"Simulated N={len(catalog)} background events")
    print(f"Model duration: {model_duration:.3f}")
    print("Inference method:", INFERENCE_METHOD.upper(), f"({GP_BACKEND} GP)")
    print("Posterior mean zonal log-intensities:", np.round(eps_mean, 3))

    grid_x, grid_y = np.meshgrid(
        np.linspace(*X_BOUNDS, PLOT_GRID_SIZE),
        np.linspace(*Y_BOUNDS, PLOT_GRID_SIZE),
    )
    estimated_intensity = posterior_background(fit, grid_x, grid_y)
    true_intensity = simulation.spatial_components(grid_x, grid_y)[3]
    fig = plot_background_comparison(
        catalog,
        grid_x,
        grid_y,
        estimated_intensity,
        true_intensity=true_intensity,
    )
    destination = save_figure(
        fig,
        "background_intensity_comparison",
        output_dir=RESULTS_DIR,
        figure_type="raster",
    )
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)
    print("Saved figure:", destination)


if __name__ == "__main__":
    main()

# %%
