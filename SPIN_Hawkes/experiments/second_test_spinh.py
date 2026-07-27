"""Minimal SPIN-H VI smoke test on Hawkes-simulated data."""

#%%
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openturns as ot
from sklearn.metrics import classification_report

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from package import (
    ETASParameters,
    GPParameters,
    SPINHModel,
    SPINHVIConfig,
    generate_voronoi_cells,
    simulate_hawkes_process,
)


# Simulation
X_BOUNDS = (0.0, 2.0)
Y_BOUNDS = (0.0, 2.0)
DURATION = 150.0
N_DOMAINS = 6
TESSELLATION_SEED = 15
SIMULATION_SEED = 42
DOMAIN_INTENSITIES = (8.0, 1.0, 2.0, 8.0, 6.0, 2.0)
MAGNITUDE_MIN = 2.0
MAGNITUDE_MAX = 6.0
TRUE_ETAS = ETASParameters(
    A=0.5,
    alpha=0.8,
    c=0.02,
    p=1.3,
    d=0.05,
    q=1.8,
    gamma=0.5,
)
TRUE_BETA = 2.3
LATENT_FIELD_WEIGHTS = (1.5, -1.5, 3.0, -3.0)
LATENT_FIELD_VARIANCE = 0.3
LATENT_FIELD_MEANS = (
    (0.5, 0.5),
    (0.5, 1.5),
    (1.5, 0.5),
    (1.5, 1.5),
)

# Model
GP_PRIOR_VARIANCE = 5.0
GP_PRIOR_LENGTH_SCALE = 0.2
EPS_PRIOR_VARIANCE = 1.0
EPS_PRIOR_LENGTH_SCALE = 0.01
MODEL_JITTER = 1e-5

# Variational inference
N_ITER = 500
TOLERANCE = 1e-6
VERBOSE = True
VERBOSE_EVERY = max(1, N_ITER // 10)
ELBO_EVERY = 5
USE_SPARSE_GP = True
USE_CALIBRATION = True
UPDATE_Z = True
UPDATE_POLYA_GAMMA = True
UPDATE_LATENT_POISSON = True
UPDATE_GP = True
UPDATE_EPS = True
UPDATE_ETAS = True
FIXED_ETAS = {}
FIXED_BETA = None
QUADRATURE_NX, QUADRATURE_NY = 20, 20
EPS_NEWTON_STEPS = 8
SPATIAL_COMPENSATOR_GRID = 10
ETAS_UPDATE_START = 5
ETAS_UPDATE_EVERY = 5
MAX_OPTIMIZER_ITER = 10
ETAS_QUADRATURE_NODES = 4
VI_JITTER = 1e-6
INITIAL_GAMMA_FACTORS = {
    "A": (5.0, 10),
    "alpha": (7.0, 10.0),
    "c": (10.0, 250.0),
    "p_minus_1": (8.0, 20.0),
    "d": (5.0, 40.0),
    "q_minus_1": (8.0, 10.0),
    "gamma": (5.0, 10.0),
    "beta": (2.0, 1.0),
}
THETA_PRIORS = {
    "a_A": 5.0, "b_A": 10.0,
    "a_alpha": 8.0, "b_alpha": 10.0,
    "a_c": 2.0, "b_c": 100.0,
    "a_p": 4.0, "b_p": 10.0,
    "a_d": 2.0, "b_d": 40.0,
    "a_q": 9.0, "b_q": 10.0,
    "a_gamma": 5.0, "b_gamma": 10.0,
}
BETA_PRIOR = {"a_beta": 2.0, "b_beta": 1.0}

# Evaluation and plots
BACKGROUND_THRESHOLD = 0.5
MAKE_PLOTS = True
PLOT_FIGSIZE = (12, 4)
PLOT_POINT_SIZE = 25
PLOT_TRUE_BACKGROUND_SIZE = 70
PLOT_TRUE_BACKGROUND_LINEWIDTH = 0.8
PLOT_CMAP = "RdYlBu"
PLOT_ELBO_GRID_ALPHA = 0.3
PLOT_SPATIAL_GRID_ALPHA = 0.25
DISPERSION_PARENT_MAGNITUDE = None


def latent_field(x, y):
    means = [ot.Point(mean) for mean in LATENT_FIELD_MEANS]
    covariance = ot.CovarianceMatrix(
        2,
        [LATENT_FIELD_VARIANCE, 0.0, 0.0, LATENT_FIELD_VARIANCE],
    )
    sample = ot.Sample(np.column_stack((x, y)))
    return sum(
        weight * np.asarray(ot.Normal(mean, covariance).computePDF(sample)).ravel()
        for weight, mean in zip(LATENT_FIELD_WEIGHTS, means)
    )


def print_parameter_summary(summary):
    print("\nVI ETAS summary")
    print(f"{'param':<8} {'true':>8} {'initial':>10} {'estimate':>10}")
    print("-" * 39)
    for name, true_value in TRUE_ETAS.as_dict().items():
        factor_name = {"p": "p_minus_1", "q": "q_minus_1"}.get(name, name)
        shape, rate = INITIAL_GAMMA_FACTORS[factor_name]
        initial_value = shape / rate + (1.0 if name in {"p", "q"} else 0.0)
        estimate = summary["theta_phi_hat"][name]
        print(
            f"{name:<8} {true_value:>8.3f} "
            f"{initial_value:>10.3f} {estimate:>10.3f}"
        )
    beta_shape, beta_rate = INITIAL_GAMMA_FACTORS["beta"]
    print(f"{'beta':<8} {TRUE_BETA:>8.3f} "
          f"{beta_shape / beta_rate:>10.3f} {summary['beta_hat']:>10.3f}")


def print_intensity_metrics(name, estimated, truth):
    error = estimated - truth
    print(f"\n{name}")
    print(f"RMSE = {np.sqrt(np.mean(error**2)):.4f}")
    print(f"MAE = {np.mean(np.abs(error)):.4f}")
    print(f"mean estim = {estimated.mean():.4f}")
    print(f"mean true  = {truth.mean():.4f}")


def main():
    polygons, _ = generate_voronoi_cells(
        n_germs=N_DOMAINS,
        X_bounds=X_BOUNDS,
        Y_bounds=Y_BOUNDS,
        rng_seed=TESSELLATION_SEED,
    )

    simulation = simulate_hawkes_process(
        X_bounds=X_BOUNDS,
        Y_bounds=Y_BOUNDS,
        T=DURATION,
        polygons=polygons,
        mus=DOMAIN_INTENSITIES,
        f=latent_field,
        etas_parameters=TRUE_ETAS,
        beta=TRUE_BETA,
        magnitude_min=MAGNITUDE_MIN,
        magnitude_max=MAGNITUDE_MAX,
        rng_seed=SIMULATION_SEED,
    )
    catalog = simulation.catalog

    print("Generated Hawkes catalog")
    print(
        f"N={len(catalog)} "
        f"({simulation.n_background} background, {simulation.n_triggered} triggered)"
    )

    model = SPINHModel.from_polygons(
        polygons=polygons,
        duration=DURATION,
        x_bounds=X_BOUNDS,
        y_bounds=Y_BOUNDS,
        gp_prior=GPParameters(
            variance=GP_PRIOR_VARIANCE,
            length_scale=GP_PRIOR_LENGTH_SCALE,
        ),
        eps_prior_variance=EPS_PRIOR_VARIANCE,
        eps_prior_length_scale=EPS_PRIOR_LENGTH_SCALE,
        jitter=MODEL_JITTER,
        etas_parameters=TRUE_ETAS,
        magnitude_min=MAGNITUDE_MIN,
        magnitude_max=MAGNITUDE_MAX,
    )

    fit = model.vi(
        catalog,
        config=SPINHVIConfig(
            n_iter=N_ITER,
            tolerance=TOLERANCE,
            gp_backend="sparse" if USE_SPARSE_GP else "exact",
            use_calibration=USE_CALIBRATION,
            verbose=VERBOSE,
            verbose_every=VERBOSE_EVERY,
            elbo_every=ELBO_EVERY,
            update_z=UPDATE_Z,
            update_polya_gamma=UPDATE_POLYA_GAMMA,
            update_latent_poisson=UPDATE_LATENT_POISSON,
            update_gp=UPDATE_GP,
            update_eps=UPDATE_EPS,
            update_etas=UPDATE_ETAS,
            fixed_etas=FIXED_ETAS,
            fixed_beta=FIXED_BETA,
            beta_prior=BETA_PRIOR,
            quadrature_nx=QUADRATURE_NX,
            quadrature_ny=QUADRATURE_NY,
            eps_newton_steps=EPS_NEWTON_STEPS,
            spatial_compensator_grid=SPATIAL_COMPENSATOR_GRID,
            etas_update_start=ETAS_UPDATE_START,
            etas_update_every=ETAS_UPDATE_EVERY,
            theta_priors=THETA_PRIORS,
            initial_gamma_factors=INITIAL_GAMMA_FACTORS,
            max_optimizer_iter=MAX_OPTIMIZER_ITER,
            etas_quadrature_nodes=ETAS_QUADRATURE_NODES,
            jitter=VI_JITTER,
            random_seed=SIMULATION_SEED,
        ),
    )
    summary = fit.summary()
    declustering = fit.declustering(background_threshold=BACKGROUND_THRESHOLD)

    print_parameter_summary(summary)
    print(
        "\nBackground probability: "
        f"VI mean={summary['p_background'].mean():.3f}, "
        f"truth={simulation.n_background / len(catalog):.3f}"
    )
    print(
        "Expected latent Poisson count: "
        f"{fit.diagnostics['expected_latent_poisson_count']:.2f}"
    )

    y_true = (~simulation.is_background).astype(int)
    y_pred = (~declustering["is_background"]).astype(int)
    print("\nDeclustering classification report")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["background", "triggered"],
            zero_division=0,
        )
    )

    eps_true = np.log(np.asarray(DOMAIN_INTENSITIES, dtype=float))
    f_true = latent_field(catalog.x, catalog.y)
    mu_true = model.background_intensity(catalog.x, catalog.y, eps_true, f_true)
    trigger_true = model.triggering_intensity(
        catalog.t,
        catalog.x,
        catalog.y,
        history=catalog,
        parameters=TRUE_ETAS,
    )
    lambda_true = mu_true + trigger_true

    estimated_etas = fit.etas_mean()
    mu_est = model.background_intensity(
        catalog.x,
        catalog.y,
        summary["eps_mean"],
        summary["f_data_mean"],
    )
    trigger_est = model.triggering_intensity(
        catalog.t,
        catalog.x,
        catalog.y,
        history=catalog,
        parameters=estimated_etas,
    )
    lambda_est = mu_est + trigger_est

    print_intensity_metrics("Background intensity at events", mu_est, mu_true)
    print_intensity_metrics("Triggering intensity at events", trigger_est, trigger_true)
    print_intensity_metrics("Total conditional intensity at events", lambda_est, lambda_true)

    if MAKE_PLOTS:
        fig, axes = plt.subplots(1, 2, figsize=PLOT_FIGSIZE)
        axes[0].plot(
            fit.diagnostics["elbo_iterations"],
            summary["elbo_trace"],
        )
        axes[0].set_title("VI ELBO")
        axes[0].set_xlabel("iteration")
        axes[0].set_ylabel("ELBO")
        axes[0].grid(alpha=PLOT_ELBO_GRID_ALPHA)

        scatter = axes[1].scatter(
            catalog.x,
            catalog.y,
            c=summary["p_background"],
            s=PLOT_POINT_SIZE,
            cmap=PLOT_CMAP,
            vmin=0.0,
            vmax=1.0,
            edgecolors="none",
        )
        axes[1].scatter(
            catalog.x[simulation.is_background],
            catalog.y[simulation.is_background],
            facecolors="none",
            edgecolors="black",
            s=PLOT_TRUE_BACKGROUND_SIZE,
            linewidths=PLOT_TRUE_BACKGROUND_LINEWIDTH,
        )
        axes[1].set_title("VI background probability")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        axes[1].set_xlim(X_BOUNDS)
        axes[1].set_ylim(Y_BOUNDS)
        axes[1].grid(alpha=PLOT_SPATIAL_GRID_ALPHA)
        fig.colorbar(scatter, ax=axes[1], label="q(Z=background)")
        fig.tight_layout()
        plt.show()

        parent_magnitude = (
            float(np.median(catalog.magnitudes))
            if DISPERSION_PARENT_MAGNITUDE is None
            else float(DISPERSION_PARENT_MAGNITUDE)
        )
        fit.plot_etas_kernel_dispersion(
            parent_magnitude=parent_magnitude,
            reference_parameters=TRUE_ETAS,
        )


if __name__ == "__main__":
    main()

# %%
