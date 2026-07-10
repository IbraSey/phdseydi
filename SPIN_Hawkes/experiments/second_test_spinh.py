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


X_BOUNDS = (0.0, 2.0)
Y_BOUNDS = (0.0, 2.0)
DURATION = 110.0
DOMAIN_INTENSITIES = (8.0, 1.0, 2.0, 8.0, 6.0, 2.0)
SEED = 42

N_ITER = 500
TOLERANCE = 1e-6
USE_SPARSE_GP = True
MAKE_PLOTS = True
BACKGROUND_THRESHOLD = 0.5

EPS_PRIOR_VARIANCE = 1.0
EPS_PRIOR_LENGTH_SCALE = 0.01

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
MAGNITUDE_MIN = 2.0
MAGNITUDE_MAX = 6.0

THETA_PRIORS = {
    "a_A": 5.0, "b_A": 10.0,
    "a_alpha": 8.0, "b_alpha": 10.0,
    "a_c": 2.0, "b_c": 100.0,
    "a_p": 4.0, "b_p": 10.0,
    "a_d": 2.0, "b_d": 40.0,
    "a_q": 9.0, "b_q": 10.0,
    "a_gamma": 5.0, "b_gamma": 10.0,
}


def latent_field(x, y):
    weights = [1.5, -1.5, 3.0, -3.0]
    sigma2 = 0.3
    means = [
        ot.Point([0.5, 0.5]),
        ot.Point([0.5, 1.5]),
        ot.Point([1.5, 0.5]),
        ot.Point([1.5, 1.5]),
    ]
    covariance = ot.CovarianceMatrix(2, [sigma2, 0.0, 0.0, sigma2])
    sample = ot.Sample(np.column_stack((x, y)))
    return sum(
        weight * np.asarray(ot.Normal(mean, covariance).computePDF(sample)).ravel()
        for weight, mean in zip(weights, means)
    )

def print_parameter_summary(summary):
    print("\nVI ETAS summary")
    print(f"{'param':<8} {'true':>8} {'estimate':>10}")
    print("-" * 28)
    for name, true_value in TRUE_ETAS.as_dict().items():
        estimate = summary["theta_phi_hat"][name]
        print(f"{name:<8} {true_value:>8.3f} {estimate:>10.3f}")
    print(f"{'beta':<8} {TRUE_BETA:>8.3f} {summary['beta_hat']:>10.3f}")


def print_intensity_metrics(name, estimated, truth):
    error = estimated - truth
    print(f"\n{name}")
    print(f"RMSE = {np.sqrt(np.mean(error**2)):.4f}")
    print(f"MAE = {np.mean(np.abs(error)):.4f}")
    print(f"mean estim = {estimated.mean():.4f}")
    print(f"mean true = {truth.mean():.4f}")


def main():
    polygons, _ = generate_voronoi_cells(
        n_germs=6,
        X_bounds=X_BOUNDS,
        Y_bounds=Y_BOUNDS,
        rng_seed=15,
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
        rng_seed=SEED,
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
        gp_prior=GPParameters(variance=5.0, length_scale=0.2),
        eps_prior_variance=EPS_PRIOR_VARIANCE,
        eps_prior_length_scale=EPS_PRIOR_LENGTH_SCALE,
        nu_prior_rate=0.5,
        jitter=1e-5,
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
            verbose=True,
            verbose_every=max(1, N_ITER // 10),
            quadrature_nx=20,
            quadrature_ny=20,
            full_gp_max_events=1200,
            latent_poisson_damping=0.35,
            latent_poisson_max_multiplier=1.0,
            etas_update_start=25,
            eps_bounds=(-20.0, 6.0),
            f_bounds=(-12.0, 12.0),
            learn_beta=True,
            beta_init=2.0,
            theta_priors=THETA_PRIORS,
            #fixed_etas={"alpha": TRUE_ETAS.alpha},
            parameter_damping=0.5,
            max_optimizer_iter=80,
            random_seed=SEED,
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
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(summary["elbo_trace"])
        axes[0].set_title("VI ELBO")
        axes[0].set_xlabel("iteration")
        axes[0].set_ylabel("ELBO")
        axes[0].grid(alpha=0.3)

        scatter = axes[1].scatter(
            catalog.x,
            catalog.y,
            c=summary["p_background"],
            s=25,
            cmap="RdYlBu",
            vmin=0.0,
            vmax=1.0,
            edgecolors="none",
        )
        axes[1].scatter(
            catalog.x[simulation.is_background],
            catalog.y[simulation.is_background],
            facecolors="none",
            edgecolors="black",
            s=70,
            linewidths=0.8,
        )
        axes[1].set_title("VI background probability")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        axes[1].set_xlim(X_BOUNDS)
        axes[1].set_ylim(Y_BOUNDS)
        axes[1].grid(alpha=0.25)
        fig.colorbar(scatter, ax=axes[1], label="q(Z=background)")
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

# %%
