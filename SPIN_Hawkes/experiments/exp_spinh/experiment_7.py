"""Experiment 7: SPIN-H VI recovery on the scenarios from experiment 5."""

#%%

import csv
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openturns as ot
from tqdm.auto import tqdm

try:
    from joblib import Parallel, delayed
except ImportError:
    Parallel = delayed = None

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(EXPERIMENT_DIR))

from package import SPINHVIConfig, SparseGP, generate_voronoi_cells, save_figure

from experiment_5 import (
    BASE_SEED,
    DOMAIN_SEED,
    ERROR_PLOT_FLOOR,
    INITIAL_BETA,
    INITIAL_ETAS,
    N_DOMAINS,
    PARAMETER_BLOCKS,
    PARAMETER_NAMES,
    SCENARIOS,
    THETA_PRIORS,
    X_BOUNDS,
    Y_BOUNDS,
    make_model,
    regular_grid,
    relative_error,
    rmse,
    simulate_scenario,
    true_latent_state,
)


# Variational inference
N_REALIZATIONS = 1
N_JOBS = -1
N_ITER = 300
TOLERANCE = 1e-4
VERBOSE = True
VERBOSE_EVERY = 50
ELBO_EVERY = 5

GP_BACKEND = "sparse"
USE_CALIBRATION = True
UPDATE_Z = True
UPDATE_POLYA_GAMMA = True
UPDATE_LATENT_POISSON = True
UPDATE_GP = True
UPDATE_EPS = True
UPDATE_ETAS = True
FIXED_ETAS = {}
FIXED_BETA = None

QUADRATURE_NX = 20
QUADRATURE_NY = 20
EPS_NEWTON_STEPS = 8
SPATIAL_COMPENSATOR_GRID = 10
ETAS_UPDATE_START = 5
ETAS_UPDATE_EVERY = 5
MAX_OPTIMIZER_ITER = 10
ETAS_QUADRATURE_NODES = 4
VI_JITTER = 1e-6

BETA_PRIOR = {"a_beta": 2.0, "b_beta": 1.0}
INITIAL_GAMMA_FACTORS = {
    "A": (10.0 * INITIAL_ETAS.A, 10.0),
    "alpha": (10.0 * INITIAL_ETAS.alpha, 10.0),
    "c": (100.0 * INITIAL_ETAS.c, 100.0),
    "p_minus_1": (10.0 * (INITIAL_ETAS.p - 1.0), 10.0),
    "d": (50.0 * INITIAL_ETAS.d, 50.0),
    "q_minus_1": (10.0 * (INITIAL_ETAS.q - 1.0), 10.0),
    "gamma": (10.0 * INITIAL_ETAS.gamma, 10.0),
    "beta": (10.0 * INITIAL_BETA, 10.0),
}

# Evaluation
BACKGROUND_THRESHOLD = 0.5
LAMBDA_GRID_SIZE = 35
RESULTS_DIR = PACKAGE_ROOT / "figures" / "experiments" / "exp_spinh"


def make_vi_config(seed):
    return SPINHVIConfig(
        n_iter=N_ITER,
        tolerance=TOLERANCE,
        verbose=VERBOSE and N_JOBS == 1,
        verbose_every=VERBOSE_EVERY,
        elbo_every=ELBO_EVERY,
        random_seed=seed,
        gp_backend=GP_BACKEND,
        use_calibration=USE_CALIBRATION,
        update_z=UPDATE_Z,
        update_polya_gamma=UPDATE_POLYA_GAMMA,
        update_latent_poisson=UPDATE_LATENT_POISSON,
        update_gp=UPDATE_GP,
        update_eps=UPDATE_EPS,
        update_etas=UPDATE_ETAS,
        fixed_etas=FIXED_ETAS,
        fixed_beta=FIXED_BETA,
        beta_prior=BETA_PRIOR,
        theta_priors=THETA_PRIORS,
        initial_gamma_factors=INITIAL_GAMMA_FACTORS,
        quadrature_nx=QUADRATURE_NX,
        quadrature_ny=QUADRATURE_NY,
        eps_newton_steps=EPS_NEWTON_STEPS,
        spatial_compensator_grid=SPATIAL_COMPENSATOR_GRID,
        etas_update_start=ETAS_UPDATE_START,
        etas_update_every=ETAS_UPDATE_EVERY,
        max_optimizer_iter=MAX_OPTIMIZER_ITER,
        etas_quadrature_nodes=ETAS_QUADRATURE_NODES,
        jitter=VI_JITTER,
    )


def declustering_metrics(fit, simulation):
    probabilities = np.asarray(fit.state.branching.probabilities, dtype=float)
    p_background = probabilities[:, 0]
    predicted_background = p_background >= BACKGROUND_THRESHOLD
    predicted_parent = np.full(len(simulation.catalog), -1, dtype=int)

    for child in np.flatnonzero(~predicted_background):
        if child > 0:
            predicted_parent[child] = int(
                np.argmax(probabilities[child, 1 : child + 1])
            )

    true_background = simulation.is_background
    true_triggered = ~true_background
    predicted_triggered = ~predicted_background
    parent_accuracy = (
        float(
            np.mean(
                predicted_parent[true_triggered]
                == simulation.parent_indices[true_triggered]
            )
        )
        if np.any(true_triggered)
        else np.nan
    )
    return {
        "background_accuracy": float(
            np.mean(predicted_background == true_background)
        ),
        "background_recall": (
            float(np.mean(predicted_background[true_background]))
            if np.any(true_background)
            else np.nan
        ),
        "triggered_recall": (
            float(np.mean(predicted_triggered[true_triggered]))
            if np.any(true_triggered)
            else np.nan
        ),
        "parent_accuracy_triggered": parent_accuracy,
    }


def _rbf_kernel(xy1, xy2, variance, length_scale):
    differences = xy1[:, None, :] - xy2[None, :, :]
    squared_distance = np.sum(differences**2, axis=2)
    return variance * np.exp(
        -squared_distance / (2.0 * length_scale**2)
    )


def variational_gp_mean(fit, xy):
    """Evaluate the variational GP mean at arbitrary coordinates."""
    xy = np.asarray(xy, dtype=float)
    coefficients = fit.state.gp.coefficients_mean
    if coefficients is not None:
        sparse_gp = fit.config.sparse_gp
        if sparse_gp is None:
            sparse_gp = SparseGP.from_bounds(
                fit.model.x_bounds,
                fit.model.y_bounds,
                fit.model.gp_prior.variance,
                fit.model.gp_prior.length_scale,
            )
        design = np.asarray(
            sparse_gp.regressorOT(ot.Sample(xy.tolist())),
            dtype=float,
        )
        return design @ np.asarray(coefficients, dtype=float)

    observed_xy = fit.catalog.xy
    prior = fit.model.gp_prior
    covariance = _rbf_kernel(
        observed_xy,
        observed_xy,
        prior.variance,
        prior.length_scale,
    )
    covariance += fit.model.jitter * np.eye(len(observed_xy))
    cross_covariance = _rbf_kernel(
        xy,
        observed_xy,
        prior.variance,
        prior.length_scale,
    )
    weights = np.linalg.solve(covariance, fit.state.gp.f_data_mean)
    return cross_covariance @ weights


def run_realization(polygons, scenario, realization):
    seed = BASE_SEED + 1000 * int(scenario["difficulty"]) + realization
    simulation = simulate_scenario(polygons, scenario, seed)
    model = make_model(polygons, scenario, etas_parameters=INITIAL_ETAS)

    start = time.perf_counter()
    fit = model.vi(simulation.catalog, config=make_vi_config(seed))
    elapsed_seconds = time.perf_counter() - start
    summary = fit.summary()

    _, _, xy_grid = regular_grid(LAMBDA_GRID_SIZE)
    t_eval = np.full(xy_grid.shape[0], scenario["duration"])
    f_hat = variational_gp_mean(fit, xy_grid)
    mu_hat = fit.model.background_intensity(
        xy_grid[:, 0],
        xy_grid[:, 1],
        summary["eps_mean"],
        f_hat,
    )
    triggering_hat = fit.model.triggering_intensity(
        t_eval,
        xy_grid[:, 0],
        xy_grid[:, 1],
        history=simulation.catalog,
        parameters=fit.etas_mean(),
    )
    total_hat = mu_hat + triggering_hat

    eps_true, f_true = true_latent_state(xy_grid, scenario)
    mu_true, triggering_true, total_true = model.conditional_intensity(
        t_eval=t_eval,
        x_eval=xy_grid[:, 0],
        y_eval=xy_grid[:, 1],
        history=simulation.catalog,
        eps=eps_true,
        latent_gp=f_true,
        parameters=scenario["etas"],
    )

    elbo_trace = np.asarray(summary["elbo_trace"], dtype=float)
    record = {
        "scenario": scenario["name"],
        "difficulty": scenario["difficulty"],
        "realization": realization,
        "seed": seed,
        "n_events": len(simulation.catalog),
        "n_background": simulation.n_background,
        "n_triggered": simulation.n_triggered,
        "true_background_ratio": (
            simulation.n_background / max(len(simulation.catalog), 1)
        ),
        "estimated_background_probability": float(
            np.mean(summary["p_background"])
        ),
        "background_rmse": rmse(mu_hat, mu_true),
        "triggering_rmse": rmse(triggering_hat, triggering_true),
        "total_rmse": rmse(total_hat, total_true),
        "elapsed_seconds": elapsed_seconds,
        "n_iter_run": fit.diagnostics["n_iter_run"],
        "converged": fit.diagnostics["converged"],
        "final_elbo": float(elbo_trace[-1]),
        "expected_latent_poisson_count": fit.diagnostics[
            "expected_latent_poisson_count"
        ],
    }
    record.update(declustering_metrics(fit, simulation))

    theta_hat = summary["theta_phi_hat"]
    theta_true = scenario["etas"].as_dict()
    for name in PARAMETER_NAMES:
        true_value = scenario["beta"] if name == "beta" else theta_true[name]
        estimate = summary["beta_hat"] if name == "beta" else theta_hat[name]
        record[f"true_{name}"] = float(true_value)
        record[f"estimate_{name}"] = float(estimate)
        record[f"relative_error_{name}"] = relative_error(estimate, true_value)

    print(
        f"{scenario['name']:<26} realization={realization} "
        f"N={len(simulation.catalog):4d} bg={record['true_background_ratio']:.2f} "
        f"A={record['estimate_A']:.3f}/{record['true_A']:.3f} "
        f"alpha={record['estimate_alpha']:.3f}/{record['true_alpha']:.3f} "
        f"total_RMSE={record['total_rmse']:.3f} "
        f"time={elapsed_seconds:.1f}s"
    )
    return record


def print_summary(records):
    print("\nVI recovery summary by scenario")
    print(
        f"{'scenario':<26} {'N':>7} {'bg':>6} "
        f"{'prod+mag':>10} {'temporal':>10} {'spatial':>10} "
        f"{'total_RMSE':>11} {'time(s)':>9}"
    )
    print("-" * 106)
    for scenario in SCENARIOS:
        rows = [row for row in records if row["scenario"] == scenario["name"]]
        if not rows:
            continue

        def mean(key):
            return float(np.nanmean([row[key] for row in rows]))

        def block_mean(names):
            return float(
                np.nanmean(
                    [
                        abs(row[f"relative_error_{name}"])
                        for row in rows
                        for name in names
                    ]
                )
            )

        print(
            f"{scenario['name']:<26} {mean('n_events'):7.1f} "
            f"{mean('true_background_ratio'):6.3f} "
            f"{block_mean(PARAMETER_BLOCKS['Productivity + magnitude']):10.3f} "
            f"{block_mean(PARAMETER_BLOCKS['Temporal kernel']):10.3f} "
            f"{block_mean(PARAMETER_BLOCKS['Spatial kernel']):10.3f} "
            f"{mean('total_rmse'):11.3f} "
            f"{mean('elapsed_seconds'):9.1f}"
        )


def plot_recovery_summary(records):
    difficulties = [scenario["difficulty"] for scenario in SCENARIOS]
    labels = [scenario["name"].replace("_", "\n") for scenario in SCENARIOS]
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16.0, 5.2),
        sharex=True,
        layout="constrained",
    )
    for ax, (title, parameter_names) in zip(axes, PARAMETER_BLOCKS.items()):
        for name in parameter_names:
            values = []
            for scenario in SCENARIOS:
                rows = [
                    row for row in records
                    if row["scenario"] == scenario["name"]
                ]
                values.append(
                    np.nanmean(
                        [abs(row[f"relative_error_{name}"]) for row in rows]
                    )
                    if rows
                    else np.nan
                )
            values = np.maximum(np.asarray(values), ERROR_PLOT_FLOOR)
            ax.plot(
                difficulties,
                values,
                marker="o",
                linewidth=1.4,
                label=name,
            )
        ax.set_xticks(difficulties)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
        ax.legend()
    axes[0].set_ylabel("Mean absolute relative error (log scale)")
    fig.suptitle("SPIN-H VI recovery across increasing difficulty")
    save_figure(fig, "package/experiment_7/vi_posterior_recovery")
    plt.show()
    return fig


def main():
    print("Experiment 7 - SPIN-H VI posterior recovery")
    print(
        f"scenarios={len(SCENARIOS)}, realizations={N_REALIZATIONS}, "
        f"n_iter={N_ITER}, gp_backend={GP_BACKEND}, n_jobs={N_JOBS}"
    )

    polygons, _ = generate_voronoi_cells(
        n_germs=N_DOMAINS,
        X_bounds=X_BOUNDS,
        Y_bounds=Y_BOUNDS,
        rng_seed=DOMAIN_SEED,
    )
    tasks = [
        (scenario, realization)
        for scenario in SCENARIOS
        for realization in range(N_REALIZATIONS)
    ]
    if N_JOBS == 1:
        records = [
            run_realization(polygons, scenario, realization)
            for scenario, realization in tqdm(
                tasks,
                desc="Experiment 7 realizations",
                unit="run",
                dynamic_ncols=True,
            )
        ]
    else:
        if Parallel is None:
            raise ImportError("joblib is required when N_JOBS != 1.")
        completed = Parallel(n_jobs=N_JOBS, return_as="generator")(
            delayed(run_realization)(polygons, scenario, realization)
            for scenario, realization in tasks
        )
        records = list(
            tqdm(
                completed,
                total=len(tasks),
                desc="Experiment 7 realizations",
                unit="run",
                dynamic_ncols=True,
            )
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "experiment_7_vi_posterior_recovery.csv"
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    print_summary(records)
    print(f"\nSaved VI recovery table: {output_path}")
    plot_recovery_summary(records)
    return records


if __name__ == "__main__":
    main()

# %%
