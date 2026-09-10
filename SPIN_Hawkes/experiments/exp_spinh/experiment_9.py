"""Experiment 9: Gibbs versus VI for SPIN-H inference and prediction.

Predictive log-scores use posterior-mean plug-in parameters for both methods.
"""

#%%

import csv
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import truncexpon
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
)
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(EXPERIMENT_DIR))

from package import (
    ETASParameters,
    EventCatalog,
    SPINHGibbsConfig,
    SPINHVIConfig,
    generate_voronoi_cells,
    save_figure,
)

from experiment_5 import (
    BASE_SEED,
    DOMAIN_SEED,
    INITIAL_BETA,
    INITIAL_ETAS,
    N_DOMAINS,
    PARAMETER_BLOCKS,
    PARAMETER_NAMES,
    SCENARIOS,
    THETA_PRIORS,
    X_BOUNDS,
    Y_BOUNDS,
    make_model as make_base_model,
    regular_grid,
    relative_error,
    rmse,
    simulate_scenario,
    true_latent_state,
)
from experiment_7 import (
    BETA_PRIOR,
    INITIAL_GAMMA_FACTORS,
    variational_gp_mean,
)


# Common protocol
N_REALIZATIONS = 1
TRAIN_FRACTION = 0.8
GP_BACKEND = "sparse"
USE_CALIBRATION = True
VERBOSE = False
BACKGROUND_THRESHOLD = 0.5

# Gibbs
GIBBS_N_ITER = 1000
GIBBS_THIN = 2
GIBBS_BURN_IN = 0.5
GIBBS_SIGMA_MH_ETAS = 0.05
GIBBS_SIGMA_MH_BETA = 0.1

# VI
VI_N_ITER = 300
VI_TOLERANCE = 1e-4
VI_ELBO_EVERY = 5
VI_QUADRATURE_NX = 20
VI_QUADRATURE_NY = 20
VI_EPS_NEWTON_STEPS = 8
VI_SPATIAL_COMPENSATOR_GRID = 10
VI_ETAS_UPDATE_START = 5
VI_ETAS_UPDATE_EVERY = 5
VI_MAX_OPTIMIZER_ITER = 10
VI_ETAS_QUADRATURE_NODES = 4
VI_JITTER = 1e-6

# Evaluation
INTENSITY_GRID_SIZE = 25
PREDICTIVE_GRID_SIZE = 25
PREDICTIVE_SPATIAL_GRID = 20
RESULTS_DIR = PACKAGE_ROOT / "figures" / "experiments" / "exp_spinh"


def make_training_model(polygons, scenario, train_end):
    training_scenario = dict(scenario)
    training_scenario["duration"] = float(train_end)
    return make_base_model(
        polygons,
        training_scenario,
        etas_parameters=INITIAL_ETAS,
    )


def make_gibbs_config(scenario):
    return SPINHGibbsConfig(
        n_iter=GIBBS_N_ITER,
        thin=GIBBS_THIN,
        mala_step=scenario["mala_step"],
        verbose=VERBOSE,
        verbose_every=max(1, GIBBS_N_ITER // 10),
        use_calibration=USE_CALIBRATION,
        beta_init=INITIAL_BETA,
        theta_priors=THETA_PRIORS,
        sigma_mh_etas=GIBBS_SIGMA_MH_ETAS,
        sigma_mh_beta=GIBBS_SIGMA_MH_BETA,
        adaptation_start=min(200, max(1, GIBBS_N_ITER // 5)),
        proposal_jitter=1e-6,
    )


def make_vi_config(seed):
    return SPINHVIConfig(
        n_iter=VI_N_ITER,
        tolerance=VI_TOLERANCE,
        verbose=VERBOSE,
        verbose_every=max(1, VI_N_ITER // 10),
        elbo_every=VI_ELBO_EVERY,
        random_seed=seed,
        gp_backend=GP_BACKEND,
        use_calibration=USE_CALIBRATION,
        update_z=True,
        update_polya_gamma=True,
        update_latent_poisson=True,
        update_gp=True,
        update_eps=True,
        update_etas=True,
        fixed_etas={},
        fixed_beta=None,
        beta_prior=BETA_PRIOR,
        theta_priors=THETA_PRIORS,
        initial_gamma_factors=INITIAL_GAMMA_FACTORS,
        quadrature_nx=VI_QUADRATURE_NX,
        quadrature_ny=VI_QUADRATURE_NY,
        eps_newton_steps=VI_EPS_NEWTON_STEPS,
        spatial_compensator_grid=VI_SPATIAL_COMPENSATOR_GRID,
        etas_update_start=VI_ETAS_UPDATE_START,
        etas_update_every=VI_ETAS_UPDATE_EVERY,
        max_optimizer_iter=VI_MAX_OPTIMIZER_ITER,
        etas_quadrature_nodes=VI_ETAS_QUADRATURE_NODES,
        jitter=VI_JITTER,
    )


def subset_catalog(catalog, mask):
    magnitudes = (
        None if catalog.magnitudes is None else catalog.magnitudes[mask]
    )
    return EventCatalog(
        catalog.t[mask],
        catalog.x[mask],
        catalog.y[mask],
        magnitudes,
    )


def temporal_split(simulation, scenario):
    split_time = TRAIN_FRACTION * float(scenario["duration"])
    train_mask = simulation.catalog.t < split_time
    test_mask = ~train_mask
    if not np.any(train_mask) or not np.any(test_mask):
        raise ValueError("The temporal split must contain train and test events.")
    return (
        split_time,
        subset_catalog(simulation.catalog, train_mask),
        subset_catalog(simulation.catalog, test_mask),
        simulation.parent_indices[train_mask],
    )


def _classification_metrics(p_background, true_parent_indices):
    p_background = np.asarray(p_background, dtype=float)
    true_parent_indices = np.asarray(true_parent_indices, dtype=int)
    true_triggered = (true_parent_indices >= 0).astype(int)
    predicted_triggered = (p_background < BACKGROUND_THRESHOLD).astype(int)
    probabilities = np.column_stack(
        [p_background, 1.0 - p_background]
    )
    return {
        "background_accuracy": float(
            accuracy_score(true_triggered, predicted_triggered)
        ),
        "triggered_f1": float(
            f1_score(true_triggered, predicted_triggered, zero_division=0)
        ),
        "background_log_loss": float(
            log_loss(true_triggered, probabilities, labels=[0, 1])
        ),
        "background_brier": float(
            brier_score_loss(true_triggered, 1.0 - p_background)
        ),
    }


def gibbs_branching_metrics(fit, true_parent_indices):
    chain = np.asarray(fit.branching_chain, dtype=int)
    burn = int(GIBBS_BURN_IN * chain.shape[0])
    chain = chain[burn:]
    true_labels = np.where(
        true_parent_indices < 0,
        0,
        np.asarray(true_parent_indices) + 1,
    )
    p_background = np.mean(chain == 0, axis=0)
    predicted_parent = np.full(chain.shape[1], -1, dtype=int)
    true_probabilities = np.empty(chain.shape[1], dtype=float)
    smoothing = 0.5

    for child in range(chain.shape[1]):
        n_classes = child + 1
        counts = np.bincount(chain[:, child], minlength=n_classes)
        true_probabilities[child] = (
            counts[true_labels[child]] + smoothing
        ) / (chain.shape[0] + smoothing * n_classes)
        if p_background[child] < BACKGROUND_THRESHOLD and child > 0:
            parent_counts = counts[1 : child + 1]
            predicted_parent[child] = int(np.argmax(parent_counts))

    true_triggered = true_parent_indices >= 0
    metrics = _classification_metrics(p_background, true_parent_indices)
    metrics.update(
        {
            "parent_accuracy_triggered": (
                float(
                    np.mean(
                        predicted_parent[true_triggered]
                        == true_parent_indices[true_triggered]
                    )
                )
                if np.any(true_triggered)
                else np.nan
            ),
            "branching_log_score": float(
                np.mean(np.log(true_probabilities))
            ),
            "estimated_background_probability": float(
                np.mean(p_background)
            ),
        }
    )
    return metrics


def vi_branching_metrics(fit, true_parent_indices):
    probabilities = np.asarray(
        fit.state.branching.probabilities,
        dtype=float,
    )
    p_background = probabilities[:, 0]
    predicted_parent = np.full(probabilities.shape[0], -1, dtype=int)
    true_labels = np.where(
        true_parent_indices < 0,
        0,
        np.asarray(true_parent_indices) + 1,
    )

    for child in np.flatnonzero(p_background < BACKGROUND_THRESHOLD):
        if child > 0:
            predicted_parent[child] = int(
                np.argmax(probabilities[child, 1 : child + 1])
            )

    true_probability = probabilities[
        np.arange(probabilities.shape[0]),
        true_labels,
    ]
    true_triggered = true_parent_indices >= 0
    metrics = _classification_metrics(p_background, true_parent_indices)
    metrics.update(
        {
            "parent_accuracy_triggered": (
                float(
                    np.mean(
                        predicted_parent[true_triggered]
                        == true_parent_indices[true_triggered]
                    )
                )
                if np.any(true_triggered)
                else np.nan
            ),
            "branching_log_score": float(
                np.mean(
                    np.log(
                        np.maximum(
                            true_probability,
                            np.finfo(float).eps,
                        )
                    )
                )
            ),
            "estimated_background_probability": float(
                np.mean(p_background)
            ),
        }
    )
    return metrics


def parameter_metrics(summary, scenario):
    theta_hat = summary["theta_phi_hat"]
    theta_true = scenario["etas"].as_dict()
    metrics = {}
    absolute_relative_errors = []
    for name in PARAMETER_NAMES:
        truth = scenario["beta"] if name == "beta" else theta_true[name]
        estimate = summary["beta_hat"] if name == "beta" else theta_hat[name]
        error = relative_error(estimate, truth)
        metrics[f"true_{name}"] = float(truth)
        metrics[f"estimate_{name}"] = float(estimate)
        metrics[f"relative_error_{name}"] = float(error)
        absolute_relative_errors.append(abs(error))

    metrics["parameter_mae_relative"] = float(
        np.mean(absolute_relative_errors)
    )
    for block_name, names in PARAMETER_BLOCKS.items():
        safe_name = block_name.lower().replace(" + ", "_").replace(" ", "_")
        metrics[f"{safe_name}_mae_rel"] = float(
            np.mean(
                [abs(metrics[f"relative_error_{name}"]) for name in names]
            )
        )
    return metrics


def latent_and_intensity_metrics(
    method,
    fit,
    model,
    scenario,
    train_catalog,
):
    if method == "Gibbs":
        summary = fit.summary(burn_in=GIBBS_BURN_IN)
        eps_estimate = summary["eps_hat"]
        f_data_estimate = summary["f_data_hat"]
    else:
        summary = fit.summary()
        eps_estimate = summary["eps_mean"]
        f_data_estimate = summary["f_data_mean"]

    eps_true, f_data_true = true_latent_state(
        train_catalog.xy,
        scenario,
    )
    metrics = {
        "eps_rmse": rmse(eps_estimate, eps_true),
        "gp_data_rmse": rmse(f_data_estimate, f_data_true),
    }

    _, _, xy_grid = regular_grid(INTENSITY_GRID_SIZE)
    t_eval = np.full(xy_grid.shape[0], model.duration)
    if method == "Gibbs":
        mu_hat, triggering_hat, total_hat = fit.conditional_intensity(
            t=t_eval,
            x=xy_grid[:, 0],
            y=xy_grid[:, 1],
            burn_in=GIBBS_BURN_IN,
        )
    else:
        f_grid = variational_gp_mean(fit, xy_grid)
        mu_hat = fit.model.background_intensity(
            xy_grid[:, 0],
            xy_grid[:, 1],
            summary["eps_mean"],
            f_grid,
        )
        triggering_hat = fit.model.triggering_intensity(
            t_eval,
            xy_grid[:, 0],
            xy_grid[:, 1],
            history=train_catalog,
            parameters=fit.etas_mean(),
        )
        total_hat = mu_hat + triggering_hat

    eps_grid_true, f_grid_true = true_latent_state(xy_grid, scenario)
    mu_true, triggering_true, total_true = model.conditional_intensity(
        t_eval=t_eval,
        x_eval=xy_grid[:, 0],
        y_eval=xy_grid[:, 1],
        history=train_catalog,
        eps=eps_grid_true,
        latent_gp=f_grid_true,
        parameters=scenario["etas"],
    )
    metrics.update(
        {
            "background_rmse": rmse(mu_hat, mu_true),
            "triggering_rmse": rmse(triggering_hat, triggering_true),
            "total_rmse": rmse(total_hat, total_true),
        }
    )
    return metrics


def midpoint_grid(model, n_grid):
    x_edges = np.linspace(
        model.x_bounds[0],
        model.x_bounds[1],
        int(n_grid) + 1,
    )
    y_edges = np.linspace(
        model.y_bounds[0],
        model.y_bounds[1],
        int(n_grid) + 1,
    )
    x_mid = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_mid = 0.5 * (y_edges[:-1] + y_edges[1:])
    X, Y = np.meshgrid(x_mid, y_mid)
    xy = np.column_stack([X.ravel(), Y.ravel()])
    inside = model.domains.locate(xy[:, 0], xy[:, 1]) >= 0
    cell_area = (
        (model.x_bounds[1] - model.x_bounds[0])
        * (model.y_bounds[1] - model.y_bounds[0])
        / int(n_grid) ** 2
    )
    return xy[inside], cell_area


def predictive_log_scores(
    method,
    fit,
    model,
    full_catalog,
    test_catalog,
    test_start,
    test_end,
):
    if method == "Gibbs":
        summary = fit.summary(burn_in=GIBBS_BURN_IN)
        parameters = ETASParameters(**summary["theta_phi_hat"])
        beta = float(summary["beta_hat"])

        def background(xy):
            return fit.background_intensity(
                xy[:, 0],
                xy[:, 1],
                burn_in=GIBBS_BURN_IN,
            )
    else:
        summary = fit.summary()
        parameters = fit.etas_mean()
        beta = float(summary["beta_hat"])

        def background(xy):
            latent_gp = variational_gp_mean(fit, xy)
            return fit.model.background_intensity(
                xy[:, 0],
                xy[:, 1],
                summary["eps_mean"],
                latent_gp,
            )

    event_background = background(test_catalog.xy)
    event_triggering = model.triggering_intensity(
        test_catalog.t,
        test_catalog.x,
        test_catalog.y,
        history=full_catalog,
        parameters=parameters,
    )
    event_intensity = event_background + event_triggering
    event_term = float(
        np.sum(np.log(np.maximum(event_intensity, np.finfo(float).tiny)))
    )

    spatial_grid, cell_area = midpoint_grid(
        model,
        PREDICTIVE_GRID_SIZE,
    )
    background_compensator = float(
        (test_end - test_start)
        * cell_area
        * np.sum(background(spatial_grid))
    )

    magnitudes = full_catalog.magnitudes
    productivity = model.etas_kernel.productivity.evaluate(
        magnitudes,
        parameters,
        model.magnitude_min,
    )
    temporal_end = model.etas_kernel.temporal.integral_until(
        full_catalog.t,
        test_end,
        parameters,
    )
    temporal_start = model.etas_kernel.temporal.integral_until(
        full_catalog.t,
        test_start,
        parameters,
    )
    spatial_mass = model.etas_kernel.spatial.retained_mass(
        full_catalog.x,
        full_catalog.y,
        magnitudes,
        parameters,
        model.magnitude_min,
        model.x_bounds,
        model.y_bounds,
        n_grid=PREDICTIVE_SPATIAL_GRID,
        observation_domain=model.domains.observation_geometry,
    )
    triggering_compensator = float(
        np.sum(
            productivity
            * (temporal_end - temporal_start)
            * spatial_mass
        )
    )
    point_process_score = (
        event_term - background_compensator - triggering_compensator
    )

    width = model.magnitude_max - model.magnitude_min
    magnitude_score = float(
        np.sum(
            truncexpon.logpdf(
                test_catalog.magnitudes,
                b=beta * width,
                loc=model.magnitude_min,
                scale=1.0 / beta,
            )
        )
    )
    marked_score = point_process_score + magnitude_score
    n_test = max(len(test_catalog), 1)
    return {
        "predictive_event_log_term": event_term,
        "predictive_background_compensator": background_compensator,
        "predictive_triggering_compensator": triggering_compensator,
        "predictive_point_process_log_score": point_process_score,
        "predictive_magnitude_log_score": magnitude_score,
        "predictive_marked_log_score": marked_score,
        "predictive_marked_log_score_per_event": marked_score / n_test,
    }


def fit_method(
    method,
    model,
    scenario,
    train_catalog,
    seed,
):
    start = time.perf_counter()
    if method == "Gibbs":
        fit = model.gibbs(
            train_catalog,
            config=make_gibbs_config(scenario),
            gp_backend=GP_BACKEND,
            rng_seed=seed,
        )
        iterations = GIBBS_N_ITER
        objective = np.nan
        latent_count = float(
            np.mean(
                fit.latent_point_counts[
                    int(GIBBS_BURN_IN * fit.latent_point_counts.size):
                ]
            )
        )
    else:
        fit = model.vi(
            train_catalog,
            config=make_vi_config(seed),
        )
        iterations = fit.diagnostics["n_iter_run"]
        objective = float(fit.elbo_trace[-1])
        latent_count = float(
            fit.diagnostics["expected_latent_poisson_count"]
        )
    elapsed = time.perf_counter() - start
    return fit, {
        "elapsed_seconds": elapsed,
        "iterations_run": iterations,
        "seconds_per_iteration": elapsed / max(iterations, 1),
        "final_elbo": objective,
        "latent_poisson_count": latent_count,
    }


def run_method(
    method,
    polygons,
    scenario,
    simulation,
    realization,
):
    split_time, train_catalog, test_catalog, train_parents = temporal_split(
        simulation,
        scenario,
    )
    model = make_training_model(polygons, scenario, split_time)
    seed = (
        BASE_SEED
        + 2000 * int(scenario["difficulty"])
        + 100 * realization
        + (0 if method == "Gibbs" else 1)
    )
    fit, timing = fit_method(
        method,
        model,
        scenario,
        train_catalog,
        seed,
    )

    summary = (
        fit.summary(burn_in=GIBBS_BURN_IN)
        if method == "Gibbs"
        else fit.summary()
    )
    branching = (
        gibbs_branching_metrics(fit, train_parents)
        if method == "Gibbs"
        else vi_branching_metrics(fit, train_parents)
    )
    record = {
        "scenario": scenario["name"],
        "difficulty": scenario["difficulty"],
        "realization": realization,
        "method": method,
        "seed": seed,
        "n_train": len(train_catalog),
        "n_test": len(test_catalog),
        "train_end": split_time,
    }
    record.update(timing)
    record.update(branching)
    record.update(parameter_metrics(summary, scenario))
    record.update(
        latent_and_intensity_metrics(
            method,
            fit,
            model,
            scenario,
            train_catalog,
        )
    )
    record.update(
        predictive_log_scores(
            method,
            fit,
            model,
            simulation.catalog,
            test_catalog,
            split_time,
            scenario["duration"],
        )
    )

    print(
        f"{scenario['name']:<26} {method:<5} "
        f"N={len(train_catalog):4d}+{len(test_catalog):3d} "
        f"Zacc={record['background_accuracy']:.3f} "
        f"Pacc={record['parent_accuracy_triggered']:.3f} "
        f"param={record['parameter_mae_relative']:.3f} "
        f"time={record['elapsed_seconds']:.1f}s "
        f"logS={record['predictive_marked_log_score_per_event']:.3f}"
    )
    return record


def run_realization(polygons, scenario, realization):
    seed = BASE_SEED + 1000 * int(scenario["difficulty"]) + realization
    simulation = simulate_scenario(polygons, scenario, seed)
    return [
        run_method(
            method,
            polygons,
            scenario,
            simulation,
            realization,
        )
        for method in ("Gibbs", "VI")
    ]


def mean_value(rows, key):
    return float(np.nanmean([row[key] for row in rows]))


def print_summary(records):
    print("\nGibbs versus VI summary")
    print(
        f"{'scenario':<26} {'method':<6} {'Zacc':>7} {'ZlogS':>8} "
        f"{'param':>7} {'intRMSE':>9} {'time(s)':>9} {'predLogS':>9}"
    )
    print("-" * 101)
    for scenario in SCENARIOS:
        for method in ("Gibbs", "VI"):
            rows = [
                row for row in records
                if row["scenario"] == scenario["name"]
                and row["method"] == method
            ]
            if not rows:
                continue
            print(
                f"{scenario['name']:<26} {method:<6} "
                f"{mean_value(rows, 'background_accuracy'):7.3f} "
                f"{mean_value(rows, 'branching_log_score'):8.3f} "
                f"{mean_value(rows, 'parameter_mae_relative'):7.3f} "
                f"{mean_value(rows, 'total_rmse'):9.3f} "
                f"{mean_value(rows, 'elapsed_seconds'):9.1f} "
                f"{mean_value(rows, 'predictive_marked_log_score_per_event'):9.3f}"
            )


def plot_comparison(records):
    metrics = [
        ("background_log_loss", "Background log-loss (lower is better)"),
        ("parameter_mae_relative", "Parameter error (lower is better)"),
        ("elapsed_seconds", "Fit time in seconds (lower is better)"),
        (
            "predictive_marked_log_score_per_event",
            "Predictive log-score/event (higher is better)",
        ),
    ]
    labels = [scenario["name"].replace("_", "\n") for scenario in SCENARIOS]
    x_locations = np.arange(len(SCENARIOS))
    width = 0.36
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14.0, 9.0),
        layout="constrained",
    )
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        for offset, (method, color) in enumerate(
            [("Gibbs", "#4C78A8"), ("VI", "#F58518")]
        ):
            values = []
            for scenario in SCENARIOS:
                rows = [
                    row for row in records
                    if row["scenario"] == scenario["name"]
                    and row["method"] == method
                ]
                values.append(mean_value(rows, metric) if rows else np.nan)
            ax.bar(
                x_locations + (offset - 0.5) * width,
                values,
                width,
                label=method,
                color=color,
            )
        ax.set_xticks(x_locations)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
    fig.suptitle("SPIN-H Gibbs versus VI")
    save_figure(fig, "package/experiment_9/gibbs_vi_comparison")
    plt.show()
    return fig


def main():
    print("Experiment 9 - SPIN-H Gibbs versus VI")
    print(
        f"scenarios={len(SCENARIOS)}, realizations={N_REALIZATIONS}, "
        f"train_fraction={TRAIN_FRACTION}, gp_backend={GP_BACKEND}, "
        f"Gibbs_iter={GIBBS_N_ITER}, VI_iter={VI_N_ITER}"
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
    records = []
    for scenario, realization in tqdm(
        tasks,
        desc="Experiment 9 comparisons",
        unit="run",
        dynamic_ncols=True,
    ):
        records.extend(run_realization(polygons, scenario, realization))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "experiment_9_gibbs_vi_comparison.csv"
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(records[0].keys()),
        )
        writer.writeheader()
        writer.writerows(records)

    print_summary(records)
    print(f"\nSaved Gibbs/VI comparison table: {output_path}")
    plot_comparison(records)
    return records


if __name__ == "__main__":
    main()

# %%
