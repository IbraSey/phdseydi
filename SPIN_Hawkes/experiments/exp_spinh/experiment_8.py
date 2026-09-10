"""Experiment 8: SPIN-H VI oracle recovery for branching vs parameters."""

#%%

import csv
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

from package import SPINHVI, SPINHVIConfig, generate_voronoi_cells, save_figure

from experiment_5 import (
    BASE_SEED,
    DOMAIN_SEED,
    INITIAL_ETAS,
    N_DOMAINS,
    PARAMETER_BLOCKS,
    PARAMETER_NAMES,
    SCENARIOS as ALL_SCENARIOS,
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
    declustering_metrics,
    variational_gp_mean,
)


SCENARIO_NAMES = ("simple_well_conditioned", "complex_diffuse_triggering")
SCENARIOS = [
    scenario for scenario in ALL_SCENARIOS
    if scenario["name"] in SCENARIO_NAMES
]

# Variational inference
N_REALIZATIONS = 2
N_JOBS = 1
N_ITER = 300
TOLERANCE = 1e-4
VERBOSE = False
VERBOSE_EVERY = 50
ELBO_EVERY = 5

GP_BACKEND = "sparse"
USE_CALIBRATION = True
QUADRATURE_NX = 20
QUADRATURE_NY = 20
EPS_NEWTON_STEPS = 8
SPATIAL_COMPENSATOR_GRID = 10
ETAS_UPDATE_START = 5
ETAS_UPDATE_EVERY = 5
MAX_OPTIMIZER_ITER = 10
ETAS_QUADRATURE_NODES = 4
VI_JITTER = 1e-6

# Evaluation
LAMBDA_GRID_SIZE = 35
RESULTS_DIR = PACKAGE_ROOT / "figures" / "experiments" / "exp_spinh"

MODES = [
    {
        "name": "full_vi",
        "description": "update q(Z) and q(theta)",
        "update_z": True,
        "known_z": False,
        "true_theta": False,
    },
    {
        "name": "oracle_theta_update_z",
        "description": "theta known, update q(Z)",
        "update_z": True,
        "known_z": False,
        "true_theta": True,
    },
    {
        "name": "oracle_z_update_theta",
        "description": "Z known, update q(theta)",
        "update_z": False,
        "known_z": True,
        "true_theta": False,
    },
]


def make_model(polygons, scenario, mode):
    parameters = scenario["etas"] if mode["true_theta"] else INITIAL_ETAS
    return make_base_model(
        polygons,
        scenario,
        etas_parameters=parameters,
    )


def make_vi_config(scenario, mode, seed):
    fixed_theta = mode["true_theta"]
    return SPINHVIConfig(
        n_iter=N_ITER,
        tolerance=TOLERANCE,
        verbose=VERBOSE,
        verbose_every=VERBOSE_EVERY,
        elbo_every=ELBO_EVERY,
        random_seed=seed,
        gp_backend=GP_BACKEND,
        use_calibration=USE_CALIBRATION,
        update_z=mode["update_z"],
        update_polya_gamma=True,
        update_latent_poisson=True,
        update_gp=True,
        update_eps=True,
        update_etas=not fixed_theta,
        fixed_etas=scenario["etas"].as_dict() if fixed_theta else {},
        fixed_beta=scenario["beta"] if fixed_theta else None,
        beta_prior=BETA_PRIOR,
        theta_priors=THETA_PRIORS,
        initial_gamma_factors=(
            {} if fixed_theta else INITIAL_GAMMA_FACTORS
        ),
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


def _known_branching_probabilities(parent_indices):
    """Represent known branching labels as a degenerate q(Z)."""
    parent_indices = np.asarray(parent_indices, dtype=int)
    n_events = parent_indices.size
    probabilities = np.zeros((n_events, n_events), dtype=float)
    for child, parent in enumerate(parent_indices):
        probabilities[child, 0 if parent < 0 else parent + 1] = 1.0
    return probabilities


def fit_mode(model, catalog, config, known_parent_indices=None):
    """Run VI, optionally fixing q(Z) to the simulated branching structure."""
    inference_model = model
    if config.use_calibration:
        calibrated_prior = model.calibrate_gp_prior(
            catalog,
            rng_seed=config.random_seed,
            verbose=config.verbose,
        )
        inference_model = replace(model, gp_prior=calibrated_prior)

    engine = SPINHVI(inference_model, catalog, config=config)
    if known_parent_indices is not None:
        engine.state.branching.probabilities = (
            _known_branching_probabilities(known_parent_indices)
        )
    return engine.fit()


def parameter_block_errors(record):
    for block_name, names in PARAMETER_BLOCKS.items():
        safe_name = block_name.lower().replace(" + ", "_").replace(" ", "_")
        record[f"{safe_name}_mae_rel"] = float(
            np.nanmean(
                [abs(record[f"relative_error_{name}"]) for name in names]
            )
        )


def add_parameter_metrics(record, summary, scenario):
    theta_hat = summary["theta_phi_hat"]
    theta_true = scenario["etas"].as_dict()
    for name in PARAMETER_NAMES:
        true_value = scenario["beta"] if name == "beta" else theta_true[name]
        estimate = summary["beta_hat"] if name == "beta" else theta_hat[name]
        record[f"true_{name}"] = float(true_value)
        record[f"estimate_{name}"] = float(estimate)
        record[f"relative_error_{name}"] = relative_error(
            estimate,
            true_value,
        )
    parameter_block_errors(record)


def add_intensity_metrics(record, fit, model, scenario, catalog):
    _, _, xy_grid = regular_grid(LAMBDA_GRID_SIZE)
    t_eval = np.full(xy_grid.shape[0], scenario["duration"])
    f_hat = variational_gp_mean(fit, xy_grid)
    summary = fit.summary()
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
        history=catalog,
        parameters=fit.etas_mean(),
    )
    total_hat = mu_hat + triggering_hat

    eps_true, f_true = true_latent_state(xy_grid, scenario)
    mu_true, triggering_true, total_true = model.conditional_intensity(
        t_eval=t_eval,
        x_eval=xy_grid[:, 0],
        y_eval=xy_grid[:, 1],
        history=catalog,
        eps=eps_true,
        latent_gp=f_true,
        parameters=scenario["etas"],
    )
    record["background_rmse"] = rmse(mu_hat, mu_true)
    record["triggering_rmse"] = rmse(triggering_hat, triggering_true)
    record["total_rmse"] = rmse(total_hat, total_true)


def run_mode(polygons, scenario, simulation, realization, mode):
    mode_index = MODES.index(mode)
    seed = (
        BASE_SEED
        + 2000 * int(scenario["difficulty"])
        + 100 * realization
        + mode_index
    )
    catalog = simulation.catalog
    model = make_model(polygons, scenario, mode)
    config = make_vi_config(scenario, mode, seed)
    known_parents = simulation.parent_indices if mode["known_z"] else None

    start = time.perf_counter()
    fit = fit_mode(model, catalog, config, known_parents)
    elapsed_seconds = time.perf_counter() - start
    summary = fit.summary()
    elbo_trace = np.asarray(summary["elbo_trace"], dtype=float)

    record = {
        "scenario": scenario["name"],
        "difficulty": scenario["difficulty"],
        "realization": realization,
        "mode": mode["name"],
        "mode_description": mode["description"],
        "seed": seed,
        "n_events": len(catalog),
        "n_background": simulation.n_background,
        "n_triggered": simulation.n_triggered,
        "true_background_ratio": (
            simulation.n_background / max(len(catalog), 1)
        ),
        "estimated_background_probability": float(
            np.mean(summary["p_background"])
        ),
        "elapsed_seconds": elapsed_seconds,
        "n_iter_run": fit.diagnostics["n_iter_run"],
        "converged": fit.diagnostics["converged"],
        "final_elbo": float(elbo_trace[-1]),
        "expected_latent_poisson_count": fit.diagnostics[
            "expected_latent_poisson_count"
        ],
    }
    record.update(declustering_metrics(fit, simulation))
    add_parameter_metrics(record, summary, scenario)
    add_intensity_metrics(record, fit, model, scenario, catalog)

    print(
        f"{scenario['name']:<26} realization={realization} "
        f"{mode['name']:<23} "
        f"Zacc={record['background_accuracy']:.3f} "
        f"Pacc={record['parent_accuracy_triggered']:.3f} "
        f"prod={record['productivity_magnitude_mae_rel']:.3f} "
        f"temp={record['temporal_kernel_mae_rel']:.3f} "
        f"spat={record['spatial_kernel_mae_rel']:.3f} "
        f"time={elapsed_seconds:.1f}s"
    )
    return record


def simulate_catalog(polygons, scenario, realization):
    seed = BASE_SEED + 1000 * int(scenario["difficulty"]) + realization
    return simulate_scenario(polygons, scenario, seed)


def fieldnames():
    base = [
        "scenario",
        "difficulty",
        "realization",
        "mode",
        "mode_description",
        "seed",
        "n_events",
        "n_background",
        "n_triggered",
        "true_background_ratio",
        "estimated_background_probability",
        "background_accuracy",
        "background_recall",
        "triggered_recall",
        "parent_accuracy_triggered",
        "background_rmse",
        "triggering_rmse",
        "total_rmse",
        "productivity_magnitude_mae_rel",
        "temporal_kernel_mae_rel",
        "spatial_kernel_mae_rel",
        "elapsed_seconds",
        "n_iter_run",
        "converged",
        "final_elbo",
        "expected_latent_poisson_count",
    ]
    parameter_fields = []
    for name in PARAMETER_NAMES:
        parameter_fields.extend(
            [
                f"true_{name}",
                f"estimate_{name}",
                f"relative_error_{name}",
            ]
        )
    return base + parameter_fields


def save_records(records):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "experiment_8_vi_oracle_branching_parameters.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames())
        writer.writeheader()
        writer.writerows(records)
    return path


def mean_value(rows, key):
    return float(np.nanmean([row[key] for row in rows]))


def print_summary(records):
    print("\nVI oracle recovery summary")
    print(
        f"{'scenario':<26} {'mode':<23} {'Zacc':>7} {'Prec':>7} "
        f"{'prod':>7} {'temp':>7} {'spat':>7} {'totRMSE':>8} {'time':>8}"
    )
    print("-" * 108)
    for scenario in SCENARIOS:
        for mode in MODES:
            rows = [
                row for row in records
                if row["scenario"] == scenario["name"]
                and row["mode"] == mode["name"]
            ]
            if not rows:
                continue
            print(
                f"{scenario['name']:<26} {mode['name']:<23} "
                f"{mean_value(rows, 'background_accuracy'):7.3f} "
                f"{mean_value(rows, 'parent_accuracy_triggered'):7.3f} "
                f"{mean_value(rows, 'productivity_magnitude_mae_rel'):7.3f} "
                f"{mean_value(rows, 'temporal_kernel_mae_rel'):7.3f} "
                f"{mean_value(rows, 'spatial_kernel_mae_rel'):7.3f} "
                f"{mean_value(rows, 'total_rmse'):8.3f} "
                f"{mean_value(rows, 'elapsed_seconds'):8.1f}"
            )


def plot_oracle_summary(records):
    scenario_labels = [
        scenario["name"].replace("_", "\n") for scenario in SCENARIOS
    ]
    mode_labels = [mode["name"].replace("_", "\n") for mode in MODES]
    fig, axes = plt.subplots(
        len(SCENARIOS),
        2,
        figsize=(12.0, 4.4 * len(SCENARIOS)),
        layout="constrained",
    )
    axes = np.atleast_2d(axes)
    for row_index, scenario in enumerate(SCENARIOS):
        scenario_rows = [
            row for row in records if row["scenario"] == scenario["name"]
        ]
        z_values = []
        theta_values = []
        for mode in MODES:
            rows = [
                row for row in scenario_rows if row["mode"] == mode["name"]
            ]
            z_values.append(
                mean_value(rows, "parent_accuracy_triggered")
                if rows
                else np.nan
            )
            theta_values.append(
                [
                    mean_value(rows, "productivity_magnitude_mae_rel")
                    if rows
                    else np.nan,
                    mean_value(rows, "temporal_kernel_mae_rel")
                    if rows
                    else np.nan,
                    mean_value(rows, "spatial_kernel_mae_rel")
                    if rows
                    else np.nan,
                ]
            )

        ax = axes[row_index, 0]
        ax.bar(np.arange(len(MODES)), z_values, color="steelblue")
        ax.set_xticks(np.arange(len(MODES)))
        ax.set_xticklabels(mode_labels)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Parent accuracy")
        ax.set_title(f"Z recovery | {scenario_labels[row_index]}")
        ax.grid(axis="y", alpha=0.3)

        ax = axes[row_index, 1]
        theta_values = np.asarray(theta_values)
        width = 0.25
        x_locations = np.arange(len(MODES))
        for offset, (label, color) in enumerate(
            [
                ("prod+mag", "#4C78A8"),
                ("temporal", "#F58518"),
                ("spatial", "#54A24B"),
            ]
        ):
            ax.bar(
                x_locations + (offset - 1) * width,
                theta_values[:, offset],
                width,
                label=label,
                color=color,
            )
        ax.set_xticks(x_locations)
        ax.set_xticklabels(mode_labels)
        ax.set_ylabel("Mean abs. relative error")
        ax.set_title(f"theta recovery | {scenario_labels[row_index]}")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
    save_figure(fig, "package/experiment_8/vi_oracle_branching_parameters")
    plt.show()
    return fig


def run_realization_modes(polygons, scenario, realization):
    simulation = simulate_catalog(polygons, scenario, realization)
    print(
        f"{scenario['name']} realization={realization} "
        f"N={len(simulation.catalog)} "
        f"bg={simulation.n_background / max(len(simulation.catalog), 1):.3f}"
    )
    return [
        run_mode(polygons, scenario, simulation, realization, mode)
        for mode in MODES
    ]


def run_all_realizations(polygons):
    tasks = [
        (scenario, realization)
        for scenario in SCENARIOS
        for realization in range(N_REALIZATIONS)
    ]
    if N_JOBS == 1:
        nested_records = [
            run_realization_modes(polygons, scenario, realization)
            for scenario, realization in tqdm(
                tasks,
                desc="Experiment 8 realizations",
                unit="run",
                dynamic_ncols=True,
            )
        ]
    else:
        if Parallel is None:
            raise ImportError("joblib is required when N_JOBS != 1.")
        print(
            f"Running {len(tasks)} independent realizations "
            f"with N_JOBS={N_JOBS}"
        )
        completed = Parallel(n_jobs=N_JOBS, return_as="generator")(
            delayed(run_realization_modes)(polygons, scenario, realization)
            for scenario, realization in tasks
        )
        nested_records = list(
            tqdm(
                completed,
                total=len(tasks),
                desc="Experiment 8 realizations",
                unit="run",
                dynamic_ncols=True,
            )
        )
    return [record for records in nested_records for record in records]


def main():
    print("Experiment 8 - SPIN-H VI oracle Z/theta recovery")
    print(
        f"scenarios={len(SCENARIOS)}, modes={len(MODES)}, "
        f"realizations={N_REALIZATIONS}, n_iter={N_ITER}, "
        f"gp_backend={GP_BACKEND}, n_jobs={N_JOBS}"
    )
    polygons, _ = generate_voronoi_cells(
        n_germs=N_DOMAINS,
        X_bounds=X_BOUNDS,
        Y_bounds=Y_BOUNDS,
        rng_seed=DOMAIN_SEED,
    )

    records = run_all_realizations(polygons)
    path = save_records(records)
    print_summary(records)
    print(f"\nSaved VI oracle recovery table: {path}")
    plot_oracle_summary(records)
    return records


if __name__ == "__main__":
    main()

# %%
