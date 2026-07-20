"""Experiment 6 — SPIN-H oracle recovery for branching vs parameters.

This experiment separates two sources of difficulty:

1. Can SPIN-H recover branching labels Z when ETAS parameters theta_-Z are known?
2. Can SPIN-H recover ETAS parameters theta_-Z when branching labels Z are known?

The experiment uses one structured scenario and one harder diffuse-triggering
scenario from the shared scenario definitions.
"""

#%%
import csv
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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


from package import (
    GPParameters,
    SPINHGibbsConfig,
    SPINHModel,
    generate_voronoi_cells,
    save_figure,
    simulate_hawkes_process,
)

from experiment_5 import (
    BASE_SEED,
    declustering_metrics,
    DOMAIN_SEED,
    INITIAL_BETA,
    INITIAL_ETAS,
    MAGNITUDE_MAX,
    MAGNITUDE_MIN,
    N_DOMAINS,
    PARAMETER_BLOCKS,
    PARAMETER_NAMES,
    SCENARIOS as ALL_SCENARIOS,
    THETA_PRIORS,
    X_BOUNDS,
    Y_BOUNDS,
    latent_field,
    regular_grid,
    relative_error,
    rmse,
    true_latent_state,
)


SCENARIO_NAMES = ("simple_well_conditioned", "complex_diffuse_triggering")
SCENARIOS = [scenario for scenario in ALL_SCENARIOS if scenario["name"] in SCENARIO_NAMES]

N_REALIZATIONS = 2
N_JOBS = 1
N_ITER = 2500
THIN = 3
BURN_IN = 0.5
SIGMA_MH_ETAS = 0.05
SIGMA_MH_BETA = 0.1
USE_SPARSE_GP = True
VERBOSE = False
VERBOSE_EVERY = 250
LAMBDA_GRID_SIZE = 35

MODES = [
    {
        "name": "full_gibbs",
        "description": "sample Z and theta",
        "sample_z": True,
        "known_z": False,
        "true_theta": False,
    },
    {
        "name": "oracle_theta_sample_z",
        "description": "theta known, sample Z",
        "sample_z": True,
        "known_z": False,
        "true_theta": True,
    },
    {
        "name": "oracle_z_sample_theta",
        "description": "Z known, sample theta",
        "sample_z": False,
        "known_z": True,
        "true_theta": False,
    },
]

RESULTS_DIR = PACKAGE_ROOT / "figures" / "experiments" / "exp_spinh"



def make_model(polygons, scenario, mode):
    etas_parameters = scenario["etas"] if mode["true_theta"] else INITIAL_ETAS
    return SPINHModel.from_polygons(
        polygons=polygons,
        duration=scenario["duration"],
        x_bounds=X_BOUNDS,
        y_bounds=Y_BOUNDS,
        gp_prior=GPParameters(variance=5.0, length_scale=0.2),
        eps_prior_variance=1.0,
        eps_prior_length_scale=0.01,
        nu_prior_rate=0.5,
        jitter=1e-5,
        etas_parameters=etas_parameters,
        magnitude_min=MAGNITUDE_MIN,
        magnitude_max=MAGNITUDE_MAX,
    )


def parameter_block_errors(record):
    for block_name, names in PARAMETER_BLOCKS.items():
        safe_name = block_name.lower().replace(" + ", "_").replace(" ", "_")
        record[f"{safe_name}_mae_rel"] = float(
            np.nanmean([abs(record[f"relative_error_{name}"]) for name in names])
        )


def add_parameter_metrics(record, summary, scenario):
    theta_hat = summary["theta_phi_hat"]
    theta_true = scenario["etas"].as_dict()
    for name in PARAMETER_NAMES:
        true_value = scenario["beta"] if name == "beta" else theta_true[name]
        estimate = summary["beta_hat"] if name == "beta" else theta_hat[name]
        record[f"true_{name}"] = float(true_value)
        record[f"estimate_{name}"] = float(estimate)
        record[f"relative_error_{name}"] = relative_error(estimate, true_value)
    parameter_block_errors(record)


def add_intensity_metrics(record, fit, model, scenario, catalog):
    _, _, xy_grid = regular_grid(LAMBDA_GRID_SIZE)
    t_eval = np.full(xy_grid.shape[0], scenario["duration"])
    mu_hat, triggering_hat, total_hat = fit.conditional_intensity(
        t=t_eval,
        x=xy_grid[:, 0],
        y=xy_grid[:, 1],
        burn_in=BURN_IN,
    )
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
    seed = BASE_SEED + 2000 * int(scenario["difficulty"]) + 100 * realization + mode_index
    catalog = simulation.catalog
    model = make_model(polygons, scenario, mode)
    mala_step = scenario["mala_step"]
    fit = model.gibbs(
        catalog,
        config=SPINHGibbsConfig(
            n_iter=N_ITER,
            thin=THIN,
            mala_step=mala_step,
            verbose=VERBOSE,
            verbose_every=VERBOSE_EVERY,
            use_calibration=True,
            beta_init=INITIAL_BETA,
            fixed_beta=scenario["beta"] if mode["true_theta"] else None,
            theta_priors=THETA_PRIORS,
            sample_z=mode["sample_z"],
            known_z=simulation.branching_labels if mode["known_z"] else None,
            fixed_etas=(
                scenario["etas"].as_dict() if mode["true_theta"] else {}
            ),
            sigma_mh_etas=SIGMA_MH_ETAS,
            sigma_mh_beta=SIGMA_MH_BETA,
            adaptation_start=200,
            proposal_jitter=1e-6,
        ),
        gp_backend="sparse" if USE_SPARSE_GP else "exact",
        rng_seed=seed,
    )
    summary = fit.summary(burn_in=BURN_IN)

    record = {
        "scenario": scenario["name"],
        "difficulty": scenario["difficulty"],
        "realization": realization,
        "mode": mode["name"],
        "mode_description": mode["description"],
        "seed": seed,
        "mala_step": mala_step,
        "n_events": len(catalog),
        "n_background": simulation.n_background,
        "n_triggered": simulation.n_triggered,
        "true_background_ratio": simulation.n_background / max(len(catalog), 1),
        "estimated_background_probability": float(summary["p_background"].mean()),
    }
    record.update(declustering_metrics(fit, simulation, BURN_IN))
    add_parameter_metrics(record, summary, scenario)
    add_intensity_metrics(record, fit, model, scenario, catalog)

    print(
        f"{scenario['name']:<26} realization={realization} {mode['name']:<23} "
        f"h={mala_step:.3f} "
        f"Zacc={record['background_accuracy']:.3f} "
        f"Pacc={record['parent_accuracy_triggered']:.3f} "
        f"prod={record['productivity_magnitude_mae_rel']:.3f} "
        f"temp={record['temporal_kernel_mae_rel']:.3f} "
        f"spat={record['spatial_kernel_mae_rel']:.3f}"
    )
    return record


def simulate_catalog(polygons, scenario, realization):
    seed = BASE_SEED + 1000 * int(scenario["difficulty"]) + realization
    field = lambda x, y: latent_field(x, y, scenario["field_scale"])
    return simulate_with_seed(polygons, scenario, field, seed)


def simulate_with_seed(polygons, scenario, field, seed):
    return simulate_hawkes_process(
        X_bounds=X_BOUNDS,
        Y_bounds=Y_BOUNDS,
        T=scenario["duration"],
        polygons=polygons,
        mus=scenario["mus"],
        f=field,
        etas_parameters=scenario["etas"],
        beta=scenario["beta"],
        magnitude_min=MAGNITUDE_MIN,
        magnitude_max=MAGNITUDE_MAX,
        rng_seed=seed,
    )


def fieldnames():
    base = [
        "scenario", "difficulty", "realization", "mode", "mode_description", "seed",
        "mala_step", "n_events", "n_background", "n_triggered", "true_background_ratio",
        "estimated_background_probability", "background_accuracy", "background_recall",
        "triggered_recall", "parent_accuracy_triggered", "background_rmse",
        "triggering_rmse", "total_rmse", "productivity_magnitude_mae_rel",
        "temporal_kernel_mae_rel", "spatial_kernel_mae_rel",
    ]
    parameter_fields = []
    for name in PARAMETER_NAMES:
        parameter_fields.extend([
            f"true_{name}", f"estimate_{name}", f"relative_error_{name}",
        ])
    return base + parameter_fields


def save_records(records):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "experiment_6_oracle_branching_parameters.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames())
        writer.writeheader()
        writer.writerows(records)
    return path


def mean_value(rows, key):
    return float(np.nanmean([row[key] for row in rows]))


def print_summary(records):
    print("\nOracle recovery summary")
    print(
        f"{'scenario':<26} {'mode':<23} {'Zacc':>7} {'Prec':>7} "
        f"{'prod':>7} {'temp':>7} {'spat':>7} {'totRMSE':>8}"
    )
    print("-" * 98)
    for scenario in SCENARIOS:
        for mode in MODES:
            rows = [
                row for row in records
                if row["scenario"] == scenario["name"] and row["mode"] == mode["name"]
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
                f"{mean_value(rows, 'total_rmse'):8.3f}"
            )


def plot_oracle_summary(records):
    scenario_labels = [scenario["name"].replace("_", "\n") for scenario in SCENARIOS]
    mode_labels = [mode["name"].replace("_", "\n") for mode in MODES]
    fig, axes = plt.subplots(len(SCENARIOS), 2, figsize=(12.0, 4.2 * len(SCENARIOS)))
    axes = np.atleast_2d(axes)
    for row_index, scenario in enumerate(SCENARIOS):
        scenario_rows = [row for row in records if row["scenario"] == scenario["name"]]
        z_values = []
        theta_values = []
        for mode in MODES:
            rows = [row for row in scenario_rows if row["mode"] == mode["name"]]
            z_values.append(mean_value(rows, "parent_accuracy_triggered") if rows else np.nan)
            theta_values.append([
                mean_value(rows, "productivity_magnitude_mae_rel") if rows else np.nan,
                mean_value(rows, "temporal_kernel_mae_rel") if rows else np.nan,
                mean_value(rows, "spatial_kernel_mae_rel") if rows else np.nan,
            ])

        ax = axes[row_index, 0]
        ax.bar(np.arange(len(MODES)), z_values, color="steelblue")
        ax.set_xticks(np.arange(len(MODES)))
        ax.set_xticklabels(mode_labels)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Parent accuracy")
        ax.set_title(f"Z recovery | {scenario_labels[row_index]}")
        ax.grid(axis="y", alpha=0.3)

        ax = axes[row_index, 1]
        theta_values = np.asarray(theta_values, dtype=float)
        width = 0.25
        xloc = np.arange(len(MODES))
        for offset, (label, color) in enumerate([
            ("prod+mag", "#4C78A8"),
            ("temporal", "#F58518"),
            ("spatial", "#54A24B"),
        ]):
            ax.bar(xloc + (offset - 1) * width, theta_values[:, offset], width, label=label, color=color)
        ax.set_xticks(xloc)
        ax.set_xticklabels(mode_labels)
        ax.set_ylabel("Mean abs. relative error")
        ax.set_title(f"theta recovery | {scenario_labels[row_index]}")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    save_figure(fig, "package/experiment_6/oracle_branching_parameters")
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
            for scenario, realization in tasks
        ]
    else:
        if Parallel is None:
            raise ImportError("joblib is required when N_JOBS != 1.")
        print(f"Running {len(tasks)} independent realizations with N_JOBS={N_JOBS}")
        nested_records = Parallel(n_jobs=N_JOBS)(
            delayed(run_realization_modes)(polygons, scenario, realization)
            for scenario, realization in tasks
        )
    return [record for records in nested_records for record in records]


def main():
    print("Experiment 6 — SPIN-H oracle Z/theta recovery")
    print(
        f"scenarios={len(SCENARIOS)}, modes={len(MODES)}, "
        f"realizations={N_REALIZATIONS}, n_iter={N_ITER}, sparse={USE_SPARSE_GP}, "
        f"n_jobs={N_JOBS}"
    )
    print({scenario["name"]: scenario["mala_step"] for scenario in SCENARIOS})
    polygons, _ = generate_voronoi_cells(
        n_germs=N_DOMAINS,
        X_bounds=X_BOUNDS,
        Y_bounds=Y_BOUNDS,
        rng_seed=DOMAIN_SEED,
    )

    records = run_all_realizations(polygons)
    path = save_records(records)
    print_summary(records)
    print(f"\nSaved oracle recovery table: {path}")
    plot_oracle_summary(records)
    return records


if __name__ == "__main__":
    main()
