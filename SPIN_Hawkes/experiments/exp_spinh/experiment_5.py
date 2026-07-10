"""Experiment 5"""

#%%

import csv
import sys
import warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import openturns as ot
from package import ETASParameters
from sklearn.exceptions import ConvergenceWarning

try:
    from joblib import Parallel, delayed
except ImportError:
    Parallel = delayed = None

warnings.filterwarnings("ignore", category=ConvergenceWarning)
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



X_BOUNDS = (0.0, 2.0)
Y_BOUNDS = (0.0, 2.0)
N_DOMAINS = 6
DOMAIN_SEED = 15
BASE_SEED = 42

MAGNITUDE_MIN = 2.0
MAGNITUDE_MAX = 6.0

INITIAL_ETAS = ETASParameters(
    A=0.4,
    alpha=0.6,
    c=0.03,
    p=1.35,
    d=0.06,
    q=1.7,
    gamma=0.3,
)
INITIAL_BETA = 2.0

THETA_PRIORS = {
    "a_A": 5.0, "b_A": 10.0,
    "a_alpha": 8.0, "b_alpha": 10.0,
    "a_c": 2.0, "b_c": 100.0,
    "a_p": 4.0, "b_p": 10.0,
    "a_d": 2.0, "b_d": 40.0,
    "a_q": 9.0, "b_q": 10.0,
    "a_gamma": 5.0, "b_gamma": 10.0,
}

SCENARIOS = [
    {
        "name": "easy_compact_structured",
        "difficulty": 1,
        "duration": 55.0,
        "mus": (8.0, 1.0, 2.0, 8.0, 7.0, 2.0),
        "field_scale": 0.9,
        "mala_step": 0.11,
        "etas": ETASParameters(
            A=0.55, alpha=0.8, c=0.04, p=1.55,
            d=0.025, q=2.25, gamma=0.35,
        ),
        "beta": 2.3,
    },
    {
        "name": "medium_reference",
        "difficulty": 2,
        "duration": 60.0,
        "mus": (10.0, 1.0, 2.0, 10.0, 8.0, 2.0),
        "field_scale": 1.0,
        "mala_step": 0.115,
        "etas": ETASParameters(
            A=0.5, alpha=0.8, c=0.02, p=1.3,
            d=0.05, q=1.8, gamma=0.5,
        ),
        "beta": 2.3,
    },
    {
        "name": "hard_diffuse_triggering",
        "difficulty": 3,
        "duration": 60.0,
        "mus": (5.0, 4.0, 4.5, 5.0, 4.0, 4.5),
        "field_scale": 0.45,
        "mala_step": 0.095,
        "etas": ETASParameters(
            A=0.35, alpha=0.35, c=0.08, p=1.12,
            d=0.16, q=1.25, gamma=0.05,
        ),
        "beta": 2.15,
    },
    {
        "name": "hard_weak_triggering",
        "difficulty": 4,
        "duration": 60.0,
        "mus": (6.0, 5.5, 5.0, 6.0, 5.5, 5.0),
        "field_scale": 0.25,
        "mala_step": 0.085,
        "etas": ETASParameters(
            A=0.18, alpha=0.2, c=0.10, p=1.08,
            d=0.20, q=1.18, gamma=0.02,
        ),
        "beta": 2.0,
    },
]

PARAMETER_NAMES = ["A", "c", "p", "d", "q", "alpha", "gamma", "beta"]
PARAMETER_BLOCKS = {
    "Productivity + magnitude": ["beta", "A", "alpha"],
    "Temporal kernel": ["c", "p"],
    "Spatial kernel": ["d", "q", "gamma"],
}


def latent_field(x, y, scale=1.0):
    weights = scale * np.array([1.5, -1.5, 3.0, -3.0])
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


def true_latent_state(xy, scenario):
    eps_true = np.log(np.asarray(scenario["mus"], dtype=float))
    f_true = latent_field(xy[:, 0], xy[:, 1], scenario["field_scale"])
    return eps_true, f_true


def regular_grid(n_grid):
    x_grid = np.linspace(X_BOUNDS[0], X_BOUNDS[1], int(n_grid))
    y_grid = np.linspace(Y_BOUNDS[0], Y_BOUNDS[1], int(n_grid))
    X, Y = np.meshgrid(x_grid, y_grid)
    return X, Y, np.column_stack((X.ravel(), Y.ravel()))


def relative_error(estimate, truth):
    denominator = max(abs(float(truth)), 1e-12)
    return (float(estimate) - float(truth)) / denominator


def rmse(estimate, truth):
    error = np.asarray(estimate, dtype=float) - np.asarray(truth, dtype=float)
    return float(np.sqrt(np.mean(error**2)))

def declustering_metrics(fit, simulation, burn_in):
    z_chain = fit.branching_chain
    if z_chain is None:
        return {
            "background_accuracy": np.nan,
            "background_recall": np.nan,
            "triggered_recall": np.nan,
            "parent_accuracy_triggered": np.nan,
        }

    z_chain = np.asarray(z_chain, dtype=int)
    z_post = z_chain[int(z_chain.shape[0] * burn_in):]
    true_background = simulation.is_background
    true_labels = simulation.branching_labels

    p_background = np.mean(z_post == 0, axis=0)
    predicted_background = p_background >= 0.5
    parent_mode = np.zeros(z_post.shape[1], dtype=int)

    for child in np.flatnonzero(~predicted_background):
        labels = z_post[:, child]
        labels = labels[labels > 0]
        if labels.size == 0:
            predicted_background[child] = True
            continue
        unique, counts = np.unique(labels, return_counts=True)
        parent_mode[child] = int(unique[np.argmax(counts)])

    true_triggered = ~true_background
    predicted_triggered = ~predicted_background
    parent_accuracy = np.nan
    if np.any(true_triggered):
        parent_accuracy = np.mean(parent_mode[true_triggered] == true_labels[true_triggered])

    return {
        "background_accuracy": float(np.mean(predicted_background == true_background)),
        "background_recall": float(np.mean(predicted_background[true_background])),
        "triggered_recall": float(np.mean(predicted_triggered[true_triggered])),
        "parent_accuracy_triggered": float(parent_accuracy),
    }


N_REALIZATIONS = 1
N_JOBS = -1
N_ITER = 3000
THIN = 3
BURN_IN = 0.5
SIGMA_MH_ETAS = 0.05
SIGMA_MH_BETA = 0.1
USE_SPARSE_GP = True
VERBOSE = True
VERBOSE_EVERY = 300
LAMBDA_GRID_SIZE = 35

RESULTS_DIR = PACKAGE_ROOT / "figures" / "experiments" / "exp_spinh"
ERROR_PLOT_FLOOR = 1e-4


def run_realization(polygons, scenario, realization):
    seed = BASE_SEED + 1000 * int(scenario["difficulty"]) + realization
    field = lambda x, y: latent_field(x, y, scenario["field_scale"])
    simulation = simulate_hawkes_process(
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

    model = SPINHModel.from_polygons(
        polygons=polygons,
        duration=scenario["duration"],
        x_bounds=X_BOUNDS,
        y_bounds=Y_BOUNDS,
        gp_prior=GPParameters(variance=5.0, length_scale=0.2),
        eps_prior_variance=1.0,
        eps_prior_length_scale=0.01,
        nu_prior_rate=0.5,
        jitter=1e-5,
        etas_parameters=INITIAL_ETAS,
        magnitude_min=MAGNITUDE_MIN,
        magnitude_max=MAGNITUDE_MAX,
    )

    fit = model.gibbs(
        simulation.catalog,
        config=SPINHGibbsConfig(
            n_iter=N_ITER,
            thin=THIN,
            mala_step=scenario["mala_step"],
            verbose=VERBOSE,
            verbose_every=VERBOSE_EVERY,
            use_calibration=True,
            learn_beta=True,
            beta_init=INITIAL_BETA,
            theta_priors=THETA_PRIORS,
            fixed_etas={"alpha": scenario["etas"].alpha},
            sigma_mh_etas=SIGMA_MH_ETAS,
            sigma_mh_beta=SIGMA_MH_BETA,
            adaptation_start=200,
            proposal_jitter=1e-6,
        ),
        gp_backend="sparse" if USE_SPARSE_GP else "exact",
        rng_seed=seed,
    )
    summary = fit.summary(burn_in=BURN_IN)

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
        history=simulation.catalog,
        eps=eps_true,
        latent_gp=f_true,
        parameters=scenario["etas"],
    )

    record = {
        "scenario": scenario["name"],
        "difficulty": scenario["difficulty"],
        "realization": realization,
        "seed": seed,
        "mala_step": scenario["mala_step"],
        "n_events": len(simulation.catalog),
        "n_background": simulation.n_background,
        "n_triggered": simulation.n_triggered,
        "true_background_ratio": simulation.n_background / max(len(simulation.catalog), 1),
        "estimated_background_probability": float(summary["p_background"].mean()),
        "background_rmse": rmse(mu_hat, mu_true),
        "triggering_rmse": rmse(triggering_hat, triggering_true),
        "total_rmse": rmse(total_hat, total_true),
    }
    record.update(declustering_metrics(fit, simulation, BURN_IN))

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
        f"h={scenario['mala_step']:.3f} "
        f"N={len(simulation.catalog):4d} bg={record['true_background_ratio']:.2f} "
        f"A={record['estimate_A']:.3f}/{record['true_A']:.3f} "
        f"alpha={record['estimate_alpha']:.3f}/{record['true_alpha']:.3f} "
        f"total_RMSE={record['total_rmse']:.3f}"
    )
    return record


def print_summary(records):
    print("\nPosterior recovery summary by scenario")
    print(
        f"{'scenario':<26} {'N':>7} {'bg':>6} "
        f"{'prod+mag':>10} {'temporal':>10} {'spatial':>10} "
        f"{'total_RMSE':>11}"
    )
    print("-" * 95)
    for scenario in SCENARIOS:
        rows = [row for row in records if row["scenario"] == scenario["name"]]
        if not rows:
            continue
        mean = lambda key: float(np.nanmean([row[key] for row in rows]))
        block_mean = lambda names: float(np.nanmean([
            abs(row[f"relative_error_{name}"])
            for row in rows
            for name in names
        ]))
        print(
            f"{scenario['name']:<26} {mean('n_events'):7.1f} "
            f"{mean('true_background_ratio'):6.3f} "
            f"{block_mean(PARAMETER_BLOCKS['Productivity + magnitude']):10.3f} "
            f"{block_mean(PARAMETER_BLOCKS['Temporal kernel']):10.3f} "
            f"{block_mean(PARAMETER_BLOCKS['Spatial kernel']):10.3f} "
            f"{mean('total_rmse'):11.3f}"
        )


def plot_recovery_summary(records):
    difficulties = [scenario["difficulty"] for scenario in SCENARIOS]
    labels = [scenario["name"].replace("_", "\n") for scenario in SCENARIOS]
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8), sharex=True)
    for ax, (title, parameter_names) in zip(axes, PARAMETER_BLOCKS.items()):
        for name in parameter_names:
            values = []
            for scenario in SCENARIOS:
                rows = [row for row in records if row["scenario"] == scenario["name"]]
                values.append(
                    np.nanmean([abs(row[f"relative_error_{name}"]) for row in rows])
                    if rows else np.nan
                )
            values = np.maximum(np.asarray(values, dtype=float), ERROR_PLOT_FLOOR)
            ax.plot(difficulties, values, marker="o", linewidth=1.4, label=name)
        ax.set_xticks(difficulties)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
        ax.legend()
    axes[0].set_ylabel("Mean absolute relative error (log scale)")
    fig.suptitle("SPIN-H posterior recovery across increasing difficulty")
    fig.tight_layout()
    save_figure(fig, "package/experiment_5/posterior_recovery")
    plt.show()
    return fig


def main():
    print("Experiment 5 — SPIN-H posterior recovery")
    print(
        f"scenarios={len(SCENARIOS)}, realizations={N_REALIZATIONS}, "
        f"n_iter={N_ITER}, thin={THIN}, sparse={USE_SPARSE_GP}, n_jobs={N_JOBS}"
    )
    print({scenario["name"]: scenario["mala_step"] for scenario in SCENARIOS})

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
            for scenario, realization in tasks
        ]
    else:
        if Parallel is None:
            raise ImportError("joblib is required when N_JOBS != 1.")
        records = Parallel(n_jobs=N_JOBS)(
            delayed(run_realization)(polygons, scenario, realization)
            for scenario, realization in tasks
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "experiment_5_posterior_recovery.csv"
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    print_summary(records)
    print(f"\nSaved posterior recovery table: {output_path}")
    plot_recovery_summary(records)
    return records


if __name__ == "__main__":
    main()

# %%
