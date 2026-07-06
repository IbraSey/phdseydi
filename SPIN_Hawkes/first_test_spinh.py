#%%

import sys
import warnings
from pathlib import Path

import numpy as np
import openturns as ot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from SPIN_Hawkes import (
    ETASParameters,
    GPParameters,
    SPINHGibbsConfig,
    SPINHModel,
    generate_voronoi_cells,
    simulate_hawkes_process,
)


X_BOUNDS = (0.0, 2.0)
Y_BOUNDS = (0.0, 2.0)
DURATION = 50.0
DOMAIN_INTENSITIES = (10.0, 1.0, 2.0, 10.0, 8.0, 2.0)
SEED = 42

N_ITER = 10000
THIN = 4
BURN_IN = 0.5
MALA_STEP = 0.11
SIGMA_MH_ETAS = 0.05
SIGMA_MH_BETA = 0.1
USE_SPARSE_GP = True
MAKE_PLOTS = True
VERBOSE = True
LAMBDA_GRID_SIZE = 60
POST_GRID = 120

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

def true_latent_state(lambda_xy):
    eps_true = np.log(np.asarray(DOMAIN_INTENSITIES, dtype=float))
    f_true = latent_field(lambda_xy[:, 0], lambda_xy[:, 1])
    return eps_true, f_true

def print_parameter_summary(summary):
    print("\nPosterior ETAS summary")
    print(f"{'param':<8} {'true':>8} {'estimate':>10}")
    print("-" * 28)
    for name, true_value in TRUE_ETAS.as_dict().items():
        estimate = summary["theta_phi_hat"][name]
        print(f"{name:<8} {true_value:>8.3f} {estimate:>10.3f}")
    print(f"{'beta':<8} {TRUE_BETA:>8.3f} {summary['beta_hat']:>10.3f}")

def print_intensity_metrics(name, estimated, truth):
    error = estimated - truth
    rmse = np.sqrt(np.mean(error**2))
    mae = np.mean(np.abs(error))
    print(f"\n{name}")
    print(f"RMSE = {rmse:.4f}")
    print(f"MAE = {mae:.4f}")
    print(f"mean estim = {estimated.mean():.4f}")
    print(f"mean true = {truth.mean():.4f}")


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
    f"N={len(catalog)}"
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

gibbs = model.gibbs(
    catalog,
    config=SPINHGibbsConfig(
        n_iter=N_ITER,
        thin=THIN,
        mala_step=MALA_STEP,
        verbose=VERBOSE,
        verbose_every=max(1, N_ITER // 10),
        use_calibration=True,
        learn_beta=True,
        theta_priors=THETA_PRIORS,
        sigma_mh_etas=SIGMA_MH_ETAS,
        sigma_mh_beta=SIGMA_MH_BETA,
        adaptation_start=200,
    ),
    gp_backend="sparse" if USE_SPARSE_GP else "exact",
    rng_seed=SEED,
)
summary = gibbs.summary(burn_in=BURN_IN)

print_parameter_summary(summary)
print(
    "\nBackground probability: "
    f"posterior mean={summary['p_background'].mean():.3f}, "
    f"truth={simulation.n_background / len(catalog):.3f}"
)

mesh = ot.IntervalMesher([LAMBDA_GRID_SIZE - 1, LAMBDA_GRID_SIZE - 1]).build(
    ot.Interval([X_BOUNDS[0], Y_BOUNDS[0]], [X_BOUNDS[1], Y_BOUNDS[1]]))
lambda_xy = np.asarray(mesh.getVertices(), dtype=float)

t_eval = np.full(lambda_xy.shape[0], DURATION)
mu_eval, trigger_eval, lambda_eval = gibbs.conditional_intensity(
    t=t_eval, x=lambda_xy[:, 0], y=lambda_xy[:, 1], burn_in=BURN_IN,
)

eps_true, f_true = true_latent_state(lambda_xy)
mu_true, trigger_true, lambda_true = model.conditional_intensity(
    t_eval=t_eval,
    x_eval=lambda_xy[:, 0],
    y_eval=lambda_xy[:, 1],
    history=catalog,
    eps=eps_true,
    latent_gp=f_true,
    parameters=TRUE_ETAS,
)

print_intensity_metrics("Background intensity", mu_eval, mu_true)
print_intensity_metrics("Triggering intensity", trigger_eval, trigger_true)
print_intensity_metrics("Total conditional intensity", lambda_eval, lambda_true)

if MAKE_PLOTS:
    gibbs.plot_traces(burn_in=BURN_IN)
    gibbs.plot_acf(burn_in=BURN_IN, max_lag=100)
    gibbs.plot_declustering(
        burn_in=BURN_IN,
        true_parent=simulation.branching_labels,
    )
    snapshot_times = np.linspace(0.2 * DURATION, DURATION, 4)
    snapshot_x = np.linspace(X_BOUNDS[0], X_BOUNDS[1], POST_GRID)
    snapshot_y = np.linspace(Y_BOUNDS[0], Y_BOUNDS[1], POST_GRID)
    X_snapshot, Y_snapshot = np.meshgrid(snapshot_x, snapshot_y)
    snapshot_xy = np.column_stack((X_snapshot.ravel(), Y_snapshot.ravel()))
    eps_snapshot, f_snapshot = true_latent_state(snapshot_xy)
    true_background = model.background_intensity(
        snapshot_xy[:, 0], snapshot_xy[:, 1], eps_snapshot, f_snapshot
    ).reshape(Y_snapshot.shape)
    true_triggering = []
    true_total = []
    for time in snapshot_times:
        _, trigger_frame, total_frame = model.conditional_intensity(
            t_eval=np.full(snapshot_xy.shape[0], time),
            x_eval=snapshot_xy[:, 0],
            y_eval=snapshot_xy[:, 1],
            history=catalog,
            eps=eps_snapshot,
            latent_gp=f_snapshot,
            parameters=TRUE_ETAS,
        )
        true_triggering.append(trigger_frame.reshape(Y_snapshot.shape))
        true_total.append(total_frame.reshape(Y_snapshot.shape))

    gibbs.plot_conditional_intensity_snapshots(
        times=snapshot_times,
        burn_in=BURN_IN,
        nx=POST_GRID,
        ny=POST_GRID,
        true_background=true_background,
        true_triggering=np.asarray(true_triggering),
        true_total=np.asarray(true_total),
    )


# %%
