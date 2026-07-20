#%%

"""Minimal SSGC Gibbs test matching Experiment 1, Profile 1, Setting A."""

import sys
import warnings
from pathlib import Path

import numpy as np
import openturns as ot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from package import (
    GPParameters,
    GibbsConfig,
    SSGCModel,
    generate_voronoi_cells,
    simulate_spatial_process,
)


X_BOUNDS = (0.0, 2.0)
Y_BOUNDS = (0.0, 2.0)
DURATION = 180.0
DOMAIN_INTENSITIES = (10.0, 1.0, 2.0, 10.0, 8.0, 2.0)
VORONOI_SEED = 15
SIMULATION_SEED = 15
MCMC_SEED = 42

N_ITER = 300
THIN = 3
BURN_IN = 0.4
MALA_STEP = 0.06
USE_SPARSE_GP = True
MAKE_PLOTS = True
USE_CALIBRATION = True
GRID_SIZE = 60
POST_GRID = 60
POSTERIOR_N_MC = 100

EPS_PRIOR_VARIANCE = 2
EPS_PRIOR_LENGTH_SCALE = 0.01
GP_PRIOR_VARIANCE = 5.0
GP_PRIOR_LENGTH_SCALE = 0.2


def latent_field(x, y):
    """Latent field used by Experiment 1, Profile 1, Setting A."""
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


def main():
    polygons, _ = generate_voronoi_cells(
        n_germs=len(DOMAIN_INTENSITIES),
        X_bounds=X_BOUNDS,
        Y_bounds=Y_BOUNDS,
        rng_seed=VORONOI_SEED,
    )

    simulation = simulate_spatial_process(
        X_bounds=X_BOUNDS,
        Y_bounds=Y_BOUNDS,
        T=DURATION,
        polygons=polygons,
        mus=DOMAIN_INTENSITIES,
        f=latent_field,
        rng_seed=SIMULATION_SEED,
        grid_res=100,
    )
    catalog = simulation.catalog
    print(f"Generated SSGC catalog: N={len(catalog)}")

    def true_intensity(x, y):
        return simulation.spatial_components(x, y)[3]

    model = SSGCModel.from_polygons(
        polygons=polygons,
        duration=DURATION,
        x_bounds=X_BOUNDS,
        y_bounds=Y_BOUNDS,
        initial_log_intensities=0.0,
        gp_prior=GPParameters(
            variance=GP_PRIOR_VARIANCE,
            length_scale=GP_PRIOR_LENGTH_SCALE,
        ),
        eps_prior_variance=EPS_PRIOR_VARIANCE,
        eps_prior_length_scale=EPS_PRIOR_LENGTH_SCALE,
        nu_prior_rate=0.5,
        jitter=1e-5,
    )

    fit = model.gibbs(
        catalog,
        config=GibbsConfig(
            n_iter=N_ITER,
            thin=THIN,
            mala_step=MALA_STEP,
            verbose=True,
            verbose_every=max(1, N_ITER // 10),
            use_calibration=USE_CALIBRATION,
            grid_nx=30,
            grid_ny=30,
            compute_emu=False,
        ),
        gp_backend="sparse" if USE_SPARSE_GP else "exact",
        rng_seed=MCMC_SEED,
        reference_intensity=true_intensity,
    )

    summary = fit.summary(burn_in=BURN_IN)
    print("\nPosterior SSGC summary")
    print(f"eps estimate = {np.round(summary['eps_hat'], 4)}")
    print(f"nu estimate = {np.round(summary['nu_hat'], 4)}")
    print(f"acceptance rates = {fit.acceptance_rates}")

    mesh = ot.IntervalMesher([GRID_SIZE - 1, GRID_SIZE - 1]).build(
        ot.Interval(
            [X_BOUNDS[0], Y_BOUNDS[0]],
            [X_BOUNDS[1], Y_BOUNDS[1]],
        )
    )
    grid = np.asarray(mesh.getVertices(), dtype=float)
    intensity_estimated = fit.background_intensity(
        grid[:, 0],
        grid[:, 1],
        burn_in=BURN_IN,
    )
    intensity_true = true_intensity(grid[:, 0], grid[:, 1])
    error = intensity_estimated - intensity_true

    print("\nBackground intensity")
    print(f"RMSE = {np.sqrt(np.mean(error**2)):.4f}")
    print(f"MAE = {np.mean(np.abs(error)):.4f}")
    print(f"mean estim = {intensity_estimated.mean():.4f}")
    print(f"mean true = {intensity_true.mean():.4f}")
    print(f"min estim = {intensity_estimated.min():.4f}")
    print(f"min true = {intensity_true.min():.4f}")
    print(f"max estim = {intensity_estimated.max():.4f}")
    print(f"max true = {intensity_true.max():.4f}")

    if MAKE_PLOTS:
        fit.plot_traces(burn_in=BURN_IN)
        fit.plot_acf(burn_in=BURN_IN)
        fit.posterior_intensity(
            burn_in=BURN_IN,
            nx=POST_GRID,
            ny=POST_GRID,
            n_mc=POSTERIOR_N_MC,
            cmap="inferno",
            mu_star_func=true_intensity,
        )


if __name__ == "__main__":
    main()

# %%
