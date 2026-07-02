#%%

# import sys
# import os
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pandas as pd
import openturns as ot
from SPIN_Hawkes import EventCatalog, GPParameters, MCMCConfig, SSGCModel


DATA_PATH = ".../.../....csv"

df = pd.read_csv(DATA_PATH)
x_arr = df["x"].to_numpy()
y_arr = df["y"].to_numpy()
t_arr = df["t"].to_numpy()

domains = [
    # domain_1,
    # domain_2,
]

X_BOUNDS = ...
Y_BOUNDS = ...
DURATION = ...


# Model and Gibbs configuration
NU_INIT = (5.0, 0.2)
LAMBDA_NU = 0.5
DELTA = (1.5, 0.01)
JITTER = 1e-5

MALA_STEP = 0.095
LEARN_NU = False
USE_CALIBRATION = True
T0_NU = 50
STEP_NU_INIT = 0.0009

N_ITER = 2000
THIN = 2
BURN_IN = 0.5
NX, NY = 30, 30
NX_POST, NY_POST = 60, 60

SEED = 42
VERBOSE = True
VERBOSE_EVERY = 100
SAVE_FIGURE = False
USE_SPARSEGP = True


catalog = EventCatalog(t=t_arr, x=x_arr, y=y_arr)

model = SSGCModel.from_polygons(
    polygons=domains,
    duration=DURATION,
    x_bounds=X_BOUNDS,
    y_bounds=Y_BOUNDS,
    initial_log_intensities=0.0,
    gp_prior=GPParameters(
        variance=NU_INIT[0],
        length_scale=NU_INIT[1],
    ),
    eps_prior_variance=DELTA[0],
    eps_prior_length_scale=DELTA[1],
    nu_prior_rate=LAMBDA_NU,
    jitter=JITTER,
)

config = MCMCConfig(
    n_iter=N_ITER,
    thin=THIN,
    mala_step=MALA_STEP,
    learn_nu=LEARN_NU,
    use_calibration=USE_CALIBRATION,
    t0_nu=T0_NU,
    step_nu_init=STEP_NU_INIT,
    verbose=VERBOSE,
    verbose_every=VERBOSE_EVERY,
    grid_nx=NX,
    grid_ny=NY,
    compute_emu=False,
)

fit = model.gibbs(
    catalog,
    config=config,
    gp_backend="sparse" if USE_SPARSEGP else "exact",
    rng_seed=SEED,
)

summary = fit.summary(burn_in=BURN_IN)
print("\nPosterior means")
print(f"eps = {summary['eps_hat']}")
print(f"nu = {summary['nu_hat']}")
print(f"acceptance rates = {fit.acceptance_rates}")

fit.plot_traces(
    burn_in=BURN_IN,
    savefigure=SAVE_FIGURE,
    title_savefig="ssgc/template/traces",
)
fit.plot_acf(
    burn_in=BURN_IN,
    savefigure=SAVE_FIGURE,
    title_savefig="ssgc/template/acf",
)
fit.posterior_intensity(
    nx=NX_POST,
    ny=NY_POST,
    burn_in=BURN_IN,
    cmap="inferno",
    savefigure=SAVE_FIGURE,
    title_savefig="ssgc/template/posterior_intensity",
)
