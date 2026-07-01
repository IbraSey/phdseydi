#%%

# import sys
# import os
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pandas as pd
import openturns as ot
from shapely.geometry import box as shapely_box
from shapely.prepared import prep

from gp.gibbs_sampler import SSGC_GibbsSampler


DATA_PATH = ".../.../....csv"

df = pd.read_csv(DATA_PATH)
x_arr = df["x"].to_numpy()
y_arr = df["y"].to_numpy()
t_arr = df["t"].to_numpy()

N = len(df)
print(f"{N} events chargés depuis {DATA_PATH}")

X_BOUNDS = (x_arr.min(), x_arr.max())
Y_BOUNDS = (y_arr.min(), y_arr.max())
T = float(t_arr.max())


# Zonage spatial
# Test 1 peut-être une seule zone qui couvre tout le domaine
domain_poly = shapely_box(X_BOUNDS[0], Y_BOUNDS[0], X_BOUNDS[1], Y_BOUNDS[1])
zones_raw_list = [domain_poly]
zones_prep = [prep(z) for z in zones_raw_list]
Areas = [(zp, 0.0) for zp in zones_prep]


NU_INIT = [5.0, 0.2]
LAMBDA_NU = 0.5
DELTA = [1.5, 0.01]
JITTER = 1e-5

MALA_STEP = 0.095
LEARN_NU = False
USE_CALIBRATION = True
T0_NU = 50
STEP_NU_INIT = 0.0009

N_ITER = 2000
THIN = 3
BURN_IN = 0.5
NX, NY = 30, 30
NX_POST, NY_POST = 60, 60

SEED = 42
VERBOSE = True
VERBOSE_EVERY = 100
SAVE_FIGURE = False

x_pt = ot.Point(x_arr.tolist())
y_pt = ot.Point(y_arr.tolist())
t_pt = ot.Point(t_arr.tolist())

sampler = SSGC_GibbsSampler(
    X_bounds=X_BOUNDS,
    Y_bounds=Y_BOUNDS,
    T=T,
    Areas=Areas,
    polygons=zones_raw_list,
    lambda_nu=LAMBDA_NU,
    nu=NU_INIT,
    delta=DELTA,
    jitter=JITTER,
    rng_seed=SEED,
)

results = sampler.run(
    t=t_pt,
    x=x_pt,
    y=y_pt,
    mala_step=MALA_STEP,
    learn_nu=LEARN_NU,
    t0_nu=T0_NU,
    step_nu_init=STEP_NU_INIT,
    n_iter=N_ITER,
    thin=THIN,
    verbose=VERBOSE,
    verbose_every=VERBOSE_EVERY,
    use_calibration=USE_CALIBRATION,
    mu_star_func=None,
    grid_nx=NX,
    grid_ny=NY,
)

# print(f"Run terminé — {N_ITER} itérations, thin={THIN} "
#       f"=> {N_ITER // THIN} échantillons conservés")

sampler.plot_chains(results, savefigure=SAVE_FIGURE)
sampler.plot_acf(results, burn_in=BURN_IN, savefigure=SAVE_FIGURE)

out = sampler.plot_posterior_intensity(
    x=x_arr, y=y_arr, t=t_arr,
    results=results,
    nx=NX_POST, ny=NY_POST,
    burn_in=BURN_IN,
    cmap="inferno",
    mu_star_func=None,
    savefigure=SAVE_FIGURE,
)

# print("Terminé.")