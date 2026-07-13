# %%

#from SIGMA_GP_PP_Gibbs import *

import os
import sys

# phebus_path = os.getenv("PHEBUS_PATH")
# if not phebus_path or not os.path.isdir(phebus_path):
#     raise EnvironmentError(
#         "PHEBUS_PATH must be set to the directory containing the phebus package."
#     )
from pathlib import Path
file_path = Path(__file__).resolve()
ROOT = file_path.parent.parent
phebus_path = ROOT / "lib_py"
sys.path.insert(0, str(phebus_path) )


import phebus

# import sys, os
# sys.path.append( os.getenv("PHEBUS_PATH")) # for importing phebus
from phebus.pybus.frclass import FrenchDomainsSourceModel

import pandas as pd


#%%
#############
# Load Case #
#############

import os
from phebus.pybus.frclass import FrenchDomainsSourceModel
# phebus_path = os.path.abspath("./phebus")
# phebus_root = "/home/g80884/Documents/phebus"
# phebus_root = os.path.join( phebus_path, "phebus" )
demo_path =  phebus_path / "phebus" / "demos" / "FrenchDomainsAnalysis"

SM = FrenchDomainsSourceModel(
    Mmin=3.,
    PWD=demo_path,
    FILE_DOMAINS= demo_path / "data" / "domains" / "domaines_xy.csv"
)

catalog = SM.catalog[SM.catalog.year >= 1965]

D = np.vstack((catalog.X, catalog.Y, catalog.magnitude)).T

T = max(catalog.year) - min(catalog.year)

# Domains
zones = [zone.get_polygon_xy() for zone in SM.zones]
areas = np.array([zone.get_area_km2() for zone in SM.zones])
zone_names = [zone.name for zone in SM.zones]

# SM.reduce_catalog(SM.catalog, SM.magnitudes, SM.first_years, SM.last_year)

# props = np.array([zone.reduced_catalog.counts.sum() for zone in SM.zones])

# values = props / (T * areas) * 1e6

# values = np.arange(len(zone_names))**2 + 1

values = areas
SM.plot_values_map( values, FIGURE_PATH=os.getcwd(),  FIGURE_NAME="lambdas_and_domains", catalog=catalog, coastline=SM.coastlines, scale=5., xticks= zone_names)




#%%

# import sys
# import os
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pandas as pd
import openturns as ot
from package import EventCatalog, GPParameters, GibbsConfig, SSGCModel


x_arr = catalog.X 
y_arr = catalog.Y
m_arr = catalog.magnitude
t_arr = catalog.year

domains = zones

X_BOUNDS = float()
Y_BOUNDS = float()
DURATION = T


# Prior 
NU_INIT = (5.0, 0.2)
LAMBDA_NU = 0.5
DELTA = (1.5, 0.01)

# Model and Gibbs configuration
MALA_STEP = 0.095
JITTER = 1e-5
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

config = GibbsConfig(
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
