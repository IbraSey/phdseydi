# %%
# =================================================================================================
# -------------------------------------------- IMPORTS --------------------------------------------
# =================================================================================================
from pathlib import Path
import os, sys
ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT))
import openturns as ot
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import math
from polyagamma import random_polyagamma
from shapely.geometry import Polygon, Point as ShapelyPoint
from shapely.prepared import prep
from visualizations.plot import plot_field, plot_poisson_zones_data
from gp.gibbs_sampler import SGCP_GibbsSampler
from gp.data_generation import generate_data

ot.RandomGenerator.SetSeed(42)


# %%
# ========================================================================================================
# ------------------------------------------- DONNÉES SIMULÉES -------------------------------------------
# ========================================================================================================

X, zones, X_bounds, Y_bounds, T, mus_vec = generate_data(X_bounds=(0, 2), Y_bounds=(0, 2), T=10, 
    n_cols=2, n_rows=2, mus=[1.5, 3.0, 5.0, 0.5], rng_seed=42, shuffle=True)

plot_poisson_zones_data(X=X, zones=zones, mus_vec=mus_vec, X_bounds=X_bounds, Y_bounds=Y_bounds,
    savefigure=True, title_savefig='figure_données_simulées_4_zones.pdf'
)





# %%








# %%








# %%






