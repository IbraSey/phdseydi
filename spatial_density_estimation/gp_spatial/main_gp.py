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







# %%







# %%








# %%









# %%






