# %%
# =================================================================================================
# -------------------------------------------- IMPORTS --------------------------------------------
# =================================================================================================
from pathlib import Path
import os, sys
ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT))
import openturns as ot
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from polyagamma import random_polyagamma
from shapely.geometry import box, Polygon, Point as ShapelyPoint
from shapely.prepared import prep
from visualizations.plot import plot_field
from gp.gibbs_sampler import SGCP_GibbsSampler
ot.RandomGenerator.SetSeed(42)


# %%
# =================================================================================================
# ------------------------------------- GÉNÉRATION DE DONNÉES -------------------------------------
# =================================================================================================

def generate_data(
    X_bounds=(0.0, 2.0),
    Y_bounds=(0.0, 2.0),
    T=10.0,
    n_cols=3,
    n_rows=1,
    mus=6.0,
    rng_seed=0,
    shuffle=True,
):
    """
    
    """
    if rng_seed is not None:
        ot.RandomGenerator.SetSeed(rng_seed)
    
    xmin, xmax = X_bounds
    ymin, ymax = Y_bounds
    dx = (xmax - xmin) / n_cols
    dy = (ymax - ymin) / n_rows
    J = n_rows * n_cols
    
    if np.isscalar(mus):
        mus_vec = ot.Point([float(mus)] * J)
    else:
        mus_vec = ot.Point(mus) if not isinstance(mus, ot.Point) else mus
    
    if len(mus_vec) != J:
        raise ValueError(f"mus doit avoir {J} éléments, pour l'instant mus a {len(mus_vec)} éléments")
    
    zones = []
    for r in range(n_rows):
        y0 = ymin + r * dy
        y1 = y0 + dy
        for c in range(n_cols):
            x0 = xmin + c * dx
            x1 = x0 + dx
            zones.append(box(x0, y0, x1, y1))
    
    all_samples = []
    for zone, mu in zip(zones, mus_vec):
        bounds = zone.bounds       # (min_x, min_y, max_x, max_y)
        mean = mu * T * zone.area
        n = int(ot.Poisson(mean).getRealization()[0])
        
        dist = ot.ComposedDistribution([
            ot.Uniform(bounds[0], bounds[2]), 
            ot.Uniform(bounds[1], bounds[3]), 
            ot.Uniform(0, T)
        ])
        all_samples.append(dist.getSample(n))
    
    if not all_samples:
        X = ot.Sample(0, 3)
    else:
        X_array = np.vstack([np.array(s) for s in all_samples])
        if shuffle:
            np.random.shuffle(X_array)
        X = ot.Sample(X_array.tolist())
    
    print("="*35)
    print("-"*5 + " JEU DE DONNÉES SIMULÉES " + "-"*5)
    print(f"\nNombre d'événements : {X.getSize()}")
    print(f"Nombre de zones régulières : {len(zones)}")
    print(f"Intensités : {mus_vec}")
    #print(f"Premiers événements :\n{X[:5]}")
    print("="*35)
    
    return X, zones, X_bounds, Y_bounds, T, mus_vec


# %%




# %%


