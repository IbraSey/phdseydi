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
def sigma(z):
    z_array = np.array(z)
    return ot.Point(1.0 / (1.0 + np.exp(-z_array)))

def generate_data(
    X_bounds=(0.0, 2.0),
    Y_bounds=(0.0, 2.0),
    T=10.0,
    n_cols=3,
    n_rows=1,
    mus=6.0,
    f=None,             
    rng_seed=0
):
    if rng_seed is not None:
        ot.RandomGenerator.SetSeed(rng_seed)
        np.random.seed(rng_seed) 
    
    xmin, xmax = X_bounds
    ymin, ymax = Y_bounds
    dx = (xmax - xmin) / n_cols
    dy = (ymax - ymin) / n_rows
    J = n_rows * n_cols
    
    if np.isscalar(mus) :
        mus_vec = ot.Point([mus] * J)
    else:
        mus_vec = ot.Point(mus) 

    if not isinstance(f, list):
        funcs = [f] * J       
    else:
        funcs = f

    zones = []
    for r in range(n_rows):
        y0 = ymin + r * dy
        y1 = y0 + dy
        for c in range(n_cols):
            x0 = xmin + c * dx
            x1 = x0 + dx
            zones.append(box(x0, y0, x1, y1))
    
    all_samples = []
    for i, (zone, mu) in enumerate(zip(zones, mus_vec)) :
        bounds = zone.bounds
        mean_candidates = mu * T * zone.area
        n_candidates = int(ot.Poisson(mean_candidates).getRealization()[0])
        func = funcs[i]
        
        if n_candidates > 0:
            cand_x = np.random.uniform(bounds[0], bounds[2], n_candidates)
            cand_y = np.random.uniform(bounds[1], bounds[3], n_candidates)
            cand_t = np.random.uniform(0, T, n_candidates)
            
            # Calcul des probas d'acceptation
            intensities_pt = sigma(func(cand_x, cand_y))
            probs = np.array(intensities_pt).flatten() 
            
            decision = np.random.uniform(0, 1, n_candidates)
            mask_accepted = decision < probs
            
            if np.any(mask_accepted):
                accepted_points = np.column_stack((
                    cand_x[mask_accepted], 
                    cand_y[mask_accepted], 
                    cand_t[mask_accepted]
                ))
                all_samples.append(accepted_points)

    if not all_samples:
        X = ot.Sample(0, 3)
    else:
        X_array = np.vstack(all_samples)
        #np.random.shuffle(X_array)
        X = ot.Sample(X_array.tolist())
    
    print("="*35)
    print("-"*5 + " JEU DE DONNÉES SIMULÉES " + "-"*5)
    print(f"\nNombre d'événements retenus : {X.getSize()}")
    print(f"Nombre de zones : {len(zones)}")
    print(f"Bornes d'intensité : {mus_vec}")
    print(f"Premiers événements :\n{X[:5]}")
    print("="*35)

    return X, zones, X_bounds, Y_bounds, T



# %%




# %%


