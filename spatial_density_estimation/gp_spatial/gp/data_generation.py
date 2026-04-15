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
from shapely.geometry import box, Polygon as ShapelyPolygon, Point as ShapelyPoint
from shapely.prepared import prep
from visualizations.plot import plot_field
from scipy.spatial import Voronoi
from scipy.special import expit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
ot.RandomGenerator.SetSeed(42)



# %%
# =================================================================================================
# ------------------------------------- GÉNÉRATION DE DONNÉES -------------------------------------
# =================================================================================================

def simulate_process(
    X_bounds=(0.0, 2.0),
    Y_bounds=(0.0, 2.0),
    T=20.0,
    polygons=None,
    n_cols=2,
    n_rows=2,
    mus=5.0,
    f=None,
    rng_seed=0,
    grid_res=100, 
    **f_kwargs
):
    if rng_seed is not None:
        ot.RandomGenerator.SetSeed(rng_seed)
        np.random.seed(rng_seed)

    xmin, xmax = X_bounds
    ymin, ymax = Y_bounds

    # Gestion des zones
    if polygons is not None:
        zones = polygons
        J = len(zones)
    else:
        J  = n_rows * n_cols
        dx = (xmax - xmin) / n_cols
        dy = (ymax - ymin) / n_rows
        zones =[]
        for r in range(n_rows):
            y0 = ymin + r * dy
            y1 = y0 + dy
            for c in range(n_cols):
                zones.append(box(xmin + c * dx, y0, xmin + c * dx + dx, y1))

    prep_zones =[prep(z) for z in zones]

    # Gestion des mus
    mus_vec = [float(mus)] * J if np.isscalar(mus) else list(mus)
    if len(mus_vec) != J:
        raise ValueError(f"Le nombre d'intensités ({len(mus_vec)}) doit correspondre au nombre de zones ({J})")

    # Gestion de f
    is_global_f = False
    if f is None:
        is_global_f = True
        global_f = lambda x, y, **kwargs: np.zeros_like(x, dtype=float)
    elif callable(f):
        is_global_f = True
        global_f = f
    else:
        if len(f) != J:
            raise ValueError(f"Le nombre de fonctions f ({len(f)}) doit correspondre au nombre de zones ({J})")
        funcs = f

    # Fonction interne
    def get_spatial_components(x_arr, y_arr):
        """Calcule et renvoie mu_tilde, f_star, sig_f et mu_star"""
        x_flat, y_flat = np.array(x_arr, dtype=float).flatten(), np.array(y_arr, dtype=float).flatten()
        
        mu_tilde = np.zeros_like(x_flat)
        f_star = np.zeros_like(x_flat)
        
        # Si unique fonction f donnée 
        if is_global_f:
            f_all = global_f(x_flat, y_flat, **f_kwargs)
            f_star[:] = np.full_like(x_flat, f_all) if np.isscalar(f_all) else f_all

        # Optim algorithmique 
        # Au lieu de tester chaque point (dans chaque polygone), on marquage tous les points comme "non assignés" (1=True)
        # Dès qu'un point est trouvé dans une zone, on le retire des tests suivants
        unassigned = np.ones(len(x_flat), dtype=bool)
        
        for j, pzone in enumerate(prep_zones):
            idx_to_check = np.where(unassigned)[0]
            if len(idx_to_check) == 0: break
            
            in_zone = np.array([pzone.covers(ShapelyPoint(x_flat[i], y_flat[i])) for i in idx_to_check])
            inside_indices = idx_to_check[in_zone]
            
            if len(inside_indices) > 0:
                mu_tilde[inside_indices] = mus_vec[j]
                if not is_global_f:
                    f_star[inside_indices] = funcs[j](x_flat[inside_indices], y_flat[inside_indices], **f_kwargs)
                unassigned[inside_indices] = False
                
        sig_f = expit(f_star)
        mu_star = mu_tilde * sig_f
        return mu_tilde, f_star, sig_f, mu_star

    GX, GY = np.meshgrid(np.linspace(xmin, xmax, grid_res), np.linspace(ymin, ymax, grid_res))
    grid_mu_tilde, grid_f_star, grid_sig_f, grid_mu_star = get_spatial_components(GX, GY)
    
    # Reshape en matrices 2D pour le tracé
    grids = {
        "GX" : GX, "GY" : GY,
        "mu_tilde" : grid_mu_tilde.reshape(grid_res, grid_res),
        "f_star" : grid_f_star.reshape(grid_res, grid_res),
        "sig_f" : grid_sig_f.reshape(grid_res, grid_res),
        "mu_star" : grid_mu_star.reshape(grid_res, grid_res)
    }

    # Simulation des points avec thonning
    lambda_max = np.max(grids["mu_star"])
    n_proposed_total = int(ot.Poisson(lambda_max * (xmax - xmin) * (ymax - ymin) * T).getRealization()[0])
    
    if n_proposed_total == 0:
        X, n_accepted_total = ot.Sample(0, 3), 0
    else:
        cand_x = np.random.uniform(xmin, xmax, n_proposed_total)
        cand_y = np.random.uniform(ymin, ymax, n_proposed_total)
        cand_t = np.random.uniform(0.0, T, n_proposed_total)
        
        _, _, _, intensities = get_spatial_components(cand_x, cand_y)
        mask = np.random.uniform(0.0, 1.0, n_proposed_total) < (intensities / lambda_max)
        
        n_accepted_total = mask.sum()
        X = ot.Sample(np.column_stack((cand_x[mask], cand_y[mask], cand_t[mask])).tolist()) if n_accepted_total > 0 else ot.Sample(0, 3)

    print("=" * 54 + f"\n{' SIMULATED DATA SUMMARY ':-^54}\n" + "=" * 54)
    print(f"Zones (J): {J} | Time (T): {T} | Nb events (N): {n_accepted_total}")
    print("=" * 54 + "\n")

    return {
        "X" : X,
        "zones" : zones,
        "mus_vec" : mus_vec,
        "bounds" : (X_bounds, Y_bounds, T),
        "mu_tilde_func": lambda x, y: get_spatial_components(x, y)[0], 
        "f_star_func" : lambda x, y: get_spatial_components(x, y)[1], 
        "mu_star_func" : lambda x, y: get_spatial_components(x, y)[3], 
    }, grids


# GÉNÉRATION DE CELLULES DE VORONOÏ
def generate_voronoi_cells(
    n_germs = 5,
    X_bounds = (0.0, 2.0),
    Y_bounds = (0.0, 2.0),
    rng_seed = None,
):
    """

    """
    if rng_seed is not None:
        np.random.seed(rng_seed)

    xmin, xmax = X_bounds
    ymin, ymax = Y_bounds
    domain = box(xmin, ymin, xmax, ymax)

    # Germes : tirés uniformément 
    germs_x = np.random.uniform(xmin, xmax, n_germs)
    germs_y = np.random.uniform(ymin, ymax, n_germs)
    germs = np.column_stack([germs_x, germs_y])

    # Points miroirs (réflexions par rapport aux 4 bords)
    mirror_left = np.column_stack([2*xmin - germs_x,  germs_y])
    mirror_right = np.column_stack([2*xmax - germs_x,  germs_y])
    mirror_bottom = np.column_stack([germs_x, 2*ymin - germs_y])
    mirror_top = np.column_stack([germs_x, 2*ymax - germs_y])
    all_points = np.vstack([germs, mirror_left, mirror_right, mirror_bottom, mirror_top])

    vor = Voronoi(all_points)

    # Construction des polygones
    def _voronoi_cell(point_idx):
        region_idx = vor.point_region[point_idx]
        region = vor.regions[region_idx]
        polygon = ShapelyPolygon(vor.vertices[region])
        return polygon.intersection(domain)

    cells = [_voronoi_cell(i) for i in range(n_germs)]

    return cells, germs



# %%






# %%






# %%