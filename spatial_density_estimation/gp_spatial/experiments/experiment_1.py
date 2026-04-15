#%%
"""
Experiment 1 — iSGCP spatial intensity estimation
Settings A, B, C
"""

# =============================================================================
# Imports
# =============================================================================
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import openturns as ot
# import matplotlib.pyplot as plt
# import arviz as az
from joblib import Parallel, delayed
from functools import partial
# from pathlib import Path
# from scipy.special import expit
from shapely.geometry import Point as ShapelyPoint
from shapely.prepared import prep
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
# from polyagamma import random_polyagamma

from gp.gibbs_sampler import iSGCP_GibbsSampler
from gp.data_generation import generate_voronoi_cells, simulate_process
from visualizations.plot import plot_voronoi_cells, plot_process_dashboard


#%%
# =============================================================================
# Paramètres globaux
# =============================================================================
X_BOUNDS = (0.0, 2.0)
Y_BOUNDS = (0.0, 2.0)
N_GERMS = 6
RNG_SEED = 15
MUS_VORONOI = (10.0, 1.0, 2.0, 10.0, 8.0, 2.0)

NU_INIT = [5.0, 0.2]
LAMBDA_NU = 0.5
DELTA = [1.0, 0.01]
JITTER = 1e-5
BURN_IN = 0.4
N_ITER = 100
THIN = 2
MALA_STEP = 0.1
LEARN_NU = False
USE_CALIBRATION = True
T0_NU = 50
STEP_NU_INIT = 0.0009
VERBOSE = True
VERBOSE_EVERY = 10
SEED = 42
NX, NY = 3, 3
NX_POST, NY_POST = 12, 12
XB, YB = (0.0, 2.0), (0.0, 2.0)
N_CHAINS = 5


# =============================================================================
# Fonctions latentes
# =============================================================================
def f_star_A(x, y):
    weights = [1.5, -1.5, 3.0, -3.0]
    sigma2 = 0.3
    means = [
        ot.Point([0.5, 0.5]),
        ot.Point([0.5, 1.5]),
        ot.Point([1.5, 0.5]),
        ot.Point([1.5, 1.5]),
    ]
    Sigma = ot.CovarianceMatrix(2, [sigma2, 0.0, 0.0, sigma2])
    dists = [ot.Normal(m, Sigma) for m in means]
    sample = ot.Sample(np.column_stack((x, y)))
    return sum(
        w * np.array(d.computePDF(sample)).flatten()
        for w, d in zip(weights, dists)
    )


def f_star_B(x, y):
    centers = np.array([
        [0.4, 0.4],
        [0.4, 1.6],
        [1.0, 1.0],
        [1.6, 0.4],
        [1.6, 1.6],
    ])
    weights = np.array([+4.0, -3.5, +2.0, -4.5, +3.0])
    ells = np.array([0.20, 0.20, 0.35, 0.15, 0.25])

    x_flat = np.atleast_1d(x).flatten()
    y_flat = np.atleast_1d(y).flatten()
    pts = np.column_stack([x_flat, y_flat])

    f_vals = np.zeros(len(pts))
    for w, c, ell in zip(weights, centers, ells):
        diff = pts - c
        sq_dist = np.sum(diff ** 2, axis=1)
        f_vals += w * np.exp(-sq_dist / (2.0 * ell ** 2))

    return f_vals.reshape(np.shape(x))


def f_star_C(x, y):
    C_step = 3.0
    x0, y0 = 1.0, 1.0
    theta = np.pi / 5

    C_ridge = 2.5
    xr, yr = 1.0, 1.0
    phi = np.pi / 4
    ell_r = 0.15

    x_flat = np.atleast_1d(x).flatten()
    y_flat = np.atleast_1d(y).flatten()

    proj_step = (x_flat - x0) * np.cos(theta) + (y_flat - y0) * np.sin(theta)
    step = C_step * (proj_step > 0).astype(float)

    proj_ridge = -(x_flat - xr) * np.sin(phi) + (y_flat - yr) * np.cos(phi)
    ridge = C_ridge * np.exp(-proj_ridge ** 2 / (2.0 * ell_r ** 2))

    return (step + ridge).reshape(np.shape(x))


# =============================================================================
# Helpers picklables
# =============================================================================
def mu_star_func_picklable(x, y, zones_raw, mus_vec, f_func):
    x_flat = np.atleast_1d(x).flatten()
    y_flat = np.atleast_1d(y).flatten()
    mu_tilde = np.zeros(len(x_flat))
    unassigned = np.ones(len(x_flat), dtype=bool)

    for j, pz in enumerate([prep(z) for z in zones_raw]):
        idx = np.where(unassigned)[0]
        if len(idx) == 0:
            break
        inside = idx[[pz.covers(ShapelyPoint(x_flat[i], y_flat[i])) for i in idx]]
        if len(inside) > 0:
            mu_tilde[inside] = mus_vec[j]
            unassigned[inside] = False

    return (mu_tilde * expit(f_func(x_flat, y_flat))).reshape(np.shape(x))


def run_chain(k, seed, zones_raw, x_arr, y_arr, t_arr, T,
              Xb, Yb, nu_init, lambda_nu, delta, jitter,
              mala_step, learn_nu, t0_nu, step_nu_init,
              n_iter, thin, verbose_every, use_calibration,
              mu_star_func, nx, ny):

    chain_seed = seed + k
    zones_prep_local = [prep(p) for p in zones_raw]
    Areas_local = [(zp, 0.0) for zp in zones_prep_local]

    x_pt = ot.Point(x_arr.tolist())
    y_pt = ot.Point(y_arr.tolist())
    t_pt = ot.Point(t_arr.tolist())

    sampler_k = iSGCP_GibbsSampler(
        X_bounds=Xb,
        Y_bounds=Yb,
        T=T,
        Areas=Areas_local,
        polygons=zones_raw,
        lambda_nu=lambda_nu,
        nu=nu_init,
        delta=delta,
        jitter=jitter,
        rng_seed=chain_seed,
    )

    results_k = sampler_k.run(
        t=t_pt,
        x=x_pt,
        y=y_pt,
        mala_step=mala_step,
        learn_nu=learn_nu,
        t0_nu=t0_nu,
        step_nu_init=step_nu_init,
        n_iter=n_iter,
        thin=thin,
        verbose=VERBOSE,
        verbose_every=verbose_every,
        use_calibration=use_calibration,
        mu_star_func=mu_star_func,
        grid_nx=nx,
        grid_ny=ny,
    )

    print(f"[Chaîne {k+1}] terminée (seed={chain_seed})"
          f" — {n_iter} itérations, thin={thin}"
          f" => {n_iter // thin} échantillons conservés")

    return results_k, list(sampler_k.nu)


# =============================================================================
# Affichage des diagnostics
# =============================================================================
def print_diagnostics(r_hat, ess_bulk, ess_tail, J, n_chains):
    print("\n" + "=" * 50)
    print(f"Diagnostics multi-chaînes ({n_chains} chaînes)")
    print("=" * 50)
    print(f"{'Zone':<7} {'R-hat':>8} {'ESS bulk':>14} {'ESS tail':>11}")
    print("-" * 50)
    for j in range(J):
        rhat_flag = "(!)" if r_hat[j] > 1.01 else "(v)"
        ess_flag = "(!)" if ess_bulk[j] < 400 else "(v)"
        print(
            f"eps_{j:<4} "
            f"{r_hat[j]:>6.4f} {rhat_flag}  "
            f"{ess_bulk[j]:>7.1f} {ess_flag}  "
            f"{ess_tail[j]:>7.1f}"
        )
    print("=" * 50)
    print(f"R-hat max    : {r_hat.max():.4f} (seuil : 1.01)")
    print(f"ESS bulk min : {ess_bulk.min():.1f} (seuil : 400)")
    print(f"ESS tail min : {ess_tail.min():.1f} (seuil : 400)")
    print("=" * 50)


# =============================================================================
# Fonction principale par setting
# =============================================================================
def run_setting(setting_name, f_star_func, T, savefigure, grid_res=100):
    print(f"\n{'#'*70}")
    print(f"  SETTING {setting_name}")
    print(f"{'#'*70}\n")

    # ------------------------------------------------------------------
    # Génération du pavage et des données
    # ------------------------------------------------------------------
    cells, germs = generate_voronoi_cells(
        n_germs=N_GERMS,
        X_bounds=X_BOUNDS,
        Y_bounds=Y_BOUNDS,
        rng_seed=RNG_SEED,
    )

    plot_voronoi_cells(
        cells, germs,
        cmap_name="cividis",
        title=f"Pavage de Voronoï — Setting {setting_name}",
        savefigure=savefigure,
    )

    sim_data, grids = simulate_process(
        X_bounds=X_BOUNDS,
        Y_bounds=Y_BOUNDS,
        T=T,
        polygons=cells,
        mus=MUS_VORONOI,
        f=f_star_func,
        grid_res=grid_res,
        rng_seed=RNG_SEED,
    )

    plot_process_dashboard(sim_data, grids, savefigure=savefigure)

    X_data = sim_data["X"]
    zones = sim_data["zones"]
    N = X_data.getSize()

    x_arr = np.array([float(X_data[i, 0]) for i in range(N)])
    y_arr = np.array([float(X_data[i, 1]) for i in range(N)])
    t_arr = np.array([float(X_data[i, 2]) for i in range(N)])

    zones_raw_list = list(zones)
    mus_vec_list = list(sim_data["mus_vec"])

    mu_star_for_workers = partial(
        mu_star_func_picklable,
        zones_raw=zones_raw_list,
        mus_vec=mus_vec_list,
        f_func=f_star_func,
    )

    # ------------------------------------------------------------------
    # Lancement des chaînes en parallèle
    # ------------------------------------------------------------------
    print(f"Lancement de {N_CHAINS} chaînes en parallèle")

    chain_outputs = Parallel(n_jobs=-1, prefer="processes")(
        delayed(run_chain)(
            k, SEED, zones_raw_list,
            x_arr, y_arr, t_arr, T,
            XB, YB, NU_INIT, LAMBDA_NU, DELTA, JITTER,
            MALA_STEP, LEARN_NU, T0_NU, STEP_NU_INIT,
            N_ITER, THIN, VERBOSE_EVERY, USE_CALIBRATION,
            mu_star_for_workers, NX, NY,
        )
        for k in range(N_CHAINS)
    )

    all_results, all_nu_finals = zip(*chain_outputs)
    all_results = list(all_results)
    all_nu_finals = list(all_nu_finals)

    print(f"\n{N_CHAINS} chaînes terminées")

    # ------------------------------------------------------------------
    # Analyse postérieure
    # ------------------------------------------------------------------
    zones_prep_main = [prep(p) for p in zones_raw_list]
    Areas_main = [(zp, 0.0) for zp in zones_prep_main]

    all_outputs = []

    for k, results_k in enumerate(all_results):
        print(f"\n{'='*35}")
        print(f"Analyse chaîne {k+1}/{N_CHAINS} — Setting {setting_name}")
        print(f"{'='*35}")

        sampler_k = iSGCP_GibbsSampler(
            X_bounds=XB,
            Y_bounds=YB,
            T=T,
            Areas=Areas_main,
            polygons=zones_raw_list,
            lambda_nu=LAMBDA_NU,
            nu=all_nu_finals[k],
            delta=DELTA,
            jitter=JITTER,
            rng_seed=SEED + k,
        )

        out_k = sampler_k.plot_posterior_intensity(
            x=x_arr,
            y=y_arr,
            t=t_arr,
            results=results_k,
            nx=NX_POST,
            ny=NY_POST,
            burn_in=BURN_IN,
            cmap="viridis",
            mu_star_func=mu_star_for_workers,
            savefigure=savefigure,
            savefigure_Emu=savefigure,
            title_savefig=f"exp1_intensity_setting{setting_name}_chain{k+1}",
            title_savefig_Emu=f"exp1_Emu_setting{setting_name}_chain{k+1}",
        )
        all_outputs.append(out_k)

    # ------------------------------------------------------------------
    # Diagnostics chaîne finale + multi-chaînes
    # ------------------------------------------------------------------
    sampler_k.plot_chains(
        results_k,
        savefigure=savefigure,
        title_savefig=f"exp1_traces_setting{setting_name}",
    )

    sampler_k.plot_acf(
        results_k,
        burn_in=BURN_IN,
        savefigure=savefigure,
        title_savefig=f"exp1_acf_setting{setting_name}",
    )

    sampler_k.plot_rhat_arviz(
        all_results,
        burn_in=BURN_IN,
        savefigure=savefigure,
        title_savefig=f"exp1_rhat_setting{setting_name}",
    )

    r_hat, ess_bulk, ess_tail = sampler_k.compute_diagnostics_multichain(
        all_results, burn_in=BURN_IN
    )
    print_diagnostics(r_hat, ess_bulk, ess_tail, sampler_k.J, N_CHAINS)

    return all_results, all_outputs, all_nu_finals


# ================
# ----- Main -----
# ================
if __name__ == "__main__":

    results_A, outputs_A, nu_A = run_setting(
        setting_name="A",
        f_star_func=f_star_A,
        #T=60.0,
        T=6.0,
        savefigure=False,
        grid_res=100,
    )

    results_B, outputs_B, nu_B = run_setting(
        setting_name="B",
        f_star_func=f_star_B,
        #T=45.0,
        T=6.0,
        savefigure=False,
        grid_res=300,
    )

    results_C, outputs_C, nu_C = run_setting(
        setting_name="C",
        f_star_func=f_star_C,
        #T=30.0,
        T=6.0,
        savefigure=False,
        grid_res=300,
    )

    print("\nExperiment 1 terminé.")










#%%










#%%




