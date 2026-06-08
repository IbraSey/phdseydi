#%%
"""
Experiment 4 — Robustness under prior misspecification
Scenarios M1, M2, M3 × Settings A, B

M1 : données homogènes (J*=1), inférence J=1 (oracle) vs J=6 (superflues)
M2 : données J*=6 Profile 1, inférence J=6 oracle vs J=5 mauvaise partition
M3 : données J*=6 Profile 1, inférence J=6 oracle vs J=1 (zones manquantes)

Utilise plot_posterior_intensity pour figures + métriques (RMSE/MAE/CRPS/ECP).
"""

# =============================================================================
# Imports
# =============================================================================
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import openturns as ot
import matplotlib.pyplot as plt
from functools import partial
from pathlib import Path
from scipy.special import expit
from shapely.geometry import Point as ShapelyPoint, box as shapely_box
from shapely.prepared import prep
from joblib import Parallel, delayed

from gp.gibbs_sampler import iSGCP_GibbsSampler
from gp.data_generation import generate_voronoi_cells, simulate_process
from visualizations.plot import plot_voronoi_cells, plot_process_dashboard


# =============================================================================
# Paramètres MCMC
# =============================================================================
NU_INIT           = [5.0, 0.2]
LAMBDA_NU         = 0.5
DELTA             = [1.0, 0.01]
JITTER            = 1e-5
BURN_IN           = 0.4
N_ITER            = 3000
THIN              = 3
MALA_STEP_J       = 0.085     # iSGCP (J > 1)
MALA_STEP_1       = 0.075     # SGCP homogène (J = 1)
LEARN_NU          = False
USE_CALIB         = True
T0_NU             = 50
STEP_NU_INIT      = 0.0009
VERBOSE           = True
VERBOSE_EVERY     = 500
SEED              = 42
NX, NY            = 30, 30
NX_POST, NY_POST  = 60, 60
N_CHAINS          = 5
XB, YB            = (0.0, 2.0), (0.0, 2.0)

T_PROFILE_1  = {"A": 30.0, "B": 25.0}
T_HOMOGENE   = {"A": 30.0, "B": 25.0}

GRID_RES_BY_SETTING = {"A": 100, "B": 300} 


# =============================================================================
# Fonctions latentes
# =============================================================================
def f_star_A(x, y):
    weights = [1.5, -1.5, 3.0, -3.0]
    sigma2  = 0.3
    means   = [
        ot.Point([0.5, 0.5]), ot.Point([0.5, 1.5]),
        ot.Point([1.5, 0.5]), ot.Point([1.5, 1.5]),
    ]
    Sigma = ot.CovarianceMatrix(2, [sigma2, 0.0, 0.0, sigma2])
    dists = [ot.Normal(m, Sigma) for m in means]
    sample = ot.Sample(np.column_stack((x, y)))
    return sum(
        w * np.array(d.computePDF(sample)).flatten()
        for w, d in zip(weights, dists)
    )


def f_star_B(x, y):
    centers = np.array([[0.4,0.4],[0.4,1.6],[1.0,1.0],[1.6,0.4],[1.6,1.6]])
    weights = np.array([+4.0, -3.5, +2.0, -4.5, +3.0])
    ells    = np.array([ 0.20,  0.20,  0.35,  0.15,  0.25])
    x_flat  = np.atleast_1d(x).flatten()
    y_flat  = np.atleast_1d(y).flatten()
    pts     = np.column_stack([x_flat, y_flat])
    f_vals  = np.zeros(len(pts))
    for w, c, ell in zip(weights, centers, ells):
        diff = pts - c
        f_vals += w * np.exp(-np.sum(diff**2, axis=1) / (2.0 * ell**2))
    return f_vals.reshape(np.shape(x))


F_STAR = {"A": f_star_A, "B": f_star_B}


# =============================================================================
# Profils de zones
# =============================================================================
# Profile 1 : J=6, high contrast — utilisé pour générer les données M2/M3
PROFILE_1 = {
    "n_germs": 6, "rng_seed": 15,
    "mus": (10.0, 1.0, 2.0, 10.0, 8.0, 2.0),
}

# Profile 2 : J=5 — partition DIFFÉRENTE pour l'inférence M2
PROFILE_2 = {
    "n_germs": 5, "rng_seed": 42,   # seed différent → frontières croisées
    "mus": (3.5, 2.0, 4.0, 3.0, 2.5),
}

# Homogène : J*=1 pour générer les données M1
MU_HOMOGENE = 5.0   # eps* = log(5)


# =============================================================================
# Helpers picklables
# =============================================================================
def mu_star_func_picklable(x, y, zones_raw, mus_vec, f_func):
    from scipy.special import expit
    from shapely.prepared import prep
    from shapely.geometry import Point as ShapelyPoint
    import numpy as np
    x_flat    = np.atleast_1d(x).flatten()
    y_flat    = np.atleast_1d(y).flatten()
    mu_tilde  = np.zeros(len(x_flat))
    unassigned = np.ones(len(x_flat), dtype=bool)
    for j, pz in enumerate([prep(z) for z in zones_raw]):
        idx = np.where(unassigned)[0]
        if len(idx) == 0:
            break
        inside = idx[[pz.covers(ShapelyPoint(x_flat[i], y_flat[i])) for i in idx]]
        if len(inside) > 0:
            mu_tilde[inside]   = mus_vec[j]
            unassigned[inside] = False
    return (mu_tilde * expit(f_func(x_flat, y_flat))).reshape(np.shape(x))


def run_chain(k, seed, zones_raw, x_arr, y_arr, t_arr, T,
              Xb, Yb, nu_init, lambda_nu, delta, jitter,
              mala_step, t0_nu, step_nu_init,
              n_iter, thin, verbose, verbose_every,
              mu_star_func, nx, ny):
    from shapely.prepared import prep
    import openturns as ot
    chain_seed       = seed + k
    zones_prep_local = [prep(p) for p in zones_raw]
    Areas_local      = [(zp, 0.0) for zp in zones_prep_local]
    sampler_k = iSGCP_GibbsSampler(
        X_bounds=Xb, Y_bounds=Yb, T=T,
        Areas=Areas_local, polygons=zones_raw,
        lambda_nu=lambda_nu, nu=nu_init,
        delta=delta, jitter=jitter, rng_seed=chain_seed,
    )
    results_k = sampler_k.run(
        t=ot.Point(t_arr.tolist()),
        x=ot.Point(x_arr.tolist()),
        y=ot.Point(y_arr.tolist()),
        mala_step=mala_step, learn_nu=LEARN_NU,
        t0_nu=t0_nu, step_nu_init=step_nu_init,
        n_iter=n_iter, thin=thin,
        verbose=verbose, verbose_every=verbose_every,
        use_calibration=USE_CALIB,
        mu_star_func=mu_star_func,
        grid_nx=nx, grid_ny=ny,
    )
    print(f"  [Chain {k+1}] done (seed={chain_seed})")
    return results_k, list(sampler_k.nu)


def launch_chains(zones_raw_list, x_arr, y_arr, t_arr, T,
                  mala_step, mu_star_func):
    chain_outputs = Parallel(n_jobs=-1, prefer="processes")(
        delayed(run_chain)(
            k, SEED, zones_raw_list,
            x_arr, y_arr, t_arr, T,
            XB, YB, NU_INIT, LAMBDA_NU, DELTA, JITTER,
            mala_step, T0_NU, STEP_NU_INIT,
            N_ITER, THIN, VERBOSE, VERBOSE_EVERY,
            mu_star_func, NX, NY,
        )
        for k in range(N_CHAINS)
    )
    all_results, all_nu = zip(*chain_outputs)
    return list(all_results), list(all_nu)


def build_sampler(zones_raw_list, nu_hat, T):
    zones_prep = [prep(p) for p in zones_raw_list]
    Areas      = [(zp, 0.0) for zp in zones_prep]
    return iSGCP_GibbsSampler(
        X_bounds=XB, Y_bounds=YB, T=T,
        Areas=Areas, polygons=zones_raw_list,
        lambda_nu=LAMBDA_NU, nu=nu_hat,
        delta=DELTA, jitter=JITTER, rng_seed=SEED,
    )


def get_nu_hat(all_results):
    burn   = int(BURN_IN * all_results[0]["nu"].shape[0])
    nu_all = np.concatenate([r["nu"][burn:] for r in all_results], axis=0)
    return nu_all.mean(axis=0).tolist()


# =============================================================================
# Fit + eval générique — appelle plot_posterior_intensity
# =============================================================================
def fit_and_eval(label, zones_raw_infer, x_arr, y_arr, t_arr, T,
                 mala_step, mu_star_func_true, scenario, setting,
                 cmap_intensities="inferno", savefigure=False):
    """
    Lance N_CHAINS, reconstruit le sampler, appelle plot_posterior_intensity.
    mu_star_func_true est la VRAIE intensité (celle qui a généré les données).
    zones_raw_infer est la partition utilisée pour l'INFÉRENCE (peut différer).
    """
    J_infer = len(zones_raw_infer)
    print(f"\n  >> {label} (J_infer={J_infer}) fitting")

    all_results, _ = launch_chains(
        zones_raw_infer, x_arr, y_arr, t_arr, T,
        mala_step, mu_star_func_true,
    )

    nu_hat  = get_nu_hat(all_results)
    sampler = build_sampler(zones_raw_infer, nu_hat, T)

    k_ref = N_CHAINS // 2
    out = sampler.plot_posterior_intensity(
        x=x_arr, y=y_arr, t=t_arr,
        results=all_results[k_ref],
        nx=NX_POST, ny=NY_POST,
        burn_in=BURN_IN,
        cmap=cmap_intensities,
        mu_star_func=mu_star_func_true,
        savefigure=savefigure,
        savefigure_Emu=savefigure,
        title_savefig=f"exp4_{scenario}_{label}_{setting}",
        title_savefig_Emu=f"exp4_{scenario}_{label}_Emu_{setting}",
    )

    return {
        "rmse": out["rmse"],
        "mae":  out["mae"],
        "crps": out.get("crps", None),
        "ecp":  out["ecp"],
        "out":  out,
        "all_results": all_results,
        "sampler": sampler,
    }


# =============================================================================
# Génération des données
# =============================================================================
def generate_data_profile1(setting_name, f_star_func):
    T = T_PROFILE_1[setting_name]
    grid_res = GRID_RES_BY_SETTING[setting_name]

    cells, germs = generate_voronoi_cells(
        n_germs=PROFILE_1["n_germs"],
        X_bounds=XB, Y_bounds=YB,
        rng_seed=PROFILE_1["rng_seed"],
    )

    sim_data, grids = simulate_process(
        X_bounds=XB, Y_bounds=YB, T=T,
        polygons=cells, mus=PROFILE_1["mus"],
        f=f_star_func, grid_res=grid_res, rng_seed=15,
    )

    X_data = sim_data["X"]
    N      = X_data.getSize()
    x_arr  = np.array([float(X_data[i, 0]) for i in range(N)])
    y_arr  = np.array([float(X_data[i, 1]) for i in range(N)])
    t_arr  = np.array([float(X_data[i, 2]) for i in range(N)])

    zones_raw_true = list(sim_data["zones"])
    mus_vec_true   = list(sim_data["mus_vec"])

    mu_star_true = partial(
        mu_star_func_picklable,
        zones_raw=zones_raw_true,
        mus_vec=mus_vec_true,
        f_func=f_star_func,
    )

    return x_arr, y_arr, t_arr, T, zones_raw_true, mu_star_true, cells, germs, sim_data, grids


def generate_data_homogeneous(setting_name, f_star_func):
    T = T_HOMOGENE[setting_name]
    grid_res = GRID_RES_BY_SETTING[setting_name]

    # Domaine entier comme zone unique
    domain_poly = shapely_box(XB[0], YB[0], XB[1], YB[1])

    sim_data, grids = simulate_process(
        X_bounds=XB, Y_bounds=YB, T=T,
        polygons=[domain_poly],
        mus=(MU_HOMOGENE,),
        f=f_star_func,
        grid_res=grid_res, rng_seed=15,
    )

    X_data = sim_data["X"]
    N      = X_data.getSize()
    x_arr  = np.array([float(X_data[i, 0]) for i in range(N)])
    y_arr  = np.array([float(X_data[i, 1]) for i in range(N)])
    t_arr  = np.array([float(X_data[i, 2]) for i in range(N)])

    zones_raw_true = [domain_poly]
    mus_vec_true   = [MU_HOMOGENE]

    mu_star_true = partial(
        mu_star_func_picklable,
        zones_raw=zones_raw_true,
        mus_vec=mus_vec_true,
        f_func=f_star_func,
    )

    return x_arr, y_arr, t_arr, T, zones_raw_true, mu_star_true, sim_data, grids


def get_inference_partition(profile):
    """Génère la partition Voronoï pour l'inférence."""
    cells, germs = generate_voronoi_cells(
        n_germs=profile["n_germs"],
        X_bounds=XB, Y_bounds=YB,
        rng_seed=profile["rng_seed"],
    )
    return list(cells), germs


# =============================================================================
# Tableau récapitulatif
# =============================================================================
def print_metrics_table(records):
    print(f"\n{'='*80}")
    print(f"  Experiment 4 — Métriques quantitatives")
    print(f"{'='*80}")
    print(f"  {'Scen.':<8} {'Setting':<9} {'Modèle':<32}"
          f" {'RMSE':>8} {'MAE':>8} {'CRPS':>8} {'ECP95':>8}")
    print(f"  {'-'*76}")

    from itertools import groupby
    for (sc, st), group in groupby(records, key=lambda r: (r["scenario"], r["setting"])):
        for r in group:
            crps_str = f"{r['crps']:.4f}" if r["crps"] is not None else "      --"
            ecp_str  = f"{r['ecp']:.4f}"  if r["ecp"]  is not None else "      --"
            print(f"  {r['scenario']:<8} {r['setting']:<9} {r['model']:<32}"
                  f" {r['rmse']:>8.4f} {r['mae']:>8.4f} {crps_str:>8} {ecp_str:>8}")
    print(f"{'='*80}\n")


# =============================================================================
# Scénarios
# =============================================================================
def run_scenario_M1(setting_name, cmap_voronoi="cividis", cmap_intensities="inferno", savefigure=False):
    """
    M1 : données homogènes J*=1, inférence J=1 (oracle) vs J=6 (superflues).
    """
    f_star_func = F_STAR[setting_name]

    print(f"\n{'#'*70}")
    print(f"  EXP4 — Scenario M1 (superfluous zones), Setting {setting_name}")
    print(f"{'#'*70}")

    # Données homogènes
    x_arr, y_arr, t_arr, T, zones_true, mu_star_true, sim_data, grids = \
        generate_data_homogeneous(setting_name, f_star_func)

    plot_process_dashboard(
        sim_data, grids,
        cmap=cmap_intensities,
        title=f"M1 données homogènes — Setting {setting_name}",
        savefigure=savefigure,
        title_savefig=f"exp4_M1_dashboard_{setting_name}",
    )

    records = []

    # Oracle : J=1 homogène (vraie structure)
    domain_poly  = shapely_box(XB[0], YB[0], XB[1], YB[1])
    res_oracle = fit_and_eval(
        "oracle_J1", [domain_poly],
        x_arr, y_arr, t_arr, T,
        MALA_STEP_1, mu_star_true,
        "M1", setting_name, cmap_intensities, savefigure,
    )
    records.append({
        "scenario": "M1", "setting": setting_name,
        "model": "Homogeneous SGCP (oracle)",
        **{k: res_oracle[k] for k in ["rmse", "mae", "crps", "ecp"]},
    })

    # Misspecified : J=6 superflues
    zones_J6, germs_J6 = get_inference_partition(PROFILE_1)

    plot_voronoi_cells(
        zones_J6, germs_J6,
        X_bounds=XB, Y_bounds=YB,
        cmap_name=cmap_voronoi,
        title=f"M1 — Partition superflue J=6 (Setting {setting_name})",
        savefigure=savefigure,
        title_savefig=f"exp4_M1_voronoi_J6_{setting_name}",
    )

    res_J6 = fit_and_eval(
        "superfluous_J6", zones_J6,
        x_arr, y_arr, t_arr, T,
        MALA_STEP_J, mu_star_true,
        "M1", setting_name, cmap_intensities, savefigure,
    )
    records.append({
        "scenario": "M1", "setting": setting_name,
        "model": "iSGCP, superfluous zones (J=6)",
        **{k: res_J6[k] for k in ["rmse", "mae", "crps", "ecp"]},
    })

    print_metrics_table(records)
    return records


def run_scenario_M2(setting_name, cmap_voronoi="cividis", cmap_intensities="inferno", savefigure=False):
    """
    M2 : données J*=6 Profile 1, inférence J=6 oracle vs J=5 mauvaise partition.
    """
    f_star_func = F_STAR[setting_name]

    print(f"\n{'#'*70}")
    print(f"  EXP4 — Scenario M2 (wrong partition), Setting {setting_name}")
    print(f"{'#'*70}")

    # Données Profile 1
    x_arr, y_arr, t_arr, T, zones_true, mu_star_true, cells_true, germs_true, sim_data, grids = \
        generate_data_profile1(setting_name, f_star_func)

    plot_voronoi_cells(
        cells_true, germs_true,
        X_bounds=XB, Y_bounds=YB,
        cmap_name=cmap_voronoi,
        title=f"M2 — Vraie partition J*=6 (Setting {setting_name})",
        savefigure=savefigure,
        title_savefig=f"exp4_M2_voronoi_true_{setting_name}",
    )

    plot_process_dashboard(
        sim_data, grids,
        cmap=cmap_intensities,
        title=f"M2 données Profile 1 — Setting {setting_name}",
        savefigure=savefigure,
        title_savefig=f"exp4_M2_dashboard_{setting_name}",
    )

    records = []

    # Oracle : J=6 vraie partition
    res_oracle = fit_and_eval(
        "oracle_J6", zones_true,
        x_arr, y_arr, t_arr, T,
        MALA_STEP_J, mu_star_true,
        "M2", setting_name, cmap_intensities, savefigure,
    )
    records.append({
        "scenario": "M2", "setting": setting_name,
        "model": "iSGCP, oracle (J=6)",
        **{k: res_oracle[k] for k in ["rmse", "mae", "crps", "ecp"]},
    })

    # Misspecified : J=5 mauvaise partition (Profile 2 avec seed différent)
    zones_wrong, germs_wrong = get_inference_partition(PROFILE_2)

    plot_voronoi_cells(
        zones_wrong, germs_wrong,
        X_bounds=XB, Y_bounds=YB,
        cmap_name=cmap_voronoi,
        title=f"M2 — Mauvaise partition J=5 (Setting {setting_name})",
        savefigure=savefigure,
        title_savefig=f"exp4_M2_voronoi_wrong_{setting_name}",
    )

    res_wrong = fit_and_eval(
        "wrong_J5", zones_wrong,
        x_arr, y_arr, t_arr, T,
        MALA_STEP_J, mu_star_true,
        "M2", setting_name, cmap_intensities, savefigure,
    )
    records.append({
        "scenario": "M2", "setting": setting_name,
        "model": "iSGCP, wrong partition (J=5)",
        **{k: res_wrong[k] for k in ["rmse", "mae", "crps", "ecp"]},
    })

    print_metrics_table(records)
    return records


def run_scenario_M3(setting_name, cmap_voronoi="cividis", cmap_intensities="inferno", savefigure=False):
    """
    M3 : données J*=6 Profile 1, inférence J=6 oracle vs J=1 (zones manquantes).
    """
    f_star_func = F_STAR[setting_name]

    print(f"\n{'#'*70}")
    print(f"  EXP4 — Scenario M3 (missing zones), Setting {setting_name}")
    print(f"{'#'*70}")

    # Données Profile 1
    x_arr, y_arr, t_arr, T, zones_true, mu_star_true, cells_true, germs_true, sim_data, grids = \
        generate_data_profile1(setting_name, f_star_func)

    plot_process_dashboard(
        sim_data, grids,
        cmap=cmap_intensities,
        title=f"M3 données Profile 1 — Setting {setting_name}",
        savefigure=savefigure,
        title_savefig=f"exp4_M3_dashboard_{setting_name}",
    )

    records = []

    # Oracle : J=6 vraie partition
    res_oracle = fit_and_eval(
        "oracle_J6", zones_true,
        x_arr, y_arr, t_arr, T,
        MALA_STEP_J, mu_star_true,
        "M3", setting_name, cmap_intensities, savefigure,
    )
    records.append({
        "scenario": "M3", "setting": setting_name,
        "model": "iSGCP, oracle (J=6)",
        **{k: res_oracle[k] for k in ["rmse", "mae", "crps", "ecp"]},
    })

    # Misspecified : J=1 homogène (zones manquantes)
    domain_poly = shapely_box(XB[0], YB[0], XB[1], YB[1])

    res_missing = fit_and_eval(
        "missing_J1", [domain_poly],
        x_arr, y_arr, t_arr, T,
        MALA_STEP_1, mu_star_true,
        "M3", setting_name, cmap_intensities, savefigure,
    )
    records.append({
        "scenario": "M3", "setting": setting_name,
        "model": "Homogeneous SGCP, missing zones (J=1)",
        **{k: res_missing[k] for k in ["rmse", "mae", "crps", "ecp"]},
    })

    print_metrics_table(records)
    return records


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":

    SAVEFIGURE  = True
    all_records = []

    for setting in ["A", "B"]:
        all_records.extend(run_scenario_M1(setting, savefigure=SAVEFIGURE))
        all_records.extend(run_scenario_M2(setting, savefigure=SAVEFIGURE))
        all_records.extend(run_scenario_M3(setting, savefigure=SAVEFIGURE))

    print("\n" + "=" * 90)
    print("  RÉCAPITULATIF GLOBAL — Experiment 4")
    print("=" * 90)
    print_metrics_table(all_records)

    print("Experiment 4 terminé.")



    #%%