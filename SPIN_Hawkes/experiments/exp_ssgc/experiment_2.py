# %%
"""
Experiment 2 — Joint SSGC vs Zone-wise independent SGCPs
Profiles 1 and 2 x Settings A, B

Joint SSGC: un seul modèle et son résultat Gibbs.
Zone-wise: un modèle par zone, puis assemblage sur la grille globale.
"""

# =============================================================================
# Imports
# =============================================================================
import warnings
import sys
warnings.filterwarnings("ignore")

import time
import numpy as np
import openturns as ot
import matplotlib.pyplot as plt
import properscoring as ps
from functools import partial
from pathlib import Path
from scipy.special import expit
from shapely.geometry import Point as ShapelyPoint
from shapely.prepared import prep
from joblib import Parallel, delayed

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT))

try:
    from package import (
        EventCatalog,
        GPParameters,
        GibbsConfig,
        SSGCModel,
        generate_voronoi_cells,
        simulate_spatial_process,
    )
    from visualization import (
        plot_process_dashboard,
        plot_voronoi_cells,
        save_figure,
    )
except ImportError:
    from package import (
        EventCatalog,
        GPParameters,
        GibbsConfig,
        SSGCModel,
        generate_voronoi_cells,
        simulate_spatial_process,
    )
    from visualization import (
        plot_process_dashboard,
        plot_voronoi_cells,
        save_figure,
    )


# =============================================================================
# Paramètres globaux
# =============================================================================
X_BOUNDS = (0.0, 2.0)
Y_BOUNDS = (0.0, 2.0)
JITTER   = 1e-5

NU_INIT      = [5.0, 0.2]
LAMBDA_NU    = 0.5
DELTA        = [1.5, 0.01]
BURN_IN      = 0.4
N_ITER       = 200
THIN         = 5
LEARN_NU     = False
USE_CALIB    = True
T0_NU        = 50
STEP_NU_INIT = 0.0009
VERBOSE      = True
VERBOSE_EVERY = 50
SEED         = 42
NX, NY       = 30, 30
NX_POST, NY_POST = 60, 60
N_CHAINS     = 2
N_JOBS       = 1
GP_BACKEND   = "sparse"
COMPUTE_EMU  = False
MIN_ZONE_CALIB_EVENTS = 30


# =============================================================================
# Profils
# =============================================================================
PROFILE_1 = {
    "name":            "1",
    "n_germs":         6,
    "rng_seed_voronoi": 15,
    "mus":             (10.0, 1.0, 2.0, 10.0, 8.0, 2.0),
}
PROFILE_2 = {
    "name":            "2",
    "n_germs":         5,
    "rng_seed_voronoi": 15,
    "mus":             (3.5, 2.0, 4.0, 3.0, 2.5),
}

T_BY_PROFILE_SETTING = {
    # Calibrated with rng_seed=15 to generate about 1000 events per catalog.
    ("1", "A"): 180.7,
    ("1", "B"): 98.2,
    ("2", "A"): 163.9,
    ("2", "B"): 139.4,
}
GRID_RES_BY_SETTING = {"A": 100, "B": 300}


# =============================================================================
# MALA step par (profil, setting, modèle)
#
# Motivations :
#   - Profile 1 (contraste 10) : postérieure de ε plus raide.
#     Joint : step modéré ; zone-wise : chaque sampler ne voit qu'une zone,
#     ce qui simplifie la géométrie de la postérieure → step légèrement plus grand.
#   - Profile 2 (contraste ~2) : zones quasi-homogènes, gradient de ε faible
#     → on peut se permettre un step plus grand pour explorer plus vite.
#   - Setting B : amplitude élevée du champ latent, M_j plus variable
#     → step réduit par rapport à A pour stabiliser l'acceptance rate.
#   - Zone-wise (J=1 local) : gradient très simple dans chaque zone
#     → step identique entre profils pour le même setting.
# =============================================================================
MALA_STEP = {
    # Calibrated on 100--200 sparse-GP Gibbs iterations for about 1000 events.
    ("1", "A", "joint"):     0.065,
    ("1", "B", "joint"):     0.065,
    ("2", "A", "joint"):     0.070,
    ("2", "B", "joint"):     0.070,
    # Zone-wise chains are J=1 local fits; small zones tend to accept almost all
    # proposals, so these values target the larger local catalogs.
    ("1", "A", "zonewise"):  0.060,
    ("1", "B", "zonewise"):  0.055,
    ("2", "A", "zonewise"):  0.060,
    ("2", "B", "zonewise"):  0.055,
}


def get_mala_step(profile_name, setting_name, model):
    """Retourne le MALA step pour la configuration (profil, setting, modèle).

    Parameters
    ----------
    profile_name : str
        Nom du profil ("1" ou "2").
    setting_name : str
        Setting du champ latent ("A" ou "B").
    model : str
        "joint" ou "zonewise".

    Returns
    -------
    float
    """
    key = (profile_name, setting_name, model)
    if key not in MALA_STEP:
        raise KeyError(
            f"Pas de MALA step défini pour {key}. "
            f"Ajouter une entrée dans MALA_STEP."
        )
    return MALA_STEP[key]


# =============================================================================
# Fonctions de champ latent
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
    centers = np.array([[0.4, 0.4], [0.4, 1.6], [1.0, 1.0], [1.6, 0.4], [1.6, 1.6]])
    weights = np.array([+4.0, -3.5, +2.0, -4.5, +3.0])
    ells    = np.array([ 0.20,  0.20,  0.35,  0.15,  0.25])
    pts = np.column_stack([np.atleast_1d(x).flatten(), np.atleast_1d(y).flatten()])
    f_vals = np.zeros(len(pts))
    for w, c, ell in zip(weights, centers, ells):
        diff = pts - c
        f_vals += w * np.exp(-np.sum(diff ** 2, axis=1) / (2.0 * ell ** 2))
    return f_vals.reshape(np.shape(x))


F_STAR = {"A": f_star_A, "B": f_star_B}


# =============================================================================
# Helpers picklables
# =============================================================================
def mu_star_func_picklable(x, y, zones_raw, mus_vec, f_func):
    from scipy.special import expit
    from shapely.prepared import prep
    from shapely.geometry import Point as ShapelyPoint
    import numpy as np
    x_flat     = np.atleast_1d(x).flatten()
    y_flat     = np.atleast_1d(y).flatten()
    mu_tilde   = np.zeros(len(x_flat))
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
              n_iter, thin, verbose, verbose_every, use_calib,
              mu_star_func, nx, ny):
    chain_seed       = seed + k
    model = SSGCModel.from_polygons(
        polygons=zones_raw,
        duration=T,
        x_bounds=Xb,
        y_bounds=Yb,
        initial_log_intensities=0.0,
        gp_prior=GPParameters(*nu_init),
        eps_prior_variance=delta[0],
        eps_prior_length_scale=delta[1],
        nu_prior_rate=lambda_nu,
        jitter=jitter,
    )
    config = GibbsConfig(
        mala_step=mala_step, learn_nu=LEARN_NU,
        t0_nu=t0_nu, step_nu_init=step_nu_init,
        n_iter=n_iter, thin=thin,
        verbose=verbose, verbose_every=verbose_every,
        use_calibration=use_calib,
        grid_nx=nx, grid_ny=ny,
        compute_emu=COMPUTE_EMU,
    )
    results_k = model.gibbs(
        EventCatalog(t=t_arr, x=x_arr, y=y_arr),
        config=config,
        rng_seed=chain_seed,
        reference_intensity=mu_star_func,
        gp_backend=GP_BACKEND,
    )
    print(f"  [Chain {k+1}] done (seed={chain_seed})")
    return results_k


def launch_chains(zones_raw_list, x_arr, y_arr, t_arr, T,
                  mala_step, mu_star_func,
                  Xb=None, Yb=None, seed_offset=0, use_calib=USE_CALIB):
    Xb = Xb or X_BOUNDS
    Yb = Yb or Y_BOUNDS
    chain_outputs = Parallel(n_jobs=N_JOBS, prefer="processes")(
        delayed(run_chain)(
            k, SEED + seed_offset, zones_raw_list,
            x_arr, y_arr, t_arr, T,
            Xb, Yb, NU_INIT, LAMBDA_NU, DELTA, JITTER,
            mala_step, T0_NU, STEP_NU_INIT,
            N_ITER, THIN, VERBOSE, VERBOSE_EVERY, use_calib,
            mu_star_func, NX, NY,
        )
        for k in range(N_CHAINS)
    )
    return list(chain_outputs)


# =============================================================================
# Fit joint SSGC
# =============================================================================
def fit_and_eval_joint(x_arr, y_arr, t_arr, T, zones_raw_list,
                       mu_star_func, profile_name, setting_name,
                       savefigure, cmap_intensities="inferno"):
    step = get_mala_step(profile_name, setting_name, "joint")
    print(f"\n  >> Joint SSGC (Profile {profile_name})  — mala_step={step}")
    t0 = time.time()

    all_results = launch_chains(
        zones_raw_list, x_arr, y_arr, t_arr, T,
        step, mu_star_func,
    )
    elapsed = time.time() - t0
    print(f"  >> Joint SSGC : {elapsed:.1f}s")

    k_ref   = N_CHAINS // 2
    tag     = f"P{profile_name}_{setting_name}"

    out = all_results[k_ref].posterior_intensity(
        nx=NX_POST, ny=NY_POST, burn_in=BURN_IN, cmap=cmap_intensities,
        mu_star_func=mu_star_func,
        savefigure=savefigure, savefigure_Emu=savefigure and COMPUTE_EMU,
        title_savefig=f"ssgc/experiment_2/exp2_intensity_joint_{tag}",
        title_savefig_Emu=f"ssgc/experiment_2/exp2_Emu_joint_{tag}",
    )
    plt.close("all")
    return out, all_results, elapsed


# =============================================================================
# Fit zone-wise SGCP
# =============================================================================
def fit_and_eval_zonewise(x_arr, y_arr, t_arr, T, zones_raw_list,
                          mu_star_func, profile_name, setting_name,
                          savefigure, cmap_intensities="inferno"):
    """Un sampler SGCP (J=1) par zone, stitch sur la grille globale."""
    step = get_mala_step(profile_name, setting_name, "zonewise")
    print(f"\n  >> Zone-wise SGCP (Profile {profile_name})  — mala_step={step}")
    t0 = time.time()

    J = len(zones_raw_list)
    zones_prep_full = [prep(z) for z in zones_raw_list]
    tag = f"P{profile_name}_{setting_name}"

    # Partition des observations par zone
    points_per_zone = [[] for _ in range(J)]
    for i in range(len(x_arr)):
        pt = ShapelyPoint(float(x_arr[i]), float(y_arr[i]))
        for j, pz in enumerate(zones_prep_full):
            if pz.covers(pt):
                points_per_zone[j].append(i)
                break

    # Grille globale pour le stitching
    xmin, xmax = X_BOUNDS
    ymin, ymax = Y_BOUNDS
    interval   = ot.Interval([xmin, ymin], [xmax, ymax])
    mesh_global = ot.IntervalMesher([NX_POST - 1, NY_POST - 1]).build(interval)
    XY_grid_global = mesh_global.getVertices()
    grid_xy = np.array(XY_grid_global)
    M, n_mc = len(grid_xy), 500

    mu_hat_zw      = np.zeros(M)
    mu_var_zw      = np.zeros(M)
    mu_hat_sims_zw = np.zeros((M, n_mc))
    per_zone_metrics = []

    for j in range(J):
        idx_j = points_per_zone[j]
        N_j   = len(idx_j)
        print(f"     Zone {j+1}/{J} (N_j={N_j})")

        if N_j < 3:
            per_zone_metrics.append(None)
            continue

        x_j  = x_arr[idx_j]
        y_j  = y_arr[idx_j]
        t_j  = t_arr[idx_j]
        poly = zones_raw_list[j]
        pz   = zones_prep_full[j]
        bx, by, bx2, by2 = poly.bounds

        zone_use_calib = USE_CALIB and N_j >= MIN_ZONE_CALIB_EVENTS
        if not zone_use_calib:
            print("       calibration skipped for this small zone")
        all_results_j = launch_chains(
            [poly], x_j, y_j, t_j, T,
            step, mu_star_func,
            Xb=(bx, bx2), Yb=(by, by2),
            seed_offset=1000 * j,
            use_calib=zone_use_calib,
        )
        k_ref = N_CHAINS // 2
        result_j = all_results_j[k_ref]
        out_j = result_j.posterior_intensity(
            nx=NX_POST, ny=NY_POST, burn_in=BURN_IN, cmap=cmap_intensities,
            mu_star_func=mu_star_func,
            savefigure=savefigure, savefigure_Emu=savefigure and COMPUTE_EMU,
            title_savefig=f"ssgc/experiment_2/exp2_intensity_zw_zone{j}_{tag}",
            title_savefig_Emu=f"ssgc/experiment_2/exp2_Emu_zw_zone{j}_{tag}",
        )
        plt.close("all")
        per_zone_metrics.append({
            "rmse": out_j["rmse"],
            "mae":  out_j["mae"],
            "crps": out_j.get("crps", None),
        })

        # Stitch sur les points de grille appartenant à cette zone
        in_zone = np.array([
            pz.covers(ShapelyPoint(float(grid_xy[k, 0]), float(grid_xy[k, 1])))
            for k in range(M)
        ])
        if not in_zone.any():
            continue

        idx_local  = np.where(in_zone)[0]
        mu_sims_l = result_j.background_intensity_samples(
            grid_xy[idx_local, 0],
            grid_xy[idx_local, 1],
            burn_in=BURN_IN,
            n_samples=n_mc,
        )

        mu_hat_zw[idx_local]      = mu_sims_l.mean(axis=1)
        mu_var_zw[idx_local]      = mu_sims_l.var(axis=1)
        mu_hat_sims_zw[idx_local] = mu_sims_l

    elapsed = time.time() - t0
    print(f"  >> Zone-wise SGCP : {elapsed:.1f}s")

    # Métriques globales
    mu_star_global = mu_star_func(grid_xy[:, 0], grid_xy[:, 1])
    rmse_zw = float(np.sqrt(np.mean((mu_hat_zw - mu_star_global) ** 2)))
    mae_zw  = float(np.mean(np.abs(mu_hat_zw - mu_star_global)))
    crps_zw = float(ps.crps_ensemble(mu_star_global, mu_hat_sims_zw).mean())
    return {
        "rmse": rmse_zw, "mae": mae_zw, "crps": crps_zw,
        "mu_hat": mu_hat_zw, "mu_var": mu_var_zw,
        "mu_star": mu_star_global, "mu_hat_sims": mu_hat_sims_zw,
        "grid_xy": grid_xy, "mesh": mesh_global,
    }, per_zone_metrics, elapsed


# =============================================================================
# Visualisations comparatives
# =============================================================================
def _save(fig, name):
    path = save_figure(fig, f"ssgc/experiment_2/{name}")
    print(f"  Figure sauvegardée : {path}")


# =============================================================================
# Fonction principale par (profil, setting)
# =============================================================================
def run_exp2_config(profile, setting_name, cmap_voronoi="cividis",
                    cmap_intensities="inferno", savefigure=False):
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    profile_name = profile["name"]
    f_star_func  = F_STAR[setting_name]
    T            = T_BY_PROFILE_SETTING[(profile_name, setting_name)]
    grid_res     = GRID_RES_BY_SETTING[setting_name]

    step_j  = get_mala_step(profile_name, setting_name, "joint")
    step_zw = get_mala_step(profile_name, setting_name, "zonewise")

    print(f"\n{'#'*70}")
    print(f"  EXP2 — Profile {profile_name}, Setting {setting_name}")
    print(f"  MALA step : joint={step_j}, zone-wise={step_zw}")
    print(f"{'#'*70}")

    cells, germs = generate_voronoi_cells(
        n_germs=profile["n_germs"],
        X_bounds=X_BOUNDS, Y_bounds=Y_BOUNDS,
        rng_seed=profile["rng_seed_voronoi"],
    )
    plot_voronoi_cells(
        cells, germs, X_bounds=X_BOUNDS, Y_bounds=Y_BOUNDS,
        cmap_name=cmap_voronoi,
        title=f"Voronoï — Profile {profile_name}, Setting {setting_name}",
        savefigure=savefigure,
        title_savefig=f"ssgc/experiment_2/exp2_voronoi_P{profile_name}_{setting_name}",
    )

    simulation = simulate_spatial_process(
        X_bounds=X_BOUNDS, Y_bounds=Y_BOUNDS, T=T,
        polygons=cells, mus=profile["mus"],
        f=f_star_func, grid_res=grid_res, rng_seed=15,
    )
    plot_process_dashboard(
        simulation, cmap=cmap_intensities,
        title=f"Processus — Profile {profile_name}, Setting {setting_name}",
        savefigure=savefigure,
        title_savefig=f"ssgc/experiment_2/exp2_dashboard_P{profile_name}_{setting_name}",
    )

    X_data         = simulation.sample
    N              = X_data.getSize()
    x_arr          = np.array([float(X_data[i, 0]) for i in range(N)])
    y_arr          = np.array([float(X_data[i, 1]) for i in range(N)])
    t_arr          = np.array([float(X_data[i, 2]) for i in range(N)])
    zones_raw_list = list(simulation.domains.polygons)
    mus_vec_list   = list(simulation.baseline_intensities)

    mu_star_for_workers = partial(
        mu_star_func_picklable,
        zones_raw=zones_raw_list,
        mus_vec=mus_vec_list,
        f_func=f_star_func,
    )

    joint_out, joint_results, time_joint = fit_and_eval_joint(
        x_arr, y_arr, t_arr, T, zones_raw_list,
        mu_star_for_workers, profile_name, setting_name,
        savefigure, cmap_intensities,
    )
    zw_metrics, zw_per_zone, time_zw = fit_and_eval_zonewise(
        x_arr, y_arr, t_arr, T, zones_raw_list,
        mu_star_for_workers, profile_name, setting_name,
        savefigure, cmap_intensities,
    )

    print_summary(joint_out, zw_metrics, zw_per_zone,
                  time_joint, time_zw, profile_name, setting_name)
    plot_intensity_comparison(joint_out, zw_metrics, profile_name, setting_name,
                              cmap_intensities=cmap_intensities, savefigure=savefigure)
    plot_variance_comparison(joint_out, zw_metrics, profile_name, setting_name,
                             savefigure=savefigure)
    plot_calibration_curves(joint_out, zw_metrics, profile_name, setting_name,
                            savefigure=savefigure)

    return {
        "profile":       profile_name,
        "setting":       setting_name,
        "joint_out":     joint_out,
        "zw_metrics":    zw_metrics,
        "zw_per_zone":   zw_per_zone,
        "time_joint":    time_joint,
        "time_zw":       time_zw,
        "joint_results": joint_results,
    }


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":

    SAVEFIGURE  = True
    all_outputs = []

    configs = [
        (PROFILE_1, "A"),
        (PROFILE_1, "B"),
        (PROFILE_2, "A"),
        (PROFILE_2, "B"),
    ]

    for profile, setting in configs:
        out = run_exp2_config(profile, setting, savefigure=SAVEFIGURE)
        all_outputs.append(out)

    print("\nExperiment 2 terminé.")
