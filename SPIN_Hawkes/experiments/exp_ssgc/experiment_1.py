# %%
"""
Experiment 1 — SSGC vs Homogeneous SGCP vs KDE
Profiles 1, 2, 3 x Settings A, B
"""

# =========
# Imports
# =========
import warnings
import gc
import sys
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
import openturns as ot
import matplotlib.pyplot as plt
from functools import partial
from pathlib import Path
from scipy.special import expit
from shapely.geometry import Point as ShapelyPoint, box as shapely_box
from shapely.prepared import prep
from joblib import Parallel, delayed

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT))

from package import (
    EventCatalog,
    GPParameters,
    GibbsConfig,
    SSGCModel,
    generate_voronoi_cells,
    simulate_spatial_process,
)
from visualization import (
    plot_process_dashboard, plot_voronoi_cells, save_figure,
)


# ========================
# Paramètres MCMC
# ========================
NU_INIT          = [5.0, 0.2]
LAMBDA_NU        = 0.5
DELTA            = [1.5, 0.01]
JITTER           = 1e-5
BURN_IN          = 0.4
N_ITER           = 200
THIN             = 3
LEARN_NU         = False
USE_CALIB        = True
USE_SPARSE_GP    = True
T0_NU            = 50
STEP_NU_INIT     = 0.0009
VERBOSE          = True
VERBOSE_EVERY    = 50
SEED             = 42
NX, NY           = 30, 30
NX_POST, NY_POST = 60, 60
POSTERIOR_N_MC = 150
CLOSE_FIGURES  = True
XB, YB           = (0.0, 2.0), (0.0, 2.0)
N_CHAINS         = 2
N_JOBS           = 1


MALA_STEP = {
    #  (profile_name, setting, model) -> step
    ("1", "A", "SSGC"):  0.055,
    ("1", "B", "SSGC"):  0.060,
    ("2", "A", "SSGC"):  0.070,
    ("2", "B", "SSGC"):  0.064,
    ("3", "A", "SSGC"):  0.055,
    ("3", "B", "SSGC"):  0.056,
    # SGCP homogène : J=1, step identique quel que soit le profil "vrai"
    ("1", "A", "SGCP"):  0.050,
    ("1", "B", "SGCP"):  0.047,
    ("2", "A", "SGCP"):  0.047,
    ("2", "B", "SGCP"):  0.050,
    ("3", "A", "SGCP"):  0.045,
    ("3", "B", "SGCP"):  0.045,
}

def get_mala_step(profile_name, setting_name, model):
    """Retourne le MALA step pour la configuration (profil, setting, modèle).

    Parameters
    ----------
    profile_name : str
        Nom du profil ("1", "2", "3").
    setting_name : str
        Setting du champ latent ("A" ou "B").
    model : str
        Modèle cible ("SSGC" ou "SGCP").

    Returns
    -------
    float
        MALA step size h.
    """
    key = (profile_name, setting_name, model)
    if key not in MALA_STEP:
        raise KeyError(
            f"Pas de MALA step défini pour {key}. "
            f"Ajouter une entrée dans MALA_STEP."
        )
    return MALA_STEP[key]


# ==================
# Profils de zones
# ==================
PROFILE_1 = {
    "name": "1", "n_germs": 6, "rng_seed_voronoi": 20,
    "mus": (10.0, 1.0, 2.0, 10.0, 8.0, 2.0), "J": 6
}
PROFILE_2 = {
    "name": "2", "n_germs": 5, "rng_seed_voronoi": 20,
    "mus": (3.5, 2.0, 4.0, 3.0, 2.5), "J": 5
}
PROFILE_3 = {
    "name": "3", "n_germs": 7, "rng_seed_voronoi": 20,
    "mus": (20.0, 1.0, 1.0, 1.0, 1.0, 1.0, 20.0), "J": 7
}

T_BY_PROFILE = {
    "1": {"A": 180, "B": 95}, 
    "2": {"A": 160, "B": 135}, 
    "3": {"A": 50, "B": 50},
}
GRID_RES_BY_SETTING = {"A": 100, "B": 200}


# ===========================
# Fonctions de champ latent
# ===========================
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


# =============================================
# Helpers picklables (parallélisation joblib)
# =============================================
def mu_star_func_picklable(x, y, zones_raw, mus_vec, f_func):
    from scipy.special import expit
    from shapely.prepared import prep
    from shapely.geometry import Point as ShapelyPoint
    import numpy as np

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
              mala_step, t0_nu, step_nu_init,
              n_iter, thin, verbose, verbose_every,
              mu_star_func, nx, ny):
    chain_seed = seed + k
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
        mala_step=mala_step,
        learn_nu=LEARN_NU,
        t0_nu=t0_nu,
        step_nu_init=step_nu_init,
        n_iter=n_iter,
        thin=thin,
        verbose=verbose,
        verbose_every=verbose_every,
        use_calibration=USE_CALIB,
        grid_nx=nx,
        grid_ny=ny,
        compute_emu=False,
    )
    catalog = EventCatalog(t=t_arr, x=x_arr, y=y_arr)
    results_k = model.gibbs(
        catalog,
        config=config,
        rng_seed=chain_seed,
        reference_intensity=mu_star_func,
        gp_backend="sparse" if USE_SPARSE_GP else "exact",
    )
    print(f"  [Chain {k+1}] done (seed={chain_seed})")
    return results_k


def launch_chains(zones_raw_list, x_arr, y_arr, t_arr, T,
                  mala_step, mu_star_func):
    """Lance N_CHAINS chaînes indépendantes.

    OpenTURNS objects stored in GibbsResults are not robustly picklable, so the
    default is sequential execution. Use process parallelism only if the worker
    return object is reduced to pure NumPy/Python data.
    """
    if N_JOBS == 1:
        return [
            run_chain(
                k, SEED, zones_raw_list,
                x_arr, y_arr, t_arr, T,
                XB, YB, NU_INIT, LAMBDA_NU, DELTA, JITTER,
                mala_step, T0_NU, STEP_NU_INIT,
                N_ITER, THIN, VERBOSE, VERBOSE_EVERY,
                mu_star_func, NX, NY,
            )
            for k in range(N_CHAINS)
        ]

    chain_outputs = Parallel(n_jobs=N_JOBS, prefer="processes")(
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
    return list(chain_outputs)


# ===============
# KDE référence
# ===============
def compute_kde_intensity(x_arr, y_arr, T, nx=NX_POST, ny=NY_POST):
    """Estime μ̂_KDE(x,y) = N/T · p̂_h(x,y), bandwidth Silverman.

    Returns
    -------
    mu_kde : ndarray, shape (nx*ny,)
    GX, GY : ndarray, shape (ny, nx)
    """
    N = len(x_arr)
    sample = ot.Sample([[float(x_arr[i]), float(y_arr[i])] for i in range(N)])
    kde = ot.KernelSmoothing().build(sample)

    gx = np.linspace(XB[0], XB[1], nx)
    gy = np.linspace(YB[0], YB[1], ny)
    GX, GY = np.meshgrid(gx, gy)
    grid_pts = ot.Sample(np.column_stack([GX.ravel(), GY.ravel()]).tolist())

    p_hat = np.array(kde.computePDF(grid_pts)).flatten()
    return (N / T) * p_hat, GX, GY


def compute_kde_metrics(mu_kde, mu_star_grid):
    rmse = float(np.sqrt(np.mean((mu_kde - mu_star_grid) ** 2)))
    mae  = float(np.mean(np.abs(mu_kde - mu_star_grid)))
    return {"rmse": rmse, "mae": mae, "crps": None}


# =========
# Figures
# =========
def plot_kde_vs_true(mu_star_grid, mu_kde, GX, GY,
                     profile_name, setting_name, cmap="inferno", savefigure=False):
    ny_r, nx_r = GX.shape
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, field, title in zip(
        axes,
        [mu_star_grid, mu_kde],
        [r"Vraie intensité $\mu^\star(s)$", r"KDE $\hat{\mu}_{\mathrm{KDE}}(s)$"],
    ):
        im = ax.contourf(GX, GY, field.reshape(ny_r, nx_r), levels=30, cmap=cmap)
        plt.colorbar(im, ax=ax)
        ax.set_title(title)
        ax.set_xlim(XB); ax.set_ylim(YB)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3, color="white", linewidth=0.5)
    plt.suptitle(
        f"Experiment 1 — Profile {profile_name}, Setting {setting_name} — KDE",
        fontsize=12,
    )
    plt.tight_layout()
    if savefigure:
        _save_figure(fig, f"exp1_true_intensity_{profile_name}{setting_name}")
    plt.show()


def _save_figure(fig, filename):
    path = save_figure(fig, f"ssgc/experiment_1/{filename}", figure_type="raster")
    print(f"  Figure sauvegardée : {path}")


# ===============
# Tableau récap
# ===============
def print_metrics_table(records):
    print(f"\n{'='*80}")
    print(f"  Experiment 1 — Métriques quantitatives")
    print(f"{'='*80}")
    print(f"  {'Profile':<9} {'Setting':<9} {'Modèle':<22}"
          f" {'RMSE':>8} {'MAE':>8} {'CRPS':>8}")
    print(f"  {'-'*76}")
    for r in records:
        crps_str = f"{r['crps']:.4f}" if r["crps"] is not None else "      --"
        print(f"  {r['profile']:<9} {r['setting']:<9} {r['model']:<22}"
              f" {r['rmse']:>8.4f} {r['mae']:>8.4f} {crps_str:>8}")
    print(f"{'='*80}\n")


# ==========================================
# Fonction principale par (profil, setting)
# ==========================================
def run_exp1_config(profile, setting_name, cmap_voronoi="cividis",
                    cmap_intensities="inferno", savefigure=False):
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    f_star_func  = F_STAR[setting_name]
    T            = T_BY_PROFILE[profile["name"]][setting_name]
    profile_name = profile["name"]
    mus_voronoi  = profile["mus"]

    step_ssgc = get_mala_step(profile_name, setting_name, "SSGC")
    step_sgcp = get_mala_step(profile_name, setting_name, "SGCP")

    print(f"\n{'#'*70}")
    print(f"  EXP1 — Profile {profile_name}, Setting {setting_name}")
    print(f"  MALA step : SSGC={step_ssgc}, SGCP={step_sgcp}")
    print(f"{'#'*70}")

    # --- Génération pavage ---
    cells, germs = generate_voronoi_cells(
        n_germs=profile["n_germs"],
        X_bounds=XB, Y_bounds=YB,
        rng_seed=profile["rng_seed_voronoi"],
    )
    plot_voronoi_cells(
        cells, germs, X_bounds=XB, Y_bounds=YB, cmap_name=cmap_voronoi,
        title=f"Pavage Voronoï — Profile {profile_name}, Setting {setting_name}",
        savefigure=savefigure,
        title_savefig=f"ssgc/experiment_1/exp1_voronoi_{profile_name}{setting_name}",
    )

    # --- Simulation ---
    simulation = simulate_spatial_process(
        X_bounds=XB, Y_bounds=YB, T=T,
        polygons=cells, mus=mus_voronoi,
        f=f_star_func, grid_res=GRID_RES_BY_SETTING[setting_name], rng_seed=15,
    )
    plot_process_dashboard(
        simulation, cmap=cmap_intensities,
        title=f"Processus spatial — Profile {profile_name}, Setting {setting_name}",
        savefigure=savefigure,
        title_savefig=f"ssgc/experiment_1/exp1_dashboard_{profile_name}{setting_name}",
    )

    X_data = simulation.sample
    N = X_data.getSize()
    x_arr = np.array([float(X_data[i, 0]) for i in range(N)])
    y_arr = np.array([float(X_data[i, 1]) for i in range(N)])
    t_arr = np.array([float(X_data[i, 2]) for i in range(N)])
    zones_raw_list = list(simulation.domains.polygons)
    mus_vec_list   = list(simulation.baseline_intensities)

    mu_star_for_workers = partial(
        mu_star_func_picklable,
        zones_raw=zones_raw_list,
        mus_vec=mus_vec_list,
        f_func=f_star_func,
    )

    records = []
    k_ref = N_CHAINS // 2

    # =========================================================
    # Modèle 1 — SSGC (J zones)
    # =========================================================
    print(f"\n  >> SSGC (J={len(zones_raw_list)})  — mala_step={step_ssgc}")
    results_ssgc = launch_chains(
        zones_raw_list, x_arr, y_arr, t_arr, T,
        step_ssgc, mu_star_for_workers,
    )
    out_ssgc = results_ssgc[k_ref].posterior_intensity(
        nx=NX_POST, ny=NY_POST, burn_in=BURN_IN, cmap=cmap_intensities,
        mu_star_func=mu_star_for_workers,
        savefigure=savefigure, savefigure_Emu=savefigure,
        title_savefig=f"ssgc/experiment_1/exp1_intensity_SSGC_{profile_name}{setting_name}",
        title_savefig_Emu=f"ssgc/experiment_1/exp1_Emu_SSGC_{profile_name}{setting_name}",
        n_mc=POSTERIOR_N_MC,
    )
    records.append({
        "profile": profile_name, "setting": setting_name,
        "model": "SSGC", "mala_step": step_ssgc,
        "rmse": out_ssgc["rmse"], "mae": out_ssgc["mae"],
        "crps": out_ssgc["crps"],
    })

    # =========================================================
    # Modèle 2 — SGCP homogène (J=1)
    # =========================================================
    print(f"\n  >> Homogeneous SGCP (J=1)  — mala_step={step_sgcp}")
    domain_poly  = shapely_box(XB[0], YB[0], XB[1], YB[1])
    zones_single = [domain_poly]

    results_sgcp = launch_chains(
        zones_single, x_arr, y_arr, t_arr, T,
        step_sgcp, mu_star_for_workers,
    )
    out_sgcp = results_sgcp[k_ref].posterior_intensity(
        nx=NX_POST, ny=NY_POST, burn_in=BURN_IN, cmap=cmap_intensities,
        mu_star_func=mu_star_for_workers,
        savefigure=savefigure, savefigure_Emu=savefigure,
        title_savefig=f"ssgc/experiment_1/exp1_intensity_SGCP_{profile_name}{setting_name}",
        title_savefig_Emu=f"ssgc/experiment_1/exp1_Emu_SGCP_{profile_name}{setting_name}",
        n_mc=POSTERIOR_N_MC,
    )
    records.append({
        "profile": profile_name, "setting": setting_name,
        "model": "Homogeneous SGCP", "mala_step": step_sgcp,
        "rmse": out_sgcp["rmse"], "mae": out_sgcp["mae"],
        "crps": out_sgcp["crps"],
    })

    # =========================================================
    # Modèle 3 — KDE (pas de posterior)
    # =========================================================
    print(f"\n  >> KDE")
    mu_kde, GX, GY = compute_kde_intensity(x_arr, y_arr, T)
    grid_pts = np.column_stack([GX.ravel(), GY.ravel()])
    mu_star_grid = mu_star_for_workers(grid_pts[:, 0], grid_pts[:, 1])
    kde_metrics = compute_kde_metrics(mu_kde, mu_star_grid)
    records.append({
        "profile": profile_name, "setting": setting_name,
        "model": "KDE", "mala_step": None,
        **kde_metrics,
    })
    plot_kde_vs_true(
        mu_star_grid, mu_kde, GX, GY,
        profile_name, setting_name,
        cmap=cmap_intensities, savefigure=savefigure,
    )

    print_metrics_table(records)
    return records


# ========
# Main
# ========
if __name__ == "__main__":

    SAVEFIGURE  = False
    all_records = []

    configs = [
        (PROFILE_1, "A"),
        (PROFILE_1, "B"),
        (PROFILE_2, "A"),
        (PROFILE_2, "B"),
        (PROFILE_3, "A"),
        (PROFILE_3, "B"),
    ]

    for profile, setting in configs:
        records = run_exp1_config(profile, setting, savefigure=SAVEFIGURE)
        all_records.extend(records)
        if CLOSE_FIGURES:
            plt.close("all")
            gc.collect()

    print("\n" + "=" * 70)
    print("  RÉCAPITULATIF GLOBAL — Experiment 1")
    print("=" * 70)
    print_metrics_table(all_records)
    print("Experiment 1 terminé !!!")

# %%
