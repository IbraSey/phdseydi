# %% 
"""
Experiment 2 — SSGC vs Homogeneous SGCP vs KDE
Profiles 1, 2, 3 x Settings A, B
"""

# =============================================================================
# Imports
# =============================================================================
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)   # sklearn GP
warnings.filterwarnings("ignore", category=RuntimeWarning)        # numpy overflow/underflow
warnings.filterwarnings("ignore", category=UserWarning)           # matplotlib, openturns
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
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


# =========================
# Paramètres MCMC communs
# =========================
NU_INIT            = [5.0, 0.2]
LAMBDA_NU          = 0.5
DELTA              = [1.5, 0.01]
JITTER             = 1e-5
BURN_IN            = 0.4
N_ITER             = 5000
THIN               = 3
LEARN_NU           = False
USE_CALIB          = True
T0_NU              = 50
MALA_STEP_SSGC     = 0.095      # Taux d'acceptation trop bas pour dataset 3
MALA_STEP_SGCP     = 0.07
STEP_NU_INIT       = 0.0009
VERBOSE            = True
VERBOSE_EVERY      = 500
SEED               = 42
NX, NY             = 30, 30
NX_POST, NY_POST   = 60, 60
XB, YB             = (0.0, 2.0), (0.0, 2.0)
N_CHAINS           = 5


### Profils de zones

# Profile 1 : J=6, contraste élevé (ratio 10)
PROFILE_1 = {
    "name": "1",
    "n_germs": 6,
    "rng_seed_voronoi": 15,
    "mus": (10.0, 1.0, 2.0, 10.0, 8.0, 2.0),
    "J": 6,
}

# Profile 2 : J=5, faible contraste (ratio ~ 2)
PROFILE_2 = {
    "name": "2",
    "n_germs": 5,
    "rng_seed_voronoi": 15,
    "mus": (3.5, 2.0, 4.0, 3.0, 2.5),
    "J": 5,
}

# Profile 3 : J=7, deux zones dominantes, background sparse (ratio 20)
PROFILE_3 = {
    "name": "3",
    "n_germs": 7,
    "rng_seed_voronoi": 15,
    "mus": (20.0, 1.0, 1.0, 1.0, 1.0, 1.0, 20.0),
    "J": 7,
}

T_BY_PROFILE = {
    "1": {"A": 30.0, "B": 25.0},
    "2": {"A": 60.0, "B": 40.0},
    "3": {"A": 20.0, "B": 25.0},
}
GRID_RES_BY_SETTING = {"A": 100, "B": 300}


# ====================
# Fonctions latentes
# ====================
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
# Helpers picklables
# =============================================================================
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
 
    from shapely.prepared import prep
    import openturns as ot
 
    chain_seed = seed + k
    zones_prep_local = [prep(p) for p in zones_raw]
    Areas_local = [(zp, 0.0) for zp in zones_prep_local]
 
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
        mala_step=mala_step,
        learn_nu=LEARN_NU,
        t0_nu=t0_nu,
        step_nu_init=step_nu_init,
        n_iter=n_iter,
        thin=thin,
        verbose=verbose,
        verbose_every=verbose_every,
        use_calibration=USE_CALIB,
        mu_star_func=mu_star_func,
        grid_nx=nx,
        grid_ny=ny,
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
 
 
# =============================================================================
# Reconstruction sampler post-run (pour appeler plot_posterior_intensity)
# =============================================================================
def build_sampler(zones_raw_list, nu_hat, T):
    zones_prep = [prep(p) for p in zones_raw_list]
    Areas = [(zp, 0.0) for zp in zones_prep]
    return iSGCP_GibbsSampler(
        X_bounds=XB, Y_bounds=YB, T=T,
        Areas=Areas, polygons=zones_raw_list,
        lambda_nu=LAMBDA_NU, nu=nu_hat,
        delta=DELTA, jitter=JITTER, rng_seed=SEED,
    )
 
 
def get_nu_hat(all_results, all_nu_finals):
    """Moyenne de nu sur toutes les chaînes post burn-in."""
    burn = int(BURN_IN * all_results[0]["nu"].shape[0])
    nu_all = np.concatenate([r["nu"][burn:] for r in all_results], axis=0)
    return nu_all.mean(axis=0).tolist()
 
 
# =============================================================================
# KDE reference — seule partie qui ne peut pas utiliser la classe
# =============================================================================
def compute_kde_intensity(x_arr, y_arr, T, nx=NX_POST, ny=NY_POST):
    """
    mu_hat_KDE(x,y) = N/T * p_hat_h(x,y), bandwidth Silverman.
    Retourne (mu_kde_flat, GX, GY).
    """
    N = len(x_arr)
    sample = ot.Sample([[float(x_arr[i]), float(y_arr[i])] for i in range(N)])
    ks = ot.KernelSmoothing()
    kde = ks.build(sample)

    gx = np.linspace(XB[0], XB[1], nx)
    gy = np.linspace(YB[0], YB[1], ny)
    GX, GY = np.meshgrid(gx, gy)
    grid_pts = ot.Sample(np.column_stack([GX.ravel(), GY.ravel()]).tolist())

    p_hat = np.array(kde.computePDF(grid_pts)).flatten()
    mu_kde = (N / T) * p_hat

    return mu_kde, GX, GY


def compute_kde_metrics(mu_kde, mu_star_grid):
    """RMSE, MAE pour le KDE. Pas d'ECP ni de CRPS (pas de posterior)."""
    rmse = float(np.sqrt(np.mean((mu_kde - mu_star_grid) ** 2)))
    mae = float(np.mean(np.abs(mu_kde - mu_star_grid)))
    return {"rmse": rmse, "mae": mae, "crps": None, "ecp": None}


# =============================================================================
# Figure comparative KDE
# =============================================================================
def plot_kde_vs_true(mu_star_grid, mu_kde, GX, GY,
                     profile_name, setting_name, cmap="inferno", savefigure=False):
    ny_r, nx_r = GX.shape

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, field, title in zip(
        axes,
        [mu_star_grid, mu_kde],
        [r"Vraie intensité $\mu^\star(s)$",
         r"KDE $\hat{\mu}_{\mathrm{KDE}}(s)$"],
    ):
        im = ax.contourf(GX, GY, field.reshape(ny_r, nx_r),
                         levels=30, cmap=cmap)
        plt.colorbar(im, ax=ax)
        ax.set_title(title)
        ax.set_xlim(XB); ax.set_ylim(YB)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3, color="white", linewidth=0.5)

    plt.suptitle(
        f"Experiment 2 — Profile {profile_name}, Setting {setting_name} — KDE",
        fontsize=12,
    )
    plt.tight_layout()

    if savefigure :
        _save_figure(fig, f"exp2_true_intensity_{profile_name}{setting_name}")
    plt.show()

def _save_figure(fig, filename):
    try:
        try:
            ROOT = Path(__file__).resolve().parent.parent
        except NameError:
            ROOT = Path(".").resolve()
        FIGURES_DIR = ROOT / "visualizations" / "figures"
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / (filename + ".pdf")
        fig.savefig(path, format="pdf", dpi=150, bbox_inches="tight")
        print(f"  Figure sauvegardée : {path}")
    except Exception as e:
        print(f"  Erreur sauvegarde : {e}")


# =======================
# Tableau récapitulatif 
# =======================
def print_metrics_table(records):
    print(f"\n{'='*80}")
    print(f"  Experiment 2 — Métriques quantitatives")
    print(f"{'='*80}")
    print(f"  {'Profile':<9} {'Setting':<9} {'Modèle':<22}"
          f" {'RMSE':>8} {'MAE':>8} {'CRPS':>8} {'ECP(0.95)':>10}")
    print(f"  {'-'*76}")
    for r in records:
        ecp_str  = f"{r['ecp']:.4f}"  if r["ecp"]  is not None else "      --"
        crps_str = f"{r['crps']:.4f}" if r["crps"] is not None else "      --"
        print(f"  {r['profile']:<9} {r['setting']:<9} {r['model']:<22}"
              f" {r['rmse']:>8.4f} {r['mae']:>8.4f} {crps_str:>8} {ecp_str:>10}")
    print(f"{'='*80}\n")


# ===========================================
# Fonction principale par (profil, setting)
# ===========================================
def run_exp2_config(profile, setting_name, cmap_voronoi="cividis",
                    cmap_intensities="inferno", savefigure=False) :
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning) 
    warnings.filterwarnings("ignore", category=RuntimeWarning) 
    warnings.filterwarnings("ignore", category=UserWarning) 

    f_star_func = F_STAR[setting_name]
    T = T_BY_PROFILE[profile["name"]][setting_name]
    grid_res = GRID_RES_BY_SETTING[setting_name]

    profile_name = profile["name"]
    mus_voronoi  = profile["mus"]

    print(f"\n{'#'*70}")
    print(f"  EXP2 — Profile {profile_name}, Setting {setting_name}")
    print(f"{'#'*70}")

    # --- Génération pavage ---
    cells, germs = generate_voronoi_cells(
        n_germs=profile["n_germs"],
        X_bounds=XB, Y_bounds=YB,
        rng_seed=profile["rng_seed_voronoi"],
    )

    plot_voronoi_cells(
        cells, germs,
        X_bounds=XB, Y_bounds=YB,
        cmap_name=cmap_voronoi,
        title=f"Pavage de Voronoï — Profile {profile_name}, Setting {setting_name}",
        savefigure=savefigure,
        title_savefig=f"exp2_voronoi_{profile_name}{setting_name}",
    )

    # --- Génération données ---
    sim_data, grids = simulate_process(
        X_bounds=XB, Y_bounds=YB, T=T,
        polygons=cells, mus=mus_voronoi,
        f=f_star_func, grid_res=grid_res, rng_seed=15,
    )

    plot_process_dashboard(
        sim_data, grids, cmap=cmap_intensities,
        title=f"Processus spatial — Profile {profile_name}, Setting {setting_name}",
        savefigure=savefigure,
        title_savefig=f"exp2_dashboard_{profile_name}{setting_name}",
    )

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

    records = []
    k_ref = N_CHAINS // 2

    # ===========================
    # Modèle 1 — iSGCP (J zones)
    # ===========================
    print(f"\n  >> iSGCP (J={len(zones_raw_list)}) fitting")
    results_isgcp, nu_isgcp = launch_chains(
        zones_raw_list, x_arr, y_arr, t_arr, T,
        MALA_STEP_SSGC, mu_star_for_workers,
    )

    nu_hat_isgcp  = get_nu_hat(results_isgcp, nu_isgcp)
    sampler_isgcp = build_sampler(zones_raw_list, nu_hat_isgcp, T)

    # plot_posterior_intensity fait tout : figures + RMSE/MAE/CRPS/ECP
    out_isgcp = sampler_isgcp.plot_posterior_intensity(
        x=x_arr, y=y_arr, t=t_arr,
        results=results_isgcp[k_ref],
        nx=NX_POST, ny=NY_POST,
        burn_in=BURN_IN,
        cmap=cmap_intensities,
        mu_star_func=mu_star_for_workers,
        savefigure=savefigure,
        savefigure_Emu=savefigure,
        title_savefig=f"exp2_intensity_iSGCP_{profile_name}{setting_name}",
        title_savefig_Emu=f"exp2_Emu_iSGCP_{profile_name}{setting_name}",
    )
    records.append({
        "profile": profile_name, "setting": setting_name,
        "model": "iSGCP",
        "rmse": out_isgcp["rmse"],
        "mae": out_isgcp["mae"],
        "crps": out_isgcp["crps"],
        "ecp": out_isgcp["ecp"],
    })

    # ===============================
    # Modèle 2 — SGCP homogène (J=1)
    # ===============================
    print(f"\n  >> Homogeneous SGCP (J=1) fitting")

    domain_poly = shapely_box(XB[0], YB[0], XB[1], YB[1])
    zones_single = [domain_poly]
    mus_single = [float(np.mean(mus_voronoi))]

    results_sgcp, nu_sgcp = launch_chains(
        zones_single, x_arr, y_arr, t_arr, T,
        MALA_STEP_SGCP, mu_star_for_workers,
    )

    nu_hat_sgcp = get_nu_hat(results_sgcp, nu_sgcp)
    sampler_sgcp = build_sampler(zones_single, nu_hat_sgcp, T)

    # mu_star_func = mu_star_for_workers (la VRAIE intensité multi-zones)
    out_sgcp = sampler_sgcp.plot_posterior_intensity(
        x=x_arr, y=y_arr, t=t_arr,
        results=results_sgcp[k_ref],
        nx=NX_POST, ny=NY_POST,
        burn_in=BURN_IN,
        cmap=cmap_intensities,
        mu_star_func=mu_star_for_workers,
        savefigure=savefigure,
        savefigure_Emu=savefigure,
        title_savefig=f"exp2_intensity_SGCP_{profile_name}{setting_name}",
        title_savefig_Emu=f"exp2_Emu_SGCP_{profile_name}{setting_name}",
    )
    records.append({
        "profile": profile_name, "setting": setting_name,
        "model": "Homogeneous SGCP",
        "rmse": out_sgcp["rmse"],
        "mae":  out_sgcp["mae"],
        "crps": out_sgcp["crps"],
        "ecp":  out_sgcp["ecp"],
    })

    # ============================================================
    # Modèle 3 — KDE (pas de posterior -> pas d'ECP, pas de CRPS)
    # ============================================================
    print(f"\n  >> KDE fitting")
    mu_kde, GX, GY = compute_kde_intensity(x_arr, y_arr, T)

    grid_pts = np.column_stack([GX.ravel(), GY.ravel()])
    mu_star_grid = mu_star_for_workers(grid_pts[:, 0], grid_pts[:, 1])

    kde_metrics = compute_kde_metrics(mu_kde, mu_star_grid)
    records.append({
        "profile": profile_name, "setting": setting_name,
        "model": "KDE",
        **kde_metrics,
    })

    plot_kde_vs_true(
        mu_star_grid, mu_kde, GX, GY,
        profile_name, setting_name,
        cmap=cmap_intensities, savefigure=savefigure,
    )

    print_metrics_table(records)
    return records


# ======
# Main
# ======
if __name__ == "__main__":
 
    SAVEFIGURE  = True
    all_records = []
 
    configs = [
        (PROFILE_1, "A"),
        (PROFILE_1, "B"),
        (PROFILE_2, "A"),
        (PROFILE_2, "B"),
        (PROFILE_3, "A"),
        (PROFILE_3, "B"),
    ]
 
    for profile, setting in configs :
        records = run_exp2_config(profile, setting, savefigure=SAVEFIGURE)
        all_records.extend(records)
 
    print("\n" + "=" * 70)
    print("  RÉCAPITULATIF GLOBAL — Experiment 2")
    print("=" * 70)
    print_metrics_table(all_records)
 
    print("Experiment 2 terminé.")



# %%



