#%%
"""
Experiment 3 — Joint iSGCP vs Zone-wise independent SGCPs
Profile 1 x Settings A, B, C

On l'utilise directement pour le joint iSGCP.
Pour le zone-wise, on instancie un sampler par zone et on appelle
plot_posterior_intensity zone par zone, puis on stitch les résultats.
"""

# ==========
# Imports
# ==========
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import warnings
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

from gp.gibbs_sampler import iSGCP_GibbsSampler
from gp.data_generation import generate_voronoi_cells, simulate_process
from visualizations.plot import plot_voronoi_cells, plot_process_dashboard


# =============================================================================
# Paramètres globaux
# =============================================================================
X_BOUNDS = (0.0, 2.0)
Y_BOUNDS = (0.0, 2.0)
N_GERMS  = 6
RNG_SEED = 15
MUS_VORONOI = (10.0, 1.0, 2.0, 10.0, 8.0, 2.0)

NU_INIT           = [5.0, 0.2]
LAMBDA_NU         = 0.5
DELTA             = [1.5, 0.01]
JITTER            = 1e-5
BURN_IN           = 0.4
N_ITER            = 5000
THIN              = 3
MALA_STEP         = 0.09
MALA_STEP_J       = [0.11, 0.15, 0.16, 0.125, 0.125, 0.135, 0.18]
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

T_BY_SETTING = {"A": 30.0, "B": 25.0, "C": 15.0}
GRID_RES_BY_SETTING = {"A": 100,  "B": 200,  "C": 200}


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


def f_star_C(x, y):
    x_flat = np.atleast_1d(x).flatten()
    y_flat = np.atleast_1d(y).flatten()
    proj_step = (x_flat - 1.0) * np.cos(np.pi/5) + (y_flat - 1.0) * np.sin(np.pi/5)
    step  = 3.0 * (proj_step > 0).astype(float)
    proj_ridge = -(x_flat - 1.0) * np.sin(np.pi/4) + (y_flat - 1.0) * np.cos(np.pi/4)
    ridge = 2.5 * np.exp(-proj_ridge**2 / (2.0 * 0.15**2))
    return (step + ridge).reshape(np.shape(x))


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
              n_iter, thin, verbose, verbose_every, use_calib,
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
        use_calibration=use_calib,
        mu_star_func=mu_star_func,
        grid_nx=nx, grid_ny=ny,
    )
    print(f"  [Chain {k+1}] done (seed={chain_seed})")
    return results_k, list(sampler_k.nu)


def launch_chains(zones_raw_list, x_arr, y_arr, t_arr, T,
                  mala_step, mu_star_func,
                  Xb=None, Yb=None, seed_offset=0):
    Xb = Xb or X_BOUNDS
    Yb = Yb or Y_BOUNDS
    chain_outputs = Parallel(n_jobs=-1, prefer="processes")(
        delayed(run_chain)(
            k, SEED + seed_offset, zones_raw_list,
            x_arr, y_arr, t_arr, T,
            Xb, Yb, NU_INIT, LAMBDA_NU, DELTA, JITTER,
            mala_step, T0_NU, STEP_NU_INIT,
            N_ITER, THIN, VERBOSE, VERBOSE_EVERY, USE_CALIB,
            mu_star_func, NX, NY,
        )
        for k in range(N_CHAINS)
    )
    all_results, all_nu = zip(*chain_outputs)
    return list(all_results), list(all_nu)


def build_sampler(zones_raw_list, nu_hat, T, Xb=None, Yb=None, seed=SEED):
    Xb = Xb or X_BOUNDS
    Yb = Yb or Y_BOUNDS
    zones_prep = [prep(p) for p in zones_raw_list]
    Areas      = [(zp, 0.0) for zp in zones_prep]
    return iSGCP_GibbsSampler(
        X_bounds=Xb, Y_bounds=Yb, T=T,
        Areas=Areas, polygons=zones_raw_list,
        lambda_nu=LAMBDA_NU, nu=nu_hat,
        delta=DELTA, jitter=JITTER, rng_seed=seed,
    )


def get_nu_hat(all_results):
    burn   = int(BURN_IN * all_results[0]["nu"].shape[0])
    nu_all = np.concatenate([r["nu"][burn:] for r in all_results], axis=0)
    return nu_all.mean(axis=0).tolist()


# =============================================================================
# Fit joint iSGCP — utilise directement plot_posterior_intensity
# =============================================================================
def fit_and_eval_joint(x_arr, y_arr, t_arr, T, zones_raw_list,
                       mu_star_func, setting_name, savefigure, cmap_intensities="inferno"):
    print("\n  >> Joint iSGCP fitting")
    t0 = time.time()

    all_results, _ = launch_chains(
        zones_raw_list, x_arr, y_arr, t_arr, T,
        MALA_STEP, mu_star_func,
    )

    elapsed = time.time() - t0
    print(f"  >> Joint iSGCP : {elapsed:.1f}s")

    nu_hat  = get_nu_hat(all_results)
    sampler = build_sampler(zones_raw_list, nu_hat, T)

    # plot_posterior_intensity fait tout : figures + RMSE/MAE/CRPS/ECP
    k_ref = N_CHAINS // 2
    out = sampler.plot_posterior_intensity(
        x=x_arr, y=y_arr, t=t_arr,
        results=all_results[k_ref],
        nx=NX_POST, ny=NY_POST,
        burn_in=BURN_IN,
        cmap=cmap_intensities,
        mu_star_func=mu_star_func,
        savefigure=savefigure,
        savefigure_Emu=savefigure,
        title_savefig=f"exp3_intensity_joint_{setting_name}",
        title_savefig_Emu=f"exp3_Emu_joint_{setting_name}",
    )
    plt.close('all')

    return out, all_results, sampler, elapsed


# =============================================================================
# Fit zone-wise SGCP — un sampler par zone, stitch des résultats
# =============================================================================
def fit_and_eval_zonewise(x_arr, y_arr, t_arr, T, zones_raw_list,
                          mu_star_func, setting_name, savefigure, cmap_intensities="inferno"):
    """
    Pour chaque zone S_j :
      1. Sélectionne les observations dans S_j
      2. Lance N_CHAINS chaînes avec J=1 (polygone = S_j)
      3. Appelle plot_posterior_intensity sur la grille locale
    Puis stitch mu_hat, mu_hat_sims sur la grille globale et calcule
    les métriques globales via properscoring.
    """
    print("\n  >> Zone-wise SGCP fitting")
    t0 = time.time()

    J = len(zones_raw_list)
    zones_prep_full = [prep(z) for z in zones_raw_list]

    # Partition des observations par zone
    points_per_zone = [[] for _ in range(J)]
    for i in range(len(x_arr)):
        pt = ShapelyPoint(float(x_arr[i]), float(y_arr[i]))
        for j, pz in enumerate(zones_prep_full):
            if pz.covers(pt):
                points_per_zone[j].append(i)
                break

    # Grille globale pour le stitching final
    xmin, xmax = X_BOUNDS
    ymin, ymax = Y_BOUNDS
    interval = ot.Interval([xmin, ymin], [xmax, ymax])
    mesh_global = ot.IntervalMesher([NX_POST - 1, NY_POST - 1]).build(interval)
    XY_grid_global = mesh_global.getVertices()
    grid_xy = np.array(XY_grid_global)
    M = len(grid_xy)
    n_mc = 500   

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

        x_j   = x_arr[idx_j]
        y_j   = y_arr[idx_j]
        t_j   = t_arr[idx_j]
        poly  = zones_raw_list[j]
        pz    = zones_prep_full[j]
        bx, by, bx2, by2 = poly.bounds

        # Chaînes locales (J=1, domaine = zone j)
        all_results_j, _ = launch_chains(
            [poly], x_j, y_j, t_j, T,
            MALA_STEP, mu_star_func,
            Xb=(bx, bx2), Yb=(by, by2),
            seed_offset=1000 * j,
        )

        nu_hat_j  = get_nu_hat(all_results_j)
        sampler_j = build_sampler(
            [poly], nu_hat_j, T,
            Xb=(bx, bx2), Yb=(by, by2),
            seed=SEED + 1000 * j,
        )

        # plot_posterior_intensity sur la zone locale
        # mu_star_func est la vraie intensité globale — correcte aussi sur S_j
        k_ref = N_CHAINS // 2
        out_j = sampler_j.plot_posterior_intensity(
            x=x_j, y=y_j, t=t_j,
            results=all_results_j[k_ref],
            nx=NX_POST, ny=NY_POST,
            burn_in=BURN_IN,
            cmap=cmap_intensities,
            mu_star_func=mu_star_func,
            savefigure=savefigure,
            savefigure_Emu=savefigure,
            title_savefig=f"exp3_intensity_zw_zone{j}_{setting_name}",
            title_savefig_Emu=f"exp3_Emu_zw_zone{j}_{setting_name}",
        )
        plt.close('all')
        per_zone_metrics.append({
            "rmse": out_j["rmse"],
            "mae":  out_j["mae"],
            "crps": out_j.get("crps", None),
            "ecp":  out_j["ecp"],
        })

        # Stitch sur la grille globale — indices des points de grille dans S_j
        in_zone = np.array([
            pz.covers(ShapelyPoint(float(grid_xy[k, 0]), float(grid_xy[k, 1])))
            for k in range(M)
        ])
        if not in_zone.any():
            continue

        # Reconstruire mu_hat_sims sur les points de la zone via posterior_gp
        # (plot_posterior_intensity a calculé sur la grille locale,
        #  on reutilise posterior_gp sur les points globaux dans S_j)
        post_sum   = sampler_j.posterior_summary(all_results_j[k_ref], BURN_IN)
        eps_hat_j  = post_sum["eps_hat"]
        f_hat_j    = post_sum["f_data_hat"]
        sampler_j.nu = ot.Point(post_sum["nu_hat"])

        idx_local  = np.where(in_zone)[0]
        XY_local   = ot.Sample(grid_xy[idx_local].tolist())
        XY_obs_j   = ot.Sample([[x_j[i], y_j[i]] for i in range(len(x_j))])

        mu_post_l, Sigma_post_l = sampler_j.posterior_gp(
            XY_obs_j, ot.Point(list(f_hat_j)), 
            ot.Mesh(XY_local, []),   # mesh local pour posterior_gp
            eps_hat_j,
        )

        # posterior_gp attend un mesh — on passe par compute_kernel directement
        K_dd, K_gd, K_gg = sampler_j.compute_kernel(XY_obs_j, XY_local)
        K_dd_reg = ot.CovarianceMatrix(K_dd)
        for ii in range(len(x_j)):
            K_dd_reg[ii, ii] += JITTER
        K_inv  = K_dd_reg.inverse()
        f_pt   = ot.Point(list(f_hat_j))
        mu_g   = np.array(K_gd * (K_inv * f_pt)).flatten()
        Sig_g  = (np.array(K_gg)
                  - np.array(K_gd) @ np.array(K_inv) @ np.array(K_gd).T)
        Sig_g  = 0.5 * (Sig_g + Sig_g.T) + JITTER * np.eye(len(idx_local))
        std_g  = np.sqrt(np.diagonal(Sig_g))

        noise_l     = np.random.randn(len(idx_local), n_mc)
        f_sims_l    = mu_g[:, None] + std_g[:, None] * noise_l
        mu_tilde_l  = sampler_j.compute_mu_tilde(XY_local, eps=eps_hat_j)
        mu_sims_l   = mu_tilde_l[:, None] * (1.0 / (1.0 + np.exp(-f_sims_l)))

        mu_hat_zw[idx_local]       = mu_sims_l.mean(axis=1)
        mu_var_zw[idx_local]       = mu_sims_l.var(axis=1)
        mu_hat_sims_zw[idx_local]  = mu_sims_l

    elapsed = time.time() - t0
    print(f"  >> Zone-wise SGCP : {elapsed:.1f}s")

    # Métriques globales
    mu_star_global = mu_star_func(grid_xy[:, 0], grid_xy[:, 1])
    crps_zw  = float(ps.crps_ensemble(mu_star_global, mu_hat_sims_zw).mean())
    rmse_zw  = float(np.sqrt(np.mean((mu_hat_zw - mu_star_global) ** 2)))
    mae_zw   = float(np.mean(np.abs(mu_hat_zw - mu_star_global)))
    q_lo     = np.quantile(mu_hat_sims_zw, 0.025, axis=1)
    q_hi     = np.quantile(mu_hat_sims_zw, 0.975, axis=1)
    ecp_zw   = float(np.mean((mu_star_global >= q_lo) & (mu_star_global <= q_hi)))

    # Courbe ECP(alpha)
    alpha_levels = np.linspace(0.05, 0.95, 19)
    ecp_curve_zw = np.array([
        np.mean(
            (mu_star_global >= np.quantile(mu_hat_sims_zw, (1-a)/2, axis=1)) &
            (mu_star_global <= np.quantile(mu_hat_sims_zw, 1-(1-a)/2, axis=1))
        )
        for a in alpha_levels
    ])

    global_metrics = {
        "rmse": rmse_zw, "mae": mae_zw, "crps": crps_zw, "ecp_95": ecp_zw,
        "ecp_curve": ecp_curve_zw, "alpha_levels": alpha_levels,
        "mu_hat": mu_hat_zw, "mu_var": mu_var_zw,
        "mu_star": mu_star_global, "mu_hat_sims": mu_hat_sims_zw,
        "grid_xy": grid_xy, "mesh": mesh_global,
    }

    return global_metrics, per_zone_metrics, elapsed


# =============================================================================
# Visualisations comparatives (inchangées dans la logique)
# =============================================================================
def _save(fig, name):
    try:
        ROOT = Path(".").resolve()
        d    = ROOT / "visualizations" / "figures"
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / (name + ".pdf"), format="pdf", dpi=150, bbox_inches="tight")
        print(f"  Figure sauvegardée : {d / (name + '.pdf')}")
    except Exception as e:
        print(f"  Erreur sauvegarde : {e}")


def plot_intensity_comparison(joint_out, zw_metrics, setting_name,
                              cmap_intensities="inferno", savefigure=False):
    grid_xy  = zw_metrics["grid_xy"]
    GX = grid_xy[:, 0].reshape(NY_POST, NX_POST)
    GY = grid_xy[:, 1].reshape(NY_POST, NX_POST)

    mu_star  = zw_metrics["mu_star"].reshape(NY_POST, NX_POST)
    mu_joint = joint_out["mu_hat"].reshape(NY_POST, NX_POST)
    mu_zw    = zw_metrics["mu_hat"].reshape(NY_POST, NX_POST)
    err_j    = np.abs(mu_joint - mu_star) / (mu_star + JITTER)
    err_zw   = np.abs(mu_zw   - mu_star) / (mu_star + JITTER)

    vmax     = max(mu_star.max(), mu_joint.max(), mu_zw.max())
    vmax_err = max(err_j.max(), err_zw.max())

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    for ax, field, title, vmax_use, cmap in [
        (axes[0,0], mu_star,  r"Vraie intensité $\mu^\star(s)$",      vmax,     cmap_intensities),
        (axes[0,1], mu_joint, r"Joint iSGCP $\hat{\mu}(s)$",          vmax,     cmap_intensities),
        (axes[0,2], mu_zw,    r"Zone-wise SGCP $\hat{\mu}_{zw}(s)$",  vmax,     cmap_intensities),
        (axes[1,1], err_j,    r"Erreur relative — joint",              vmax_err, "magma"),
        (axes[1,2], err_zw,   r"Erreur relative — zone-wise",          vmax_err, "magma"),
    ]:
        im = ax.contourf(GX, GY, field, levels=30, cmap=cmap, vmin=0, vmax=vmax_use)
        plt.colorbar(im, ax=ax)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3, color="white", linewidth=0.5)

    axes[1, 0].set_visible(False)
    plt.suptitle(f"Setting {setting_name} — Joint vs Zone-wise", fontsize=13)
    plt.tight_layout()
    if savefigure:
        _save(fig, f"exp3_intensity_setting{setting_name}")
    plt.show()


def plot_variance_comparison(joint_out, zw_metrics, setting_name, cmap_intensities="Greys", savefigure=False):
    grid_xy  = zw_metrics["grid_xy"]
    GX = grid_xy[:, 0].reshape(NY_POST, NX_POST)
    GY = grid_xy[:, 1].reshape(NY_POST, NX_POST)

    # joint_out contient mu_hat_sims → variance
    var_joint = joint_out["mu_hat_sims"].var(axis=1).reshape(NY_POST, NX_POST) \
                if "mu_hat_sims" in joint_out else \
                np.zeros((NY_POST, NX_POST))
    var_zw    = zw_metrics["mu_var"].reshape(NY_POST, NX_POST)
    vmax      = max(var_joint.max(), var_zw.max())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, field, title in [
        (axes[0], var_joint, r"$\mathrm{Var}[\mu \mid \mathcal{D}]$ — joint"),
        (axes[1], var_zw,    r"$\mathrm{Var}[\mu \mid \mathcal{D}]$ — zone-wise"),
    ]:
        im = ax.contourf(GX, GY, field, levels=30, cmap=cmap_intensities, vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax)
        ax.set_title(title)
        ax.set_aspect("equal")
    plt.suptitle(f"Setting {setting_name} — Variance posterior", fontsize=13)
    plt.tight_layout()
    if savefigure:
        _save(fig, f"exp3_variance_setting{setting_name}")
    plt.show()


def plot_calibration_curves(joint_out, zw_metrics, setting_name, savefigure=False):
    # Courbe ECP pour joint : recalcul depuis mu_hat_sims + mu_star
    alpha_levels = zw_metrics["alpha_levels"]
    mu_star = zw_metrics["mu_star"]

    if "mu_hat_sims" in joint_out and joint_out["mu_hat_sims"] is not None:
        sims_j = joint_out["mu_hat_sims"]
        ecp_joint = np.array([
            np.mean(
                (mu_star >= np.quantile(sims_j, (1-a)/2,   axis=1)) &
                (mu_star <= np.quantile(sims_j, 1-(1-a)/2, axis=1))
            )
            for a in alpha_levels
        ])
    else:
        ecp_joint = np.full_like(alpha_levels, np.nan)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Diagonale")
    ax.plot(alpha_levels, ecp_joint, "o-", color="steelblue", label="Joint iSGCP")
    ax.plot(alpha_levels, zw_metrics["ecp_curve"], "s-", color="crimson",  label="Zone-wise SGCP")
    ax.set_xlabel(r"Niveau nominal $\alpha$")
    ax.set_ylabel(r"$\mathrm{ECP}(\alpha)$")
    ax.set_title(f"Setting {setting_name} — Courbes de calibration")
    ax.legend(); ax.grid(alpha=0.3); ax.set_xlim(0,1); ax.set_ylim(0,1)
    plt.tight_layout()
    if savefigure:
        _save(fig, f"exp3_calibration_setting{setting_name}")
    plt.show()


def print_summary(joint_out, zw_metrics, zw_per_zone,
                  time_joint, time_zw, setting_name):
    print(f"\n{'='*72}")
    print(f"  Setting {setting_name} — Métriques globales")
    print(f"{'='*72}")
    print(f"  {'Métrique':<12} {'Joint':>12} {'Zone-wise':>12} {'Δ (J-ZW)':>12}")
    print(f"  {'-'*52}")

    # Métriques joint extraites de plot_posterior_intensity
    m_joint = {
        "RMSE" : joint_out["rmse"],
        "MAE"  : joint_out["mae"],
        "CRPS" : joint_out.get("crps", None),
        "ECP95": joint_out["ecp"],
    }
    m_zw = {
        "RMSE" : zw_metrics["rmse"],
        "MAE"  : zw_metrics["mae"],
        "CRPS" : zw_metrics["crps"],
        "ECP95": zw_metrics["ecp_95"],
    }

    for k in ["RMSE", "MAE", "CRPS", "ECP95"]:
        vj = m_joint[k]
        vz = m_zw[k]
        if vj is None or vz is None:
            print(f"  {k:<12} {'--':>12} {'--':>12} {'--':>12}")
        else:
            print(f"  {k:<12} {vj:>12.4f} {vz:>12.4f} {vj-vz:>+12.4f}")

    print(f"\n  Temps (s) : joint={time_joint:.1f}  zone-wise={time_zw:.1f}")

    print(f"\n  Métriques par zone (zone-wise) :")
    print(f"  {'Zone':<6} {'RMSE':>10} {'MAE':>10} {'CRPS':>10} {'ECP95':>10}")
    for j, mj in enumerate(zw_per_zone):
        if mj is None:
            print(f"  {j:<6} {'--':>10}")
            continue
        crps_str = f"{mj['crps']:.4f}" if mj["crps"] is not None else "        --"
        print(f"  {j:<6} {mj['rmse']:>10.4f} {mj['mae']:>10.4f} "
              f"{crps_str:>10} {mj['ecp']:>10.4f}")
    print(f"{'='*72}\n")


# =================================
# Fonction principale par setting
# =================================
def run_setting_exp3(setting_name, f_star_func, cmap_voronoi="cividis",
                     cmap_intensities="inferno", savefigure=False):
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning) 
    warnings.filterwarnings("ignore", category=RuntimeWarning) 
    warnings.filterwarnings("ignore", category=UserWarning) 

    T = T_BY_SETTING[setting_name]
    grid_res = GRID_RES_BY_SETTING[setting_name] 

    print(f"\n{'#'*70}")
    print(f"  EXPERIMENT 3 — SETTING {setting_name}")
    print(f"{'#'*70}\n")

    cells, germs = generate_voronoi_cells(
        n_germs=N_GERMS, X_bounds=X_BOUNDS, Y_bounds=Y_BOUNDS, rng_seed=RNG_SEED,
    )

    plot_voronoi_cells(
        cells, germs,
        X_bounds=X_BOUNDS, Y_bounds=Y_BOUNDS,
        cmap_name=cmap_voronoi,
        title=f"Pavage de Voronoï — Setting {setting_name}",
        savefigure=savefigure,
        title_savefig=f"exp3_voronoi_{setting_name}",
    )

    sim_data, grids = simulate_process(
        X_bounds=X_BOUNDS, Y_bounds=Y_BOUNDS, T=T,
        polygons=cells, mus=MUS_VORONOI,
        f=f_star_func, grid_res=grid_res, rng_seed=RNG_SEED,
    )

    plot_process_dashboard(
        sim_data, grids,
        cmap=cmap_intensities,
        title=f"Processus spatial — Setting {setting_name}",
        savefigure=savefigure,
        title_savefig=f"exp3_dashboard_{setting_name}",
    )

    X_data         = sim_data["X"]
    N              = X_data.getSize()
    x_arr          = np.array([float(X_data[i, 0]) for i in range(N)])
    y_arr          = np.array([float(X_data[i, 1]) for i in range(N)])
    t_arr          = np.array([float(X_data[i, 2]) for i in range(N)])
    zones_raw_list = list(sim_data["zones"])
    mus_vec_list   = list(sim_data["mus_vec"])

    mu_star_for_workers = partial(
        mu_star_func_picklable,
        zones_raw=zones_raw_list,
        mus_vec=mus_vec_list,
        f_func=f_star_func,
    )

    # Joint iSGCP — plot_posterior_intensity fait tout
    joint_out, joint_results, joint_sampler, time_joint = fit_and_eval_joint(
        x_arr, y_arr, t_arr, T, zones_raw_list,
        mu_star_for_workers, setting_name, savefigure,
    )

    # Zone-wise SGCP — stitch + métriques globales
    zw_metrics, zw_per_zone, time_zw = fit_and_eval_zonewise(
        x_arr, y_arr, t_arr, T, zones_raw_list,
        mu_star_for_workers, setting_name, savefigure,
    )

    print_summary(joint_out, zw_metrics, zw_per_zone,
                  time_joint, time_zw, setting_name)

    plot_intensity_comparison(joint_out, zw_metrics, setting_name,
                          cmap_intensities=cmap_intensities, savefigure=savefigure)
    plot_variance_comparison(joint_out, zw_metrics, setting_name, savefigure=savefigure)
    plot_calibration_curves(joint_out, zw_metrics, setting_name, savefigure=savefigure)

    return {
        "setting"      : setting_name,
        "joint_out"    : joint_out,
        "zw_metrics"   : zw_metrics,
        "zw_per_zone"  : zw_per_zone,
        "time_joint"   : time_joint,
        "time_zw"      : time_zw,
        "joint_results": joint_results,
    }


# ======
# Main
# ======
if __name__ == "__main__":

    SAVEFIGURE = True

    out_A = run_setting_exp3("A", f_star_A, cmap_voronoi="cividis", cmap_intensities="inferno", savefigure=SAVEFIGURE)
    out_B = run_setting_exp3("B", f_star_B, cmap_voronoi="cividis", cmap_intensities="inferno", savefigure=SAVEFIGURE)
    out_C = run_setting_exp3("C", f_star_C, cmap_voronoi="cividis", cmap_intensities="inferno", savefigure=SAVEFIGURE)

    print("\nExperiment 3 terminé.")




# %%
