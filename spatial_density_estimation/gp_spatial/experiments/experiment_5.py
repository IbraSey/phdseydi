#%%
"""
Experiment 5 — Sensitivity to the prior hyperparameters (delta_0, delta_1)
Profile 1, Setting A

Grille 6×6 de (delta_0, delta_1), 36 configurations.
Chaque config : N_CHAINS chaînes × N_ITER itérations.
Métriques via plot_posterior_intensity (RMSE/MAE/CRPS/ECP).
Figures : heatmaps, profils marginaux, traces sélectionnées.
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
from shapely.geometry import Point as ShapelyPoint
from shapely.prepared import prep
from joblib import Parallel, delayed

from gp.gibbs_sampler import iSGCP_GibbsSampler
from gp.data_generation import generate_voronoi_cells, simulate_process
from visualizations.plot import plot_voronoi_cells, plot_process_dashboard


# =============================================================================
# Paramètres MCMC (référence)
# =============================================================================
NU_INIT            = [5.0, 0.2]
LAMBDA_NU          = 0.5
DELTA_REF          = [1.0, 0.01]      # (delta0*, delta1*) référence
JITTER             = 1e-5
BURN_IN            = 0.4
N_ITER             = 3000
THIN               = 3
MALA_STEP          = 0.095
LEARN_NU           = False
USE_CALIB          = True
T0_NU              = 50
STEP_NU_INIT       = 0.0009
VERBOSE            = True
VERBOSE_EVERY      = 500
SEED               = 42
NX, NY             = 30, 30
NX_POST, NY_POST   = 60, 60
N_CHAINS           = 5
XB, YB             = (0.0, 2.0), (0.0, 2.0)

# Données fixes : Profile 1, Setting A
N_GERMS = 6
RNG_SEED = 15
MUS_VOR = (10.0, 1.0, 2.0, 10.0, 8.0, 2.0)
T = 30.0
GRID_RES = 100

# Grille de sensibilité
DELTA0_GRID = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
DELTA1_GRID = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]

# Configurations représentatives pour les traces
REPRESENTATIVE_CONFIGS = {
    "Over-shrinkage":    (0.01, 0.01),
    "Independent zones": (5.0,  0.001),
    "Uniform shrinkage": (5.0,  1.0),
    "Reference":         (1.0,  0.01),
}


# =============================================================================
# Fonction latente Setting A
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


def run_chain(k, seed, zones_raw, x_arr, y_arr, t_arr, T_val,
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
        X_bounds=Xb, Y_bounds=Yb, T=T_val,
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
    return results_k, list(sampler_k.nu)


def launch_chains(zones_raw_list, x_arr, y_arr, t_arr, T_val,
                  mala_step, mu_star_func, delta):
    """Lance N_CHAINS avec un delta spécifique."""
    chain_outputs = Parallel(n_jobs=-1, prefer="processes")(
        delayed(run_chain)(
            k, SEED, zones_raw_list,
            x_arr, y_arr, t_arr, T_val,
            XB, YB, NU_INIT, LAMBDA_NU, delta, JITTER,
            mala_step, T0_NU, STEP_NU_INIT,
            N_ITER, THIN, VERBOSE, VERBOSE_EVERY,
            mu_star_func, NX, NY,
        )
        for k in range(N_CHAINS)
    )
    all_results, all_nu = zip(*chain_outputs)
    return list(all_results), list(all_nu)


def build_sampler(zones_raw_list, nu_hat, T_val, delta):
    zones_prep = [prep(p) for p in zones_raw_list]
    Areas      = [(zp, 0.0) for zp in zones_prep]
    return iSGCP_GibbsSampler(
        X_bounds=XB, Y_bounds=YB, T=T_val,
        Areas=Areas, polygons=zones_raw_list,
        lambda_nu=LAMBDA_NU, nu=nu_hat,
        delta=delta, jitter=JITTER, rng_seed=SEED,
    )


def get_nu_hat(all_results):
    burn   = int(BURN_IN * all_results[0]["nu"].shape[0])
    nu_all = np.concatenate([r["nu"][burn:] for r in all_results], axis=0)
    return nu_all.mean(axis=0).tolist()


# =============================================================================
# Sauvegarde figures
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


# =============================================================================
# Fit + eval pour un (delta0, delta1) donné
# =============================================================================
def fit_single_config(d0, d1, zones_raw_list, x_arr, y_arr, t_arr,
                      mu_star_func, cmap_intensities="inferno", savefigure=False, save_traces=False):
    """
    Lance l'inférence pour un couple (delta0, delta1).
    Retourne un dict avec rmse, mae, crps, ecp, et optionnellement
    les résultats bruts pour les traces.
    """
    delta = [d0, d1]
    tag   = f"d0={d0}_d1={d1}"
    print(f"    ({d0}, {d1})", end=" ", flush=True)

    all_results, _ = launch_chains(
        zones_raw_list, x_arr, y_arr, t_arr, T,
        MALA_STEP, mu_star_func, delta,
    )

    nu_hat  = get_nu_hat(all_results)
    sampler = build_sampler(zones_raw_list, nu_hat, T, delta)

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
        title_savefig=f"exp5_intensity_{tag}",
        title_savefig_Emu=f"exp5_Emu_{tag}",
    )

    entry = {
        "d0":   d0,
        "d1":   d1,
        "rmse": out["rmse"],
        "mae":  out["mae"],
        "crps": out.get("crps", None),
        "ecp":  out["ecp"],
    }

    # Conserver les résultats bruts pour les traces représentatives
    if save_traces:
        entry["all_results"] = all_results
        entry["sampler"]     = sampler
        entry["out"]         = out

    print(f"→ RMSE={out['rmse']:.4f}  MAE={out['mae']:.4f}  ECP={out['ecp']:.4f}")
    return entry


# =============================================================================
# Heatmaps
# =============================================================================
def plot_heatmaps(grid_results, savefigure=False):
    """Heatmaps de RMSE, MAE, ECP(0.95) sur la grille (delta0, delta1)."""
    n0 = len(DELTA0_GRID)
    n1 = len(DELTA1_GRID)

    rmse_mat = np.zeros((n1, n0))
    mae_mat  = np.zeros((n1, n0))
    ecp_mat  = np.zeros((n1, n0))

    for r in grid_results:
        i0 = DELTA0_GRID.index(r["d0"])
        i1 = DELTA1_GRID.index(r["d1"])
        rmse_mat[i1, i0] = r["rmse"]
        mae_mat[i1, i0]  = r["mae"]
        ecp_mat[i1, i0]  = r["ecp"] if r["ecp"] is not None else np.nan

    # Position de la référence
    i0_ref = DELTA0_GRID.index(DELTA_REF[0])
    i1_ref = DELTA1_GRID.index(DELTA_REF[1])

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for ax, mat, title, cmap in [
        (axes[0], rmse_mat, r"$\mathrm{RMSE}_\mu$",       "YlOrRd"),
        (axes[1], mae_mat,  r"$\mathrm{MAE}_\mu$",        "YlOrRd"),
        (axes[2], ecp_mat,  r"$\mathrm{ECP}_\mu(0.95)$",  "RdYlGn"),
    ]:
        im = ax.imshow(mat, origin="lower", aspect="auto", cmap=cmap)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xticks(range(n0))
        ax.set_xticklabels([str(v) for v in DELTA0_GRID], fontsize=9)
        ax.set_yticks(range(n1))
        ax.set_yticklabels([str(v) for v in DELTA1_GRID], fontsize=9)
        ax.set_xlabel(r"$\delta_0$")
        ax.set_ylabel(r"$\delta_1$")
        ax.set_title(title)

        # Marquer la référence
        ax.plot(i0_ref, i1_ref, marker="*", color="white", markersize=18,
                markeredgecolor="black", markeredgewidth=1.5)

        # Annoter les valeurs
        for i in range(n1):
            for j in range(n0):
                val = mat[i, j]
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=7, color="black")

    plt.suptitle(r"Experiment 5 — Heatmaps $(\delta_0, \delta_1)$", fontsize=13)
    plt.tight_layout()

    if savefigure:
        _save(fig, "exp5_heatmaps")
    plt.show()


# =============================================================================
# Profils marginaux de Delta_RMSE
# =============================================================================
def plot_marginal_profiles(grid_results, savefigure=False):
    """Delta_RMSE marginalisé sur delta0 et delta1."""
    n0 = len(DELTA0_GRID)
    n1 = len(DELTA1_GRID)

    rmse_mat = np.zeros((n1, n0))
    for r in grid_results:
        i0 = DELTA0_GRID.index(r["d0"])
        i1 = DELTA1_GRID.index(r["d1"])
        rmse_mat[i1, i0] = r["rmse"]

    # RMSE de référence
    rmse_ref = None
    for r in grid_results:
        if r["d0"] == DELTA_REF[0] and r["d1"] == DELTA_REF[1]:
            rmse_ref = r["rmse"]
            break

    if rmse_ref is None or rmse_ref == 0:
        print("  [marginal_profiles] Référence introuvable, skip.")
        return

    # Delta_RMSE = (RMSE - RMSE*) / RMSE*
    delta_rmse_mat = (rmse_mat - rmse_ref) / rmse_ref

    # Marginal sur delta0 (moyenne + IQR sur delta1)
    mean_d0   = delta_rmse_mat.mean(axis=0)
    q25_d0    = np.quantile(delta_rmse_mat, 0.25, axis=0)
    q75_d0    = np.quantile(delta_rmse_mat, 0.75, axis=0)

    # Marginal sur delta1 (moyenne + IQR sur delta0)
    mean_d1   = delta_rmse_mat.mean(axis=1)
    q25_d1    = np.quantile(delta_rmse_mat, 0.25, axis=1)
    q75_d1    = np.quantile(delta_rmse_mat, 0.75, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    x_pos = np.arange(n0)
    ax.fill_between(x_pos, q25_d0, q75_d0, alpha=0.25, color="steelblue")
    ax.plot(x_pos, mean_d0, "o-", color="steelblue", linewidth=2)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(v) for v in DELTA0_GRID])
    ax.set_xlabel(r"$\delta_0$")
    ax.set_ylabel(r"$\Delta_{\mathrm{RMSE}}$")
    ax.set_title(r"Marginal sur $\delta_0$ (moyenné sur $\delta_1$)")
    ax.grid(alpha=0.3)

    ax = axes[1]
    x_pos = np.arange(n1)
    ax.fill_between(x_pos, q25_d1, q75_d1, alpha=0.25, color="crimson")
    ax.plot(x_pos, mean_d1, "s-", color="crimson", linewidth=2)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(v) for v in DELTA1_GRID])
    ax.set_xlabel(r"$\delta_1$")
    ax.set_ylabel(r"$\Delta_{\mathrm{RMSE}}$")
    ax.set_title(r"Marginal sur $\delta_1$ (moyenné sur $\delta_0$)")
    ax.grid(alpha=0.3)

    plt.suptitle(r"Experiment 5 — Profils marginaux $\Delta_{\mathrm{RMSE}}$",
                 fontsize=13)
    plt.tight_layout()

    if savefigure:
        _save(fig, "exp5_marginal_profiles")
    plt.show()


# =============================================================================
# Traces eps pour les 4 configurations représentatives
# =============================================================================
def plot_representative_traces(grid_results, savefigure=False):
    """Traces eps et E_mu pour les 4 régimes représentatifs."""
    configs = {}
    for r in grid_results:
        key = (r["d0"], r["d1"])
        for name, ref_key in REPRESENTATIVE_CONFIGS.items():
            if key == ref_key and "all_results" in r:
                configs[name] = r

    if not configs:
        print("  [traces] Aucune configuration représentative trouvée.")
        return

    n_repr = len(configs)

    # --- Traces eps ---
    fig, axes = plt.subplots(n_repr, 1, figsize=(12, 3.5 * n_repr))
    if n_repr == 1:
        axes = [axes]

    for ax, (name, r) in zip(axes, configs.items()):
        results_k = r["all_results"][N_CHAINS // 2]
        eps_chain = np.asarray(results_k["eps"])
        thin      = results_k.get("thin", 1)
        iters     = np.arange(eps_chain.shape[0]) * thin
        J         = eps_chain.shape[1]
        for j in range(J):
            ax.plot(iters, eps_chain[:, j], linewidth=0.8,
                    label=rf"$\varepsilon_{j}$")
        ax.set_title(f"{name} — $(\\delta_0, \\delta_1) = ({r['d0']}, {r['d1']})$")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$\varepsilon_j$")
        ax.legend(ncol=J, fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)

    plt.suptitle(r"Experiment 5 — Traces $\varepsilon_j$", fontsize=13)
    plt.tight_layout()
    if savefigure:
        _save(fig, "exp5_traces_eps")
    plt.show()

    # --- Traces E_mu ---
    fig, axes = plt.subplots(n_repr, 1, figsize=(12, 3 * n_repr))
    if n_repr == 1:
        axes = [axes]

    for ax, (name, r) in zip(axes, configs.items()):
        results_k = r["all_results"][N_CHAINS // 2]
        E_mu = results_k["E_mu"]
        mask = ~np.isnan(E_mu)
        if mask.any():
            ax.plot(np.where(mask)[0], E_mu[mask], linewidth=0.8, color="steelblue")
        ax.set_title(f"{name} — $(\\delta_0, \\delta_1) = ({r['d0']}, {r['d1']})$")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$\mathcal{E}_\mu^{(t)}$")
        ax.grid(alpha=0.3)

    plt.suptitle(r"Experiment 5 — Traces $\mathcal{E}_\mu^{(t)}$", fontsize=13)
    plt.tight_layout()
    if savefigure:
        _save(fig, "exp5_traces_Emu")
    plt.show()


# =============================================================================
# Tableau des 4 configs représentatives
# =============================================================================
def print_representative_table(grid_results):
    rmse_ref = None
    for r in grid_results:
        if r["d0"] == DELTA_REF[0] and r["d1"] == DELTA_REF[1]:
            rmse_ref = r["rmse"]
            break

    print(f"\n{'='*85}")
    print(f"  Experiment 5 — Configurations représentatives")
    print(f"{'='*85}")
    print(f"  {'Régime':<22} {'(δ0, δ1)':<16}"
          f" {'RMSE':>8} {'MAE':>8} {'CRPS':>8} {'ECP95':>8} {'ΔRMSE':>10}")
    print(f"  {'-'*80}")

    for name, (d0, d1) in REPRESENTATIVE_CONFIGS.items():
        for r in grid_results:
            if r["d0"] == d0 and r["d1"] == d1:
                crps_str = f"{r['crps']:.4f}" if r["crps"] is not None else "      --"
                ecp_str  = f"{r['ecp']:.4f}"  if r["ecp"]  is not None else "      --"
                if rmse_ref and rmse_ref > 0 and name != "Reference":
                    delta_rmse = (r["rmse"] - rmse_ref) / rmse_ref
                    delta_str  = f"{delta_rmse:>+10.2%}"
                else:
                    delta_str = "        --"
                star = "*" if name == "Reference" else " "
                print(f"  {name + star:<22} ({d0}, {d1}){'':<6}"
                      f" {r['rmse']:>8.4f} {r['mae']:>8.4f} {crps_str:>8}"
                      f" {ecp_str:>8} {delta_str:>10}")
                break

    print(f"{'='*85}\n")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":

    SAVEFIGURE = True

    print(f"\n{'#'*70}")
    print(f"  EXPERIMENT 5 — Sensitivity (delta_0, delta_1)")
    print(f"{'#'*70}\n")

    # --- Génération des données (fixées pour tout l'experiment) ---
    cells, germs = generate_voronoi_cells(
        n_germs=N_GERMS, X_bounds=XB, Y_bounds=YB, rng_seed=RNG_SEED,
    )

    plot_voronoi_cells(
        cells, germs,
        X_bounds=XB, Y_bounds=YB,
        cmap_name="cividis",
        title="Exp5 — Pavage de Voronoï (Profile 1)",
        savefigure=SAVEFIGURE,
        title_savefig="exp5_voronoi",
    )

    sim_data, grids = simulate_process(
        X_bounds=XB, Y_bounds=YB, T=T,
        polygons=cells, mus=MUS_VOR,
        f=f_star_A, grid_res=GRID_RES, rng_seed=15,
    )

    plot_process_dashboard(
        sim_data, grids,
        title="Exp5 — Données (Profile 1, Setting A)",
        savefigure=SAVEFIGURE,
        title_savefig="exp5_dashboard",
    )

    X_data         = sim_data["X"]
    N              = X_data.getSize()
    x_arr          = np.array([float(X_data[i, 0]) for i in range(N)])
    y_arr          = np.array([float(X_data[i, 1]) for i in range(N)])
    t_arr          = np.array([float(X_data[i, 2]) for i in range(N)])
    zones_raw_list = list(sim_data["zones"])
    mus_vec_list   = list(sim_data["mus_vec"])

    mu_star_func = partial(
        mu_star_func_picklable,
        zones_raw=zones_raw_list,
        mus_vec=mus_vec_list,
        f_func=f_star_A,
    )

    # --- Boucle sur la grille (delta0, delta1) ---
    n_total = len(DELTA0_GRID) * len(DELTA1_GRID)
    grid_results = []
    count = 0

    for d0 in DELTA0_GRID:
        for d1 in DELTA1_GRID:
            count += 1
            print(f"\n  [{count}/{n_total}]", end=" ")

            # Sauvegarder les traces pour les 4 configs représentatives
            save_traces = (d0, d1) in REPRESENTATIVE_CONFIGS.values()

            entry = fit_single_config(
                d0, d1,
                zones_raw_list, x_arr, y_arr, t_arr,
                mu_star_func,
                savefigure=False,          # pas de figure par config
                save_traces=save_traces,
            )
            grid_results.append(entry)

    # --- Visualisations ---
    print(f"\n{'='*70}")
    print(f"  Grille terminée — {n_total} configurations")
    print(f"{'='*70}")

    plot_heatmaps(grid_results, savefigure=SAVEFIGURE)
    plot_marginal_profiles(grid_results, savefigure=SAVEFIGURE)
    plot_representative_traces(grid_results, savefigure=SAVEFIGURE)
    print_representative_table(grid_results)

    # --- Figures d'intensité pour les 4 configs représentatives ---
    print("\n  >> Figures d'intensité pour les configurations représentatives")
    for name, (d0, d1) in REPRESENTATIVE_CONFIGS.items():
        for r in grid_results:
            if r["d0"] == d0 and r["d1"] == d1 and "sampler" in r:
                k_ref = N_CHAINS // 2
                r["sampler"].plot_posterior_intensity(
                    x=x_arr, y=y_arr, t=t_arr,
                    results=r["all_results"][k_ref],
                    nx=NX_POST, ny=NY_POST,
                    burn_in=BURN_IN,
                    cmap="inferno",
                    mu_star_func=mu_star_func,
                    savefigure=SAVEFIGURE,
                    savefigure_Emu=SAVEFIGURE,
                    title_savefig=f"exp5_intensity_{name.replace(' ', '_')}",
                    title_savefig_Emu=f"exp5_Emu_{name.replace(' ', '_')}",
                )
                break

    print("\nExperiment 5 terminé.")




    #%%




