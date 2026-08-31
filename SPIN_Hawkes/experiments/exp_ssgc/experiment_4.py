# %%
"""
Experiment 4 — Sensitivity to the prior hyperparameters (delta_0, delta_1)
Profile 1, Setting A

Grille 6×6 de (delta_0, delta_1), 36 configurations.
Chaque config : N_CHAINS chaînes × N_ITER itérations.
Métriques via GibbsResults.posterior_intensity (RMSE/MAE/CRPS).
Figures : heatmaps, profils marginaux, traces sélectionnées.
"""

# =============================================================================
# Imports
# =============================================================================
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import openturns as ot
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.special import expit
from shapely.geometry import Point as ShapelyPoint
from shapely.prepared import prep
from tqdm.auto import tqdm

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT))

from package import (
    GPParameters,
    GibbsConfig,
    SSGCModel,
    generate_voronoi_cells,
    simulate_spatial_process,
)
from visualization import plot_process_dashboard, plot_voronoi_cells, save_figure


# =============================================================================
# Paramètres MCMC
# =============================================================================
NU_INIT       = [5.0, 0.2]
LAMBDA_NU     = 0.5
DELTA_REF     = [1.0, 0.01]     # (delta0*, delta1*) référence
JITTER        = 1e-5
BURN_IN       = 0.4
N_ITER        = 200
THIN          = 5
LEARN_NU      = False
USE_CALIB     = True
T0_NU         = 50
STEP_NU_INIT  = 0.0009
VERBOSE       = True
VERBOSE_EVERY = 50
SEED          = 42
NX_POST, NY_POST = 60, 60
N_CHAINS      = 2
XB, YB        = (0.0, 2.0), (0.0, 2.0)
GP_BACKEND   = "sparse"
COMPUTE_EMU  = False
POSTERIOR_N_MC = 500

# Données fixes : Profile 1, Setting A
N_GERMS  = 6
RNG_SEED = 15
MUS_VOR  = (10.0, 1.0, 2.0, 10.0, 8.0, 2.0)
T        = 180.7
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
# MALA step par (delta0, delta1)
#
# Les valeurs ci-dessous ont été recalibrées avec le backend sparse GP, environ
# 1000 événements simulés et des runs courts de 100--200 itérations.
# La table reste explicite pour pouvoir ajuster un couple (delta0, delta1) sans
# modifier une règle cachée.
# =============================================================================
MALA_STEP = {
    # Calibrated on sparse-GP Gibbs runs for about 1000 events.
    # Keys are kept explicit so each grid point can be adjusted by hand.
    (0.01, 0.001): 0.080,
    (0.01, 0.010): 0.080,
    (0.01, 0.050): 0.080,
    (0.01, 0.100): 0.080,
    (0.01, 0.500): 0.080,
    (0.01, 1.000): 0.080,
    (0.10, 0.001): 0.070,
    (0.10, 0.010): 0.070,
    (0.10, 0.050): 0.070,
    (0.10, 0.100): 0.070,
    (0.10, 0.500): 0.070,
    (0.10, 1.000): 0.070,
    (0.50, 0.001): 0.060,
    (0.50, 0.010): 0.060,
    (0.50, 0.050): 0.060,
    (0.50, 0.100): 0.060,
    (0.50, 0.500): 0.060,
    (0.50, 1.000): 0.060,
    (1.00, 0.001): 0.065,
    (1.00, 0.010): 0.065,
    (1.00, 0.050): 0.065,
    (1.00, 0.100): 0.065,
    (1.00, 0.500): 0.065,
    (1.00, 1.000): 0.065,
    (2.00, 0.001): 0.065,
    (2.00, 0.010): 0.065,
    (2.00, 0.050): 0.065,
    (2.00, 0.100): 0.065,
    (2.00, 0.500): 0.065,
    (2.00, 1.000): 0.065,
    (5.00, 0.001): 0.065,
    (5.00, 0.010): 0.065,
    (5.00, 0.050): 0.065,
    (5.00, 0.100): 0.065,
    (5.00, 0.500): 0.065,
    (5.00, 1.000): 0.065,
}


def get_mala_step(d0, d1):
    """Retourne le MALA step pour la configuration (delta0, delta1).

    Parameters
    ----------
    d0 : float
        Valeur de delta0 (doit appartenir à DELTA0_GRID).
    d1 : float
        Valeur de delta1 (doit appartenir à DELTA1_GRID).

    Returns
    -------
    float
    """
    # Normalise les clés pour éviter les erreurs d'arrondi flottant
    key = (round(d0, 10), round(d1, 10))
    if key not in MALA_STEP:
        # Recherche de la clé la plus proche si arrondi flottant
        best = min(MALA_STEP.keys(),
                   key=lambda k: abs(k[0] - d0) + abs(k[1] - d1))
        if abs(best[0] - d0) < 1e-9 and abs(best[1] - d1) < 1e-9:
            return MALA_STEP[best]
        raise KeyError(
            f"Pas de MALA step défini pour (delta0={d0}, delta1={d1}). "
            f"Clés disponibles : {sorted(MALA_STEP.keys())}"
        )
    return MALA_STEP[key]


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
# Construction des modèles et chaînes
# =============================================================================
def make_reference_intensity(zones, mus, latent_field):
    prepared_zones = [prep(zone) for zone in zones]

    def reference_intensity(x, y):
        x_values, y_values = np.broadcast_arrays(
            np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        )
        flat_x = x_values.reshape(-1)
        flat_y = y_values.reshape(-1)
        baseline = np.zeros(flat_x.size, dtype=float)
        unassigned = np.ones(flat_x.size, dtype=bool)

        for intensity, zone in zip(mus, prepared_zones):
            indices = np.flatnonzero(unassigned)
            if indices.size == 0:
                break
            inside = np.fromiter(
                (
                    zone.covers(ShapelyPoint(flat_x[i], flat_y[i]))
                    for i in indices
                ),
                dtype=bool,
                count=indices.size,
            )
            selected = indices[inside]
            baseline[selected] = intensity
            unassigned[selected] = False

        values = baseline * expit(latent_field(flat_x, flat_y))
        return values.reshape(x_values.shape)

    return reference_intensity


def make_model(zones, duration, delta):
    return SSGCModel.from_polygons(
        polygons=zones,
        duration=duration,
        x_bounds=XB,
        y_bounds=YB,
        initial_log_intensities=0.0,
        gp_prior=GPParameters(
            variance=NU_INIT[0], length_scale=NU_INIT[1]
        ),
        eps_prior_variance=delta[0],
        eps_prior_length_scale=delta[1],
        nu_prior_rate=LAMBDA_NU,
        jitter=JITTER,
    )


def make_gibbs_config(mala_step):
    return GibbsConfig(
        n_iter=N_ITER,
        thin=THIN,
        mala_step=mala_step,
        learn_nu=LEARN_NU,
        use_calibration=USE_CALIB,
        verbose=VERBOSE,
        verbose_every=VERBOSE_EVERY,
        t0_nu=T0_NU,
        step_nu_init=STEP_NU_INIT,
        compute_emu=COMPUTE_EMU,
    )


def run_chain(
    chain_index,
    zones,
    catalog,
    duration,
    delta,
    mala_step,
    reference_intensity,
):
    chain_seed = SEED + chain_index
    model = make_model(zones, duration, delta)
    result = model.gibbs(
        catalog,
        config=make_gibbs_config(mala_step),
        rng_seed=chain_seed,
        reference_intensity=reference_intensity,
        gp_backend=GP_BACKEND,
    )
    print(f"  [Chain {chain_index + 1}] done (seed={chain_seed})")
    return result


def launch_chains(
    zones, catalog, duration, delta, mala_step, reference_intensity
):
    """Run independent chains sequentially; Gibbs results contain OT objects."""
    return [
        run_chain(
            chain_index,
            zones,
            catalog,
            duration,
            delta,
            mala_step,
            reference_intensity,
        )
        for chain_index in tqdm(
            range(N_CHAINS),
            desc="Gibbs chains",
            unit="chain",
            leave=False,
            dynamic_ncols=True,
        )
    ]


def reference_chain(results):
    return results[len(results) // 2]


# =============================================================================
# Sauvegarde figures
# =============================================================================
def _save(fig, name):
    path = save_figure(fig, f"ssgc/experiment_4/{name}")
    print(f"  Figure sauvegardée : {path}")


# =============================================================================
# Fit + eval pour un (delta0, delta1) donné
# =============================================================================
def fit_single_config(d0, d1, zones, catalog, reference_intensity,
                      cmap_intensities="inferno",
                      savefigure=False, save_traces=False):
    """Lance l'inférence pour un couple (delta0, delta1).

    Le MALA step est récupéré depuis le dict MALA_STEP via get_mala_step(d0, d1).

    Parameters
    ----------
    d0, d1 : float
        Hyperparamètres du prior Σ_ε.
    save_traces : bool
        Si True, conserve les GibbsResults des chaînes
        (utile pour les 4 configurations représentatives).
    """
    delta     = [d0, d1]
    mala_step = get_mala_step(d0, d1)
    tag       = f"d0={d0}_d1={d1}"
    tqdm.write(f"    ({d0}, {d1})  step={mala_step}")

    all_results = launch_chains(
        zones, catalog, T, delta, mala_step, reference_intensity,
    )
    out = reference_chain(all_results).posterior_intensity(
        nx=NX_POST, ny=NY_POST, burn_in=BURN_IN,
        cmap=cmap_intensities,
        mu_star_func=reference_intensity,
        savefigure=savefigure, savefigure_Emu=savefigure and COMPUTE_EMU,
        title_savefig=f"ssgc/experiment_4/exp4_intensity_{tag}",
        title_savefig_Emu=f"ssgc/experiment_4/exp4_Emu_{tag}",
        n_mc=POSTERIOR_N_MC,
    )

    tqdm.write(f"    RMSE={out['rmse']:.4f}  MAE={out['mae']:.4f}")

    entry = {
        "d0": d0, "d1": d1,
        "mala_step": mala_step,
        "rmse": out["rmse"], "mae": out["mae"],
        "crps": out.get("crps"),
    }
    if save_traces:
        entry["all_results"] = all_results
        entry["out"]         = out

    return entry


# =============================================================================
# Heatmaps
# =============================================================================
def plot_heatmaps(grid_results, savefigure=False):
    """Heatmaps de RMSE, MAE et MALA step sur la grille."""
    n0 = len(DELTA0_GRID)
    n1 = len(DELTA1_GRID)

    rmse_mat = np.zeros((n1, n0))
    mae_mat  = np.zeros((n1, n0))
    step_mat = np.zeros((n1, n0))

    for r in grid_results:
        i0 = DELTA0_GRID.index(r["d0"])
        i1 = DELTA1_GRID.index(r["d1"])
        rmse_mat[i1, i0] = r["rmse"]
        mae_mat[i1, i0]  = r["mae"]
        step_mat[i1, i0] = r["mala_step"]

    i0_ref = DELTA0_GRID.index(DELTA_REF[0])
    i1_ref = DELTA1_GRID.index(DELTA_REF[1])

    def _panel(ax, mat, title, cmap, annotate=True):
        im = ax.imshow(mat, origin="lower", aspect="auto", cmap=cmap)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xticks(range(n0))
        ax.set_xticklabels([str(v) for v in DELTA0_GRID], fontsize=9)
        ax.set_yticks(range(n1))
        ax.set_yticklabels([str(v) for v in DELTA1_GRID], fontsize=9)
        ax.set_xlabel(r"$\delta_0$")
        ax.set_ylabel(r"$\delta_1$")
        ax.set_title(title)
        ax.plot(i0_ref, i1_ref, marker="*", color="white", markersize=18,
                markeredgecolor="black", markeredgewidth=1.5)
        if annotate:
            for i in range(n1):
                for j in range(n0):
                    ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                            fontsize=7, color="black")

    # Panel 1 : métriques
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        layout="constrained",
    )
    _panel(axes[0], rmse_mat, r"$\mathrm{RMSE}_\mu$",      "YlOrRd")
    _panel(axes[1], mae_mat,  r"$\mathrm{MAE}_\mu$",       "YlOrRd")
    plt.suptitle(r"Experiment 4 — Heatmaps $($\delta_0, \delta_1)$", fontsize=13)
    if savefigure:
        save_figure(fig, "ssgc/experiment_4/exp4_heatmaps", figure_type="raster")
    plt.show()

    # Panel 2 : MALA step utilisé (diagnostic)
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    _panel(ax, step_mat, "MALA step par config", "Blues", annotate=True)
    plt.suptitle(r"Experiment 4 — MALA step $($\delta_0, \delta_1)$", fontsize=13)
    if savefigure:
        save_figure(fig, "ssgc/experiment_4/exp4_heatmap_mala_step", figure_type="raster")
    plt.show()


# =============================================================================
# Profils marginaux de Delta_RMSE
# =============================================================================
def plot_marginal_profiles(grid_results, savefigure=False,
                           delta0_color="steelblue", delta1_color="crimson"):
    n0 = len(DELTA0_GRID)
    n1 = len(DELTA1_GRID)

    rmse_mat = np.zeros((n1, n0))
    for r in grid_results:
        i0 = DELTA0_GRID.index(r["d0"])
        i1 = DELTA1_GRID.index(r["d1"])
        rmse_mat[i1, i0] = r["rmse"]

    rmse_ref = next(
        (r["rmse"] for r in grid_results
         if r["d0"] == DELTA_REF[0] and r["d1"] == DELTA_REF[1]),
        None,
    )
    if rmse_ref is None or rmse_ref == 0:
        print("  [marginal_profiles] Référence introuvable, skip.")
        return

    delta_rmse_mat = (rmse_mat - rmse_ref) / rmse_ref

    mean_d0 = delta_rmse_mat.mean(axis=0)
    q25_d0  = np.quantile(delta_rmse_mat, 0.25, axis=0)
    q75_d0  = np.quantile(delta_rmse_mat, 0.75, axis=0)

    mean_d1 = delta_rmse_mat.mean(axis=1)
    q25_d1  = np.quantile(delta_rmse_mat, 0.25, axis=1)
    q75_d1  = np.quantile(delta_rmse_mat, 0.75, axis=1)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
        layout="constrained",
    )

    ax = axes[0]
    x_pos = np.arange(n0)
    ax.fill_between(x_pos, q25_d0, q75_d0, alpha=0.25, color=delta0_color)
    ax.plot(x_pos, mean_d0, "o-", color=delta0_color, linewidth=2)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(v) for v in DELTA0_GRID])
    ax.set_xlabel(r"$\delta_0$")
    ax.set_ylabel(r"$\Delta_{\mathrm{RMSE}}$")
    ax.set_title(r"Marginal sur $\delta_0$ (moyenné sur $\delta_1$)")
    ax.grid(alpha=0.3)

    ax = axes[1]
    x_pos = np.arange(n1)
    ax.fill_between(x_pos, q25_d1, q75_d1, alpha=0.25, color=delta1_color)
    ax.plot(x_pos, mean_d1, "s-", color=delta1_color, linewidth=2)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(v) for v in DELTA1_GRID])
    ax.set_xlabel(r"$\delta_1$")
    ax.set_ylabel(r"$\Delta_{\mathrm{RMSE}}$")
    ax.set_title(r"Marginal sur $\delta_1$ (moyenné sur $\delta_0$)")
    ax.grid(alpha=0.3)

    plt.suptitle(r"Experiment 4 — Profils marginaux $\Delta_{\mathrm{RMSE}}$",
                 fontsize=13)
    if savefigure:
        _save(fig, "exp4_marginal_profiles")
    plt.show()


# =============================================================================
# Traces eps et E_mu pour les 4 configurations représentatives
# =============================================================================
def plot_representative_traces(grid_results, savefigure=False, emu_color="steelblue"):
    configs = {
        name: r
        for r in grid_results
        for name, ref_key in REPRESENTATIVE_CONFIGS.items()
        if (r["d0"], r["d1"]) == ref_key and "all_results" in r
    }
    if not configs:
        print("  [traces] Aucune configuration représentative trouvée.")
        return

    n_repr = len(configs)

    # Traces eps
    fig, axes = plt.subplots(
        n_repr,
        1,
        figsize=(12, 3.5 * n_repr),
        layout="constrained",
    )
    if n_repr == 1:
        axes = [axes]
    for ax, (name, r) in zip(axes, configs.items()):
        result = reference_chain(r["all_results"])
        eps_chain = result.eps_chain
        thin = result.raw.get("thin", 1)
        iters     = np.arange(eps_chain.shape[0]) * thin
        for j in range(eps_chain.shape[1]):
            ax.plot(
                iters,
                eps_chain[:, j],
                lw=0.8,
                label=rf"$\varepsilon_{{{j}}}$",
            )
        ax.set_title(
            f"{name} — $($\\delta_0, \\delta_1) = ({r['d0']}, {r['d1']})$"
            f"  [step={r['mala_step']}]"
        )
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$\varepsilon_j$")
        ax.legend(
            ncol=min(3, eps_chain.shape[1]),
            fontsize=7,
            loc="upper right",
        )
        ax.grid(alpha=0.3)
    plt.suptitle(r"Experiment 4 — Traces $\varepsilon_j$", fontsize=13)
    if savefigure:
        _save(fig, "exp4_traces_eps")
    plt.show()

    # Traces E_mu
    fig, axes = plt.subplots(
        n_repr,
        1,
        figsize=(12, 3 * n_repr),
        layout="constrained",
    )
    if n_repr == 1:
        axes = [axes]
    for ax, (name, r) in zip(axes, configs.items()):
        result = reference_chain(r["all_results"])
        E_mu = np.asarray(result.raw.get("E_mu", []), dtype=float)
        mask = ~np.isnan(E_mu)
        if mask.any():
            ax.plot(np.where(mask)[0], E_mu[mask], lw=0.8, color=emu_color)
        ax.set_title(
            f"{name} — $($\\delta_0, \\delta_1) = ({r['d0']}, {r['d1']})$"
            f"  [step={r['mala_step']}]"
        )
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$\mathcal{E}_\mu^{(t)}$")
        ax.grid(alpha=0.3)
    plt.suptitle(r"Experiment 4 — Traces $\mathcal{E}_\mu^{(t)}$", fontsize=13)
    if savefigure:
        _save(fig, "exp4_traces_Emu")
    plt.show()


# =============================================================================
# Tableau des 4 configurations représentatives
# =============================================================================
def print_representative_table(grid_results):
    rmse_ref = next(
        (r["rmse"] for r in grid_results
         if r["d0"] == DELTA_REF[0] and r["d1"] == DELTA_REF[1]),
        None,
    )
    print(f"\n{'='*95}")
    print(f"  Experiment 4 — Configurations représentatives")
    print(f"{'='*95}")
    print(f"  {'Régime':<22} {'(δ0, δ1)':<14} {'step':>6}"
          f" {'RMSE':>8} {'MAE':>8} {'CRPS':>8} {'ΔRMSE':>10}")
    print(f"  {'-'*90}")
    for name, (d0, d1) in REPRESENTATIVE_CONFIGS.items():
        for r in grid_results:
            if r["d0"] == d0 and r["d1"] == d1:
                cs = f"{r['crps']:.4f}" if r["crps"] is not None else "      --"
                if rmse_ref and rmse_ref > 0 and name != "Reference":
                    ds = f"{(r['rmse'] - rmse_ref) / rmse_ref:>+10.2%}"
                else:
                    ds = "        --"
                star = " *" if name == "Reference" else "  "
                print(f"  {name + star:<22} ({d0}, {d1}){'':<4}"
                      f" {r['mala_step']:>6.3f}"
                      f" {r['rmse']:>8.4f} {r['mae']:>8.4f}"
                      f" {cs:>8} {ds:>10}")
                break
    print(f"{'='*95}\n")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":

    SAVEFIGURE = True

    print(f"\n{'#'*70}")
    print(f"  EXPERIMENT 4 — Sensitivity (delta_0, delta_1)")
    print(f"{'#'*70}\n")

    # Données fixes pour tout l'experiment
    cells, germs = generate_voronoi_cells(
        n_germs=N_GERMS, X_bounds=XB, Y_bounds=YB, rng_seed=RNG_SEED,
    )
    plot_voronoi_cells(cells, germs, X_bounds=XB, Y_bounds=YB, cmap_name="cividis",
        title="Exp4 — Pavage de Voronoï (Profile 1)",
        savefigure=SAVEFIGURE, title_savefig="ssgc/experiment_4/exp4_voronoi")

    simulation = simulate_spatial_process(
        X_bounds=XB, Y_bounds=YB, T=T,
        polygons=cells, mus=MUS_VOR,
        f=f_star_A, grid_res=GRID_RES, rng_seed=15,
    )
    plot_process_dashboard(simulation,
        title="Exp4 — Données (Profile 1, Setting A)",
        savefigure=SAVEFIGURE, title_savefig="ssgc/experiment_4/exp4_dashboard")

    catalog = simulation.catalog
    zones = list(simulation.domains.polygons)
    reference_intensity = make_reference_intensity(
        zones, simulation.baseline_intensities, f_star_A
    )

    # Boucle sur la grille (delta0, delta1)
    parameter_grid = [
        (d0, d1)
        for d0 in DELTA0_GRID
        for d1 in DELTA1_GRID
    ]
    n_total      = len(parameter_grid)
    grid_results = []

    for d0, d1 in tqdm(
        parameter_grid,
        desc="Experiment 4 hyperparameter grid",
        unit="config",
        dynamic_ncols=True,
    ):
        save_traces = (d0, d1) in REPRESENTATIVE_CONFIGS.values()
        entry = fit_single_config(
            d0, d1, zones, catalog, reference_intensity,
            savefigure=False,
            save_traces=save_traces,
        )
        grid_results.append(entry)

    print(f"\n{'='*70}")
    print(f"  Grille terminée — {n_total} configurations")
    print(f"{'='*70}")

    plot_heatmaps(grid_results,           savefigure=SAVEFIGURE)
    plot_marginal_profiles(grid_results,  savefigure=SAVEFIGURE)
    plot_representative_traces(grid_results, savefigure=SAVEFIGURE)
    print_representative_table(grid_results)

    # Figures d'intensité pour les 4 configurations représentatives
    print("\n  >> Figures d'intensité — configurations représentatives")
    for name, (d0, d1) in REPRESENTATIVE_CONFIGS.items():
        for r in grid_results:
            if r["d0"] == d0 and r["d1"] == d1 and "all_results" in r:
                reference_chain(r["all_results"]).posterior_intensity(
                    nx=NX_POST, ny=NY_POST, burn_in=BURN_IN, cmap="inferno",
                    mu_star_func=reference_intensity,
                    savefigure=SAVEFIGURE, savefigure_Emu=SAVEFIGURE and COMPUTE_EMU,
                    title_savefig=f"ssgc/experiment_4/exp4_intensity_{name.replace(' ', '_')}",
                    title_savefig_Emu=f"ssgc/experiment_4/exp4_Emu_{name.replace(' ', '_')}",
                    n_mc=POSTERIOR_N_MC,
                )
                break

    print("\nExperiment 4 terminé.")
