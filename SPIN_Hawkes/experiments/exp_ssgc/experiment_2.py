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
import sys
import warnings

warnings.filterwarnings("ignore")

import time
import numpy as np
import openturns as ot
import matplotlib.pyplot as plt
import properscoring as ps
from pathlib import Path
from scipy.special import expit
from shapely.geometry import Point as ShapelyPoint
from shapely.prepared import prep

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
GP_BACKEND   = "sparse"
COMPUTE_EMU  = False
MIN_ZONE_CALIB_EVENTS = 30
POSTERIOR_N_MC = 500


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


def make_model(zones, duration, x_bounds, y_bounds):
    return SSGCModel.from_polygons(
        polygons=zones,
        duration=duration,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        initial_log_intensities=0.0,
        gp_prior=GPParameters(
            variance=NU_INIT[0], length_scale=NU_INIT[1]
        ),
        eps_prior_variance=DELTA[0],
        eps_prior_length_scale=DELTA[1],
        nu_prior_rate=LAMBDA_NU,
        jitter=JITTER,
    )


def make_gibbs_config(mala_step, use_calibration):
    return GibbsConfig(
        n_iter=N_ITER,
        thin=THIN,
        mala_step=mala_step,
        learn_nu=LEARN_NU,
        use_calibration=use_calibration,
        verbose=VERBOSE,
        verbose_every=VERBOSE_EVERY,
        t0_nu=T0_NU,
        step_nu_init=STEP_NU_INIT,
        grid_nx=NX,
        grid_ny=NY,
        compute_emu=COMPUTE_EMU,
    )


def run_chain(
    chain_index,
    zones,
    catalog,
    duration,
    mala_step,
    reference_intensity,
    x_bounds,
    y_bounds,
    seed_offset=0,
    use_calibration=USE_CALIB,
):
    chain_seed = SEED + seed_offset + chain_index
    model = make_model(zones, duration, x_bounds, y_bounds)
    result = model.gibbs(
        catalog,
        config=make_gibbs_config(mala_step, use_calibration),
        rng_seed=chain_seed,
        reference_intensity=reference_intensity,
        gp_backend=GP_BACKEND,
    )
    print(f"  [Chain {chain_index + 1}] done (seed={chain_seed})")
    return result


def launch_chains(
    zones,
    catalog,
    duration,
    mala_step,
    reference_intensity,
    x_bounds=X_BOUNDS,
    y_bounds=Y_BOUNDS,
    seed_offset=0,
    use_calibration=USE_CALIB,
):
    """Run independent chains sequentially; Gibbs results contain OT objects."""
    return [
        run_chain(
            chain_index,
            zones,
            catalog,
            duration,
            mala_step,
            reference_intensity,
            x_bounds,
            y_bounds,
            seed_offset=seed_offset,
            use_calibration=use_calibration,
        )
        for chain_index in range(N_CHAINS)
    ]


def reference_chain(results):
    return results[len(results) // 2]


# =============================================================================
# Fit joint SSGC
# =============================================================================
def fit_and_eval_joint(
    catalog,
    duration,
    zones,
    reference_intensity,
    profile_name,
    setting_name,
    savefigure,
    cmap_intensities="inferno",
):
    step = get_mala_step(profile_name, setting_name, "joint")
    print(f"\n  >> Joint SSGC (Profile {profile_name})  — mala_step={step}")
    t0 = time.time()

    all_results = launch_chains(
        zones, catalog, duration, step, reference_intensity,
    )
    elapsed = time.time() - t0
    print(f"  >> Joint SSGC : {elapsed:.1f}s")

    tag = f"P{profile_name}_{setting_name}"

    out = reference_chain(all_results).posterior_intensity(
        nx=NX_POST, ny=NY_POST, burn_in=BURN_IN, cmap=cmap_intensities,
        mu_star_func=reference_intensity,
        savefigure=savefigure, savefigure_Emu=savefigure and COMPUTE_EMU,
        title_savefig=f"ssgc/experiment_2/exp2_intensity_joint_{tag}",
        title_savefig_Emu=f"ssgc/experiment_2/exp2_Emu_joint_{tag}",
        n_mc=POSTERIOR_N_MC,
    )
    plt.close("all")
    return out, all_results, elapsed


# =============================================================================
# Fit zone-wise SGCP
# =============================================================================
def fit_and_eval_zonewise(
    catalog,
    duration,
    zones,
    reference_intensity,
    profile_name,
    setting_name,
    savefigure,
    cmap_intensities="inferno",
):
    """Un sampler SGCP (J=1) par zone, stitch sur la grille globale."""
    step = get_mala_step(profile_name, setting_name, "zonewise")
    print(f"\n  >> Zone-wise SGCP (Profile {profile_name})  — mala_step={step}")
    t0 = time.time()

    J = len(zones)
    zones_prep_full = [prep(zone) for zone in zones]
    tag = f"P{profile_name}_{setting_name}"

    # Partition des observations par zone
    points_per_zone = [[] for _ in range(J)]
    for i, (x_value, y_value) in enumerate(zip(catalog.x, catalog.y)):
        pt = ShapelyPoint(float(x_value), float(y_value))
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
    M, n_mc = len(grid_xy), POSTERIOR_N_MC

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

        zone_catalog = EventCatalog(
            t=catalog.t[idx_j],
            x=catalog.x[idx_j],
            y=catalog.y[idx_j],
        )
        poly = zones[j]
        pz   = zones_prep_full[j]
        bx, by, bx2, by2 = poly.bounds

        zone_use_calib = USE_CALIB and N_j >= MIN_ZONE_CALIB_EVENTS
        if not zone_use_calib:
            print("       calibration skipped for this small zone")
        all_results_j = launch_chains(
            [poly], zone_catalog, duration,
            step, reference_intensity,
            x_bounds=(bx, bx2), y_bounds=(by, by2),
            seed_offset=1000 * j,
            use_calibration=zone_use_calib,
        )
        result_j = reference_chain(all_results_j)

        def zone_reference(x, y, prepared_zone=pz):
            x_values, y_values = np.broadcast_arrays(
                np.asarray(x, dtype=float), np.asarray(y, dtype=float)
            )
            flat_x = x_values.reshape(-1)
            flat_y = y_values.reshape(-1)
            inside = np.fromiter(
                (
                    prepared_zone.covers(ShapelyPoint(xi, yi))
                    for xi, yi in zip(flat_x, flat_y)
                ),
                dtype=bool,
                count=flat_x.size,
            )
            truth = np.asarray(
                reference_intensity(flat_x, flat_y), dtype=float
            ).reshape(-1)
            return np.where(inside, truth, 0.0).reshape(x_values.shape)

        result_j.posterior_intensity(
            nx=NX_POST, ny=NY_POST, burn_in=BURN_IN, cmap=cmap_intensities,
            mu_star_func=zone_reference,
            savefigure=savefigure, savefigure_Emu=savefigure and COMPUTE_EMU,
            title_savefig=f"ssgc/experiment_2/exp2_intensity_zw_zone{j}_{tag}",
            title_savefig_Emu=f"ssgc/experiment_2/exp2_Emu_zw_zone{j}_{tag}",
            n_mc=n_mc,
        )
        plt.close("all")

        # Stitch sur les points de grille appartenant à cette zone
        in_zone = np.array([
            pz.covers(ShapelyPoint(float(grid_xy[k, 0]), float(grid_xy[k, 1])))
            for k in range(M)
        ])
        if not in_zone.any():
            per_zone_metrics.append(None)
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

        truth_local = np.asarray(
            reference_intensity(
                grid_xy[idx_local, 0], grid_xy[idx_local, 1]
            ),
            dtype=float,
        )
        estimate_local = mu_hat_zw[idx_local]
        per_zone_metrics.append({
            "rmse": float(np.sqrt(np.mean((estimate_local - truth_local) ** 2))),
            "mae": float(np.mean(np.abs(estimate_local - truth_local))),
            "crps": float(ps.crps_ensemble(truth_local, mu_sims_l).mean()),
        })

    elapsed = time.time() - t0
    print(f"  >> Zone-wise SGCP : {elapsed:.1f}s")

    # Métriques globales
    mu_star_global = reference_intensity(grid_xy[:, 0], grid_xy[:, 1])
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


def _format_metric(value):
    return "--" if value is None else f"{value:.4f}"


def print_summary(
    joint,
    zonewise,
    per_zone,
    time_joint,
    time_zonewise,
    profile_name,
    setting_name,
):
    print(f"\n{'=' * 72}")
    print(f"Experiment 2 — Profile {profile_name}, Setting {setting_name}")
    print(f"{'=' * 72}")
    print(f"{'Model':<22} {'RMSE':>10} {'MAE':>10} {'CRPS':>10} {'Time (s)':>12}")
    print("-" * 72)
    print(
        f"{'Joint SSGC':<22} {_format_metric(joint['rmse']):>10} "
        f"{_format_metric(joint['mae']):>10} "
        f"{_format_metric(joint['crps']):>10} {time_joint:>12.1f}"
    )
    print(
        f"{'Zone-wise SGCP':<22} {_format_metric(zonewise['rmse']):>10} "
        f"{_format_metric(zonewise['mae']):>10} "
        f"{_format_metric(zonewise['crps']):>10} {time_zonewise:>12.1f}"
    )
    print("\nZone-wise metrics evaluated only inside each polygon:")
    for zone_index, metrics in enumerate(per_zone, start=1):
        if metrics is None:
            print(f"  Zone {zone_index}: insufficient data or no grid point")
            continue
        print(
            f"  Zone {zone_index}: RMSE={metrics['rmse']:.4f}, "
            f"MAE={metrics['mae']:.4f}, CRPS={metrics['crps']:.4f}"
        )


def _plot_fields(grid_xy, fields, titles, cmap, colorbar_label):
    finite_values = np.concatenate([
        np.asarray(field, dtype=float)[np.isfinite(field)] for field in fields
    ])
    vmin = float(np.min(finite_values))
    vmax = float(np.max(finite_values))
    if vmax <= vmin:
        vmax = vmin + 1.0

    fig, axes = plt.subplots(1, len(fields), figsize=(6 * len(fields), 5))
    axes = np.atleast_1d(axes)
    for ax, field, title in zip(axes, fields, titles):
        image = ax.tricontourf(
            grid_xy[:, 0],
            grid_xy[:, 1],
            np.asarray(field, dtype=float),
            levels=30,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        fig.colorbar(image, ax=ax, label=colorbar_label)
        ax.set_title(title)
        ax.set_xlim(X_BOUNDS)
        ax.set_ylim(Y_BOUNDS)
        ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig


def plot_intensity_comparison(
    joint,
    zonewise,
    profile_name,
    setting_name,
    cmap_intensities="inferno",
    savefigure=False,
):
    grid_xy = np.asarray(joint["mesh"].getVertices(), dtype=float)
    fig = _plot_fields(
        grid_xy,
        [joint["mu_star"], joint["mu_hat"], zonewise["mu_hat"]],
        ["True intensity", "Joint SSGC", "Zone-wise SGCP"],
        cmap_intensities,
        "Intensity",
    )
    fig.suptitle(f"Profile {profile_name}, Setting {setting_name} — intensity", y=1.02)
    if savefigure:
        _save(fig, f"exp2_intensity_comparison_P{profile_name}_{setting_name}")
    plt.show()


def plot_variance_comparison(
    joint,
    zonewise,
    profile_name,
    setting_name,
    savefigure=False,
):
    grid_xy = np.asarray(joint["mesh"].getVertices(), dtype=float)
    fig = _plot_fields(
        grid_xy,
        [joint["var_mu_hat"], zonewise["mu_var"]],
        ["Joint SSGC", "Zone-wise SGCP"],
        "viridis",
        "Posterior variance",
    )
    fig.suptitle(
        f"Profile {profile_name}, Setting {setting_name} — uncertainty",
        y=1.02,
    )
    if savefigure:
        _save(fig, f"exp2_variance_comparison_P{profile_name}_{setting_name}")
    plt.show()


def _calibration_curve(truth, estimate, n_bins=10):
    ordered_indices = np.argsort(truth)
    bins = np.array_split(ordered_indices, n_bins)
    truth_means = np.asarray([np.mean(truth[index]) for index in bins if index.size])
    estimate_means = np.asarray([
        np.mean(estimate[index]) for index in bins if index.size
    ])
    return truth_means, estimate_means


def plot_calibration_curves(
    joint,
    zonewise,
    profile_name,
    setting_name,
    savefigure=False,
):
    truth = np.asarray(joint["mu_star"], dtype=float)
    joint_x, joint_y = _calibration_curve(truth, joint["mu_hat"])
    zonewise_x, zonewise_y = _calibration_curve(truth, zonewise["mu_hat"])
    upper = float(max(np.max(truth), np.max(joint_y), np.max(zonewise_y)))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0.0, upper], [0.0, upper], "k--", label="Ideal")
    ax.plot(joint_x, joint_y, "o-", label="Joint SSGC")
    ax.plot(zonewise_x, zonewise_y, "s-", label="Zone-wise SGCP")
    ax.set_xlabel("Mean true intensity by bin")
    ax.set_ylabel("Mean posterior intensity by bin")
    ax.set_title(f"Profile {profile_name}, Setting {setting_name} — calibration")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if savefigure:
        _save(fig, f"exp2_calibration_P{profile_name}_{setting_name}")
    plt.show()


# =============================================================================
# Fonction principale par (profil, setting)
# =============================================================================
def run_exp2_config(profile, setting_name, cmap_voronoi="cividis",
                    cmap_intensities="inferno", savefigure=False):
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

    catalog = simulation.catalog
    zones = list(simulation.domains.polygons)
    reference_intensity = make_reference_intensity(
        zones,
        simulation.baseline_intensities,
        f_star_func,
    )

    joint_out, joint_results, time_joint = fit_and_eval_joint(
        catalog, T, zones,
        reference_intensity, profile_name, setting_name,
        savefigure, cmap_intensities,
    )
    zw_metrics, zw_per_zone, time_zw = fit_and_eval_zonewise(
        catalog, T, zones,
        reference_intensity, profile_name, setting_name,
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
