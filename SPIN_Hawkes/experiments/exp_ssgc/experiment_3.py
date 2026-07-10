# %%
"""
Experiment 3 — Robustness under prior misspecification
Scenarios M1, M2, M2b, M3 × Settings A, B

M1  : données homogènes (J*=1), inférence J=1 (oracle) vs J=6 (zones superflues)
M2  : données J*=6 Profile 1, inférence J=6 oracle vs J=5 mauvaise partition
      (frontières ne coïncident pas avec les frontières vraies)
M2b : données J*=6 Profile 1, inférence J=4 partition incomplète
      (les 4 zones d'inférence sont des unions exactes de zones vraies — frontières
       conservées mais 2 paires de zones voisines fusionnées → partition emboîtée)
M3  : données J*=6 Profile 1, inférence J=6 oracle vs J=1 (zones manquantes)

MALA step par (scénario, modèle, J_infer) :
  - J=1  : gradient de ε scalaire, step plus petit
  - J=4  : intermédiaire
  - J=5  : proche de J=6, step légèrement réduit (moins de données par zone)
  - J=6  : référence
"""

# =============================================================================
# Imports
# =============================================================================
import warnings
import sys
warnings.filterwarnings("ignore")

import numpy as np
import openturns as ot
from functools import partial
from pathlib import Path
from scipy.special import expit
from shapely.geometry import Point as ShapelyPoint, box as shapely_box
from shapely.ops import unary_union
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
# Paramètres MCMC
# =============================================================================
NU_INIT       = [5.0, 0.2]
LAMBDA_NU     = 0.5
DELTA         = [1.0, 0.01]
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
NX, NY        = 30, 30
NX_POST, NY_POST = 60, 60
N_CHAINS      = 2
XB, YB        = (0.0, 2.0), (0.0, 2.0)
N_JOBS       = 1
GP_BACKEND   = "sparse"
COMPUTE_EMU  = False

# Calibrated with rng_seed=15 to generate about 1000 events per catalog.
T_PROFILE_1 = {"A": 180.7, "B": 98.2}
T_HOMOGENE  = {"A": 100.0, "B": 83.0}
GRID_RES_BY_SETTING = {"A": 100, "B": 300}


# =============================================================================
# MALA step par (scénario, label_modèle)
#
# Motivations :
#   J=1  : une seule composante ε, postérieure très concentrée, gradient simple.
#           Step plus petit pour ne pas dépasser le mode.
#   J=4  : 4 composantes, zones plus grandes donc plus de données localement,
#           gradient moins bruité qu'avec J=6 → step légèrement plus grand.
#   J=5  : proche de J=6 mais chaque zone reçoit en moyenne moins d'observations
#           (mauvaise partition → zones potentiellement déséquilibrées).
#           Step légèrement réduit pour les zones les plus petites.
#   J=6  : référence, step calibré empiriquement sur Profile 1.
#
# Schéma : MALA_STEP[(scenario, label)] où label identifie le modèle testé.
# On utilise le label plutôt que J seul car deux modèles peuvent avoir le même J
# mais des postérieures différentes (oracle vs misspec).
# =============================================================================
MALA_STEP = {
    # Calibrated on 100--200 sparse-GP Gibbs iterations for about 1000 events.
    # J=1 posteriors are sharper with this catalog size; multi-zone fits use
    # the same scale as Profile 1 in Experiment 2.
    ("M1",  "oracle_J1"):       0.050,
    ("M1",  "superfluous_J6"):  0.065,
    ("M2",  "oracle_J6"):       0.065,
    ("M2",  "wrong_J5"):        0.065,
    ("M2b", "oracle_J6"):       0.065,
    ("M2b", "partial_J4"):      0.065,
    ("M3",  "oracle_J6"):       0.065,
    ("M3",  "missing_J1"):      0.050,
}


def get_mala_step(scenario, label):
    """Retourne le MALA step pour (scénario, label modèle).

    Parameters
    ----------
    scenario : str
        Identifiant du scénario ("M1", "M2", "M2b", "M3").
    label : str
        Identifiant du modèle inféré (ex. "oracle_J6", "wrong_J5").

    Returns
    -------
    float
    """
    key = (scenario, label)
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
# Profils de zones
# =============================================================================
PROFILE_1 = {
    "n_germs": 6, "rng_seed": 15,
    "mus": (10.0, 1.0, 2.0, 10.0, 8.0, 2.0),
}

# Profile 2 : J=5, seed DIFFÉRENT de Profile 1 → frontières croisées
PROFILE_2 = {
    "n_germs": 5, "rng_seed": 42,
    "mus": (3.5, 2.0, 4.0, 3.0, 2.5),
}

MU_HOMOGENE = 5.0


# =============================================================================
# Construction de la partition partielle M2b
#
# Principe : on part des J=6 vraies zones et on en fusionne 2 paires de voisines
# (zones 0+1 et zones 2+3) pour obtenir J=4 zones.  Les 2 zones restantes
# (4 et 5) sont conservées intactes.  Les frontières INTERNES des paires sont
# effacées mais toutes les frontières externes sont identiques à l'oracle.
# L'intensité de la zone fusionnée est la moyenne pondérée par l'aire.
#
# Critère de voisinage : deux zones sont "voisines" si leurs polygones partagent
# une frontière (intersection de dimension ≥ 1).
# =============================================================================

def find_adjacent_pairs(zones_raw):
    """Retourne les paires de zones adjacentes (frontière commune).

    Parameters
    ----------
    zones_raw : list of shapely.Polygon

    Returns
    -------
    pairs : list of tuple(int, int)
        Paires (i, j) avec i < j telles que zones_raw[i] et zones_raw[j]
        partagent une frontière de longueur > 0.
    """
    pairs = []
    J = len(zones_raw)
    for i in range(J):
        for j in range(i + 1, J):
            inter = zones_raw[i].boundary.intersection(zones_raw[j].boundary)
            if not inter.is_empty and inter.length > 1e-8:
                pairs.append((i, j))
    return pairs


def build_partial_partition(zones_raw, mus_vec):
    """Construit une partition à J=4 zones par fusion de 2 paires adjacentes.

    Sélectionne les deux premières paires adjacentes distinctes (sans chevauchement
    d'indices), fusionne chaque paire, et conserve les zones restantes.

    Parameters
    ----------
    zones_raw : list of shapely.Polygon, len = 6
    mus_vec   : sequence of float, len = 6
        Intensités vraies par zone.

    Returns
    -------
    zones_partial : list of shapely.Polygon, len = 4
        Nouvelles zones après fusion.
    mus_partial : list of float, len = 4
        Intensités moyennées par l'aire pour les zones fusionnées.
    merge_map : list of tuple
        Liste des paires fusionnées : [(i0, j0), (i1, j1)].
    """
    adj = find_adjacent_pairs(zones_raw)
    if len(adj) < 2:
        raise ValueError(
            "Pas assez de paires adjacentes pour construire une partition à J=4."
        )

    # Sélection de 2 paires disjointes (aucun indice partagé)
    chosen = [adj[0]]
    used   = set(adj[0])
    for p in adj[1:]:
        if p[0] not in used and p[1] not in used:
            chosen.append(p)
            used.update(p)
            if len(chosen) == 2:
                break

    if len(chosen) < 2:
        raise ValueError(
            "Impossible de trouver 2 paires adjacentes disjointes parmi les 6 zones."
        )

    merged_indices = set(used)
    remaining      = [j for j in range(len(zones_raw)) if j not in merged_indices]

    zones_partial = []
    mus_partial   = []
    mus_arr       = np.array([float(m) for m in mus_vec])

    for (i, j) in chosen:
        poly_merged = unary_union([zones_raw[i], zones_raw[j]])
        zones_partial.append(poly_merged)
        ai, aj  = zones_raw[i].area, zones_raw[j].area
        mu_fuse = (mus_arr[i] * ai + mus_arr[j] * aj) / (ai + aj)
        mus_partial.append(float(mu_fuse))

    for j in remaining:
        zones_partial.append(zones_raw[j])
        mus_partial.append(float(mus_arr[j]))

    return zones_partial, mus_partial, chosen


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
              n_iter, thin, verbose, verbose_every,
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
        use_calibration=USE_CALIB,
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
                  mala_step, mu_star_func):
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


# =============================================================================
# Fit + eval générique
# =============================================================================
def fit_and_eval(scenario, label, zones_raw_infer, x_arr, y_arr, t_arr, T,
                 mu_star_func_true, setting,
                 cmap_intensities="inferno", savefigure=False):
    """Run independent chains and evaluate one representative posterior.

    Parameters
    ----------
    scenario : str
        Identifiant du scénario ("M1", "M2", "M2b", "M3").
    label : str
        Identifiant du modèle (clé pour get_mala_step et noms de fichiers).
    zones_raw_infer : list
        Partition utilisée pour l'inférence.
    mu_star_func_true : callable
        Vraie intensité (générée avec la vraie partition).
    """
    step = get_mala_step(scenario, label)
    J    = len(zones_raw_infer)
    print(f"\n  >> [{scenario}] {label} (J_infer={J})  — mala_step={step}")

    all_results = launch_chains(
        zones_raw_infer, x_arr, y_arr, t_arr, T,
        step, mu_star_func_true,
    )
    k_ref = N_CHAINS // 2
    out = all_results[k_ref].posterior_intensity(
        nx=NX_POST, ny=NY_POST, burn_in=BURN_IN,
        cmap=cmap_intensities,
        mu_star_func=mu_star_func_true,
        savefigure=savefigure, savefigure_Emu=savefigure and COMPUTE_EMU,
        title_savefig=f"ssgc/experiment_3/exp3_{scenario}_{label}_{setting}",
        title_savefig_Emu=f"ssgc/experiment_3/exp3_{scenario}_{label}_Emu_{setting}",
    )
    return {
        "rmse": out["rmse"], "mae": out["mae"],
        "crps": out.get("crps"),
        "out": out, "all_results": all_results,
    }


# =============================================================================
# Génération des données
# =============================================================================
def generate_data_profile1(setting_name, f_star_func):
    T = T_PROFILE_1[setting_name]
    cells, germs = generate_voronoi_cells(
        n_germs=PROFILE_1["n_germs"], X_bounds=XB, Y_bounds=YB,
        rng_seed=PROFILE_1["rng_seed"],
    )
    simulation = simulate_spatial_process(
        X_bounds=XB, Y_bounds=YB, T=T,
        polygons=cells, mus=PROFILE_1["mus"],
        f=f_star_func, grid_res=GRID_RES_BY_SETTING[setting_name], rng_seed=15,
    )
    X_data = simulation.sample; N = X_data.getSize()
    x_arr  = np.array([float(X_data[i, 0]) for i in range(N)])
    y_arr  = np.array([float(X_data[i, 1]) for i in range(N)])
    t_arr  = np.array([float(X_data[i, 2]) for i in range(N)])
    zones_raw_true = list(simulation.domains.polygons)
    mus_vec_true   = list(simulation.baseline_intensities)
    mu_star_true   = partial(mu_star_func_picklable,
                             zones_raw=zones_raw_true, mus_vec=mus_vec_true,
                             f_func=f_star_func)
    return x_arr, y_arr, t_arr, T, zones_raw_true, mus_vec_true, mu_star_true, \
           cells, germs, simulation


def generate_data_homogeneous(setting_name, f_star_func):
    T = T_HOMOGENE[setting_name]
    domain_poly = shapely_box(XB[0], YB[0], XB[1], YB[1])
    simulation = simulate_spatial_process(
        X_bounds=XB, Y_bounds=YB, T=T,
        polygons=[domain_poly], mus=(MU_HOMOGENE,),
        f=f_star_func, grid_res=GRID_RES_BY_SETTING[setting_name], rng_seed=15,
    )
    X_data = simulation.sample; N = X_data.getSize()
    x_arr = np.array([float(X_data[i, 0]) for i in range(N)])
    y_arr = np.array([float(X_data[i, 1]) for i in range(N)])
    t_arr = np.array([float(X_data[i, 2]) for i in range(N)])
    mu_star_true = partial(mu_star_func_picklable,
                           zones_raw=[domain_poly], mus_vec=[MU_HOMOGENE],
                           f_func=f_star_func)
    return x_arr, y_arr, t_arr, T, [domain_poly], mu_star_true, simulation


def get_inference_partition(profile):
    cells, germs = generate_voronoi_cells(
        n_germs=profile["n_germs"], X_bounds=XB, Y_bounds=YB,
        rng_seed=profile["rng_seed"],
    )
    return list(cells), germs


# =============================================================================
# Tableau récapitulatif
# =============================================================================
def _save(fig, name):
    path = save_figure(fig, f"ssgc/experiment_3/{name}")
    print(f"  Figure sauvegardée : {path}")


# =============================================================================
# Scénario M1
# =============================================================================
def run_scenario_M1(setting_name, cmap_voronoi="cividis",
                    cmap_intensities="inferno", savefigure=False):
    """Données homogènes J*=1 vs inférence J=1 (oracle) et J=6 (zones superflues)."""
    f_star_func = F_STAR[setting_name]
    print(f"\n{'#'*70}")
    print(f"  EXP3 — Scenario M1 (superfluous zones), Setting {setting_name}")
    print(f"{'#'*70}")

    x_arr, y_arr, t_arr, T, zones_true, mu_star_true, simulation = \
        generate_data_homogeneous(setting_name, f_star_func)
    plot_process_dashboard(simulation, cmap=cmap_intensities,
        title=f"M1 données homogènes — Setting {setting_name}",
        savefigure=savefigure, title_savefig=f"ssgc/experiment_3/exp3_M1_dashboard_{setting_name}")

    records = []
    domain_poly = shapely_box(XB[0], YB[0], XB[1], YB[1])

    # Oracle J=1
    r = fit_and_eval("M1", "oracle_J1", [domain_poly],
                     x_arr, y_arr, t_arr, T, mu_star_true,
                     setting_name, cmap_intensities, savefigure)
    records.append({"scenario": "M1", "setting": setting_name,
                    "model": "Homogeneous SGCP (oracle J=1)",
                    **{k: r[k] for k in ["rmse", "mae", "crps"]}})

    # Superfluous J=6
    zones_J6, germs_J6 = get_inference_partition(PROFILE_1)
    plot_voronoi_cells(zones_J6, germs_J6, X_bounds=XB, Y_bounds=YB,
        cmap_name=cmap_voronoi,
        title=f"M1 — Partition superflue J=6 (Setting {setting_name})",
        savefigure=savefigure, title_savefig=f"ssgc/experiment_3/exp3_M1_voronoi_J6_{setting_name}")
    r = fit_and_eval("M1", "superfluous_J6", zones_J6,
                     x_arr, y_arr, t_arr, T, mu_star_true,
                     setting_name, cmap_intensities, savefigure)
    records.append({"scenario": "M1", "setting": setting_name,
                    "model": "SSGC, superfluous zones (J=6)",
                    **{k: r[k] for k in ["rmse", "mae", "crps"]}})

    print_metrics_table(records)
    return records


# =============================================================================
# Scénario M2
# =============================================================================
def run_scenario_M2(setting_name, cmap_voronoi="cividis",
                    cmap_intensities="inferno", savefigure=False):
    """Données J*=6 Profile 1 vs oracle J=6 et mauvaise partition J=5."""
    f_star_func = F_STAR[setting_name]
    print(f"\n{'#'*70}")
    print(f"  EXP3 — Scenario M2 (wrong partition), Setting {setting_name}")
    print(f"{'#'*70}")

    x_arr, y_arr, t_arr, T, zones_true, _, mu_star_true, \
        cells_true, germs_true, simulation = \
        generate_data_profile1(setting_name, f_star_func)
    plot_voronoi_cells(cells_true, germs_true, X_bounds=XB, Y_bounds=YB,
        cmap_name=cmap_voronoi, title=f"M2 — Vraie partition J*=6 (Setting {setting_name})",
        savefigure=savefigure, title_savefig=f"ssgc/experiment_3/exp3_M2_voronoi_true_{setting_name}")
    plot_process_dashboard(simulation, cmap=cmap_intensities,
        title=f"M2 données Profile 1 — Setting {setting_name}",
        savefigure=savefigure, title_savefig=f"ssgc/experiment_3/exp3_M2_dashboard_{setting_name}")

    records = []

    # Oracle J=6
    r = fit_and_eval("M2", "oracle_J6", zones_true,
                     x_arr, y_arr, t_arr, T, mu_star_true,
                     setting_name, cmap_intensities, savefigure)
    records.append({"scenario": "M2", "setting": setting_name,
                    "model": "SSGC, oracle (J=6)",
                    **{k: r[k] for k in ["rmse", "mae", "crps"]}})

    # Wrong partition J=5
    zones_wrong, germs_wrong = get_inference_partition(PROFILE_2)
    plot_voronoi_cells(zones_wrong, germs_wrong, X_bounds=XB, Y_bounds=YB,
        cmap_name=cmap_voronoi,
        title=f"M2 — Mauvaise partition J=5 (Setting {setting_name})",
        savefigure=savefigure, title_savefig=f"ssgc/experiment_3/exp3_M2_voronoi_wrong_{setting_name}")
    r = fit_and_eval("M2", "wrong_J5", zones_wrong,
                     x_arr, y_arr, t_arr, T, mu_star_true,
                     setting_name, cmap_intensities, savefigure)
    records.append({"scenario": "M2", "setting": setting_name,
                    "model": "SSGC, wrong partition (J=5)",
                    **{k: r[k] for k in ["rmse", "mae", "crps"]}})

    print_metrics_table(records)
    return records


# =============================================================================
# Scénario M2b — partition incomplète (J=4, zones emboîtées)
# =============================================================================
def run_scenario_M2b(setting_name, cmap_voronoi="cividis",
                     cmap_intensities="inferno", savefigure=False):
    """Données J*=6 Profile 1 vs oracle J=6 et partition partielle J=4.

    Les J=4 zones d'inférence sont obtenues en fusionnant deux paires de zones
    vraies adjacentes.  Les frontières conservées sont un sous-ensemble exact des
    frontières vraies (partition emboîtée), contrairement à M2 où les frontières
    se croisent.  Ce scénario teste si le GP peut compenser l'information perdue
    sur les ε intra-paires fusionnées.
    """
    f_star_func = F_STAR[setting_name]
    print(f"\n{'#'*70}")
    print(f"  EXP3 — Scenario M2b (partial/nested partition), Setting {setting_name}")
    print(f"{'#'*70}")

    x_arr, y_arr, t_arr, T, zones_true, mus_vec_true, mu_star_true, \
        cells_true, germs_true, simulation = \
        generate_data_profile1(setting_name, f_star_func)

    # Construction de la partition à J=4
    zones_partial, mus_partial, merge_map = build_partial_partition(
        zones_true, mus_vec_true,
    )
    J_partial = len(zones_partial)
    assert J_partial == 4, f"Attendu J=4, obtenu J={J_partial}"
    print(f"  >> Paires fusionnées : {merge_map}")

    # Visualisation oracle
    plot_voronoi_cells(cells_true, germs_true, X_bounds=XB, Y_bounds=YB,
        cmap_name=cmap_voronoi,
        title=f"M2b — Vraie partition J*=6 (Setting {setting_name})",
        savefigure=savefigure,
        title_savefig=f"ssgc/experiment_3/exp3_M2b_voronoi_oracle_{setting_name}")

    # Visualisation partition partielle
    # germs factices = centroïdes des zones partielles
    germs_partial = ot.Sample([
        [float(z.centroid.x), float(z.centroid.y)] for z in zones_partial
    ])
    plot_voronoi_cells(zones_partial, germs_partial, X_bounds=XB, Y_bounds=YB,
        cmap_name=cmap_voronoi,
        title=f"M2b — Partition partielle J=4 (Setting {setting_name})",
        savefigure=savefigure,
        title_savefig=f"ssgc/experiment_3/exp3_M2b_voronoi_partial_{setting_name}")

    plot_process_dashboard(simulation, cmap=cmap_intensities,
        title=f"M2b données Profile 1 — Setting {setting_name}",
        savefigure=savefigure, title_savefig=f"ssgc/experiment_3/exp3_M2b_dashboard_{setting_name}")

    records = []

    # Oracle J=6
    r = fit_and_eval("M2b", "oracle_J6", zones_true,
                     x_arr, y_arr, t_arr, T, mu_star_true,
                     setting_name, cmap_intensities, savefigure)
    records.append({"scenario": "M2b", "setting": setting_name,
                    "model": "SSGC, oracle (J=6)",
                    **{k: r[k] for k in ["rmse", "mae", "crps"]}})

    # Partition partielle J=4
    r = fit_and_eval("M2b", "partial_J4", zones_partial,
                     x_arr, y_arr, t_arr, T, mu_star_true,
                     setting_name, cmap_intensities, savefigure)
    records.append({"scenario": "M2b", "setting": setting_name,
                    "model": "SSGC, partial partition (J=4)",
                    **{k: r[k] for k in ["rmse", "mae", "crps"]}})

    print_metrics_table(records)
    return records


# =============================================================================
# Scénario M3
# =============================================================================
def run_scenario_M3(setting_name, cmap_voronoi="cividis",
                    cmap_intensities="inferno", savefigure=False):
    """Données J*=6 Profile 1 vs oracle J=6 et J=1 (zones manquantes)."""
    f_star_func = F_STAR[setting_name]
    print(f"\n{'#'*70}")
    print(f"  EXP3 — Scenario M3 (missing zones), Setting {setting_name}")
    print(f"{'#'*70}")

    x_arr, y_arr, t_arr, T, zones_true, _, mu_star_true, \
        cells_true, germs_true, simulation = \
        generate_data_profile1(setting_name, f_star_func)
    plot_process_dashboard(simulation, cmap=cmap_intensities,
        title=f"M3 données Profile 1 — Setting {setting_name}",
        savefigure=savefigure, title_savefig=f"ssgc/experiment_3/exp3_M3_dashboard_{setting_name}")

    records = []

    # Oracle J=6
    r = fit_and_eval("M3", "oracle_J6", zones_true,
                     x_arr, y_arr, t_arr, T, mu_star_true,
                     setting_name, cmap_intensities, savefigure)
    records.append({"scenario": "M3", "setting": setting_name,
                    "model": "SSGC, oracle (J=6)",
                    **{k: r[k] for k in ["rmse", "mae", "crps"]}})

    # J=1 (zones manquantes)
    domain_poly = shapely_box(XB[0], YB[0], XB[1], YB[1])
    r = fit_and_eval("M3", "missing_J1", [domain_poly],
                     x_arr, y_arr, t_arr, T, mu_star_true,
                     setting_name, cmap_intensities, savefigure)
    records.append({"scenario": "M3", "setting": setting_name,
                    "model": "Homogeneous SGCP, missing zones (J=1)",
                    **{k: r[k] for k in ["rmse", "mae", "crps"]}})

    print_metrics_table(records)
    return records


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":

    SAVEFIGURE  = True
    all_records = []

    for setting in ["A", "B"]:
        all_records.extend(run_scenario_M1(setting,  savefigure=SAVEFIGURE))
        all_records.extend(run_scenario_M2(setting,  savefigure=SAVEFIGURE))
        all_records.extend(run_scenario_M2b(setting, savefigure=SAVEFIGURE))
        all_records.extend(run_scenario_M3(setting,  savefigure=SAVEFIGURE))

    print("\n" + "=" * 90)
    print("  RÉCAPITULATIF GLOBAL — Experiment 3")
    print("=" * 90)
    print_metrics_table(all_records)
    print("Experiment 3 terminé.")
