"""FCAT-17 random and temporal validation of SSGC, SGCP and Silverman KDE.

FCAT-17 is treated as a declustered background catalogue. The default
campaign compares Gibbs sparse fits of the SSGC component of SPIN-H and its
zoneless SGCP special case, together with a fixed-bandwidth Silverman KDE.
Other implemented backends remain available as
explicit command-line or Python API options.

For an editor workflow, change the ``EDITOR SETTINGS`` block below and run the
file without arguments.  Command-line arguments remain available for batch
execution and reproducible campaigns.
"""

from __future__ import annotations

# %% Imports
import argparse
import sys
import time
from dataclasses import asdict
from itertools import combinations
from numbers import Integral
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
from shapely.geometry import Polygon
from shapely.ops import unary_union

start = Path(globals().get("__file__", Path.cwd() / "interactive.py")).resolve().parent
search_roots = [start, *start.parents]
for candidate in search_roots + [path / "SPIN_Hawkes" for path in search_roots]:
    if (candidate / "package/models/spinh.py").is_file():
        REPO_ROOT = candidate
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        break
else:
    raise RuntimeError("Open the repository folder before running this script.")

from data import EventCatalog
from experiments.exp_spinh.fcat_validation import (
    ExposureNormalizedKDE,
    PROTOCOL_LABELS,
    predictive_scores,
    training_catalog,
    validation_splits,
    zonal_quadrature,
)
from experiments.exp_spinh.fcat_settings import CAMPAIGNS, configure_campaign
from experiments.exp_spinh.fcat_runner_utils import (
    mcmc_diagnostics,
    write_campaign,
    write_records,
)
from experiments.exp_spinh.runner_utils import (
    calibration_slot,
    checkpoint_directory,
    effective_worker_count,
    parallel_map,
    resolve_n_jobs,
)
from experiments.exp_ssgc.deliverable_utils import (
    GPParameters,
    fit_intensity_method,
    make_model as make_ssgc_model,
    points_in_geometry,
)
from spatial import DomainPartition

RESULTS_ROOT = REPO_ROOT / "results" / "spinh_test"

# %% ========================================================================
# FCAT-17 SCIENTIFIC SETTINGS
# =============================================================================
# These values define the catalogue selection and SSGC prior used by every
# execution mode.  Change them only when changing the scientific protocol.

YEAR_MIN = 1965
OBSERVATION_END_YEAR = 2017  # exclusive upper boundary of the catalogue exposure
MAGNITUDE_MIN = 3.0
EPS_PRIOR_VARIANCE = 10.0
EPS_PRIOR_LENGTH_SCALE_KM = 3.0
INITIAL_GP = GPParameters(variance=2.0, length_scale=50.0)
FINAL_INTENSITY_GRID_SIZE = 90
SSGC_METHODS = {
    "ssgc_gibbs_sparse": "gibbs_sparse",
    "ssgc_vi_sparse": "vi_sparse",
}
MODELS = (*SSGC_METHODS, "sgcp", "kde")
DEFAULT_MODELS = ("ssgc_gibbs_sparse", "sgcp", "kde")
MODEL_LABELS = {
    "ssgc_gibbs_sparse": "SSGC (Gibbs sparse)",
    "ssgc_vi_sparse": "SSGC (VI sparse)",
    "sgcp": "SGCP (J=1)",
    "kde": "KDE",
}
SGCP_INFERENCE_METHODS = (
    "gibbs_exact",
    "gibbs_sparse",
    "vi_exact",
    "vi_sparse",
)


# %% ========================================================================
# EDITOR SETTINGS
# =============================================================================
# Edit only this block for a normal "Run Python File" / "Run in Interactive
# Window" workflow.  Command-line arguments take precedence when provided.
# ``None`` means: keep the selected profile's default value.

EDITOR_PROFILE = "full"                     # "smoke" or "full"
EDITOR_MODELS = DEFAULT_MODELS               # SSGC, SGCP (both Gibbs sparse), KDE
EDITOR_PROTOCOLS = ("random", "temporal")
EDITOR_INFERENCE_METHOD = "gibbs_sparse"     # SGCP only; SSGC variants are fixed
EDITOR_N_JOBS = -1                            # simultaneous train/test fits
EDITOR_RESUME = True                         # reuse completed task checkpoints
EDITOR_SAVE_FIGURES = True
EDITOR_SHOW_FIGURES = True
EDITOR_OUTPUT_DIR = None                     # None: results/spinh_test/<profile>/fcat17

EDITOR_CAMPAIGN_OVERRIDES = {
    "random_replicates": None,               # full: 10 reproducible 50/50 splits
    "validation_seed": None,
    "temporal_train_fraction": None,         # full: 0.8
    "cv_n_chains": None,
    "full_n_chains": None,
    "cv_gibbs_iterations": None,
    "full_gibbs_iterations": None,
    "gibbs_thin": None,
    "gibbs_burn_in": None,
    "mala_initial_step": None,
    "mala_adaptation_start": None,
    "mala_target_acceptance": None,
    "mala_adaptation_decay": None,
    "mala_precondition": None,
    "vi_iterations": None,
    "vi_tolerance": None,
    "quadrature_space_grid": None,  # full: 100 x 100; scoring and VI integration
    "score_posterior_draws": None,  # full: 1000
    "map_posterior_draws": None,    # full: 500
    "max_parallel_calibrations": None,
    "exact_max_events": None,
    "use_calibration": None,
}

# %% End of editor settings


def editor_run_options():
    """Return an isolated copy of the options from the editor settings block."""
    return {
        "profile": EDITOR_PROFILE,
        "models": tuple(EDITOR_MODELS),
        "protocols": tuple(EDITOR_PROTOCOLS),
        "inference_method": EDITOR_INFERENCE_METHOD,
        "n_jobs": EDITOR_N_JOBS,
        "resume": EDITOR_RESUME,
        "save_figures": EDITOR_SAVE_FIGURES,
        "show_figures": EDITOR_SHOW_FIGURES,
        "output_dir": EDITOR_OUTPUT_DIR,
        "campaign_overrides": dict(EDITOR_CAMPAIGN_OVERRIDES),
    }


def should_use_editor_settings(arguments=None, in_ipykernel=None):
    """Select editor mode for notebooks or direct launches without CLI options."""
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    if in_ipykernel is None:
        in_ipykernel = "ipykernel" in sys.modules
    return bool(in_ipykernel) or not arguments


def run_from_editor():
    """Run with the single user-editable settings block at the top of this file."""
    return run(**editor_run_options())


def validate_fcat_settings():
    """Validate the editable FCAT protocol constants before loading the data."""
    if isinstance(YEAR_MIN, bool) or not isinstance(YEAR_MIN, Integral):
        raise ValueError("YEAR_MIN must be an integer.")
    if (
        isinstance(OBSERVATION_END_YEAR, bool)
        or not isinstance(OBSERVATION_END_YEAR, Integral)
        or OBSERVATION_END_YEAR <= YEAR_MIN
    ):
        raise ValueError("OBSERVATION_END_YEAR must be an integer after YEAR_MIN.")
    for name, value in (
        ("MAGNITUDE_MIN", MAGNITUDE_MIN),
        ("EPS_PRIOR_VARIANCE", EPS_PRIOR_VARIANCE),
        ("EPS_PRIOR_LENGTH_SCALE_KM", EPS_PRIOR_LENGTH_SCALE_KM),
        ("INITIAL_GP.variance", INITIAL_GP.variance),
        ("INITIAL_GP.length_scale", INITIAL_GP.length_scale),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    if not MODELS or len(set(MODELS)) != len(MODELS):
        raise ValueError("MODELS must contain unique model identifiers.")
    if not SGCP_INFERENCE_METHODS:
        raise ValueError("SGCP_INFERENCE_METHODS cannot be empty.")
    if (
        isinstance(FINAL_INTENSITY_GRID_SIZE, bool)
        or not isinstance(FINAL_INTENSITY_GRID_SIZE, Integral)
        or FINAL_INTENSITY_GRID_SIZE < 20
    ):
        raise ValueError("FINAL_INTENSITY_GRID_SIZE must be an integer >= 20.")


def _validate_posterior_draw_budget(campaign, models, inference_method):
    """Prevent repeated Gibbs states when posterior draws exceed stored states."""
    gibbs_models = [
        model_name
        for model_name in models
        if _inference_for_model(model_name, inference_method).startswith("gibbs")
    ]
    if not gibbs_models:
        return
    for scope, iterations, chains, requested in (
        ("validation", campaign.cv_gibbs_iterations, campaign.cv_n_chains,
         campaign.score_posterior_draws),
        ("maps", campaign.full_gibbs_iterations, campaign.full_n_chains,
         campaign.map_posterior_draws),
    ):
        n_stored = (iterations + campaign.gibbs_thin - 1) // campaign.gibbs_thin
        warmup = int(campaign.gibbs_burn_in * iterations)
        burn = (warmup + campaign.gibbs_thin - 1) // campaign.gibbs_thin
        available = chains * (n_stored - burn)
        if requested > available:
            raise ValueError(
                f"{scope}: {requested} posterior draws requested but only {available} "
                "distinct post-warm-up Gibbs states are available. Increase iterations, "
                "reduce gibbs_thin, or request fewer draws."
            )


def _resolve_figure_display(show_figures):
    """Disable window display cleanly when Matplotlib uses a file-only backend."""
    file_backends = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}
    backend = str(plt.get_backend()).lower()
    can_show = "ipykernel" in sys.modules or backend not in file_backends
    if show_figures and not can_show:
        print(
            f"Figure display is unavailable with the {plt.get_backend()} backend; "
            "saved figures are unaffected."
        )
    return bool(show_figures and can_show)


def _print_run_settings(campaign, models, inference_method, n_jobs, data, splits, output):
    print("\n" + "=" * 78)
    print("FCAT-17 RANDOM AND TEMPORAL VALIDATION")
    print("=" * 78)
    print(f"Profile={campaign.name} | models={', '.join(MODEL_LABELS[m] for m in models)}")
    print(
        f"Catalogue: N={len(data['catalog'])}, magnitude >= {MAGNITUDE_MIN:g}, "
        f"observation=[{YEAR_MIN}, {OBSERVATION_END_YEAR}), "
        f"exposure={data['duration']:g} years"
    )
    for protocol in dict.fromkeys(split.protocol for split in splits):
        selected = [split for split in splits if split.protocol == protocol]
        first = selected[0]
        print(
            f"{first.label}: {len(selected)} split(s); "
            f"train/test exposure={first.training_duration:g}/{first.test_duration:g} yr"
        )
        if first.test_start is not None:
            print(f"  Temporal cutoff: decimal year {first.test_start:.4f}")
    print(
        f"Quadrature={campaign.quadrature_space_grid}x{campaign.quadrature_space_grid} "
        f"with exact zone intersections | score draws={campaign.score_posterior_draws} "
        f"| map draws={campaign.map_posterior_draws} | workers={effective_worker_count(n_jobs)}"
    )
    print(
        f"GP calibration={'on' if campaign.use_calibration else 'off'} "
        f"(training events only; {campaign.max_parallel_calibrations} simultaneous fit)"
    )
    if any(_inference_for_model(m, inference_method).startswith('gibbs') for m in models):
        print(
            f"Gibbs: validation={campaign.cv_n_chains} x {campaign.cv_gibbs_iterations}; "
            f"full data={campaign.full_n_chains} x {campaign.full_gibbs_iterations}; "
            f"burn-in={campaign.gibbs_burn_in:.0%}, thin={campaign.gibbs_thin}"
        )
        print(
            f"MALA: initial step={campaign.mala_initial_step:g}, "
            f"target={campaign.mala_target_acceptance:.1%}; "
            "adaptation during burn-in only"
        )
    if 'kde' in models:
        print("KDE: Silverman bandwidth with observation-domain normalization")
    print(f"Output directory: {output}")


_CHECKPOINT_SOURCES = (
    REPO_ROOT / "experiments/exp_spinh/run_fcat17_experiment.py",
    REPO_ROOT / "experiments/exp_spinh/fcat_settings.py",
    REPO_ROOT / "experiments/exp_spinh/fcat_validation.py",
    REPO_ROOT / "experiments/exp_spinh/fcat_runner_utils.py",
    REPO_ROOT / "experiments/exp_spinh/runner_utils.py",
    REPO_ROOT / "package",
    REPO_ROOT / "data",
    REPO_ROOT / "spatial",
    REPO_ROOT / "experiments" / "exp_ssgc" / "deliverable_utils.py",
)


def _print_run_summary(records, summary, full_fit_records, output):
    print("\n" + "-" * 78)
    print("VALIDATION SUMMARY (higher predictive score is better)")
    print(f"{'Protocol':<18} {'Model':<25} {'Splits':>7} {'Score/event':>12} {'SD':>9}")
    for row in summary:
        print(
            f"{row['protocol']:<18} {row['model_label']:<25} "
            f"{row['n_completed']}/{row['n_expected']:<5} "
            f"{row['predictive_log_score_per_event']:>12.4f} "
            f"{row['score_sd']:>9.4f}"
        )
    mala = [r for r in [*records, *full_fit_records]
            if np.isfinite(r.get('mala_production_acceptance_mean', np.nan))]
    if mala:
        print("\nPost-warm-up MALA acceptance")
        for model in dict.fromkeys(r['model'] for r in mala):
            rates = np.asarray([r['mala_production_acceptance_mean']
                                for r in mala if r['model'] == model])
            print(f"{MODEL_LABELS[model]:<25} mean={rates.mean():.1%}, "
                  f"range={rates.min():.1%}-{rates.max():.1%}")
    failures = [r for r in [*records, *full_fit_records] if r.get('status') != 'ok']
    for record in failures:
        print(f"Incomplete: {record.get('protocol', 'full')} / {record['model']}: "
              f"{record.get('error_message', record['status'])}")
    if failures:
        print("Partial outputs are preserved; failed fits will be retried on resume.")
    else:
        print("All requested fits completed.")
    print(f"Results: {output}")


def resolve_use_case_path():
    for candidate in (REPO_ROOT / "use_case", REPO_ROOT.parent / "use_case"):
        if (candidate / "catalog.csv").is_file():
            return candidate
    raise FileNotFoundError("Could not find the bundled FCAT-17 use_case directory.")


def project_coordinates(longitude, latitude):
    longitude, latitude = np.broadcast_arrays(
        np.asarray(longitude, dtype=float),
        np.asarray(latitude, dtype=float),
    )
    shape = longitude.shape
    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", "EPSG:2154", always_xy=True
    )
    if longitude.size == 1:
        x, y = transformer.transform(
            float(longitude.reshape(-1)[0]),
            float(latitude.reshape(-1)[0]),
        )
    else:
        x, y = transformer.transform(
            longitude.reshape(-1).tolist(),
            latitude.reshape(-1).tolist(),
        )
    return (
        np.asarray(x, dtype=float).reshape(shape) * 1e-3,
        np.asarray(y, dtype=float).reshape(shape) * 1e-3,
    )


def load_coastlines(path):
    coordinates = np.loadtxt(path)
    separators = np.where(np.any(~np.isfinite(coordinates), axis=1))[0]
    coastlines = []
    start = 0
    for stop in np.append(separators, len(coordinates)):
        segment = coordinates[start:stop]
        if len(segment):
            x, y = project_coordinates(segment[:, 0], segment[:, 1])
            coastlines.append(np.vstack((x, y)))
        start = stop + 1
    return coastlines


def load_fcat17():
    path = resolve_use_case_path()
    frame = pd.read_csv(path / "catalog.csv")
    frame = frame[
        (frame["year"] >= YEAR_MIN)
        & (frame["year"] < OBSERVATION_END_YEAR)
        & (frame["magnitude"] >= MAGNITUDE_MIN)
    ].copy()
    x, y = project_coordinates(frame["longitude"], frame["latitude"])
    frame["x_km"] = x
    frame["y_km"] = y

    domain_frame = pd.read_csv(path / "domaines_xy.csv")
    zones = []
    names = []
    for name, group in domain_frame.groupby("CODE_GTR", sort=False):
        zone_x, zone_y = project_coordinates(group["X"], group["Y"])
        polygon = Polygon(np.column_stack([zone_x, zone_y]))
        polygon = polygon if polygon.is_valid else polygon.buffer(0)
        zones.append(polygon.buffer(-1e-5))
        names.append(str(name))
    union = unary_union(zones)
    inside = points_in_geometry(frame[["x_km", "y_km"]].to_numpy(), union)
    frame = frame.loc[inside].copy()
    dates = pd.to_datetime(frame[["year", "month", "day"]], errors="raise")
    year_starts = pd.to_datetime(frame["year"].astype(str) + "-01-01")
    next_years = pd.to_datetime((frame["year"] + 1).astype(str) + "-01-01")
    frame["time_years"] = frame["year"] - YEAR_MIN + (
        (dates - year_starts) / (next_years - year_starts)
    )
    frame = frame.sort_values("time_years", kind="stable").reset_index(drop=True)
    duration = float(OBSERVATION_END_YEAR - YEAR_MIN)
    catalog = EventCatalog(
        t=frame["time_years"].to_numpy(dtype=float),
        x=frame["x_km"].to_numpy(),
        y=frame["y_km"].to_numpy(),
        magnitudes=frame["magnitude"].to_numpy(),
    )
    bounds = (
        (float(union.bounds[0]), float(union.bounds[2])),
        (float(union.bounds[1]), float(union.bounds[3])),
    )
    return {
        "catalog": catalog,
        "frame": frame,
        "zones": zones,
        "zone_names": names,
        "union": union,
        "duration": duration,
        "observation_start_year": YEAR_MIN,
        "observation_end_year": OBSERVATION_END_YEAR,
        "x_bounds": bounds[0],
        "y_bounds": bounds[1],
        "coastlines": load_coastlines(path / "coastlines_france.txt"),
    }


def _make_model(zones, data, gp_prior):
    return make_ssgc_model(
        zones,
        data["duration"],
        eps_prior_variance=EPS_PRIOR_VARIANCE,
        eps_prior_length_scale=EPS_PRIOR_LENGTH_SCALE_KM,
        gp_prior=gp_prior,
        x_bounds=data["x_bounds"],
        y_bounds=data["y_bounds"],
        magnitude_min=MAGNITUDE_MIN,
        magnitude_max=float(data["catalog"].magnitudes.max()) + 0.1,
    )


def _ssgc_campaign(
    campaign,
    *,
    gibbs_iterations,
    posterior_draws,
    n_chains,
    retain_diagnostic_chains=False,
):
    """Adapt FCAT settings to the shared SSGC fitting interface."""
    adaptation_end = int(gibbs_iterations * campaign.gibbs_burn_in)
    return SimpleNamespace(
        n_chains=n_chains,
        gibbs_iterations=gibbs_iterations,
        gibbs_thin=campaign.gibbs_thin,
        gibbs_burn_in=campaign.gibbs_burn_in,
        mala_initial_step=campaign.mala_initial_step,
        mala_adaptation_start=campaign.mala_adaptation_start,
        mala_adaptation_end=adaptation_end,
        mala_target_acceptance=campaign.mala_target_acceptance,
        mala_adaptation_decay=campaign.mala_adaptation_decay,
        mala_precondition=campaign.mala_precondition,
        mala_acceptance_bounds=campaign.mala_acceptance_bounds,
        vi_iterations=campaign.vi_iterations,
        vi_tolerance=getattr(campaign, "vi_tolerance", 1e-5),
        evaluation_grid=campaign.quadrature_space_grid,
        quadrature_grid=campaign.quadrature_space_grid,
        posterior_draws=posterior_draws,
        exact_max_events=campaign.exact_max_events,
        track_peak_memory=False,
        retain_diagnostic_chains=retain_diagnostic_chains,
    )


def _inference_for_model(model_name, sgcp_inference_method):
    if model_name in SSGC_METHODS:
        return SSGC_METHODS[model_name]
    return "silverman_kde" if model_name == "kde" else sgcp_inference_method


def _calibrate_gp(training_catalog, training_zones, data, campaign, seed):
    if not campaign.use_calibration:
        return INITIAL_GP, 0.0, False, 0
    model = _make_model(training_zones, data, INITIAL_GP)
    with calibration_slot():
        started = time.perf_counter()
        try:
            prior = model.calibrate_gp_prior(
                training_catalog,
                rng_seed=seed,
                verbose=False,
            )
            return prior, time.perf_counter() - started, True, len(training_catalog)
        except Exception as error:
            raise RuntimeError(
                f"GP calibration failed on {len(training_catalog)} training events: {error}"
            ) from error


def _intensity_map_grid(data, grid_size=FINAL_INTENSITY_GRID_SIZE):
    """Build a regular display grid and retain points inside the FCAT domain."""
    x_values = np.linspace(*data["x_bounds"], int(grid_size))
    y_values = np.linspace(*data["y_bounds"], int(grid_size))
    grid_x, grid_y = np.meshgrid(x_values, y_values)
    all_xy = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    inside = points_in_geometry(all_xy, data["union"])
    if not np.any(inside):
        raise RuntimeError("The final FCAT intensity grid does not intersect the domain.")
    return x_values, y_values, inside.reshape(grid_x.shape), all_xy[inside]


def _fit_full_intensity(
    model_name,
    data,
    evaluation_xy,
    gp_prior,
    calibration_seconds,
    calibration_succeeded,
    calibration_n_events,
    campaign,
    inference_method,
    seed,
):
    """Refit one model to the complete catalogue for the final intensity map."""
    started = time.perf_counter()
    if model_name == "kde":
        kde = ExposureNormalizedKDE(
            data["catalog"], data["duration"],
            data["quadrature_xy"], data["quadrature_weights"],
        )
        estimate = np.exp(kde.log_intensity(evaluation_xy))
        return estimate, {
            "status": "ok",
            "model": model_name,
            "model_label": MODEL_LABELS[model_name],
            "inference_method": "silverman_kde",
            "n_events": len(data["catalog"]),
            "n_posterior_draws": 1,
            "runtime_seconds": time.perf_counter() - started,
            "gp_calibration_seconds": 0.0,
            "gp_calibration_succeeded": False,
            "gp_calibration_n_events": 0,
            **kde.diagnostics(),
        }, None

    zones = data["zones"] if model_name in SSGC_METHODS else [data["union"]]
    model = _make_model(zones, data, gp_prior)
    partition = DomainPartition.from_polygons(zones)
    domain_index = partition.locate(evaluation_xy[:, 0], evaluation_xy[:, 1])
    draws, diagnostics = fit_intensity_method(
        model,
        data["catalog"],
        _inference_for_model(model_name, inference_method),
        _ssgc_campaign(
            campaign,
            gibbs_iterations=campaign.full_gibbs_iterations,
            posterior_draws=campaign.map_posterior_draws,
            n_chains=campaign.full_n_chains,
            retain_diagnostic_chains=True,
        ),
        seed,
        evaluation_xy,
        domain_index=domain_index,
        return_log_intensity=False,
        show_progress=False,
    )
    diagnostics.pop("peak_memory_mb", None)
    eps_chains = diagnostics.pop("_eps_chains", None)
    eps_burn_index = diagnostics.pop("_eps_burn_index", None)
    if draws is None:
        return None, {
            "model": model_name,
            "model_label": MODEL_LABELS[model_name],
            **diagnostics,
        }, None
    inference_seconds = diagnostics.pop("runtime_seconds")
    chain_diagnostics = None
    if eps_chains is not None:
        chain_diagnostics = {
            "eps_chains": np.asarray(eps_chains, dtype=float),
            "burn_index": int(eps_burn_index),
            "thin": int(campaign.gibbs_thin),
        }
    return draws.mean(axis=1), {
        "status": "ok",
        "model": model_name,
        "model_label": MODEL_LABELS[model_name],
        "inference_method": _inference_for_model(model_name, inference_method),
        "n_events": len(data["catalog"]),
        "n_posterior_draws": draws.shape[1],
        "runtime_seconds": inference_seconds + calibration_seconds,
        "gp_calibration_seconds": calibration_seconds,
        "gp_calibration_succeeded": calibration_succeeded,
        "gp_calibration_n_events": calibration_n_events,
        "gp_variance": gp_prior.variance,
        "gp_length_scale": gp_prior.length_scale,
        **diagnostics,
    }, chain_diagnostics


def _full_intensity_task(model_name, data, campaign, inference_method, evaluation_xy):
    """Calibrate and fit one full-data map under its own model geometry."""
    try:
        if model_name == "kde":
            gp_prior, calibration_seconds, calibration_succeeded, calibration_n_events = (
                INITIAL_GP,
                0.0,
                False,
                0,
            )
        else:
            calibration_zones = (
                data["zones"] if model_name in SSGC_METHODS else [data["union"]]
            )
            (
                gp_prior,
                calibration_seconds,
                calibration_succeeded,
                calibration_n_events,
            ) = _calibrate_gp(
                data["catalog"],
                calibration_zones,
                data,
                campaign,
                seed=91_000,
            )
        estimate, record, chain_diagnostics = _fit_full_intensity(
            model_name,
            data,
            evaluation_xy,
            gp_prior,
            calibration_seconds,
            calibration_succeeded,
            calibration_n_events,
            campaign,
            inference_method,
            seed=92_000 + sum(map(ord, model_name)),
        )
    except Exception as error:
        estimate = None
        chain_diagnostics = None
        record = {
            "status": "error",
            "model": model_name,
            "model_label": MODEL_LABELS[model_name],
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
    return model_name, estimate, record, chain_diagnostics


def fit_full_intensity_maps(
    models,
    data,
    campaign,
    inference_method,
    evaluation_xy,
    *,
    n_jobs=1,
    checkpoint_dir=None,
    resume=True,
):
    """Fit selected models on all FCAT events and return their posterior means."""
    tasks = [
        (model_name, data, campaign, inference_method, evaluation_xy)
        for model_name in models
    ]
    results = parallel_map(
        _full_intensity_task,
        tasks,
        n_jobs,
        "FCAT-17 full-data maps",
        task_keys=list(models),
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        max_parallel_calibrations=campaign.max_parallel_calibrations,
        isolate_tasks=True,
        worker_memory_reservation_gib=2.0,
    )
    estimates = {
        model_name: np.asarray(estimate, dtype=float)
        for model_name, estimate, _, _ in results
        if estimate is not None
    }
    chain_diagnostics = {
        model_name: diagnostics
        for model_name, _, _, diagnostics in results
        if diagnostics is not None
    }
    return (
        estimates,
        [record for _, _, record, _ in results],
        chain_diagnostics,
    )


def _validation_task(model_name, split, data, campaign, inference_method):
    started = time.perf_counter()
    record = {
        'protocol': split.protocol, 'repeat': split.repeat, 'model': model_name,
        'protocol_label': split.label,
        'model_label': MODEL_LABELS[model_name],
        'inference_method': _inference_for_model(model_name, inference_method),
        'training_rate_fraction': split.training_rate_fraction,
        'n_train': int(split.training_mask.sum()), 'n_held_out': int(split.test_mask.sum()),
        'train_duration': split.training_duration, 'test_duration': split.test_duration,
    }
    try:
        catalog = training_catalog(data['catalog'], split)
        model_data = dict(data, catalog=catalog, duration=split.training_duration)
        test_xy = data['catalog'].xy[split.test_mask]
        evaluation_xy = np.vstack([test_xy, data['quadrature_xy']])
        seed = int(np.random.SeedSequence([
            campaign.validation_seed, 1 if split.protocol == 'random' else 2,
            split.repeat, sum(map(ord, model_name)),
        ]).generate_state(1)[0])
        if model_name == 'kde':
            kde = ExposureNormalizedKDE(
                catalog, split.training_duration,
                data['quadrature_xy'], data['quadrature_weights'],
            )
            log_draws = kde.log_intensity(evaluation_xy)[:, None]
            draws = np.exp(log_draws)
            diagnostics = kde.diagnostics()
        else:
            zones = data['zones'] if model_name in SSGC_METHODS else [data['union']]
            prior, seconds, calibrated, calibration_n = _calibrate_gp(
                catalog, zones, model_data, campaign, seed,
            )
            model = _make_model(zones, model_data, prior)
            draws, diagnostics = fit_intensity_method(
                model, catalog, _inference_for_model(model_name, inference_method),
                _ssgc_campaign(
                    campaign, gibbs_iterations=campaign.cv_gibbs_iterations,
                    posterior_draws=campaign.score_posterior_draws,
                    n_chains=campaign.cv_n_chains,
                ),
                seed, evaluation_xy, return_log_intensity=True, show_progress=False,
            )
            if draws is None:
                return {'record': {**record, **diagnostics}, 'zones': []}
            log_draws = diagnostics.pop('_log_intensity_draws')
            diagnostics.pop('peak_memory_mb', None)
            diagnostics.pop('runtime_seconds', None)
            diagnostics.update({
                'gp_calibration_seconds': seconds, 'gp_calibration_succeeded': calibrated,
                'gp_calibration_n_events': calibration_n,
                'gp_variance': prior.variance, 'gp_length_scale': prior.length_scale,
            })
        n_test = len(test_xy)
        score, zonal = predictive_scores(
            log_draws[:n_test], draws[n_test:], data['quadrature_weights'],
            split.test_duration, data['event_zones'][split.test_mask],
            data['quadrature_zones'], len(data['zones']),
        )
        zone_rows = [{
            'protocol': split.protocol, 'repeat': split.repeat,
            'protocol_label': split.label,
            'model': model_name, 'model_label': MODEL_LABELS[model_name],
            'zone': zone, 'zone_index': index,
            'observed_count': int(zonal['observed_count'][index]),
            'predicted_count': float(zonal['predicted_count'][index]),
            'expected_log_score': float(zonal['expected_log_score'][index]),
        } for index, zone in enumerate(data['zone_names'])]
        record.update({**diagnostics, **score, 'status': 'ok',
                       'runtime_seconds': time.perf_counter() - started})
        return {'record': record, 'zones': zone_rows}
    except Exception as error:
        record.update(status='error', error_type=type(error).__name__, error_message=str(error))
        return {'record': record, 'zones': []}


def _write_zone_outputs(records, models, output, save, show):
    if not records:
        return
    frame = pd.DataFrame(records)
    summary = frame.groupby(['protocol', 'model', 'model_label', 'zone', 'zone_index'],
                            sort=False, as_index=False).agg(
        n_completed=('repeat', 'size'),
        observed_count=('observed_count', 'mean'),
        predicted_count=('predicted_count', 'mean'),
        expected_log_score=('expected_log_score', 'mean'),
    )
    frame[['protocol', 'repeat', 'model', 'zone', 'observed_count', 'predicted_count']].to_csv(
        output / 'fcat17_zone_counts_raw.csv', index=False,
    )
    summary.drop(columns='expected_log_score').to_csv(output / 'fcat17_zone_counts.csv', index=False)
    appendix = output / 'appendix'
    appendix.mkdir(exist_ok=True)
    frame.to_csv(appendix / 'fcat17_expected_log_score_by_zone_raw.csv', index=False)
    pairs = []
    reference = 'ssgc_gibbs_sparse' if 'ssgc_gibbs_sparse' in models else models[0]
    for other in models:
        if other == reference:
            continue
        paired = frame[frame.model == reference].merge(
            frame[frame.model == other], on=['protocol', 'repeat', 'zone', 'zone_index'],
            suffixes=('_reference', '_other'), validate='one_to_one',
        )
        paired['expected_log_score_difference'] = (
            paired.expected_log_score_reference - paired.expected_log_score_other
        )
        for (protocol, zone, index), group in paired.groupby(
            ['protocol', 'zone', 'zone_index'], sort=False,
        ):
            differences = group.expected_log_score_difference.to_numpy()
            pairs.append({
                'protocol': protocol, 'zone': zone, 'zone_index': index,
                'protocol_label': group.protocol_label_reference.iloc[0],
                'reference_model': reference, 'other_model': other, 'n_pairs': len(group),
                'expected_log_score_difference': float(differences.mean()),
                'difference_sd': float(differences.std(ddof=1)) if len(group) > 1 else np.nan,
            })
    write_records(appendix / 'fcat17_zone_score_differences.csv', pairs)
    if save or show:
        _plot_observed_predicted(frame, models, output, save, show)
        _plot_zone_score_differences(pairs, output, save, show)


def _plot_observed_predicted(frame, models, output, save, show):
    protocols = list(dict.fromkeys(frame.protocol))
    figure, axes = plt.subplots(len(protocols), 1, figsize=(9.3, 3.6 * len(protocols)),
                               squeeze=False, sharex=True, layout='constrained')
    colors = ('#0072B2', '#D55E00', '#009E73', '#CC79A7')
    markers = ('o', '^', 's', 'D')
    for axis, protocol in zip(axes.ravel(), protocols):
        group = frame[frame.protocol == protocol]
        # Use the same successful repetitions for all models in this figure.
        common = set.intersection(*(set(group[group.model == m]['repeat']) for m in models))
        if not common:
            axis.text(0.5, 0.5, 'No complete paired split', transform=axis.transAxes, ha='center')
            continue
        averaged = group[group['repeat'].isin(common)].groupby(
            ['model', 'zone', 'zone_index'], sort=False, as_index=False,
        )[['observed_count', 'predicted_count']].mean()
        reference = averaged[averaged.model == models[0]].sort_values('zone_index')
        positions = np.arange(len(reference))
        offsets = np.linspace(-0.25, 0.25, len(models) + 1)
        axis.scatter(positions + offsets[0], reference.observed_count, s=55,
                     facecolor='white', edgecolor='black', linewidth=1.2,
                     label='Observed', zorder=4)
        for index, model in enumerate(models):
            selected = averaged[averaged.model == model].sort_values('zone_index')
            axis.scatter(positions + offsets[index + 1], selected.predicted_count,
                         color=colors[index % len(colors)], marker=markers[index % len(markers)],
                         s=45, alpha=0.9, label=MODEL_LABELS[model], zorder=3)
        label = group.protocol_label.iloc[0]
        peak = max(1.0, averaged[['observed_count', 'predicted_count']].to_numpy().max())
        axis.set(xlim=(-0.65, len(reference) - 0.35), ylim=(-0.04 * peak, 1.15 * peak),
                 ylabel='Test events per zone',
                 title=f"{label} ({len(common)} split{'s' if len(common) > 1 else ''})")
        axis.set_xticks(positions, reference.zone, fontsize=10)
        axis.grid(axis='y', alpha=0.15)
        axis.legend(frameon=False, fontsize=9, ncol=min(4, len(models) + 1), loc='upper right')
    axes.ravel()[-1].set_xlabel('Source zone')
    if save:
        _save_pdf(figure, output / 'fcat17_observed_predicted.pdf')
    if show:
        figure.show()
    else:
        plt.close(figure)


def _plot_zone_score_differences(records, output, save, show):
    if not records:
        return
    frame = pd.DataFrame(records)
    for protocol, group in frame.groupby('protocol', sort=False):
        comparisons = list(dict.fromkeys(group.other_model))
        figure, axes = plt.subplots(1, len(comparisons), figsize=(5.3 * len(comparisons), 6),
                                   squeeze=False, layout='constrained')
        for axis, other in zip(axes.ravel(), comparisons):
            selected = group[group.other_model == other].sort_values('zone_index')
            values = selected.expected_log_score_difference.to_numpy()
            axis.barh(selected.zone, values, color=np.where(values >= 0, '#0072B2', '#D55E00'))
            axis.axvline(0, color='0.4', linewidth=0.8)
            axis.invert_yaxis()
            reference = selected.reference_model.iloc[0]
            axis.set(title=f"{MODEL_LABELS[reference]} minus {MODEL_LABELS[other]}",
                     xlabel='Difference in posterior expected log score')
        figure.suptitle(group.protocol_label.iloc[0])
        if save:
            _save_pdf(figure, output / 'appendix' / f'fcat17_zone_score_differences_{protocol}.pdf')
        if show:
            figure.show()
        else:
            plt.close(figure)


def _summarize(records, models, splits):
    summary = []
    for protocol in dict.fromkeys(split.protocol for split in splits):
        label = next(split.label for split in splits if split.protocol == protocol)
        expected = sum(split.protocol == protocol for split in splits)
        for model in models:
            rows = [r for r in records if r['protocol'] == protocol
                    and r['model'] == model and r.get('status') == 'ok']
            scores = np.asarray([r['predictive_log_score_per_event'] for r in rows])
            summary.append({
                'protocol': protocol, 'protocol_label': label,
                'model': model, 'model_label': MODEL_LABELS[model],
                'n_completed': len(rows), 'n_expected': expected,
                'complete': len(rows) == expected,
                'predictive_log_score_per_event': float(scores.mean()) if len(scores) else np.nan,
                'score_sd': float(scores.std(ddof=1)) if len(scores) > 1 else np.nan,
                'runtime_seconds_mean': float(np.mean([r['runtime_seconds'] for r in rows]))
                if rows else np.nan,
            })
    return summary


def _paired_score_comparisons(records, models):
    successful = pd.DataFrame(r for r in records if r.get('status') == 'ok')
    if successful.empty:
        return []
    comparisons = []
    for protocol, group in successful.groupby('protocol', sort=False):
        for first, second in combinations(models, 2):
            paired = group[group.model == first].merge(
                group[group.model == second], on='repeat',
                suffixes=('_first', '_second'), validate='one_to_one',
            )
            if paired.empty:
                continue
            differences = (paired.predictive_log_score_per_event_first
                           - paired.predictive_log_score_per_event_second).to_numpy()
            comparisons.append({
                'protocol': protocol, 'first_model': first, 'second_model': second,
                'n_pairs': len(paired), 'first_model_wins': int(np.sum(differences > 0)),
                'mean_score_difference_per_event': float(differences.mean()),
                'sd_score_difference_per_event': float(differences.std(ddof=1))
                if len(differences) > 1 else np.nan,
            })
    return comparisons


def _mala_adaptation_records(validation_records, full_fit_records):
    """Return one compact MALA diagnostic row per Gibbs fit."""
    fields = (
        "mala_initial_step",
        "mala_step",
        "mala_chain_final_steps",
        "mala_adaptation_start",
        "mala_adaptation_end",
        "mala_target_acceptance",
        "mala_adaptation_decay",
        "mala_precondition",
        "mala_relative_steps",
        "mala_adaptation_acceptance_mean",
        "mala_adaptation_acceptance_rates",
        "mala_production_acceptance_mean",
        "mala_production_acceptance_min",
        "mala_production_acceptance_max",
        "mala_production_acceptance_rates",
        "mala_production_acceptance_ok",
    )
    rows = []
    for scope, records in (("validation", validation_records), ("full", full_fit_records)):
        for record in records:
            if "mala_step" not in record:
                continue
            row = {
                "scope": scope,
                "model": record["model"],
                "model_label": record["model_label"],
                "inference_method": record["inference_method"],
                "n_chains": record.get("n_chains"),
                "n_events": record.get("n_train", record.get("n_events")),
            }
            if scope == "validation":
                row.update({"protocol": record["protocol"], "repeat": record["repeat"]})
            row.update({field: record.get(field) for field in fields})
            rows.append(row)
    return rows


def _write_latex_table(path, summary):
    lines = [r'\begin{tabular}{llc}', r'\toprule',
             r'Validation & Model & $S_{\mathrm{test}}$ \\', r'\midrule']
    for row in summary:
        score = f"{row['predictive_log_score_per_event']:.3f}"
        if np.isfinite(row['score_sd']):
            score += rf" $\pm$ {row['score_sd']:.3f}"
        if not row['complete']:
            score += r' $^{\dagger}$'
        lines.append(f"{row['protocol_label']} & {row['model_label']} & {score} " + r'\\')
    lines.extend([r'\bottomrule', r'\end{tabular}',
                  '% +/- denotes SD across random splits, not a confidence interval.'])
    if any(not row['complete'] for row in summary):
        lines.append('% dagger: partial result; see n_completed and n_expected in the CSV.')
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _plot_domain_boundaries(
    axis,
    zones,
    *,
    color="#303030",
    linewidth=0.65,
    alpha=1.0,
):
    for zone in zones:
        geometries = list(zone.geoms) if hasattr(zone, "geoms") else [zone]
        for geometry in geometries:
            x, y = geometry.exterior.xy
            axis.plot(
                x,
                y,
                color=color,
                linewidth=linewidth,
                alpha=alpha,
            )


def _save_pdf(figure, path, *, contains_rasterized_artists=False):
    """Save vector artwork as PDF, rasterizing marked artists at 300 dpi."""
    destination = Path(path).with_suffix(".pdf")
    destination.parent.mkdir(parents=True, exist_ok=True)
    options = {
        "bbox_inches": "tight",
        "pad_inches": 0.08,
        "facecolor": figure.get_facecolor(),
        "transparent": False,
    }
    if contains_rasterized_artists:
        options["dpi"] = 300
    figure.savefig(destination, **options)
    return destination


def _acf(values, max_lag):
    """Return the empirical autocorrelation through ``max_lag``."""
    centered = np.asarray(values, dtype=float) - np.mean(values)
    denominator = float(np.dot(centered, centered))
    if denominator <= 0.0:
        return np.ones(max_lag + 1, dtype=float)
    return np.asarray(
        [
            np.dot(centered[: centered.size - lag], centered[lag:])
            / denominator
            for lag in range(max_lag + 1)
        ],
        dtype=float,
    )


def _diagnostic_parameter_labels(model_name, data, n_parameters):
    if model_name == "ssgc_gibbs_sparse" and len(data["zone_names"]) == n_parameters:
        return [rf"$\varepsilon_{{\mathrm{{{name}}}}}$" for name in data["zone_names"]]
    if n_parameters == 1:
        return [r"$\varepsilon$"]
    return [rf"$\varepsilon_{{{index + 1}}}$" for index in range(n_parameters)]


def _diagnostic_axes(n_parameters):
    n_columns = min(3, n_parameters)
    n_rows = int(np.ceil(n_parameters / n_columns))
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(4.4 * n_columns, 2.7 * n_rows + 0.8),
        squeeze=False,
    )
    axes = axes.ravel()
    for axis in axes[n_parameters:]:
        axis.set_visible(False)
    return figure, axes


def _finish_diagnostic_figure(figure, path, save, show):
    visible_axes = sum(axis.get_visible() for axis in figure.axes)
    top = 0.77 if visible_axes <= 3 else 0.87
    figure.subplots_adjust(
        left=0.07,
        right=0.985,
        bottom=0.08,
        top=top,
        hspace=0.38,
    )
    if save:
        _save_pdf(figure, path)
    if show:
        figure.show()
    else:
        plt.close(figure)


def _plot_full_gibbs_diagnostics(
    chain_diagnostics,
    data,
    output,
    save,
    show,
    *,
    max_lag=100,
):
    """Save full-data trace and ACF plots for the retained regional effects."""
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    for model_name, payload in chain_diagnostics.items():
        chains = np.asarray(payload["eps_chains"], dtype=float)
        if chains.ndim != 3 or not chains.size:
            continue
        burn = int(payload["burn_index"])
        thin = int(payload["thin"])
        n_chains, n_draws, n_parameters = chains.shape
        labels = _diagnostic_parameter_labels(model_name, data, n_parameters)
        iterations = 1 + np.arange(n_draws) * thin
        burn_iteration = burn * thin

        figure, axes = _diagnostic_axes(n_parameters)
        for parameter_index, (axis, label) in enumerate(zip(axes, labels)):
            for chain_index in range(n_chains):
                axis.plot(
                    iterations,
                    chains[chain_index, :, parameter_index],
                    color=colors[chain_index % len(colors)],
                    linewidth=0.65,
                    alpha=0.85,
                    label=f"Chain {chain_index + 1}",
                )
            axis.axvline(
                burn_iteration,
                color="black",
                linestyle="--",
                linewidth=0.9,
                label="Burn-in end",
            )
            axis.set_title(label)
            axis.set_xlabel("Iteration")
            axis.grid(alpha=0.18)
        handles, legend_labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=min(len(legend_labels), 4),
            frameon=False,
        )
        figure.suptitle(
            f"{MODEL_LABELS[model_name]}: full-data Gibbs traces",
            y=0.995,
        )
        _finish_diagnostic_figure(
            figure,
            output / f"fcat17_gibbs_traces_{model_name}.pdf",
            save,
            show,
        )

        post_burn = chains[:, burn:, :]
        lag = min(int(max_lag), post_burn.shape[1] - 1)
        if lag < 1:
            continue
        figure, axes = _diagnostic_axes(n_parameters)
        lags = np.arange(lag + 1)
        reference = 1.96 / np.sqrt(post_burn.shape[1])
        for parameter_index, (axis, label) in enumerate(zip(axes, labels)):
            for chain_index in range(n_chains):
                axis.plot(
                    lags,
                    _acf(post_burn[chain_index, :, parameter_index], lag),
                    color=colors[chain_index % len(colors)],
                    linewidth=1.0,
                    alpha=0.9,
                    label=f"Chain {chain_index + 1}",
                )
            axis.axhline(0.0, color="black", linewidth=0.7)
            axis.axhline(reference, color="0.45", linestyle=":", linewidth=0.8)
            axis.axhline(-reference, color="0.45", linestyle=":", linewidth=0.8)
            axis.set_ylim(-1.0, 1.0)
            axis.set_title(label)
            axis.set_xlabel(f"Stored-draw lag (thin={thin})")
            axis.grid(alpha=0.18)
        handles, legend_labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=min(len(legend_labels), 4),
            frameon=False,
        )
        figure.suptitle(
            f"{MODEL_LABELS[model_name]}: full-data Gibbs ACF",
            y=0.995,
        )
        _finish_diagnostic_figure(
            figure,
            output / f"fcat17_gibbs_acf_{model_name}.pdf",
            save,
            show,
        )


def _write_full_gibbs_chains(chain_diagnostics, output):
    """Persist compact chains so diagnostics can be rebuilt without refitting."""
    for model_name, payload in chain_diagnostics.items():
        np.savez_compressed(
            output / f"fcat17_gibbs_chains_{model_name}.npz",
            eps_chains=np.asarray(payload["eps_chains"], dtype=float),
            burn_index=np.asarray(payload["burn_index"], dtype=int),
            thin=np.asarray(payload["thin"], dtype=int),
        )


def _full_gibbs_diagnostic_records(chain_diagnostics, data):
    """Return parameter-level diagnostics for the saved full-data chains."""
    partition = DomainPartition.from_polygons(data["zones"])
    zone_index = partition.locate(
        data["catalog"].x,
        data["catalog"].y,
    )
    zone_counts = np.bincount(
        zone_index[zone_index >= 0],
        minlength=len(data["zones"]),
    )
    records = []
    for model_name, payload in chain_diagnostics.items():
        chains = np.asarray(payload["eps_chains"], dtype=float)
        if chains.ndim != 3 or not chains.size:
            continue
        burn = int(payload["burn_index"])
        post_burn = chains[:, burn:, :]
        diagnostics = mcmc_diagnostics(post_burn)
        if model_name == "ssgc_gibbs_sparse":
            parameter_names = data["zone_names"]
            event_counts = zone_counts
        else:
            parameter_names = ["global"]
            event_counts = np.asarray([len(data["catalog"])], dtype=int)
        for index, parameter_name in enumerate(parameter_names):
            values = post_burn[:, :, index]
            records.append(
                {
                    "model": model_name,
                    "model_label": MODEL_LABELS[model_name],
                    "parameter": parameter_name,
                    "n_events": int(event_counts[index]),
                    "n_chains": values.shape[0],
                    "draws_per_chain": values.shape[1],
                    "posterior_mean": float(np.mean(values)),
                    "posterior_sd": float(np.std(values, ddof=1)),
                    "rhat": float(diagnostics["rhat"][index]),
                    "ess_bulk": float(diagnostics["ess_bulk"][index]),
                    "ess_tail": float(diagnostics["ess_tail"][index]),
                    "mcse_mean": float(diagnostics["mcse_mean"][index]),
                    "chain_means": ";".join(
                        f"{value:.8g}" for value in np.mean(values, axis=1)
                    ),
                }
            )
    return records


def _plot_catalogue(data, output, save, show):
    figure, axis = plt.subplots(figsize=(7.2, 7.2), layout="constrained")
    _plot_domain_boundaries(axis, data["zones"])
    scatter = axis.scatter(
        data["catalog"].x,
        data["catalog"].y,
        s=7,
        c=data["catalog"].magnitudes,
        cmap="viridis",
        alpha=0.55,
        linewidths=0,
        rasterized=True,
    )
    axis.set(xlabel="x (km)", ylabel="y (km)", aspect="equal")
    axis.set_title("FCAT-17 and French seismotectonic source domains")
    figure.colorbar(
        scatter,
        ax=axis,
        label="Magnitude",
        shrink=0.60,
        pad=0.025,
        aspect=18,
    )
    if save:
        _save_pdf(
            figure,
            output / "fcat17_catalogue.pdf",
            contains_rasterized_artists=True,
        )
    if show:
        figure.show()
    else:
        plt.close(figure)


def _plot_intensity_maps(
    data,
    evaluation_xy,
    estimates,
    output,
    save,
    show,
):
    if not estimates:
        return
    all_values = np.concatenate(list(estimates.values()))
    lower = 0.0
    upper = float(np.max(all_values))
    if not np.isfinite(upper) or upper <= lower:
        upper = 1.0
    color_map = plt.get_cmap("viridis")
    x_values, y_values, inside, map_xy = _intensity_map_grid(data)
    if not np.array_equal(evaluation_xy, map_xy):
        raise ValueError("Intensity maps must use the shared regular display grid.")

    model_names = list(estimates)
    ncols = 2 if len(model_names) == 4 else len(model_names)
    nrows = int(np.ceil(len(model_names) / ncols))
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.4 * ncols + 1.0, 4.6 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    axes = axes.ravel()
    for axis in axes[len(model_names):]:
        axis.set_visible(False)
    image = None
    for index, (axis, model_name) in enumerate(zip(axes, model_names)):
        surface = np.zeros(inside.shape)
        surface[inside] = estimates[model_name]
        image = axis.pcolormesh(
            x_values, y_values, surface,
            vmin=lower,
            vmax=upper,
            cmap=color_map,
            shading="auto",
            rasterized=True,
        )
        magnitude = data["catalog"].magnitudes
        axis.scatter(
            data["catalog"].x,
            data["catalog"].y,
            s=2.0 + 3.0 * (magnitude - MAGNITUDE_MIN),
            color="#c62828",
            alpha=0.30,
            linewidths=0,
            rasterized=True,
            zorder=3,
        )
        _plot_domain_boundaries(
            axis,
            data["zones"],
            color="white",
            linewidth=0.4,
            alpha=0.28,
        )
        for coastline in data["coastlines"]:
            axis.plot(
                coastline[0],
                coastline[1],
                color="white",
                linewidth=1.0,
                alpha=0.95,
            )
        axis.set(
            title=MODEL_LABELS[model_name],
            xlabel="x (km)",
            xlim=data["x_bounds"],
            ylim=data["y_bounds"],
            aspect="equal",
        )
        axis.set_facecolor(color_map(0.0))
        if index == 0:
            axis.set_ylabel("y (km)")
    figure.colorbar(
        image,
        ax=list(axes[:len(model_names)]),
        label=r"Estimated annual intensity (events km$^{-2}$ yr$^{-1}$)",
        shrink=0.84,
        pad=0.02,
    )
    if save:
        _save_pdf(
            figure,
            output / "fcat17_fitted_intensities.pdf",
            contains_rasterized_artists=True,
        )
    if show:
        figure.show()
    else:
        plt.close(figure)


def run(
    profile="smoke", models=DEFAULT_MODELS, inference_method="gibbs_sparse", *,
    protocols=("random", "temporal"), n_jobs=None, resume=True,
    save_figures=True, show_figures=False, campaign_overrides=None, output_dir=None,
):
    validate_fcat_settings()
    models = tuple(models)
    protocols = tuple(protocols)
    unknown = set(models) - set(MODELS)
    if not models or unknown or len(set(models)) != len(models):
        raise ValueError(f"Select unique valid models; unknown={sorted(unknown)}.")
    if inference_method not in SGCP_INFERENCE_METHODS:
        raise ValueError(f"Unknown inference method {inference_method!r}.")
    if not all(isinstance(value, bool) for value in (resume, save_figures, show_figures)):
        raise ValueError("resume, save_figures and show_figures must be boolean.")
    n_jobs = resolve_n_jobs(profile, n_jobs, worker_memory_reservation_gib=2.0)
    display = _resolve_figure_display(show_figures)
    campaign = configure_campaign(profile, **(campaign_overrides or {}))
    _validate_posterior_draw_budget(campaign, models, inference_method)
    data = load_fcat17()
    splits = validation_splits(data, campaign, protocols)
    data['quadrature_xy'], data['quadrature_weights'], data['quadrature_zones'] = (
        zonal_quadrature(data, campaign.quadrature_space_grid)
    )
    data['event_zones'] = DomainPartition.from_polygons(data['zones']).locate(
        data['catalog'].x, data['catalog'].y
    )
    if np.any(data['event_zones'] < 0):
        raise RuntimeError("Every retained event must have an original source-zone label.")
    output = Path(output_dir) if output_dir is not None else RESULTS_ROOT / profile / 'fcat17'
    output.mkdir(parents=True, exist_ok=True)
    _print_run_settings(campaign, models, inference_method, n_jobs, data, splits, output)
    settings = {
        'models': models, 'protocols': tuple(protocols), 'inference_method': inference_method,
        'year_min': YEAR_MIN, 'observation_end_year': OBSERVATION_END_YEAR,
        'magnitude_min': MAGNITUDE_MIN, 'eps_prior_variance': EPS_PRIOR_VARIANCE,
        'eps_prior_length_scale_km': EPS_PRIOR_LENGTH_SCALE_KM,
        'initial_gp_variance': INITIAL_GP.variance,
        'initial_gp_length_scale_km': INITIAL_GP.length_scale,
    }
    write_campaign(output / 'fcat17_campaign.json', campaign, {
        **settings, 'n_events': len(data['catalog']), 'duration': data['duration'],
        'n_jobs': n_jobs, 'effective_workers': effective_worker_count(n_jobs),
        'final_intensity_grid_size': FINAL_INTENSITY_GRID_SIZE,
    })
    split_rows = []
    for split in splits:
        for index, zone in enumerate(data['zone_names']):
            split_rows.append({
                'protocol': split.protocol, 'repeat': split.repeat, 'zone': zone,
                'n_train': int(np.sum(split.training_mask & (data['event_zones'] == index))),
                'n_test': int(np.sum(split.test_mask & (data['event_zones'] == index))),
                'train_duration': split.training_duration, 'test_duration': split.test_duration,
                'test_start_year': split.test_start,
            })
    write_records(output / 'fcat17_splits.csv', split_rows)
    sources = (*_CHECKPOINT_SOURCES, *(resolve_use_case_path() / name
               for name in ('catalog.csv', 'domaines_xy.csv', 'coastlines_france.txt')))
    budget = asdict(campaign)
    for key in ('map_posterior_draws', 'full_n_chains', 'full_gibbs_iterations'):
        budget.pop(key)
    checkpoint_dir = checkpoint_directory(
        output, 'fcat17_validation', budget, settings=settings, source_paths=sources,
    )
    results = parallel_map(
        _validation_task,
        [(model, split, data, campaign, inference_method) for split in splits for model in models],
        n_jobs, 'FCAT-17 validation',
        task_keys=[(split.protocol, split.repeat, model) for split in splits for model in models],
        checkpoint_dir=checkpoint_dir, resume=resume,
        max_parallel_calibrations=campaign.max_parallel_calibrations,
        isolate_tasks=True, worker_memory_reservation_gib=2.0,
    )
    records = [payload['record'] for payload in results]
    zone_records = [row for payload in results for row in payload['zones']]
    summary = _summarize(records, models, splits)
    write_records(output / 'fcat17_validation_raw.csv', records)
    write_records(output / 'fcat17_table.csv', summary)
    write_records(output / 'fcat17_paired_comparison.csv',
                  _paired_score_comparisons(records, models))
    _write_latex_table(output / 'fcat17_table.tex', summary)
    _write_zone_outputs(zone_records, models, output, save_figures, display)
    # Validation figures/tables are persisted before the costly full-data refits.
    if save_figures or display:
        _plot_catalogue(data, output, save_figures, display)

    _, _, _, map_xy = _intensity_map_grid(data)
    map_budget = asdict(campaign)
    for key in ('score_posterior_draws', 'cv_n_chains', 'cv_gibbs_iterations',
                'random_replicates', 'validation_seed', 'temporal_train_fraction'):
        map_budget.pop(key)
    map_settings = {key: value for key, value in settings.items() if key != 'protocols'}
    map_settings['grid_size'] = FINAL_INTENSITY_GRID_SIZE
    map_checkpoints = checkpoint_directory(
        output, 'fcat17_full_maps', map_budget, settings=map_settings, source_paths=sources,
    )
    estimates, full_records, chains = fit_full_intensity_maps(
        models, data, campaign, inference_method, map_xy,
        n_jobs=n_jobs, checkpoint_dir=map_checkpoints, resume=resume,
    )
    write_records(output / 'fcat17_full_fit.csv', full_records)
    _write_full_gibbs_chains(chains, output)
    write_records(output / 'fcat17_gibbs_parameter_diagnostics.csv',
                  _full_gibbs_diagnostic_records(chains, data))
    write_records(output / 'fcat17_mala_adaptation.csv',
                  _mala_adaptation_records(records, full_records))
    map_frame = pd.DataFrame({'x_km': map_xy[:, 0], 'y_km': map_xy[:, 1]})
    for model_name, estimate in estimates.items():
        map_frame[f'intensity_{model_name}'] = estimate
    map_frame.to_csv(output / 'fcat17_full_intensity_grid.csv', index=False)
    if save_figures or display:
        _plot_intensity_maps(data, map_xy, estimates, output, save_figures, display)
        _plot_full_gibbs_diagnostics(chains, data, output, save_figures, display)
    _print_run_summary(records, summary, full_records, output)
    if display and 'ipykernel' not in sys.modules:
        plt.show()
    return records, summary


# %% Command-line interface
class _HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    pass


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=_HelpFormatter,
    )
    selection = parser.add_argument_group("model selection")
    selection.add_argument(
        "--profile",
        choices=CAMPAIGNS,
        default="smoke",
        help="Numerical budget profile.",
    )
    selection.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        default=list(DEFAULT_MODELS),
        help="Models included in the random and temporal comparison.",
    )
    selection.add_argument(
        "--inference-method",
        choices=SGCP_INFERENCE_METHODS,
        default="gibbs_sparse",
        help="Inference backend for SGCP; SSGC Gibbs/VI variants are fixed.",
    )
    selection.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Simultaneous jobs (default: 1). Test 2 before requesting more; RAM is shared.",
    )
    selection.add_argument("--output-dir", type=Path, default=None)
    selection.add_argument("--protocols", nargs="+", choices=PROTOCOL_LABELS,
                           default=list(PROTOCOL_LABELS))

    budget = parser.add_argument_group("campaign budget overrides")
    budget.add_argument("--random-replicates", type=int, default=None)
    budget.add_argument("--validation-seed", type=int, default=None)
    budget.add_argument("--temporal-train-fraction", type=float, default=None)
    budget.add_argument("--cv-n-chains", type=int, default=None)
    budget.add_argument("--full-n-chains", type=int, default=None)
    budget.add_argument("--cv-gibbs-iterations", type=int, default=None)
    budget.add_argument("--full-gibbs-iterations", type=int, default=None)
    budget.add_argument("--gibbs-thin", type=int, default=None)
    budget.add_argument("--gibbs-burn-in", type=float, default=None)
    budget.add_argument("--mala-initial-step", type=float, default=None)
    budget.add_argument("--mala-adaptation-start", type=int, default=None)
    budget.add_argument("--mala-target-acceptance", type=float, default=None)
    budget.add_argument("--mala-adaptation-decay", type=float, default=None)
    budget.add_argument("--vi-iterations", type=int, default=None)
    budget.add_argument("--vi-tolerance", type=float, default=None)
    budget.add_argument("--quadrature-space-grid", type=int, default=None)
    budget.add_argument("--score-posterior-draws", type=int, default=None)
    budget.add_argument("--map-posterior-draws", type=int, default=None)
    budget.add_argument("--max-parallel-calibrations", type=int, default=None)
    budget.add_argument("--exact-max-events", type=int, default=None)
    budget.add_argument(
        "--no-calibration",
        action="store_true",
        help="Disable empirical GP-prior calibration.",
    )

    figures = parser.add_argument_group("figures")
    figures.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore compatible task checkpoints and recompute every task.",
    )
    figures.add_argument(
        "--no-figures",
        action="store_true",
        help="Do not generate or save figures.",
    )
    figures.add_argument(
        "--show-figures",
        action="store_true",
        help="Open figure windows in addition to saving figures.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    overrides = {
        "random_replicates": args.random_replicates,
        "validation_seed": args.validation_seed,
        "temporal_train_fraction": args.temporal_train_fraction,
        "cv_n_chains": args.cv_n_chains,
        "full_n_chains": args.full_n_chains,
        "cv_gibbs_iterations": args.cv_gibbs_iterations,
        "full_gibbs_iterations": args.full_gibbs_iterations,
        "gibbs_thin": args.gibbs_thin,
        "gibbs_burn_in": args.gibbs_burn_in,
        "mala_initial_step": args.mala_initial_step,
        "mala_adaptation_start": args.mala_adaptation_start,
        "mala_target_acceptance": args.mala_target_acceptance,
        "mala_adaptation_decay": args.mala_adaptation_decay,
        "vi_iterations": args.vi_iterations,
        "vi_tolerance": args.vi_tolerance,
        "quadrature_space_grid": args.quadrature_space_grid,
        "score_posterior_draws": args.score_posterior_draws,
        "map_posterior_draws": args.map_posterior_draws,
        "max_parallel_calibrations": args.max_parallel_calibrations,
        "exact_max_events": args.exact_max_events,
        "use_calibration": False if args.no_calibration else None,
    }
    return run(
        profile=args.profile,
        models=args.models,
        protocols=args.protocols,
        inference_method=args.inference_method,
        n_jobs=args.n_jobs,
        resume=not args.no_resume,
        save_figures=not args.no_figures,
        show_figures=args.show_figures,
        campaign_overrides=overrides,
        output_dir=args.output_dir,
    )


# %% Run the file
if __name__ == "__main__":
    if should_use_editor_settings():
        run_from_editor()
    else:
        main()

# %%
