"""FCAT-17 spatial block cross-validation for the SPIN-H numerical tests.

FCAT-17 is treated as a declustered background catalogue.  The script compares
the SSGC component of SPIN-H, its zoneless SGCP special case and a KDE whose
bandwidth and normalization are estimated from each training fold only.

For an editor workflow, change the ``EDITOR SETTINGS`` block below and run the
file without arguments.  Command-line arguments remain available for batch
execution and reproducible campaigns.
"""

from __future__ import annotations

# %% Imports
import argparse
import sys
import time
import warnings
from numbers import Integral
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
from scipy.special import logsumexp
from scipy.stats import gaussian_kde
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import EventCatalog
from experiments.exp_spinh.test_utils import (
    CAMPAIGNS,
    RESULTS_ROOT,
    configure_campaign,
    mean_confidence_interval,
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
    area_weighted_grid,
    fit_intensity_method,
    make_model as make_ssgc_model,
    points_in_geometry,
)
from spatial import DomainPartition


# %% ========================================================================
# FCAT-17 SCIENTIFIC SETTINGS
# =============================================================================
# These values define the catalogue selection and SSGC prior used by every
# execution mode.  Change them only when changing the scientific protocol.

YEAR_MIN = 1965
MAGNITUDE_MIN = 3.0
EPS_PRIOR_VARIANCE = 10.0
EPS_PRIOR_LENGTH_SCALE_KM = 3.0
INITIAL_GP = GPParameters(variance=2.0, length_scale=50.0)
FINAL_INTENSITY_GRID_SIZE = 90
MODELS = ("ssgc", "sgcp", "kde")
MODEL_LABELS = {
    "ssgc": "SSGC (French partition)",
    "sgcp": "SGCP (J=1)",
    "kde": "KDE",
}
SSGC_INFERENCE_METHODS = (
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

EDITOR_PROFILE = "smoke"                    # "smoke" or "full"
EDITOR_MODELS = tuple(MODELS)                # any subset of MODELS
EDITOR_INFERENCE_METHOD = "vi_sparse"        # one of SSGC_INFERENCE_METHODS
EDITOR_N_JOBS = None                         # None: one fit at a time, for both profiles
EDITOR_RESUME = True                         # reuse completed task checkpoints
EDITOR_SAVE_FIGURES = True
EDITOR_SHOW_FIGURES = True

EDITOR_CAMPAIGN_OVERRIDES = {
    "n_chains": None,
    "gibbs_iterations": None,
    "gibbs_thin": None,
    "vi_iterations": None,
    "evaluation_space_grid": None,
    "quadrature_space_grid": None,
    "posterior_draws": None,
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
        "inference_method": EDITOR_INFERENCE_METHOD,
        "n_jobs": EDITOR_N_JOBS,
        "resume": EDITOR_RESUME,
        "save_figures": EDITOR_SAVE_FIGURES,
        "show_figures": EDITOR_SHOW_FIGURES,
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
    if not SSGC_INFERENCE_METHODS:
        raise ValueError("SSGC_INFERENCE_METHODS cannot be empty.")
    if (
        isinstance(FINAL_INTENSITY_GRID_SIZE, bool)
        or not isinstance(FINAL_INTENSITY_GRID_SIZE, Integral)
        or FINAL_INTENSITY_GRID_SIZE < 20
    ):
        raise ValueError("FINAL_INTENSITY_GRID_SIZE must be an integer >= 20.")


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


def _print_run_settings(
    campaign, models, inference_method, n_jobs, data, folds, output
):
    print("\n" + "=" * 78)
    print("SPIN-H FCAT-17 TEST")
    print("=" * 78)
    print(
        f"Profile={campaign.name} | models={','.join(model.upper() for model in models)} | "
        f"SSGC inference={inference_method}"
    )
    print(
        f"Catalogue: N={len(data['catalog'])}, years >= {YEAR_MIN}, "
        f"magnitude >= {MAGNITUDE_MIN:g}, duration={data['duration']:.1f} years"
    )
    print(
        f"Spatial folds={len(folds)} | posterior draws={campaign.posterior_draws} | "
        f"GP calibration={'on' if campaign.use_calibration else 'off'} | "
        f"workers={effective_worker_count(n_jobs)} (n_jobs={n_jobs})"
    )
    if campaign.use_calibration:
        print(
            "GP calibration: complete training catalogue, "
            f"at most {campaign.max_parallel_calibrations} simultaneous fit(s)"
        )
    print(f"Output directory: {output}")
    if campaign.name == "full":
        print("Full profile selected: this campaign can require substantial compute time.")


_CHECKPOINT_SOURCES = (
    Path(__file__),
    Path(__file__).with_name("test_utils.py"),
    Path(__file__).with_name("runner_utils.py"),
    REPO_ROOT / "package",
    REPO_ROOT / "experiments" / "exp_ssgc" / "deliverable_utils.py",
)


def _print_run_summary(records, summary, full_fit_records, output):
    print("\n" + "-" * 78)
    print("RUN SUMMARY")
    print(f"{'Model':<30} {'Folds':>5} {'Score/event':>12} {'Time (s)':>10}")
    for row in summary:
        print(
            f"{row['model_label']:<30} {row['n_folds']:>5} "
            f"{row['predictive_log_score_per_event']:>12.3f} "
            f"{row['runtime_seconds']:>10.2f}"
        )
    print("\nFull-data intensity maps")
    print(f"{'Model':<30} {'Status':<22} {'Time (s)':>10}")
    for record in full_fit_records:
        runtime = float(record.get("runtime_seconds", np.nan))
        print(
            f"{record['model_label']:<30} {record.get('status', 'unknown'):<22} "
            f"{runtime:>10.2f}"
        )
    failures = [
        record
        for record in [*records, *full_fit_records]
        if record.get("status") != "ok"
    ]
    if failures:
        print(f"\nWarnings: {len(failures)} fit(s) did not complete.")
        for record in failures[:10]:
            print(
                f"  repeat={record.get('repeat', '?')}, fold={record.get('fold', '?')}, "
                f"model={record.get('model', '?')}: "
                f"{record.get('error_message', record.get('status', 'unknown'))}"
            )
    else:
        print("\nAll requested fits completed successfully.")
    print(f"Results written to: {output}")
    print("-" * 78)


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
        (frame["year"] >= YEAR_MIN) & (frame["magnitude"] >= MAGNITUDE_MIN)
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
    duration = float(frame["year"].max() - frame["year"].min())
    catalog = EventCatalog(
        t=np.zeros(len(frame), dtype=float),
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
        "x_bounds": bounds[0],
        "y_bounds": bounds[1],
        "coastlines": load_coastlines(path / "coastlines_france.txt"),
    }


def _limit_held_fraction(held_geometry, zones, fold_index, maximum=0.65):
    retained = []
    for zone in zones:
        held_part = zone.intersection(held_geometry)
        if held_part.is_empty:
            continue
        if held_part.area > maximum * zone.area:
            xmin, ymin, xmax, ymax = zone.bounds
            split = (
                box(xmin, ymin, zone.centroid.x, ymax)
                if fold_index % 2 == 0
                else box(xmin, ymin, xmax, zone.centroid.y)
            )
            limited = held_part.intersection(split)
            if not limited.is_empty:
                held_part = limited
        retained.append(held_part)
    return unary_union(retained)


def spatial_block_folds(data, profile):
    if profile == "smoke":
        n_side, n_folds, n_repeats = 3, 3, 1
    else:
        n_side, n_folds, n_repeats = 5, 5, 3
    x_edges = np.linspace(*data["x_bounds"], n_side + 1)
    y_edges = np.linspace(*data["y_bounds"], n_side + 1)
    folds = []
    for repeat in range(n_repeats):
        buckets = [[] for _ in range(n_folds)]
        multiplier = repeat + 2
        for ix, (left, right) in enumerate(zip(x_edges[:-1], x_edges[1:])):
            for iy, (lower, upper) in enumerate(zip(y_edges[:-1], y_edges[1:])):
                cell = box(left, lower, right, upper).intersection(data["union"])
                if not cell.is_empty and cell.area > 0.0:
                    buckets[(ix + multiplier * iy + repeat) % n_folds].append(cell)
        for fold, cells in enumerate(buckets):
            held = _limit_held_fraction(
                unary_union(cells), data["zones"], fold + repeat
            )
            if not held.is_empty and held.area > 0.0:
                folds.append((repeat, fold, held))
    return folds[:2] if profile == "smoke" else folds


def _training_zones(zones, held_geometry):
    clipped = []
    for zone in zones:
        training_zone = zone.difference(held_geometry)
        if training_zone.is_empty or training_zone.area <= 0.0:
            radius = max(np.sqrt(zone.area) * 1e-6, 1e-6)
            training_zone = zone.representative_point().buffer(radius).intersection(zone)
        clipped.append(training_zone)
    return clipped


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


def _ssgc_campaign(campaign):
    """Adapt the SPIN-H campaign names to the existing SSGC fit wrapper."""
    return SimpleNamespace(
        n_chains=campaign.n_chains,
        gibbs_iterations=campaign.gibbs_iterations,
        gibbs_thin=campaign.gibbs_thin,
        vi_iterations=campaign.vi_iterations,
        evaluation_grid=campaign.evaluation_space_grid,
        quadrature_grid=campaign.quadrature_space_grid,
        posterior_draws=campaign.posterior_draws,
        exact_max_events=campaign.exact_max_events,
    )


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
            warnings.warn(
                f"GP calibration failed; using INITIAL_GP: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            return (
                INITIAL_GP,
                time.perf_counter() - started,
                False,
                len(training_catalog),
            )


def _kde_intensity(training_catalog, training_geometry, evaluation_xy, data, grid_size):
    """Fit and exposure-normalize a KDE using only the training fold."""
    points = np.asarray(training_catalog.xy, dtype=float)
    quadrature_xy, quadrature_weights = area_weighted_grid(
        data["x_bounds"],
        data["y_bounds"],
        max(20, int(grid_size)),
        max(20, int(grid_size)),
        training_geometry,
    )
    if len(points) < 3 or np.linalg.matrix_rank(np.cov(points.T)) < 2:
        return np.full(
            len(evaluation_xy),
            len(points) / (data["duration"] * training_geometry.area),
        )
    kde = gaussian_kde(points.T, bw_method="scott")
    quadrature_density = kde(quadrature_xy.T)
    retained_mass = float(quadrature_weights @ quadrature_density)
    if not np.isfinite(retained_mass) or retained_mass <= 0.0:
        raise FloatingPointError("KDE training-domain normalization is not positive.")
    scale = len(points) / (data["duration"] * retained_mass)
    return scale * kde(np.asarray(evaluation_xy).T)


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
        estimate = _kde_intensity(
            data["catalog"],
            data["union"],
            evaluation_xy,
            data,
            FINAL_INTENSITY_GRID_SIZE,
        )
        return estimate, {
            "status": "ok",
            "model": model_name,
            "model_label": MODEL_LABELS[model_name],
            "inference_method": "scott_kde",
            "n_events": len(data["catalog"]),
            "n_posterior_draws": 1,
            "runtime_seconds": time.perf_counter() - started,
            "gp_calibration_seconds": 0.0,
            "gp_calibration_succeeded": False,
            "gp_calibration_n_events": 0,
        }

    zones = data["zones"] if model_name == "ssgc" else [data["union"]]
    model = _make_model(zones, data, gp_prior)
    partition = DomainPartition.from_polygons(zones)
    domain_index = partition.locate(evaluation_xy[:, 0], evaluation_xy[:, 1])
    draws, diagnostics = fit_intensity_method(
        model,
        data["catalog"],
        inference_method,
        _ssgc_campaign(campaign),
        seed,
        evaluation_xy,
        domain_index=domain_index,
        return_log_intensity=False,
        show_progress=False,
    )
    diagnostics.pop("peak_memory_mb", None)
    if draws is None:
        return None, {
            "model": model_name,
            "model_label": MODEL_LABELS[model_name],
            **diagnostics,
        }
    inference_seconds = diagnostics.pop("runtime_seconds")
    return draws.mean(axis=1), {
        "status": "ok",
        "model": model_name,
        "model_label": MODEL_LABELS[model_name],
        "inference_method": inference_method,
        "n_events": len(data["catalog"]),
        "n_posterior_draws": draws.shape[1],
        "runtime_seconds": inference_seconds + calibration_seconds,
        "gp_calibration_seconds": calibration_seconds,
        "gp_calibration_succeeded": calibration_succeeded,
        "gp_calibration_n_events": calibration_n_events,
        "gp_variance": gp_prior.variance,
        "gp_length_scale": gp_prior.length_scale,
        **diagnostics,
    }


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
                data["zones"] if model_name == "ssgc" else [data["union"]]
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
        estimate, record = _fit_full_intensity(
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
        record = {
            "status": "error",
            "model": model_name,
            "model_label": MODEL_LABELS[model_name],
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
    return model_name, estimate, record


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
    )
    estimates = {
        model_name: np.asarray(estimate, dtype=float)
        for model_name, estimate, _ in results
        if estimate is not None
    }
    return estimates, [record for _, _, record in results]


def _score_draws(event_draws, quadrature_draws, quadrature_weights, duration):
    log_likelihood = (
        np.sum(np.log(np.maximum(event_draws, np.finfo(float).tiny)), axis=0)
        - duration * (quadrature_weights @ quadrature_draws)
    )
    return float(logsumexp(log_likelihood) - np.log(log_likelihood.size))


def _fit_fold_model(
    model_name,
    training_catalog,
    held_event_xy,
    held_quadrature_xy,
    held_quadrature_weights,
    held_geometry,
    training_geometry,
    data,
    gp_prior,
    calibration_seconds,
    campaign,
    inference_method,
    seed,
):
    started = time.perf_counter()
    evaluation_xy = np.vstack([held_event_xy, held_quadrature_xy])
    n_events = len(held_event_xy)
    n_quadrature = len(held_quadrature_xy)
    if model_name == "kde":
        estimate = _kde_intensity(
            training_catalog,
            training_geometry,
            evaluation_xy,
            data,
            campaign.evaluation_space_grid,
        )
        draws = estimate[:, None]
        diagnostics = {
            "status": "ok",
            "inference_seconds": time.perf_counter() - started,
            "n_iter_run": 0,
            "converged": True,
        }
    else:
        full_zones = data["zones"] if model_name == "ssgc" else [data["union"]]
        training_zones = _training_zones(full_zones, held_geometry)
        model = _make_model(training_zones, data, gp_prior)
        partition = DomainPartition.from_polygons(full_zones)
        domain_index = partition.locate(evaluation_xy[:, 0], evaluation_xy[:, 1])
        draws, diagnostics = fit_intensity_method(
            model,
            training_catalog,
            inference_method,
            _ssgc_campaign(campaign),
            seed,
            evaluation_xy,
            domain_index=domain_index,
            return_log_intensity=False,
            show_progress=False,
        )
        diagnostics.pop("peak_memory_mb", None)
        if draws is None:
            return {
                "status": diagnostics.get("status", "skipped"),
                "model": model_name,
                "model_label": MODEL_LABELS[model_name],
                **diagnostics,
            }
        diagnostics["inference_seconds"] = diagnostics.pop("runtime_seconds")
    event_draws = draws[:n_events]
    quadrature_draws = draws[n_events : n_events + n_quadrature]
    score = _score_draws(
        event_draws,
        quadrature_draws,
        held_quadrature_weights,
        data["duration"],
    )
    record = {
        "status": "ok",
        "model": model_name,
        "model_label": MODEL_LABELS[model_name],
        "inference_method": "scott_kde" if model_name == "kde" else inference_method,
        "n_train": len(training_catalog),
        "n_held_out": n_events,
        "predictive_log_score": score,
        "predictive_log_score_per_event": score / n_events,
        "runtime_seconds": float(
            diagnostics["inference_seconds"]
            + (calibration_seconds if model_name != "kde" else 0.0)
        ),
        "gp_calibration_seconds": calibration_seconds if model_name != "kde" else 0.0,
        "gp_variance": gp_prior.variance if model_name != "kde" else float("nan"),
        "gp_length_scale": gp_prior.length_scale if model_name != "kde" else float("nan"),
        **diagnostics,
    }
    return record


def _fcat_fold_task(
    model_name,
    repeat,
    fold,
    held_geometry,
    data,
    campaign,
    inference_method,
):
    """Fit and score one model on one spatially held-out FCAT block."""
    held_mask = points_in_geometry(data["catalog"].xy, held_geometry)
    training_catalog = EventCatalog(
        t=data["catalog"].t[~held_mask],
        x=data["catalog"].x[~held_mask],
        y=data["catalog"].y[~held_mask],
        magnitudes=data["catalog"].magnitudes[~held_mask],
    )
    held_event_xy = data["catalog"].xy[held_mask]
    held_quadrature_xy, held_quadrature_weights = area_weighted_grid(
        data["x_bounds"],
        data["y_bounds"],
        max(12, campaign.evaluation_space_grid),
        max(12, campaign.evaluation_space_grid),
        held_geometry,
    )
    if not len(held_event_xy) or not len(held_quadrature_xy):
        return None

    training_geometry = data["union"].difference(held_geometry)
    seed = 80_000 + 1009 * repeat + 101 * fold + sum(map(ord, model_name))
    calibration_seed = 70_000 + 1009 * repeat + 101 * fold
    if model_name == "kde":
        gp_prior, calibration_seconds, calibration_succeeded, calibration_n_events = (
            INITIAL_GP,
            0.0,
            False,
            0,
        )
    else:
        full_zones = data["zones"] if model_name == "ssgc" else [data["union"]]
        calibration_zones = _training_zones(full_zones, held_geometry)
        (
            gp_prior,
            calibration_seconds,
            calibration_succeeded,
            calibration_n_events,
        ) = _calibrate_gp(
            training_catalog,
            calibration_zones,
            data,
            campaign,
            calibration_seed,
        )
    try:
        record = _fit_fold_model(
            model_name,
            training_catalog,
            held_event_xy,
            held_quadrature_xy,
            held_quadrature_weights,
            held_geometry,
            training_geometry,
            data,
            gp_prior,
            calibration_seconds,
            campaign,
            inference_method,
            seed,
        )
    except Exception as error:
        record = {
            "status": "error",
            "model": model_name,
            "model_label": MODEL_LABELS[model_name],
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
    record.update(
        {
            "repeat": repeat,
            "fold": fold,
            "year_min": YEAR_MIN,
            "magnitude_min": MAGNITUDE_MIN,
            "gp_calibration_succeeded": calibration_succeeded,
            "gp_calibration_n_events": calibration_n_events,
        }
    )
    return record


def _summarize(records, models=MODELS):
    summary = []
    for model in models:
        rows = [
            row for row in records if row.get("model") == model and row.get("status") == "ok"
        ]
        if not rows:
            continue
        entry = {
            "model": model,
            "model_label": MODEL_LABELS[model],
            "n_folds": len(rows),
        }
        for metric in (
            "predictive_log_score_per_event",
            "runtime_seconds",
        ):
            mean, lower, upper = mean_confidence_interval(
                [row[metric] for row in rows]
            )
            entry[metric] = mean
            entry[f"{metric}_ci_low"] = lower
            entry[f"{metric}_ci_high"] = upper
        summary.append(entry)
    return summary


def _write_latex_table(path, summary):
    lines = [
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Model & $S_{\mathrm{test}}$ \\",
        r"\midrule",
    ]
    for row in summary:
        mean = row["predictive_log_score_per_event"]
        lower = row["predictive_log_score_per_event_ci_low"]
        upper = row["predictive_log_score_per_event_ci_high"]
        score = f"{mean:.3f}"
        if np.isfinite(lower) and np.isfinite(upper):
            score += f" [{lower:.3f}, {upper:.3f}]"
        lines.append(
            f"{row['model'].upper()} & {score} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    upper = float(np.quantile(all_values, 0.99))
    if not np.isfinite(upper) or upper <= lower:
        upper = 1.0
    levels = np.linspace(lower, upper, 36)
    color_map = plt.get_cmap("viridis")

    model_names = list(estimates)
    figure, axes = plt.subplots(
        1,
        len(model_names),
        figsize=(4.4 * len(model_names) + 1.0, 6.2),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    axes = axes.ravel()
    image = None
    for index, (axis, model_name) in enumerate(zip(axes, model_names)):
        image = axis.tricontourf(
            evaluation_xy[:, 0],
            evaluation_xy[:, 1],
            np.clip(estimates[model_name], lower, upper),
            levels=levels,
            vmin=lower,
            vmax=upper,
            cmap=color_map,
            extend="max",
            antialiased=False,
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
        ax=list(axes),
        label=r"Annual posterior mean intensity (events km$^{-2}$ yr$^{-1}$)",
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
    profile="smoke",
    models=MODELS,
    inference_method="vi_sparse",
    *,
    n_jobs=None,
    resume=True,
    save_figures=True,
    show_figures=False,
    campaign_overrides=None,
):
    validate_fcat_settings()
    models = tuple(models)
    unknown = set(models) - set(MODELS)
    if not models or unknown:
        raise ValueError(f"At least one valid model is required; unknown={sorted(unknown)}.")
    if inference_method not in SSGC_INFERENCE_METHODS:
        raise ValueError(f"Unknown inference method {inference_method!r}.")
    n_jobs = resolve_n_jobs(profile, n_jobs)
    if not all(isinstance(value, bool) for value in (resume, save_figures, show_figures)):
        raise ValueError("resume, save_figures and show_figures must be boolean.")
    display_figures = _resolve_figure_display(show_figures)
    campaign = configure_campaign(profile, **(campaign_overrides or {}))
    data = load_fcat17()
    output = RESULTS_ROOT / campaign.name
    output.mkdir(parents=True, exist_ok=True)
    folds = spatial_block_folds(data, campaign.name)
    if not folds:
        raise RuntimeError("The FCAT-17 spatial blocking scheme produced no fold.")
    _print_run_settings(
        campaign, models, inference_method, n_jobs, data, folds, output
    )
    write_campaign(
        output / "fcat17_campaign.json",
        campaign,
        {
            "models": models,
            "inference_method": inference_method,
            "n_jobs": n_jobs,
            "effective_workers": effective_worker_count(n_jobs),
            "resume": resume,
            "year_min": YEAR_MIN,
            "magnitude_min": MAGNITUDE_MIN,
            "n_events": len(data["catalog"]),
            "n_spatial_folds": len(folds),
            "final_intensity_grid_size": FINAL_INTENSITY_GRID_SIZE,
        },
    )
    fold_tasks = [
        (
            model_name,
            repeat,
            fold,
            held_geometry,
            data,
            campaign,
            inference_method,
        )
        for repeat, fold, held_geometry in folds
        for model_name in models
    ]
    fold_keys = [
        (repeat, fold, model_name)
        for repeat, fold, _ in folds
        for model_name in models
    ]
    fold_checkpoints = checkpoint_directory(
        output,
        "fcat17_block_cv",
        campaign,
        settings={
            "models": models,
            "inference_method": inference_method,
            "year_min": YEAR_MIN,
            "magnitude_min": MAGNITUDE_MIN,
        },
        source_paths=_CHECKPOINT_SOURCES,
    )
    records = [
        record
        for record in parallel_map(
            _fcat_fold_task,
            fold_tasks,
            n_jobs,
            "FCAT-17 block CV",
            task_keys=fold_keys,
            checkpoint_dir=fold_checkpoints,
            resume=resume,
            max_parallel_calibrations=campaign.max_parallel_calibrations,
        )
        if record is not None
    ]
    if not records:
        raise RuntimeError("No FCAT-17 fold was evaluable.")
    summary = _summarize(records, models)
    write_records(output / "fcat17_block_cv_raw.csv", records)
    write_records(output / "fcat17_table.csv", summary)
    _write_latex_table(output / "fcat17_table.tex", summary)

    _, _, _, map_xy = _intensity_map_grid(data)
    map_checkpoints = checkpoint_directory(
        output,
        "fcat17_full_maps",
        campaign,
        settings={
            "models": models,
            "inference_method": inference_method,
            "grid_size": FINAL_INTENSITY_GRID_SIZE,
        },
        source_paths=_CHECKPOINT_SOURCES,
    )
    full_estimates, full_fit_records = fit_full_intensity_maps(
        models,
        data,
        campaign,
        inference_method,
        map_xy,
        n_jobs=n_jobs,
        checkpoint_dir=map_checkpoints,
        resume=resume,
    )
    write_records(output / "fcat17_full_fit.csv", full_fit_records)
    map_frame = pd.DataFrame({"x_km": map_xy[:, 0], "y_km": map_xy[:, 1]})
    for model_name, estimate in full_estimates.items():
        map_frame[f"intensity_{model_name}"] = estimate
    map_frame.to_csv(output / "fcat17_full_intensity_grid.csv", index=False)

    if save_figures or display_figures:
        _plot_catalogue(data, output, save_figures, display_figures)
        _plot_intensity_maps(
            data,
            map_xy,
            full_estimates,
            output,
            save_figures,
            display_figures,
        )
    _print_run_summary(records, summary, full_fit_records, output)
    if display_figures and "ipykernel" not in sys.modules:
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
        default=list(MODELS),
        help="Models included in the spatial block comparison.",
    )
    selection.add_argument(
        "--inference-method",
        choices=SSGC_INFERENCE_METHODS,
        default="vi_sparse",
        help="Inference backend used for SSGC and SGCP.",
    )
    selection.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Simultaneous jobs (default: 1). Test 2 before requesting more; RAM is shared.",
    )

    budget = parser.add_argument_group("campaign budget overrides")
    budget.add_argument("--n-chains", type=int, default=None)
    budget.add_argument("--gibbs-iterations", type=int, default=None)
    budget.add_argument("--gibbs-thin", type=int, default=None)
    budget.add_argument("--vi-iterations", type=int, default=None)
    budget.add_argument("--evaluation-space-grid", type=int, default=None)
    budget.add_argument("--quadrature-space-grid", type=int, default=None)
    budget.add_argument("--posterior-draws", type=int, default=None)
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
        "n_chains": args.n_chains,
        "gibbs_iterations": args.gibbs_iterations,
        "gibbs_thin": args.gibbs_thin,
        "vi_iterations": args.vi_iterations,
        "evaluation_space_grid": args.evaluation_space_grid,
        "quadrature_space_grid": args.quadrature_space_grid,
        "posterior_draws": args.posterior_draws,
        "max_parallel_calibrations": args.max_parallel_calibrations,
        "exact_max_events": args.exact_max_events,
        "use_calibration": False if args.no_calibration else None,
    }
    return run(
        profile=args.profile,
        models=args.models,
        inference_method=args.inference_method,
        n_jobs=args.n_jobs,
        resume=not args.no_resume,
        save_figures=not args.no_figures,
        show_figures=args.show_figures,
        campaign_overrides=overrides,
    )


# %% Run the file
if __name__ == "__main__":
    if should_use_editor_settings():
        run_from_editor()
    else:
        main()
