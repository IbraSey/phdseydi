"""Shared, reproducible utilities for the SSGC deliverable experiments.

The module keeps experimental concerns outside the inference package: common
simulation settings, posterior scores, runtime diagnostics, CSV exports and the
four Gibbs/VI configurations used in the deliverable.
"""

from __future__ import annotations

import csv
import sys
import time
import tracemalloc
from dataclasses import dataclass, replace
from numbers import Integral, Real
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openturns as ot
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.special import expit
from scipy.stats import gaussian_kde
from shapely.geometry import Point, box
from shapely.ops import unary_union
from shapely.prepared import prep
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from package import (
    EventCatalog,
    GPParameters,
    GibbsConfig,
    SSGCModel,
    SSGCVIConfig,
    SparseGP,
    generate_voronoi_cells,
    simulate_spatial_process,
)
from visualization import save_figure


RESULTS_ROOT = REPO_ROOT / "results" / "ssgc_deliverable"
X_BOUNDS = (0.0, 2.0)
Y_BOUNDS = (0.0, 2.0)

METHOD_LABELS = {
    "gibbs_exact": "Full-rank Gibbs",
    "gibbs_sparse": "Reduced-rank Gibbs",
    "vi_exact": "Full-rank MF-VI",
    "vi_sparse": "Reduced-rank MF-VI",
    "kde": "KDE",
}
PLOT_METHOD_LABELS = {
    "gibbs_exact": "Exact Gibbs",
    "gibbs_sparse": "Sparse-GP Gibbs",
    "vi_exact": "Exact MF-VI",
    "vi_sparse": "Sparse-GP MF-VI",
    "kde": "KDE",
}
INFERENCE_METHODS = tuple(name for name in METHOD_LABELS if name != "kde")

PROFILES = {
    "1": {
        "n_germs": 6,
        "mus": (10.0, 1.0, 2.0, 10.0, 8.0, 2.0),
        "seed": 13,
    },
    "2": {
        "n_germs": 5,
        "mus": (3.5, 2.0, 4.0, 3.0, 2.5),
        "seed": 13,
    },
    "3": {
        "n_germs": 7,
        "mus": (20.0, 1.0, 1.0, 1.0, 1.0, 1.0, 20.0),
        "seed": 13,
    },
}

EXPERIMENT_1_DURATIONS = {
    ("1", "A"): 100.0,
    ("1", "B"): 55.0,
    ("2", "A"): 100.0,
    ("2", "B"): 85.0,
    ("3", "A"): 35.0,
    ("3", "B"): 35.0,
}
EXPERIMENT_2_DURATIONS = {
    ("1", "A"): 180.7,
    ("1", "B"): 98.2,
    ("2", "A"): 163.9,
    ("2", "B"): 139.4,
}


@dataclass(frozen=True)
class CampaignConfig:
    """Validated numerical budget for an SSGC experimental campaign."""

    name: str
    n_replicates: int
    n_chains: int
    gibbs_iterations: int
    gibbs_thin: int
    vi_iterations: int
    evaluation_grid: int
    quadrature_grid: int
    posterior_draws: int
    duration_scale: float
    exact_max_events: int
    use_calibration: bool

    def __post_init__(self):
        integer_minima = {
            "n_replicates": 1,
            "n_chains": 1,
            "gibbs_iterations": 1,
            "gibbs_thin": 1,
            "vi_iterations": 1,
            "evaluation_grid": 2,
            "quadrature_grid": 2,
            "posterior_draws": 1,
            "exact_max_events": 1,
        }
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Campaign name must be a non-empty string.")
        for field_name, minimum in integer_minima.items():
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or int(value) < minimum
            ):
                raise ValueError(
                    f"{field_name} must be an integer greater than or equal "
                    f"to {minimum}."
                )
        if (
            isinstance(self.duration_scale, bool)
            or not isinstance(self.duration_scale, Real)
            or not np.isfinite(float(self.duration_scale))
            or float(self.duration_scale) <= 0.0
        ):
            raise ValueError("duration_scale must be a finite positive number.")
        if not isinstance(self.use_calibration, bool):
            raise ValueError("use_calibration must be a boolean.")


CAMPAIGNS = {
    "smoke": CampaignConfig(
        name="smoke",
        n_replicates=1,
        n_chains=1,
        gibbs_iterations=20,
        gibbs_thin=2,
        vi_iterations=12,
        evaluation_grid=12,
        quadrature_grid=10,
        posterior_draws=12,
        duration_scale=0.04,
        exact_max_events=100,
        use_calibration=False,
    ),
    "full": CampaignConfig(
        name="full",
        n_replicates=20,
        n_chains=5,
        gibbs_iterations=3000,
        gibbs_thin=5,
        vi_iterations=500,
        evaluation_grid=30,
        quadrature_grid=30,
        posterior_draws=500,
        duration_scale=1.0,
        exact_max_events=300,
        use_calibration=True,
    ),
}


def configure_campaign(profile, **overrides):
    """Return one named campaign with validated, non-``None`` overrides."""
    if profile not in CAMPAIGNS:
        choices = ", ".join(CAMPAIGNS)
        raise ValueError(f"Unknown profile {profile!r}; choose one of: {choices}.")
    allowed = set(CampaignConfig.__dataclass_fields__) - {"name"}
    unknown = set(overrides) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown campaign override(s): {names}.")
    updates = {
        name: value
        for name, value in overrides.items()
        if value is not None
    }
    return replace(CAMPAIGNS[profile], **updates)


def f_star_a(x, y):
    """Smooth latent field used as Setting A."""
    points = np.column_stack(
        [np.asarray(x, dtype=float).reshape(-1), np.asarray(y, dtype=float).reshape(-1)]
    )
    centers = np.array([[0.5, 0.5], [0.5, 1.5], [1.5, 0.5], [1.5, 1.5]])
    weights = np.array([1.5, -1.5, 3.0, -3.0])
    values = np.zeros(points.shape[0], dtype=float)
    for weight, center in zip(weights, centers):
        difference = points - center
        density = np.exp(-np.sum(difference**2, axis=1) / 0.6) / (0.6 * np.pi)
        values += weight * density
    return values.reshape(np.shape(x))


def f_star_b(x, y):
    """Local, contrasted latent field used as Setting B."""
    points = np.column_stack(
        [np.asarray(x, dtype=float).reshape(-1), np.asarray(y, dtype=float).reshape(-1)]
    )
    centers = np.array(
        [[0.4, 0.4], [0.4, 1.6], [1.0, 1.0], [1.6, 0.4], [1.6, 1.6]]
    )
    weights = np.array([4.0, -3.5, 2.0, -4.5, 3.0])
    length_scales = np.array([0.20, 0.20, 0.35, 0.15, 0.25])
    values = np.zeros(points.shape[0], dtype=float)
    for weight, center, length_scale in zip(weights, centers, length_scales):
        difference = points - center
        values += weight * np.exp(
            -np.sum(difference**2, axis=1) / (2.0 * length_scale**2)
        )
    return values.reshape(np.shape(x))


LATENT_FIELDS = {"A": f_star_a, "B": f_star_b}


def profile_partition(profile_name: str, seed_offset: int = 0):
    profile = PROFILES[str(profile_name)]
    cells, germs = generate_voronoi_cells(
        n_germs=profile["n_germs"],
        X_bounds=X_BOUNDS,
        Y_bounds=Y_BOUNDS,
        rng_seed=profile["seed"] + int(seed_offset),
    )
    return list(cells), np.asarray(germs, dtype=float)


def simulate_configuration(
    profile_name: str,
    setting: str,
    duration: float,
    seed: int,
    grid_res: int = 100,
):
    zones, _ = profile_partition(profile_name)
    return simulate_spatial_process(
        X_bounds=X_BOUNDS,
        Y_bounds=Y_BOUNDS,
        T=float(duration),
        polygons=zones,
        mus=PROFILES[str(profile_name)]["mus"],
        f=LATENT_FIELDS[str(setting)],
        grid_res=int(grid_res),
        rng_seed=int(seed),
    )


def reference_intensity(simulation):
    """Return the exact generating intensity callable of a simulation."""
    return lambda x, y: simulation.spatial_components(x, y)[3]


def make_model(
    zones,
    duration,
    *,
    eps_prior_variance=1.0,
    eps_prior_length_scale=0.01,
    gp_prior=GPParameters(variance=5.0, length_scale=0.2),
    x_bounds=X_BOUNDS,
    y_bounds=Y_BOUNDS,
    magnitude_min=0.0,
    magnitude_max=None,
):
    return SSGCModel.from_polygons(
        polygons=list(zones),
        duration=float(duration),
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        gp_prior=gp_prior,
        eps_prior_variance=float(eps_prior_variance),
        eps_prior_length_scale=float(eps_prior_length_scale),
        jitter=1e-5,
        magnitude_min=magnitude_min,
        magnitude_max=magnitude_max,
    )


def area_weighted_grid(x_bounds, y_bounds, nx, ny, geometry=None):
    """Build midpoint quadrature cells and retain their exact intersected areas."""
    x_edges = np.linspace(x_bounds[0], x_bounds[1], int(nx) + 1)
    y_edges = np.linspace(y_bounds[0], y_bounds[1], int(ny) + 1)
    points = []
    weights = []
    for x_left, x_right in zip(x_edges[:-1], x_edges[1:]):
        for y_lower, y_upper in zip(y_edges[:-1], y_edges[1:]):
            cell = box(x_left, y_lower, x_right, y_upper)
            intersection = cell if geometry is None else cell.intersection(geometry)
            if intersection.is_empty or intersection.area <= 0.0:
                continue
            representative = intersection.representative_point()
            points.append((representative.x, representative.y))
            weights.append(intersection.area)
    return np.asarray(points, dtype=float), np.asarray(weights, dtype=float)


def crps_ensemble(draws, truth):
    """Pointwise empirical CRPS in O(G R log R), avoiding an R by R array."""
    draws = np.asarray(draws, dtype=float)
    truth = np.asarray(truth, dtype=float).reshape(-1)
    if draws.ndim != 2 or draws.shape[0] != truth.size:
        raise ValueError("draws must have shape (n_points, n_draws).")
    sorted_draws = np.sort(draws, axis=1)
    n_draws = draws.shape[1]
    coefficients = 2.0 * np.arange(n_draws) - n_draws + 1.0
    pair_term = np.sum(sorted_draws * coefficients[None, :], axis=1) / n_draws**2
    return np.mean(np.abs(draws - truth[:, None]), axis=1) - pair_term


def posterior_metrics(draws, truth, weights):
    draws = np.asarray(draws, dtype=float)
    truth = np.asarray(truth, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if draws.shape[0] != truth.size or truth.size != weights.size:
        raise ValueError("Draws, truth and quadrature weights must align.")
    normalized_weights = weights / weights.sum()
    estimate = draws.mean(axis=1)
    squared_error = np.sum(normalized_weights * (estimate - truth) ** 2)
    truth_norm = np.sum(normalized_weights * truth**2)
    metrics = {
        "rel_l2": float(np.sqrt(squared_error / truth_norm)),
        "mae": float(np.sum(normalized_weights * np.abs(estimate - truth))),
        "crps": float(np.sum(normalized_weights * crps_ensemble(draws, truth))),
    }
    for level in (0.50, 0.90):
        tail = 0.5 * (1.0 - level)
        lower, upper = np.quantile(draws, [tail, 1.0 - tail], axis=1)
        suffix = int(round(100 * level))
        covered = (truth >= lower) & (truth <= upper)
        metrics[f"ecp_{suffix}"] = float(np.sum(normalized_weights * covered))
        metrics[f"mpiw_{suffix}"] = float(
            np.sum(normalized_weights * (upper - lower))
        )
    return metrics


def kde_draws(catalog, duration, evaluation_xy):
    points = np.asarray(catalog.xy, dtype=float)
    if points.shape[0] < 3 or np.linalg.matrix_rank(np.cov(points.T)) < 2:
        area = (X_BOUNDS[1] - X_BOUNDS[0]) * (Y_BOUNDS[1] - Y_BOUNDS[0])
        intensity = np.full(evaluation_xy.shape[0], len(catalog) / (duration * area))
    else:
        density = gaussian_kde(points.T)(np.asarray(evaluation_xy).T)
        intensity = len(catalog) * density / float(duration)
    return intensity[:, None]


def split_rhat(chains):
    chains = np.asarray(chains, dtype=float)
    if chains.ndim != 3 or chains.shape[0] < 2 or chains.shape[1] < 4:
        return float("nan")
    half = chains.shape[1] // 2
    split = np.concatenate([chains[:, :half], chains[:, -half:]], axis=0)
    within = np.mean(np.var(split, axis=1, ddof=1), axis=0)
    between = half * np.var(np.mean(split, axis=1), axis=0, ddof=1)
    variance = (half - 1.0) * within / half + between / half
    ratio = np.divide(variance, within, out=np.ones_like(variance), where=within > 0)
    return float(np.max(np.sqrt(np.maximum(ratio, 0.0))))


def effective_sample_size(chains):
    chains = np.asarray(chains, dtype=float)
    if chains.ndim != 3 or chains.shape[1] < 3:
        return float("nan")
    n_chains, n_draws, n_parameters = chains.shape
    estimates = []
    for parameter in range(n_parameters):
        values = chains[:, :, parameter]
        centered = values - values.mean(axis=1, keepdims=True)
        variance = np.mean(np.sum(centered**2, axis=1) / max(1, n_draws - 1))
        if variance <= 0.0:
            estimates.append(float(n_chains * n_draws))
            continue
        autocorrelation_sum = 0.0
        for lag in range(1, n_draws):
            covariance = np.mean(
                np.sum(centered[:, :-lag] * centered[:, lag:], axis=1)
                / (n_draws - lag)
            )
            correlation = covariance / variance
            if correlation <= 0.0:
                break
            autocorrelation_sum += correlation
        estimates.append(n_chains * n_draws / (1.0 + 2.0 * autocorrelation_sum))
    return float(np.min(estimates))


def _fit_gibbs(
    model,
    catalog,
    method,
    campaign,
    seed,
    evaluation_xy,
    domain_index=None,
    show_progress=True,
):
    chains = []
    intensity_draws = []
    draws_per_chain = max(1, int(np.ceil(campaign.posterior_draws / campaign.n_chains)))
    backend = "exact" if method == "gibbs_exact" else "sparse"
    for chain_index in tqdm(
        range(campaign.n_chains),
        desc=f"{PLOT_METHOD_LABELS[method]} chains",
        unit="chain",
        leave=False,
        dynamic_ncols=True,
        disable=not show_progress,
    ):
        sparse_gp = None
        if backend == "sparse":
            sparse_gp = SparseGP.from_bounds(
                model.x_bounds,
                model.y_bounds,
                model.gp_prior.variance,
                model.gp_prior.length_scale,
            )
        config = GibbsConfig(
            n_iter=campaign.gibbs_iterations,
            thin=campaign.gibbs_thin,
            mala_step=0.06,
            use_calibration=False,
            learn_nu=False,
            verbose=False,
        )
        fit = model.gibbs(
            catalog,
            config=config,
            gp_backend=backend,
            sparse_gp=sparse_gp,
            rng_seed=seed + 1009 * chain_index,
        )
        chains.append(fit.eps_chain)
        intensity_draws.append(
            fit.background_intensity_samples(
                evaluation_xy[:, 0],
                evaluation_xy[:, 1],
                burn_in=0.4,
                n_samples=draws_per_chain,
                domain_index=domain_index,
            )
        )
    n_stored = min(chain.shape[0] for chain in chains)
    eps_chains = np.stack([chain[-n_stored:] for chain in chains], axis=0)
    diagnostics = {
        "rhat_max": split_rhat(eps_chains),
        "ess_min": effective_sample_size(eps_chains),
        "n_iter_run": campaign.gibbs_iterations,
        "converged": bool(
            campaign.n_chains < 2 or split_rhat(eps_chains) < 1.05
        ),
    }
    return np.concatenate(intensity_draws, axis=1)[:, : campaign.posterior_draws], diagnostics


def _fit_vi(
    model,
    catalog,
    method,
    campaign,
    seed,
    evaluation_xy,
    domain_index=None,
    return_log_intensity=False,
):
    backend = "exact" if method == "vi_exact" else "sparse"
    config = SSGCVIConfig(
        n_iter=campaign.vi_iterations,
        tolerance=1e-5,
        verbose=False,
        random_seed=seed,
        fixed_beta=None,
        quadrature_nx=campaign.quadrature_grid,
        quadrature_ny=campaign.quadrature_grid,
        gp_backend=backend,
        use_calibration=False,
    )
    fit = model.vi(catalog, config=config)
    prediction_options = {
        "n_samples": campaign.posterior_draws,
        "rng_seed": seed + 7919,
        "domain_index": domain_index,
    }
    if return_log_intensity:
        log_draws = fit.background_log_intensity_samples(
            evaluation_xy[:, 0], evaluation_xy[:, 1], **prediction_options
        )
        draws = np.exp(log_draws)
    else:
        log_draws = None
        draws = fit.background_intensity_samples(
            evaluation_xy[:, 0], evaluation_xy[:, 1], **prediction_options
        )
    diagnostics = {
        "rhat_max": float("nan"),
        "ess_min": float("nan"),
        "n_iter_run": fit.diagnostics["n_iter_run"],
        "converged": bool(fit.diagnostics["converged"]),
        "final_elbo": float(fit.elbo_trace[-1]),
    }
    if log_draws is not None:
        diagnostics["_log_intensity_draws"] = log_draws
    return draws, diagnostics


def fit_intensity_method(
    model,
    catalog,
    method,
    campaign,
    seed,
    evaluation_xy,
    domain_index=None,
    return_log_intensity=False,
    show_progress=True,
):
    """Fit one method and return posterior draws plus timing diagnostics."""
    method = str(method)
    if method not in INFERENCE_METHODS:
        raise ValueError(f"Unknown inference method: {method!r}.")
    if method.endswith("exact") and len(catalog) > campaign.exact_max_events:
        return None, {
            "status": "skipped_exact_size",
            "runtime_seconds": float("nan"),
            "peak_memory_mb": float("nan"),
        }

    tracemalloc.start()
    started = time.perf_counter()
    try:
        if method.startswith("gibbs"):
            draws, diagnostics = _fit_gibbs(
                model,
                catalog,
                method,
                campaign,
                seed,
                evaluation_xy,
                domain_index,
                show_progress,
            )
            if return_log_intensity and draws is not None:
                diagnostics["_log_intensity_draws"] = np.log(
                    np.maximum(draws, np.finfo(float).tiny)
                )
        else:
            draws, diagnostics = _fit_vi(
                model,
                catalog,
                method,
                campaign,
                seed,
                evaluation_xy,
                domain_index,
                return_log_intensity,
            )
    finally:
        runtime = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    diagnostics.update(
        {
            "status": "ok",
            "runtime_seconds": float(runtime),
            "peak_memory_mb": float(peak / 1024**2),
        }
    )
    return draws, diagnostics


def calibrate_model(model, catalog, campaign, seed):
    if not campaign.use_calibration:
        return model, {
            "gp_variance": model.gp_prior.variance,
            "gp_length_scale": model.gp_prior.length_scale,
            "calibrated": False,
        }
    calibrated = model.calibrate_gp_prior(catalog, rng_seed=seed, verbose=False)
    return replace(model, gp_prior=calibrated), {
        "gp_variance": calibrated.variance,
        "gp_length_scale": calibrated.length_scale,
        "calibrated": True,
    }


def write_records(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        raise ValueError("Cannot write an empty record collection.")
    fieldnames = []
    for record in records:
        for name in record:
            if name not in fieldnames:
                fieldnames.append(name)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return path


def summarize_records(records, group_fields):
    numeric_fields = (
        "n_events",
        "rel_l2",
        "mae",
        "crps",
        "ecp_50",
        "ecp_90",
        "mpiw_50",
        "mpiw_90",
        "runtime_seconds",
        "peak_memory_mb",
        "rhat_max",
        "ess_min",
    )
    groups = {}
    for record in records:
        key = tuple(record.get(name) for name in group_fields)
        groups.setdefault(key, []).append(record)
    summaries = []
    for key, group in groups.items():
        summary = dict(zip(group_fields, key))
        summary["n_replicates"] = len(group)
        for field in numeric_fields:
            values = np.asarray(
                [record.get(field, np.nan) for record in group], dtype=float
            )
            finite = values[np.isfinite(values)]
            if finite.size:
                summary[f"{field}_mean"] = float(finite.mean())
                summary[f"{field}_se"] = float(
                    finite.std(ddof=1) / np.sqrt(finite.size)
                    if finite.size > 1
                    else 0.0
                )
            else:
                summary[f"{field}_mean"] = float("nan")
                summary[f"{field}_se"] = float("nan")
        summaries.append(summary)
    return summaries


def finish_figure(figure, filename, *, save=True, show=False):
    """Save and/or display a figure, then release it when it is not shown."""
    path = save_figure(figure, filename) if save else None
    if show:
        plt.show(block=False)
        plt.pause(0.001)
    else:
        plt.close(figure)
    return path


def save_settings_figures(*, save=True, show=False):
    """Generate the five simulation-design panels referenced by the draft."""
    for profile_name, profile in PROFILES.items():
        zones, _ = profile_partition(profile_name)
        figure, axis = plt.subplots(figsize=(5.2, 4.4), layout="constrained")
        for polygon, intensity in zip(zones, profile["mus"]):
            x, y = polygon.exterior.xy
            axis.fill(x, y, color=plt.cm.viridis(np.log1p(intensity) / np.log(21.0)))
            center = polygon.representative_point()
            axis.text(center.x, center.y, f"{intensity:g}", ha="center", va="center")
        axis.set(xlim=X_BOUNDS, ylim=Y_BOUNDS, aspect="equal", xlabel="x", ylabel="y")
        axis.set_title(f"Zonal Profile {profile_name}")
        finish_figure(
            figure,
            f"ssgc/profile_{profile_name}",
            save=save,
            show=show,
        )

    grid_x, grid_y = np.meshgrid(
        np.linspace(*X_BOUNDS, 200), np.linspace(*Y_BOUNDS, 200)
    )
    for setting, latent_field in LATENT_FIELDS.items():
        figure, axis = plt.subplots(figsize=(5.2, 4.4), layout="constrained")
        values = latent_field(grid_x, grid_y)
        limit = float(np.max(np.abs(values)))
        if limit == 0.0:
            limit = 1.0
        levels = np.linspace(-limit, limit, 31)
        image = axis.pcolormesh(
            grid_x,
            grid_y,
            values,
            shading="auto",
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
            rasterized=True,
        )
        figure.colorbar(
            image,
            ax=axis,
            label=r"$f^\star(s)$",
            ticks=np.linspace(-limit, limit, 7),
        )
        axis.set(aspect="equal", xlabel="x", ylabel="y")
        axis.set_title(f"Latent GP Setting {setting}")
        finish_figure(
            figure,
            f"ssgc/latent_setting_{setting}",
            save=save,
            show=show,
        )


def merge_adjacent_zones(zones, target_count=4):
    """Greedily merge adjacent polygons until the requested count is reached."""
    groups = list(zones)
    while len(groups) > int(target_count):
        candidates = []
        for left in range(len(groups)):
            for right in range(left + 1, len(groups)):
                shared = groups[left].boundary.intersection(groups[right].boundary).length
                if shared > 1e-10:
                    candidates.append((shared, left, right))
        if not candidates:
            break
        _, left, right = max(candidates)
        merged = unary_union([groups[left], groups[right]])
        groups = [
            polygon
            for index, polygon in enumerate(groups)
            if index not in {left, right}
        ] + [merged]
    return groups


def save_partition_figures(*, save=True, show=False):
    oracle, _ = profile_partition("1")
    crossed, _ = profile_partition("2", seed_offset=8)
    partitions = {
        "partition_oracle": oracle,
        "partition_crossed": crossed,
        "partition_merged": merge_adjacent_zones(oracle, target_count=4),
        "partition_homogeneous": [
            box(X_BOUNDS[0], Y_BOUNDS[0], X_BOUNDS[1], Y_BOUNDS[1])
        ],
    }
    for name, zones in partitions.items():
        figure, axis = plt.subplots(figsize=(4.8, 4.4), layout="constrained")
        for index, polygon in enumerate(zones):
            x, y = polygon.exterior.xy
            axis.fill(x, y, alpha=0.35, color=plt.cm.tab10(index % 10))
            axis.plot(x, y, color="black", linewidth=0.8)
        axis.set(xlim=X_BOUNDS, ylim=Y_BOUNDS, aspect="equal", xlabel="x", ylabel="y")
        finish_figure(figure, f"ssgc/{name}", save=save, show=show)


def plot_metric_summary(
    summary,
    x_field,
    group_field,
    filename,
    title,
    *,
    save=True,
    show=False,
):
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(14.8, 5.6),
        layout="constrained",
    )
    metrics = (("rel_l2", r"Relative $L_2$ error"), ("crps", "CRPS"), ("ecp_90", "ECP 90%"))
    x_values = list(dict.fromkeys(record[x_field] for record in summary))
    groups = list(dict.fromkeys(record[group_field] for record in summary))
    for axis, (metric, label) in zip(axes, metrics):
        for group in groups:
            selected = [record for record in summary if record[group_field] == group]
            means = []
            errors = []
            for value in x_values:
                matching = [record for record in selected if record[x_field] == value]
                estimates = np.asarray(
                    [record.get(f"{metric}_mean", np.nan) for record in matching],
                    dtype=float,
                )
                standard_errors = np.asarray(
                    [record.get(f"{metric}_se", np.nan) for record in matching],
                    dtype=float,
                )
                finite = np.isfinite(estimates)
                if not np.any(finite):
                    means.append(np.nan)
                    errors.append(np.nan)
                    continue
                estimates = estimates[finite]
                standard_errors = standard_errors[finite]
                means.append(float(estimates.mean()))
                between = (
                    estimates.var(ddof=1) / estimates.size
                    if estimates.size > 1
                    else 0.0
                )
                within = float(np.mean(standard_errors**2))
                errors.append(float(np.sqrt(between + within)))
            axis.errorbar(x_values, means, yerr=errors, marker="o", capsize=3, label=str(group))
        axis.set(xlabel=x_field.replace("_", " ").title(), ylabel=label)
        axis.grid(alpha=0.25)
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=min(4, len(labels)),
        fontsize=8,
        frameon=False,
    )
    figure.suptitle(title)
    finish_figure(figure, f"ssgc/{filename}", save=save, show=show)


def plot_metric_boxplots(
    records,
    *,
    x_field,
    group_field,
    color_field,
    style_field,
    panel_field,
    filename,
    title,
    save=True,
    show=False,
):
    """Plot replicate distributions without pooling distinct data settings."""
    metrics = (
        ("rel_l2", r"Relative $L_2$ error"),
        ("crps", "CRPS"),
        ("ecp_90", "ECP 90%"),
    )
    valid = [record for record in records if record.get("status") == "ok"]
    if not valid:
        return
    panels = list(dict.fromkeys(record[panel_field] for record in valid))
    x_values = list(dict.fromkeys(record[x_field] for record in valid))
    groups = list(dict.fromkeys(record[group_field] for record in valid))
    color_values = list(dict.fromkeys(record[color_field] for record in valid))
    style_values = list(dict.fromkeys(record[style_field] for record in valid))
    palette = plt.get_cmap("tab10")(
        np.linspace(0.0, 0.9, max(len(color_values), 2))
    )
    colors = dict(zip(color_values, palette))
    markers = dict(zip(style_values, ("o", "s", "D", "^", "P", "X")))
    hatches = dict(zip(style_values, ("", "///", "xx", "..", "++", "\\\\")))
    group_specs = {}
    for group in groups:
        example = next(record for record in valid if record[group_field] == group)
        group_specs[group] = (
            example[color_field],
            example[style_field],
        )
    centers = np.arange(len(x_values), dtype=float)
    cluster_width = 0.82
    box_width = 0.72 * cluster_width / max(len(groups), 1)
    offsets = (
        np.arange(len(groups), dtype=float) - 0.5 * (len(groups) - 1)
    ) * cluster_width / max(len(groups), 1)
    rng = np.random.default_rng(731)

    figure, axes = plt.subplots(
        len(panels),
        len(metrics),
        figsize=(16.0, 4.2 * len(panels)),
        squeeze=False,
        sharex="col",
        layout="constrained",
    )
    for panel_index, panel in enumerate(panels):
        panel_records = [
            record for record in valid if record[panel_field] == panel
        ]
        for metric_index, (metric, metric_label) in enumerate(metrics):
            axis = axes[panel_index, metric_index]
            for group_index, group in enumerate(groups):
                values_by_x = []
                positions = []
                for x_index, x_value in enumerate(x_values):
                    values = np.asarray(
                        [
                            record.get(metric, np.nan)
                            for record in panel_records
                            if record[x_field] == x_value
                            and record[group_field] == group
                        ],
                        dtype=float,
                    )
                    values = values[np.isfinite(values)]
                    if values.size:
                        values_by_x.append(values)
                        positions.append(centers[x_index] + offsets[group_index])
                if not values_by_x:
                    continue
                color_value, style_value = group_specs[group]
                color = colors[color_value]
                box_values = [values for values in values_by_x if values.size > 1]
                box_positions = [
                    position
                    for position, values in zip(positions, values_by_x)
                    if values.size > 1
                ]
                if box_values:
                    axis.boxplot(
                        box_values,
                        positions=box_positions,
                        widths=box_width,
                        patch_artist=True,
                        manage_ticks=False,
                        showfliers=False,
                        boxprops={
                            "facecolor": color,
                            "edgecolor": color,
                            "alpha": 0.50,
                            "hatch": hatches[style_value],
                        },
                        medianprops={"color": "black", "linewidth": 1.15},
                        whiskerprops={"color": color, "linewidth": 1.0},
                        capprops={"color": color, "linewidth": 1.0},
                    )
                for position, values in zip(positions, values_by_x):
                    jitter = (
                        rng.uniform(-0.23, 0.23, size=values.size) * box_width
                        if values.size > 1
                        else np.zeros(1)
                    )
                    axis.scatter(
                        position + jitter,
                        values,
                        s=27,
                        marker=markers[style_value],
                        color=color,
                        edgecolors="black",
                        linewidths=0.35,
                        alpha=0.76,
                        zorder=3,
                    )
            if metric == "ecp_90":
                axis.axhline(0.9, color="0.25", linestyle="--", linewidth=0.9)
                axis.set_ylim(0.0, 1.02)
            axis.set_ylabel(metric_label)
            axis.set_title(f"Setting {panel} | {metric_label}", fontsize=10)
            axis.grid(axis="y", alpha=0.22)
            axis.set_xticks(centers, [str(value) for value in x_values])
            if panel_index == len(panels) - 1:
                axis.set_xlabel(x_field.replace("_", " ").title())

    method_legend = [
        Patch(
            facecolor=colors[value],
            edgecolor=colors[value],
            alpha=0.65,
            label=f"Method: {value}",
        )
        for value in color_values
    ]
    model_legend = [
        Line2D(
            [0],
            [0],
            marker=markers[value],
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=7,
            label=f"Model: {value}",
        )
        for value in style_values
    ]
    legend = method_legend + model_legend
    figure.legend(
        handles=legend,
        loc="outside lower center",
        ncol=min(4, len(legend)),
        fontsize=8.5,
        frameon=False,
    )
    figure.suptitle(title, fontsize=13)
    finish_figure(figure, f"ssgc/{filename}", save=save, show=show)


def points_in_geometry(xy, geometry):
    prepared = prep(geometry)
    return np.fromiter(
        (prepared.covers(Point(float(x), float(y))) for x, y in np.asarray(xy)),
        dtype=bool,
        count=len(xy),
    )
