"""FCAT-17 spatial block cross-validation for the SSGC deliverable.

Only the FCAT-17 application is implemented here. The second French catalogue
mentioned in the draft is deliberately left out until it is available.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
from scipy.special import logsumexp
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from deliverable_utils import (
    CAMPAIGNS,
    INFERENCE_METHODS,
    METHOD_LABELS,
    RESULTS_ROOT,
    GPParameters,
    area_weighted_grid,
    configure_campaign,
    finish_figure,
    fit_intensity_method,
    make_model,
    merge_adjacent_zones,
    points_in_geometry,
    write_records,
)
from data import EventCatalog
from spatial import DomainPartition

# %% User settings for VS Code/Jupyter interactive execution
# Edit this block, then run the complete file. Command-line execution ignores it.
YEAR_MIN = 1965
MAGNITUDE_MIN = 3.0
EPS_PRIOR_VARIANCE = 10.0  # delta_0: marginal variance of zonal effects
EPS_PRIOR_LENGTH_SCALE_KM = 3.0  # delta_1: correlation range between zones
INITIAL_GP = GPParameters(variance=2.0, length_scale=50.0)
REGIMES = ("zoning", "hybrid", "zoneless", "aggregated")

INTERACTIVE_PROFILE = "smoke"
INTERACTIVE_METHODS = ("vi_sparse",)
INTERACTIVE_REGIMES = REGIMES
INTERACTIVE_SAVE_FIGURES = True
INTERACTIVE_SHOW_FIGURES = True

# A value of None keeps the selected profile's default.
INTERACTIVE_CAMPAIGN_OVERRIDES = {
    "n_chains": None,
    "gibbs_iterations": None,
    "gibbs_thin": None,
    "vi_iterations": None,
    "evaluation_grid": None,
    "quadrature_grid": None,
    "posterior_draws": None,
    "exact_max_events": None,
}


def resolve_use_case_path():
    repo_root = Path(__file__).resolve().parents[2]
    for candidate in (repo_root / "use_case", repo_root.parent / "use_case"):
        if (candidate / "catalog.csv").is_file():
            return candidate
    raise FileNotFoundError("Could not find the bundled FCAT-17 use_case directory.")


def project_coordinates(longitude, latitude):
    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", "EPSG:2154", always_xy=True
    )
    x, y = transformer.transform(longitude, latitude)
    return np.asarray(x, dtype=float) * 1e-3, np.asarray(y, dtype=float) * 1e-3


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
        # The source polygons share boundaries with millimetric numerical
        # overlaps after projection. A 1 cm erosion removes only those slivers.
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


def regime_partitions(data):
    zones = data["zones"]
    return {
        "zoning": list(zones),
        "hybrid": list(zones),
        "zoneless": [data["union"]],
        "aggregated": merge_adjacent_zones(
            zones, target_count=max(2, int(np.ceil(len(zones) / 2)))
        ),
    }


def _limit_held_fraction(held_geometry, zones, fold_index, maximum=0.65):
    """Prevent a fold from removing an entire zonal effect from training."""
    retained_parts = []
    for zone in zones:
        held_part = zone.intersection(held_geometry)
        if held_part.is_empty:
            continue
        if held_part.area > maximum * zone.area:
            xmin, ymin, xmax, ymax = zone.bounds
            if fold_index % 2 == 0:
                split = box(xmin, ymin, zone.centroid.x, ymax)
            else:
                split = box(xmin, ymin, xmax, zone.centroid.y)
            limited = held_part.intersection(split)
            if not limited.is_empty:
                held_part = limited
        retained_parts.append(held_part)
    return unary_union(retained_parts)


def spatial_block_folds(geometry, zones, x_bounds, y_bounds, profile):
    if profile == "smoke":
        n_side, n_folds, n_repeats = 3, 3, 1
    else:
        n_side, n_folds, n_repeats = 5, 5, 3
    x_edges = np.linspace(x_bounds[0], x_bounds[1], n_side + 1)
    y_edges = np.linspace(y_bounds[0], y_bounds[1], n_side + 1)
    folds = []
    for repeat in range(n_repeats):
        buckets = [[] for _ in range(n_folds)]
        multiplier = repeat + 2
        for ix, (left, right) in enumerate(zip(x_edges[:-1], x_edges[1:])):
            for iy, (lower, upper) in enumerate(zip(y_edges[:-1], y_edges[1:])):
                cell = box(left, lower, right, upper).intersection(geometry)
                if not cell.is_empty and cell.area > 0.0:
                    buckets[(ix + multiplier * iy + repeat) % n_folds].append(cell)
        for fold, cells in enumerate(buckets):
            held_geometry = unary_union(cells)
            held_geometry = _limit_held_fraction(
                held_geometry,
                zones,
                fold_index=fold + repeat,
            )
            folds.append((repeat, fold, held_geometry))
    return folds


def _training_zones(zones, held_geometry):
    clipped = []
    for zone in zones:
        training_zone = zone.difference(held_geometry)
        if training_zone.is_empty or training_zone.area <= 0.0:
            point = zone.representative_point()
            radius = max(np.sqrt(zone.area) * 1e-6, 1e-6)
            training_zone = point.buffer(radius).intersection(zone)
        clipped.append(training_zone)
    return clipped


def _build_model(zones, data, gp_prior):
    return make_model(
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


def _calibrate_training_gp(training_zones, training_catalog, data, seed):
    calibration_model = _build_model(training_zones, data, INITIAL_GP)
    calibrated = calibration_model.calibrate_gp_prior(
        training_catalog, rng_seed=seed, verbose=False
    )
    return calibrated


def _predictive_record(
    data,
    full_zones,
    training_zones,
    training_catalog,
    held_geometry,
    method,
    regime,
    campaign,
    seed,
    gp_prior,
):
    model = _build_model(training_zones, data, gp_prior)
    held_event_mask = points_in_geometry(data["catalog"].xy, held_geometry)
    event_xy = data["catalog"].xy[held_event_mask]
    quadrature_xy, quadrature_weights = area_weighted_grid(
        data["x_bounds"],
        data["y_bounds"],
        max(12, campaign.evaluation_grid),
        max(12, campaign.evaluation_grid),
        held_geometry,
    )
    if event_xy.size == 0 or quadrature_xy.size == 0:
        return None, None, None

    full_partition = DomainPartition.from_polygons(full_zones)
    event_domains = full_partition.locate(event_xy[:, 0], event_xy[:, 1])
    quadrature_domains = full_partition.locate(
        quadrature_xy[:, 0], quadrature_xy[:, 1]
    )
    if np.any(event_domains < 0) or np.any(quadrature_domains < 0):
        raise RuntimeError("FCAT-17 prediction points could not be assigned to a regime domain.")
    evaluation_xy = np.vstack([event_xy, quadrature_xy])
    domain_index = np.concatenate([event_domains, quadrature_domains])
    draws, diagnostics = fit_intensity_method(
        model,
        training_catalog,
        method,
        campaign,
        seed,
        evaluation_xy,
        domain_index=domain_index,
        return_log_intensity=True,
    )
    if draws is None:
        return {
            "status": diagnostics["status"],
            "regime": regime,
            "method": method,
            "method_label": METHOD_LABELS[method],
            "n_train": len(training_catalog),
            "n_held_out": len(event_xy),
            **diagnostics,
        }, None, None

    log_draws = diagnostics.pop("_log_intensity_draws")
    n_events = len(event_xy)
    event_draws = draws[:n_events]
    event_log_draws = log_draws[:n_events]
    quadrature_draws = draws[n_events:]
    log_likelihood = (
        np.sum(event_log_draws, axis=0)
        - data["duration"] * (quadrature_weights @ quadrature_draws)
    )
    log_score = float(logsumexp(log_likelihood) - np.log(log_likelihood.size))
    expected_counts = data["duration"] * (quadrature_weights @ quadrature_draws)
    rng = np.random.default_rng(seed + 37)
    predictive_counts = rng.poisson(expected_counts)
    count_interval_90 = np.quantile(predictive_counts, [0.05, 0.95])
    count_interval_50 = np.quantile(predictive_counts, [0.25, 0.75])
    observed_count = int(n_events)
    record = {
        "status": "ok",
        "regime": regime,
        "method": method,
        "method_label": METHOD_LABELS[method],
        "n_train": len(training_catalog),
        "n_held_out": observed_count,
        "log_score": log_score,
        "log_score_per_event": log_score / observed_count,
        "count_ecp_90": float(count_interval_90[0] <= observed_count <= count_interval_90[1]),
        "count_ecp_50": float(count_interval_50[0] <= observed_count <= count_interval_50[1]),
        "count_interval_width_90": float(np.diff(count_interval_90)[0]),
        "count_interval_width_50": float(np.diff(count_interval_50)[0]),
        "predictive_count_mean": float(predictive_counts.mean()),
        "gp_variance": gp_prior.variance,
        "gp_length_scale": gp_prior.length_scale,
        **diagnostics,
    }
    return record, quadrature_xy, quadrature_draws.mean(axis=1)


def _summary(records):
    summaries = []
    for regime in REGIMES:
        for method in sorted({record["method"] for record in records}):
            group = [
                record for record in records
                if record["regime"] == regime
                and record["method"] == method
                and record["status"] == "ok"
            ]
            if not group:
                continue
            summary = {
                "catalogue": "FCAT-17",
                "regime": regime,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "n_folds": len(group),
            }
            for field in (
                "log_score",
                "log_score_per_event",
                "count_ecp_90",
                "count_ecp_50",
                "count_interval_width_90",
                "runtime_seconds",
                "peak_memory_mb",
                "gp_variance",
                "gp_length_scale",
            ):
                values = np.asarray([record[field] for record in group], dtype=float)
                summary[field] = float(values.mean())
                summary[f"{field}_se"] = float(
                    values.std(ddof=1) / np.sqrt(values.size) if values.size > 1 else 0.0
                )
            summaries.append(summary)
    return summaries


def write_latex_table(path, summary):
    """Write rows matching the FCAT-17 table in the deliverable."""
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Catalogue & Regime & Log score & ECP & Time (s) \\",
        r"\midrule",
    ]
    for row in summary:
        lines.append(
            "FCAT-17 & "
            f"{row['regime']} & "
            f"{row['log_score']:.2f} & "
            f"{row['count_ecp_90']:.3f} & "
            f"{row['runtime_seconds']:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path = Path(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def plot_catalogue(data, *, save=True, show=False):
    figure, axis = plt.subplots(figsize=(8.2, 8.0), layout="constrained")
    for zone in data["zones"]:
        boundary = zone.boundary
        if boundary.geom_type == "LineString":
            x, y = boundary.xy
            axis.plot(x, y, color="0.35", linewidth=0.7)
    magnitude = data["catalog"].magnitudes
    axis.scatter(
        data["catalog"].x,
        data["catalog"].y,
        s=4.0 + 5.0 * (magnitude - MAGNITUDE_MIN),
        color="#a82828",
        alpha=0.55,
        linewidths=0,
    )
    for coastline in data["coastlines"]:
        axis.plot(coastline[0], coastline[1], color="black", linewidth=0.8)
    axis.set(
        xlim=data["x_bounds"], ylim=data["y_bounds"], aspect="equal",
        xlabel="x (km)", ylabel="y (km)",
        title=f"FCAT-17: {len(data['catalog'])} events, $M_w\\geq3$",
    )
    finish_figure(figure, "ssgc/fcat17_catalogue", save=save, show=show)


def plot_regime_predictions(predictions, data, *, save=True, show=False):
    available = [regime for regime in REGIMES if predictions.get(regime)]
    if not available:
        return
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 9.2), layout="constrained")
    axes = np.asarray(axes).reshape(-1)
    all_values = np.concatenate(
        [np.concatenate([values for _, values in predictions[regime]]) for regime in available]
    )
    lower = 0.0
    upper = float(np.quantile(all_values, 0.99))
    if upper <= lower:
        upper = 1.0
    levels = np.linspace(lower, upper, 36)
    color_map = plt.get_cmap("viridis")
    for axis, regime in zip(axes, available):
        points = np.vstack([xy for xy, _ in predictions[regime]])
        values = np.concatenate([field for _, field in predictions[regime]])
        unique_points, inverse = np.unique(points, axis=0, return_inverse=True)
        value_sum = np.bincount(inverse, weights=values)
        value_count = np.bincount(inverse)
        unique_values = value_sum / value_count
        image = axis.tricontourf(
            unique_points[:, 0],
            unique_points[:, 1],
            np.clip(unique_values, lower, upper),
            levels=levels,
            vmin=lower,
            vmax=upper,
            cmap=color_map,
            extend="max",
            antialiased=False,
        )
        magnitude = data["catalog"].magnitudes
        marker_size = 2.0 + 3.0 * (magnitude - MAGNITUDE_MIN)
        axis.scatter(
            data["catalog"].x,
            data["catalog"].y,
            s=marker_size,
            color="#c62828",
            alpha=0.30,
            linewidths=0,
            rasterized=True,
            zorder=3,
        )
        for zone in data["zones"]:
            boundary = zone.boundary
            lines = [boundary] if boundary.geom_type == "LineString" else boundary.geoms
            for line in lines:
                x_line, y_line = line.xy
                axis.plot(x_line, y_line, color="white", linewidth=0.4, alpha=0.28)
        for coastline in data["coastlines"]:
            axis.plot(
                coastline[0], coastline[1], color="white", linewidth=1.0, alpha=0.95
            )
        axis.set(
            xlim=data["x_bounds"],
            ylim=data["y_bounds"],
            aspect="equal",
            title=regime.title(),
            xlabel="x (km)",
            ylabel="y (km)",
        )
        axis.set_facecolor(color_map(0.0))
    for axis in axes[len(available):]:
        axis.set_visible(False)
    figure.colorbar(
        image,
        ax=axes[:len(available)].tolist(),
        label=r"Annual posterior mean intensity (events km$^{-2}$ yr$^{-1}$)",
        shrink=0.84,
        pad=0.02,
    )
    figure.suptitle(
        "FCAT-17 cross-validated background intensity by model regime",
        fontsize=14,
    )
    finish_figure(
        figure,
        "ssgc/fcat17_regime_intensities",
        save=save,
        show=show,
    )


def plot_scores(summary, *, save=True, show=False):
    if not summary:
        return
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.5, 4.8),
        layout="constrained",
    )
    labels = [
        f"{row['regime'].title()}\n"
        f"{textwrap.fill(row['method_label'], width=16)}"
        for row in summary
    ]
    x = np.arange(len(summary))
    axes[0].bar(x, [row["log_score_per_event"] for row in summary], color="#39706f")
    axes[0].set(ylabel="Held-out log score per event")
    axes[1].bar(x, [row["count_ecp_90"] for row in summary], color="#b65b43")
    axes[1].axhline(0.9, color="black", linestyle="--", linewidth=1)
    axes[1].set(ylabel="Predictive count ECP 90%", ylim=(0, 1.05))
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.tick_params(axis="x", labelsize=8)
        axis.grid(axis="y", alpha=0.25)
    finish_figure(
        figure,
        "ssgc/fcat17_cross_validation",
        save=save,
        show=show,
    )


def run(
    profile="smoke",
    methods=("vi_sparse",),
    regimes=REGIMES,
    *,
    save_figures=True,
    show_figures=False,
    campaign_overrides=None,
):
    """Run FCAT-17 cross-validation from Python or an interactive session."""
    methods = tuple(methods)
    regimes = tuple(regimes)
    unknown_methods = set(methods) - set(INFERENCE_METHODS)
    unknown_regimes = set(regimes) - set(REGIMES)
    if not methods or unknown_methods:
        unknown = ", ".join(sorted(unknown_methods))
        detail = f" Unknown method(s): {unknown}." if unknown else ""
        raise ValueError(f"At least one valid inference method is required.{detail}")
    if not regimes or unknown_regimes:
        unknown = ", ".join(sorted(unknown_regimes))
        detail = f" Unknown regime(s): {unknown}." if unknown else ""
        raise ValueError(f"At least one valid regime is required.{detail}")
    data = load_fcat17()
    campaign = configure_campaign(profile, **(campaign_overrides or {}))
    print(
        "FCAT-17 settings: "
        f"profile={campaign.name}, chains={campaign.n_chains}, "
        f"Gibbs={campaign.gibbs_iterations}, VI={campaign.vi_iterations}, "
        f"posterior_draws={campaign.posterior_draws}, "
        f"evaluation_grid={campaign.evaluation_grid}"
    )
    partitions = regime_partitions(data)
    folds = spatial_block_folds(
        data["union"], data["zones"], data["x_bounds"], data["y_bounds"], profile
    )
    if profile == "smoke":
        folds = folds[:2]
    records = []
    predictions = {regime: [] for regime in regimes}
    progress = tqdm(
        total=len(folds) * len(regimes) * len(methods),
        desc="FCAT-17 held-out fits",
        unit="fit",
        dynamic_ncols=True,
    )
    for repeat, fold, held_geometry in folds:
        held_mask = points_in_geometry(data["catalog"].xy, held_geometry)
        training_catalog = EventCatalog(
            t=data["catalog"].t[~held_mask],
            x=data["catalog"].x[~held_mask],
            y=data["catalog"].y[~held_mask],
            magnitudes=data["catalog"].magnitudes[~held_mask],
        )
        calibration_zones = _training_zones(data["zones"], held_geometry)
        if calibration_zones is None:
            progress.write(
                f"Skipping repeat={repeat}, fold={fold}: "
                "an entire domain is held out."
            )
            progress.update(len(regimes) * len(methods))
            continue
        calibrated_gp = _calibrate_training_gp(
            calibration_zones,
            training_catalog,
            data,
            seed=50_000 + 101 * repeat + fold,
        )
        for regime in regimes:
            full_zones = partitions[regime]
            training_zones = _training_zones(full_zones, held_geometry)
            if training_zones is None:
                progress.write(
                    f"Skipping {regime}, repeat={repeat}, fold={fold}: "
                    "empty training zone."
                )
                progress.update(len(methods))
                continue
            gp_prior = (
                GPParameters(variance=1e-4, length_scale=calibrated_gp.length_scale)
                if regime == "zoning"
                else calibrated_gp
            )
            for method in methods:
                record, points, estimate = _predictive_record(
                    data,
                    full_zones,
                    training_zones,
                    training_catalog,
                    held_geometry,
                    method,
                    regime,
                    campaign,
                    seed=60_000 + 1009 * repeat + 101 * fold + sum(map(ord, regime + method)),
                    gp_prior=gp_prior,
                )
                if record is None:
                    progress.update()
                    continue
                record.update({"repeat": repeat, "fold": fold})
                records.append(record)
                if points is not None:
                    predictions[regime].append((points, estimate))
                progress.update()
    progress.close()
    if not records:
        raise RuntimeError("No FCAT-17 cross-validation fold completed.")
    summary = _summary(records)
    output = RESULTS_ROOT / campaign.name
    write_records(output / "fcat17_block_cv_raw.csv", records)
    write_records(output / "fcat17_table.csv", summary)
    write_latex_table(output / "fcat17_table.tex", summary)
    if save_figures or show_figures:
        plot_catalogue(data, save=save_figures, show=show_figures)
        plot_regime_predictions(
            predictions,
            data,
            save=save_figures,
            show=show_figures,
        )
        plot_scores(summary, save=save_figures, show=show_figures)
    if show_figures and "ipykernel" not in sys.modules:
        plt.show()
    return records, summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=CAMPAIGNS, default="smoke")
    parser.add_argument("--methods", nargs="+", choices=INFERENCE_METHODS, default=["vi_sparse"])
    parser.add_argument("--regimes", nargs="+", choices=REGIMES, default=list(REGIMES))
    parser.add_argument("--n-chains", type=int, default=None)
    parser.add_argument("--gibbs-iterations", type=int, default=None)
    parser.add_argument("--gibbs-thin", type=int, default=None)
    parser.add_argument("--vi-iterations", type=int, default=None)
    parser.add_argument("--evaluation-grid", type=int, default=None)
    parser.add_argument("--quadrature-grid", type=int, default=None)
    parser.add_argument("--posterior-draws", type=int, default=None)
    parser.add_argument("--exact-max-events", type=int, default=None)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument(
        "--show-figures",
        action="store_true",
        help="Display figures in addition to saving them.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    overrides = {
        "n_chains": args.n_chains,
        "gibbs_iterations": args.gibbs_iterations,
        "gibbs_thin": args.gibbs_thin,
        "vi_iterations": args.vi_iterations,
        "evaluation_grid": args.evaluation_grid,
        "quadrature_grid": args.quadrature_grid,
        "posterior_draws": args.posterior_draws,
        "exact_max_events": args.exact_max_events,
    }
    records, summary = run(
        profile=args.profile,
        methods=tuple(args.methods),
        regimes=tuple(args.regimes),
        save_figures=not args.no_figures,
        show_figures=args.show_figures,
        campaign_overrides=overrides,
    )
    print(f"Completed {len(records)} FCAT-17 fold fits.")
    for row in summary:
        print(
            f"{row['regime']:<10} {row['method_label']:<22} "
            f"log_score={row['log_score']:.3f} "
            f"ECP90={row['count_ecp_90']:.3f} "
            f"time={row['runtime_seconds']:.2f}s"
        )
    print(f"Results: {RESULTS_ROOT / args.profile}")


if __name__ == "__main__":
    if "ipykernel" in sys.modules:
        run(
            profile=INTERACTIVE_PROFILE,
            methods=INTERACTIVE_METHODS,
            regimes=INTERACTIVE_REGIMES,
            save_figures=INTERACTIVE_SAVE_FIGURES,
            show_figures=INTERACTIVE_SHOW_FIGURES,
            campaign_overrides=INTERACTIVE_CAMPAIGN_OVERRIDES,
        )
    else:
        main()
