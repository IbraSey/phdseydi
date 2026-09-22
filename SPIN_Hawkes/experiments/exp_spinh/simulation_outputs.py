"""Saved arrays, tables and figures for the numerical studies."""

from __future__ import annotations

from pathlib import Path
import csv

from shapely import from_wkt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np

from .simulation_settings import (
    ETAS_PARAMETER_NAMES,
    EXPERIMENT_2_BETA,
    EXPERIMENT_2_ETAS,
    METHODS,
    N_REGIONS,
    PARAMETER_NAMES,
    PARTITION_SEED,
    PARTITION_FIGURE_REPLICATE,
    SCENARIOS,
)
from .simulation_studies import _partition_scenarios
from .test_utils import generate_partition, write_records
from package import plot_spinh_parameter_marginals
from visualization import save_figure


ACCURACY_METRICS = (
    "rel_l2_background",
    "mae_background",
    "etas_parameter_log_error",
    "log_error_A",
    "log_error_alpha",
    "log_error_c",
    "log_error_p",
    "log_error_d",
    "log_error_q",
    "log_error_gamma",
    "background_brier",
    "background_accuracy",
    "background_f1",
    "mean_true_state_probability",
    "candidate_recall",
    "runtime_seconds",
)

PARTITION_METRICS = (
    "delta_rel_l2_background",
    "delta_etas_parameter_log_error",
    "delta_background_brier",
    "delta_runtime_seconds",
)

_INTERNAL_GIBBS_FIELDS = {
    "diagnostic_status",
    "mcmc_diagnostic_method",
    "proposal_warning",
    "n_iter_run",
    "burn_in_fraction",
    "mala_step",
    "sigma_mh_etas",
    "sigma_mh_beta",
}
_INTERNAL_GIBBS_PREFIXES = (
    "rhat_",
    "ess_",
    "mcse_",
    "acceptance_",
    "proposal_scale_",
)
_LEGACY_GIBBS_DIAGNOSTIC_FILES = (
    "experiment_1_gibbs_diagnostics.csv",
    "experiment_1_gibbs_diagnostics_summary.csv",
    "experiment_1_marginal_recovery_gibbs_diagnostics.csv",
    "experiment_1_marginal_recovery_gibbs_diagnostics_summary.csv",
)

MAX_SAVED_REPLICATE_FIGURES = 5
RECONSTRUCTION_DISPLAY_GRID_SIZE = 40


def _save_figure(
    figure,
    path,
    save,
    show,
    *,
    contains_rasterized_artists=False,
):
    if save:
        path = Path(path)
        save_figure(
            figure, path.name, output_dir=path.parent,
            figure_type="raster" if contains_rasterized_artists else "vector",
        )
    if show:
        figure.show()
    else:
        plt.close(figure)


def render_accuracy_outputs(
    records, summary, backgrounds, posteriors, selection, output,
    *, save=True, show=False, replicate_boxplots=False,
):
    """Render a run from its stored data; this function performs no inference."""
    if not save and not show:
        return
    chosen = {(row["scenario"], int(row["replicate"])) for row in selection}
    representative_backgrounds = [
        payload for payload in backgrounds
        if (payload["scenario"], int(payload["replicate"])) in chosen
    ]
    truths = generating_parameters(records)
    figure_selection = select_replicates_for_figures(records)
    plot_accuracy(summary, output, save=save, show=show)
    plot_accuracy_reconstruction(
        representative_backgrounds, output, save=save, show=show,
    )
    plot_accuracy_parameter_marginals(
        posteriors, selection, output, save=save, show=show, truths=truths,
    )
    for selected in figure_selection:
        scenario = selected["scenario"]
        replicate = int(selected["replicate"])
        replicate_backgrounds = [
            payload for payload in backgrounds
            if payload["scenario"] == scenario
            and int(payload["replicate"]) == replicate
        ]
        plot_accuracy_reconstruction(
            replicate_backgrounds,
            output,
            save=save,
            show=show,
            filename=(
                f"experiment_1_reconstruction_{scenario}_rep{replicate:02d}.pdf"
            ),
        )
    plot_accuracy_parameter_marginals(
        posteriors,
        figure_selection,
        output,
        save=save,
        show=show,
        truths=truths,
        include_replicate=True,
    )
    plot_accuracy_gibbs_diagnostics(
        posteriors, selection, output, save=save, show=show, truths=truths,
    )
    if replicate_boxplots:
        plot_accuracy_replicate_boxplots(
            records, output, save=save, show=show,
        )


def select_replicates_for_figures(records, limit=MAX_SAVED_REPLICATE_FIGURES):
    """Select every available replicate up to a deterministic per-scenario cap."""
    if isinstance(limit, bool) or not isinstance(limit, (int, np.integer)) or limit < 1:
        raise ValueError("limit must be a positive integer.")
    selected = []
    for scenario in SCENARIOS:
        replicates = sorted({
            int(row["replicate"])
            for row in records
            if row.get("scenario") == scenario
            and row.get("status") == "ok"
            and row.get("replicate") not in (None, "")
        })
        selected.extend(
            {"scenario": scenario, "replicate": replicate}
            for replicate in replicates[:limit]
        )
    return selected


def select_representative_reconstructions(records, reconstructions):
    """Select M1's median-error replicate, or the first available method."""
    payloads = {
        (payload["scenario"], payload["replicate"], payload["method"]): payload
        for payload in reconstructions
    }
    selected = []
    selection_records = []
    for scenario_name in SCENARIOS:
        reference_method = next((
            method for method in METHODS
            if any(
                row.get("scenario") == scenario_name and row.get("method") == method
                and row.get("status") == "ok"
                and (scenario_name, row["replicate"], method) in payloads
                for row in records
            )
        ), None)
        candidates = [
            record
            for record in records
            if record.get("scenario") == scenario_name
            and record.get("method") == reference_method
            and record.get("status") == "ok"
            and np.isfinite(record.get("rel_l2_background", np.nan))
            and (scenario_name, record["replicate"], reference_method) in payloads
        ]
        if not candidates:
            continue
        median_error = float(
            np.median([record["rel_l2_background"] for record in candidates])
        )
        representative = min(
            candidates,
            key=lambda record: (
                abs(record["rel_l2_background"] - median_error),
                record["replicate"],
            ),
        )
        selected.extend(
            payloads[(scenario_name, representative["replicate"], method)]
            for method in METHODS
            if (scenario_name, representative["replicate"], method) in payloads
        )
        selection_records.append(
            {
                "scenario": scenario_name,
                "method": reference_method,
                "replicate": int(representative["replicate"]),
                "median_rel_l2_background": median_error,
                "selected_rel_l2_background": float(
                    representative["rel_l2_background"]
                ),
            }
        )
    return selected, selection_records


def _background_surface(payload, estimate):
    suffix = "estimate" if estimate else "true"
    values = np.asarray(payload[f"background_{suffix}"], dtype=float)
    n_space = int(payload["space_grid_size"])
    return values.reshape(n_space, n_space)


def _refine_regular_surface(x_values, y_values, surface, n_side):
    """Interpolate a regular cell-centred surface for smoother display only."""
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    surface = np.asarray(surface, dtype=float)
    target = max(int(n_side), len(x_values), len(y_values))
    if target == len(x_values) == len(y_values):
        return x_values, y_values, surface

    dx = x_values[1] - x_values[0]
    dy = y_values[1] - y_values[0]
    x_bounds = (x_values[0] - dx / 2, x_values[-1] + dx / 2)
    y_bounds = (y_values[0] - dy / 2, y_values[-1] + dy / 2)
    refined_x = np.linspace(*x_bounds, target, endpoint=False)
    refined_y = np.linspace(*y_bounds, target, endpoint=False)
    refined_x += (x_bounds[1] - x_bounds[0]) / (2 * target)
    refined_y += (y_bounds[1] - y_bounds[0]) / (2 * target)
    along_x = np.vstack([
        np.interp(refined_x, x_values, row) for row in surface
    ])
    refined = np.vstack([
        np.interp(refined_y, y_values, along_x[:, column])
        for column in range(along_x.shape[1])
    ]).T
    return refined_x, refined_y, refined


def plot_accuracy_reconstruction(
    reconstructions,
    output,
    *,
    save=True,
    show=False,
    filename="experiment_1_reconstruction.pdf",
):
    """Compare truth and posterior-mean backgrounds in 3-by-2 catalogue blocks."""
    if not reconstructions:
        return
    payloads = {
        (payload["scenario"], payload["method"]): payload
        for payload in reconstructions
    }
    scenarios = [
        scenario for scenario in SCENARIOS
        if any(key[0] == scenario for key in payloads)
    ]
    methods = [
        method for method in METHODS
        if any(key[1] == method for key in payloads)
    ]
    values = np.concatenate(
        [
            _background_surface(payload, estimate=True).ravel()
            for payload in payloads.values()
        ]
        + [
            _background_surface(next(
                payload for (name, _), payload in payloads.items()
                if name == scenario
            ), estimate=False).ravel()
            for scenario in scenarios
        ]
    )
    finite = values[np.isfinite(values)]
    upper = float(np.max(finite)) if finite.size else 1.0
    if upper <= 0.0:
        upper = 1.0

    n_panels = 1 + len(methods)
    n_rows = int(np.ceil(n_panels / 2))
    figure, axes = plt.subplots(
        n_rows,
        2 * len(scenarios),
        figsize=(5.7 * len(scenarios), 2.75 * n_rows + 0.55),
        squeeze=False,
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    image = None
    active_axes = []
    for scenario_index, scenario in enumerate(scenarios):
        scenario_payloads = {
            method: payloads[(scenario, method)]
            for method in methods
            if (scenario, method) in payloads
        }
        reference = next(iter(scenario_payloads.values()))
        zones = (
            list(from_wkt(reference["zones_wkt"]))
            if reference.get("zones_wkt") is not None
            else generate_partition(N_REGIONS, seed=PARTITION_SEED)[0]
        )
        panel_payloads = [("True", reference, False)] + [
            (method.upper(), scenario_payloads.get(method), True)
            for method in methods
        ]
        payload = reference
        spatial_xy = np.asarray(payload["spatial_xy"], dtype=float)
        n_space = int(payload["space_grid_size"])
        x_values = spatial_xy[:n_space, 0]
        y_values = spatial_xy[::n_space, 1]
        dx, dy = x_values[1] - x_values[0], y_values[1] - y_values[0]
        x_bounds = (x_values[0] - dx / 2, x_values[-1] + dx / 2)
        y_bounds = (y_values[0] - dy / 2, y_values[-1] + dy / 2)
        for panel_index, (title, panel_payload, estimate) in enumerate(panel_payloads):
            row = panel_index // 2
            column = 2 * scenario_index + panel_index % 2
            axis = axes[row, column]
            if panel_payload is None:
                axis.text(0.5, 0.5, "Unavailable", ha="center", va="center")
                axis.set_axis_off()
                continue
            refined_x, refined_y, refined_surface = _refine_regular_surface(
                x_values,
                y_values,
                _background_surface(panel_payload, estimate),
                RECONSTRUCTION_DISPLAY_GRID_SIZE,
            )
            image = axis.pcolormesh(
                refined_x,
                refined_y,
                refined_surface,
                cmap="magma",
                vmin=0.0,
                vmax=upper,
                shading="auto",
                rasterized=True,
            )
            active_axes.append(axis)
            _plot_partition_boundaries(axis, zones, color="white", linewidth=0.55)
            axis.set(xlim=x_bounds, ylim=y_bounds, aspect="equal")
            axis.set_title(
                f"{scenario.title()}\n{title}"
                if len(scenarios) > 1 and row == 0
                else title
            )
            if row == n_rows - 1:
                axis.set_xlabel("x")
            if panel_index % 2 == 0:
                axis.set_ylabel("y")
        for panel_index in range(n_panels, 2 * n_rows):
            row = panel_index // 2
            column = 2 * scenario_index + panel_index % 2
            axes[row, column].set_axis_off()
    if len(scenarios) == 1:
        reference = next(iter(payloads.values()))
        figure.suptitle(
            f"{scenarios[0].title()} scenario, replicate "
            f"{int(reference['replicate'])}"
        )
    else:
        figure.suptitle("Representative background reconstructions")
    if image is not None:
        figure.colorbar(
            image,
            ax=active_axes,
            label="Background intensity",
            shrink=0.78,
            pad=0.015,
        )
    _save_figure(
        figure,
        output / filename,
        save,
        show,
        contains_rasterized_artists=True,
    )


def save_accuracy_backgrounds(
    reconstructions,
    output,
    *,
    stem="experiment_1_background_intensities",
):
    """Store the plotted true and method-specific background surfaces."""
    if not reconstructions:
        return None, []
    arrays = {}
    index = []
    truth_saved = set()
    for payload in reconstructions:
        scenario = payload["scenario"]
        replicate = int(payload["replicate"])
        method = payload["method"]
        prefix = f"{scenario}__rep{replicate:02d}"
        arrays[f"{prefix}__{method}"] = np.asarray(
            payload["background_estimate"], dtype=float
        )
        arrays[f"{prefix}__xy"] = np.asarray(payload["spatial_xy"], dtype=float)
        if payload.get("zones_wkt") is not None:
            arrays[f"{prefix}__zones_wkt"] = np.asarray(payload["zones_wkt"], dtype=str)
        if prefix not in truth_saved:
            arrays[f"{prefix}__true"] = np.asarray(
                payload["background_true"], dtype=float
            )
            truth_saved.add(prefix)
        index.append(
            {
                "scenario": scenario,
                "replicate": replicate,
                "method": method,
                "method_label": payload["method_label"],
                "grid_size": int(payload["space_grid_size"]),
            }
        )
    path = output / f"{stem}.npz"
    np.savez_compressed(path, **arrays)
    write_records(output / f"{stem}_index.csv", index)
    return path, index


def plot_accuracy(summary, output, *, save=True, show=False):
    if not summary:
        return
    figure, axes = plt.subplots(2, 4, figsize=(13.5, 6.8), layout="constrained")
    colors = {"easy": "#0072B2", "difficult": "#D55E00"}
    parameter_labels = {"alpha": r"\alpha", "gamma": r"\gamma"}
    panels = [("rel_l2_background", r"Background $e_{L_2}$")] + [
        (
            f"log_error_{name}",
            rf"${parameter_labels.get(name, name)}$ log-error",
        )
        for name in ETAS_PARAMETER_NAMES
    ]
    for axis, (metric, title) in zip(axes.flat, panels):
        width = 0.38
        x = np.arange(len(METHODS))
        for offset, scenario in zip((-0.5, 0.5), SCENARIOS):
            rows = {row["method"]: row for row in summary if row["scenario"] == scenario}
            values = [rows.get(method, {}).get(metric, np.nan) for method in METHODS]
            axis.bar(x + offset * width, values, width, color=colors[scenario], label=scenario.title())
        axis.set_title(title)
        axis.set_xticks(x, [method.upper() for method in METHODS])
        axis.set_ylabel(
            "Relative L2 error"
            if metric == "rel_l2_background"
            else "Absolute log error"
        )
        axis.grid(axis="y", alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.07),
        ncol=len(SCENARIOS),
        frameon=False,
    )
    _save_figure(figure, output / "experiment_1_accuracy.pdf", save, show)


_REPLICATE_BOXPLOT_METRICS = (
    ("rel_l2_background", r"Background $e_{L_2}$", "Relative L2 error", False),
    ("mae_background", "Background MAE", "Mean absolute error", False),
    ("etas_parameter_log_error", "Mean ETAS log-error", "Absolute log error", False),
    ("background_brier", "Background Brier score", "Score", False),
    ("background_f1", "Background F1 score", "Score", False),
    ("mean_true_state_probability", "True-state probability", "Probability", False),
    ("candidate_recall", "Candidate recall", "Recall", False),
    ("runtime_seconds", "Wall-clock time", "Seconds", True),
)


def _grouped_metric_boxplot(axis, records, metric, title, ylabel, log_scale):
    methods = [
        method for method in METHODS
        if any(row.get("method") == method for row in records)
    ]
    scenarios = [
        scenario for scenario in SCENARIOS
        if any(row.get("scenario") == scenario for row in records)
    ]
    colors = {"easy": "#0072B2", "difficult": "#D55E00"}
    centers = np.arange(len(methods), dtype=float)
    offsets = np.linspace(-0.22, 0.22, len(scenarios)) if len(scenarios) > 1 else [0.0]
    width = min(0.32, 0.7 / max(len(scenarios), 1))
    for scenario, offset in zip(scenarios, offsets):
        values, positions = [], []
        for center, method in zip(centers, methods):
            sample = np.asarray([
                row.get(metric, np.nan)
                for row in records
                if row.get("status") == "ok"
                and row.get("scenario") == scenario
                and row.get("method") == method
            ], dtype=float)
            sample = sample[np.isfinite(sample)]
            if sample.size:
                values.append(sample)
                positions.append(center + offset)
        if not values:
            continue
        artists = axis.boxplot(
            values,
            positions=positions,
            widths=width,
            patch_artist=True,
            whis=1.5,
            manage_ticks=False,
            boxprops={"linewidth": 0.9},
            whiskerprops={"linewidth": 0.8},
            capprops={"linewidth": 0.8},
            medianprops={"color": "#1A1A1A", "linewidth": 1.4},
            flierprops={
                "marker": "o", "markersize": 2.5, "alpha": 0.45,
                "markerfacecolor": colors.get(scenario, "#777777"),
                "markeredgecolor": "none",
            },
        )
        for box in artists["boxes"]:
            box.set_facecolor(colors.get(scenario, "#777777"))
            box.set_alpha(0.72)
    axis.set_title(title)
    axis.set_xticks(centers, [method.upper() for method in methods])
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25)
    if log_scale:
        axis.set_yscale("log")


def _plot_replicate_boxplot_grid(
    records, panels, output_path, *, save, show, truth_parameters=None,
):
    figure, axes = plt.subplots(2, 4, figsize=(14.5, 7.0), layout="constrained")
    colors = {"easy": "#0072B2", "difficult": "#D55E00"}
    for index, (axis, panel) in enumerate(zip(axes.flat, panels)):
        _grouped_metric_boxplot(axis, records, *panel)
        if truth_parameters is not None:
            parameter = truth_parameters[index]
            for scenario in SCENARIOS:
                truths = {
                    float(row[f"true_{parameter}"])
                    for row in records
                    if row.get("scenario") == scenario
                    and row.get(f"true_{parameter}") not in (None, "")
                }
                if len(truths) == 1:
                    axis.axhline(
                        truths.pop(), color=colors.get(scenario, "#777777"),
                        linestyle=":", linewidth=1.2,
                    )
    scenarios = [
        scenario for scenario in SCENARIOS
        if any(row.get("scenario") == scenario for row in records)
    ]
    handles = [
        Patch(facecolor=colors.get(scenario, "#777777"), alpha=0.72,
              label=scenario.title())
        for scenario in scenarios
    ]
    handles.append(
        Line2D([], [], color="#1A1A1A", linewidth=1.4, label="Median")
    )
    if truth_parameters is not None:
        handles.append(
            Line2D([], [], color="#555555", linestyle=":", linewidth=1.2,
                   label="Generating value")
        )
    figure.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.035),
        ncol=len(handles), frameon=False,
    )
    _save_figure(figure, output_path, save, show)


def plot_accuracy_replicate_boxplots(records, output, *, save=True, show=False):
    """Plot replicate distributions; each value summarizes one fitted catalogue."""
    completed = [row for row in records if row.get("status") == "ok"]
    if not completed:
        return
    _plot_replicate_boxplot_grid(
        completed,
        _REPLICATE_BOXPLOT_METRICS,
        Path(output) / "experiment_1_replicate_boxplots.pdf",
        save=save,
        show=show,
    )
    labels = {"alpha": r"\alpha", "gamma": r"\gamma", "beta": r"\beta"}
    estimate_panels = tuple(
        (
            f"estimate_{name}",
            rf"${labels.get(name, name)}$ posterior mean",
            "Posterior mean",
            False,
        )
        for name in PARAMETER_NAMES
    )
    _plot_replicate_boxplot_grid(
        completed,
        estimate_panels,
        Path(output) / "experiment_1_parameter_estimate_boxplots.pdf",
        save=save,
        show=show,
        truth_parameters=PARAMETER_NAMES,
    )
    parameter_panels = tuple(
        (
            f"log_error_{name}",
            rf"${labels.get(name, name)}$ log-error",
            "Absolute log error",
            False,
        )
        for name in PARAMETER_NAMES
    )
    _plot_replicate_boxplot_grid(
        completed,
        parameter_panels,
        Path(output) / "experiment_1_parameter_error_boxplots.pdf",
        save=save,
        show=show,
    )


def save_accuracy_posteriors(
    posteriors,
    output,
    *,
    stem="experiment_1_posterior_samples",
):
    """Store compact posterior draws with an explicit index."""
    if not posteriors:
        return None, []
    arrays = {}
    index = []
    for payload in posteriors:
        prefix = (
            f"{payload['scenario']}__rep{int(payload['replicate']):02d}__"
            f"{payload['method']}"
        )
        sizes = []
        for parameter, values in payload["samples"].items():
            values = np.asarray(values, dtype=float).reshape(-1)
            arrays[f"{prefix}__{parameter}"] = values
            sizes.append(len(values))
        index.append(
            {
                "scenario": payload["scenario"],
                "replicate": int(payload["replicate"]),
                "method": payload["method"],
                "method_label": payload["method_label"],
                "n_draws": min(sizes) if sizes else 0,
            }
        )
    path = output / f"{stem}.npz"
    np.savez_compressed(path, **arrays)
    write_records(output / f"{stem}_index.csv", index)
    save_gibbs_traces(posteriors, output)
    return path, index


def _gibbs_trace_key(payload):
    parts = [payload["scenario"]]
    if payload.get("fit_role"):
        parts.append(payload["fit_role"])
    parts.extend((f"rep{int(payload['replicate']):02d}", payload["method"]))
    return "__".join(parts)


def save_gibbs_traces(
    posteriors,
    output,
    *,
    archive_name="experiment_1_gibbs_traces.npz",
    index_name="experiment_1_gibbs_trace_index.csv",
):
    """Preserve all unthinned traces even when no figures are requested."""
    arrays, index = {}, []
    for payload in posteriors:
        if payload.get("gibbs_trace") is None:
            continue
        traces = np.asarray(payload["gibbs_trace"], dtype=float)
        key = _gibbs_trace_key(payload)
        arrays[key] = traces
        row = {
            "array_key": key,
            "scenario": payload["scenario"],
            "replicate": payload["replicate"],
            "method": payload["method"],
            "method_label": payload["method_label"],
            "n_chains": traces.shape[0],
            "n_iterations": traces.shape[1],
            "burn_in_iteration": int(traces.shape[1] * payload["burn_in_fraction"]),
            "adaptation_end": payload.get("adaptation_end"),
            "parameter_order": ",".join(PARAMETER_NAMES),
        }
        if payload.get("fit_role"):
            row["fit_role"] = payload["fit_role"]
        if payload.get("ess_min") is not None:
            row["ess_min"] = payload["ess_min"]
        row.update({
            f"ess_bulk_{name}": value
            for name, value in payload.get("ess", {}).items()
        })
        row.update({
            f"true_{name}": payload[f"true_{name}"]
            for name in PARAMETER_NAMES if f"true_{name}" in payload
        })
        index.append(row)
    np.savez_compressed(Path(output) / archive_name, **arrays)
    write_records(Path(output) / index_name, index)
    return Path(output) / archive_name, index


def save_partition_gibbs_traces(traces, output):
    return save_gibbs_traces(
        traces,
        output,
        archive_name="experiment_2_gibbs_traces.npz",
        index_name="experiment_2_gibbs_trace_index.csv",
    )


def load_gibbs_traces(output, *, experiment=1):
    """Read scalar traces and their original burn-in and parameter order."""
    output = Path(output)
    trace_path = output / f"experiment_{experiment}_gibbs_traces.npz"
    index_path = output / f"experiment_{experiment}_gibbs_trace_index.csv"
    if not trace_path.is_file() or not index_path.is_file():
        return []
    payloads = []
    with np.load(trace_path, allow_pickle=False) as archive:
        for row in _read_records(index_path):
            key = row.get("array_key") or _gibbs_trace_key(row)
            order = row["parameter_order"].split(",")
            traces = archive[key][:, :, [order.index(p) for p in PARAMETER_NAMES]]
            payloads.append({
                **row,
                "gibbs_trace": traces,
                "burn_in_fraction": row["burn_in_iteration"] / traces.shape[1],
                "adaptation_end": row.get("adaptation_end") or None,
            })
    return payloads


def load_accuracy_posteriors(output):
    """Read saved samples and traces without importing or fitting a sampler."""
    output = Path(output)
    sample_path = output / "experiment_1_posterior_samples.npz"
    index_path = output / "experiment_1_posterior_samples_index.csv"
    if not sample_path.is_file() or not index_path.is_file():
        return []
    payloads = {}
    with np.load(sample_path, allow_pickle=False) as archive:
        for row in _read_records(index_path):
            prefix = f"{row['scenario']}__rep{int(row['replicate']):02d}__{row['method']}"
            payloads[prefix] = {
                **row,
                "samples": {name: archive[f"{prefix}__{name}"] for name in PARAMETER_NAMES},
            }
    for trace in load_gibbs_traces(output):
        key = _gibbs_trace_key(trace)
        if key in payloads:
            payloads[key].update(trace)
    return list(payloads.values())


def load_accuracy_backgrounds(output):
    output = Path(output)
    path = output / "experiment_1_background_intensities.npz"
    index = output / "experiment_1_background_intensities_index.csv"
    if not path.is_file() or not index.is_file():
        return []
    payloads = []
    with np.load(path, allow_pickle=False) as archive:
        for row in _read_records(index):
            prefix = f"{row['scenario']}__rep{int(row['replicate']):02d}"
            payloads.append({
                **row,
                "space_grid_size": row["grid_size"],
                "spatial_xy": archive[f"{prefix}__xy"],
                "background_true": archive[f"{prefix}__true"],
                "background_estimate": archive[f"{prefix}__{row['method']}"],
                "zones_wkt": archive[f"{prefix}__zones_wkt"] if f"{prefix}__zones_wkt" in archive else None,
            })
    return payloads


def generating_parameters(records):
    """Use the generating values from this run, even after settings change."""
    return {
        (row["scenario"], int(row["replicate"])): {
            name: float(row[f"true_{name}"]) for name in PARAMETER_NAMES
        }
        for row in records
        if all(row.get(f"true_{name}") not in (None, "") for name in PARAMETER_NAMES)
    }


def _acf(values, max_lag):
    values = np.asarray(values, dtype=float)
    values = values - values.mean()
    denominator = float(np.dot(values, values))
    if denominator <= 0.0:
        return np.ones(max_lag + 1)
    return np.asarray(
        [
            np.dot(values[: values.size - lag], values[lag:]) / denominator
            for lag in range(max_lag + 1)
        ]
    )


def plot_accuracy_gibbs_diagnostics(
    posteriors,
    selection_records,
    output,
    *,
    max_lag=200,
    save=True,
    show=False,
    truths=None,
    filename_prefix="experiment_1",
    include_fit_role=False,
    include_replicate=False,
):
    """Save trace plots and ACFs for representative M1--M3 fits."""
    selected = {
        (row["scenario"], int(row["replicate"])) for row in selection_records
    }
    payloads = [
        payload
        for payload in posteriors
        if (payload["scenario"], int(payload["replicate"])) in selected
        and payload.get("gibbs_trace") is not None
    ]
    if not payloads:
        return []

    display_order = ("beta", "A", "alpha", "c", "p", "d", "q", "gamma")
    labels = {
        "beta": r"$\beta$",
        "A": r"$A$",
        "alpha": r"$\alpha$",
        "c": r"$c$",
        "p": r"$p$",
        "d": r"$d$",
        "q": r"$q$",
        "gamma": r"$\gamma$",
    }
    columns = [PARAMETER_NAMES.index(name) for name in display_order]
    colors = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#000000")
    saved = []
    for payload in payloads:
        scenario = payload["scenario"]
        method = payload["method"]
        replicate = int(payload["replicate"])
        traces = np.asarray(payload["gibbs_trace"], dtype=float)[:, :, columns]
        burn = int(traces.shape[1] * float(payload["burn_in_fraction"]))
        adaptation_end = payload.get("adaptation_end")
        truth = (truths or {}).get((scenario, replicate))
        if truth is None:
            truth = {
                **SCENARIOS[scenario]["etas"].as_dict(),
                "beta": SCENARIOS[scenario]["beta"],
            }
        role = payload.get("fit_role")
        context = f"{scenario} scenario, replicate {replicate}"
        if role:
            context += f", {role} fit"
        file_parts = [scenario]
        if include_fit_role and role:
            file_parts.append(role)
        if include_replicate:
            file_parts.append(f"rep{replicate:02d}")
        file_parts.append(method)
        file_suffix = "_".join(file_parts)

        figure, axes = plt.subplots(4, 2, figsize=(12, 10.5), sharex=True)
        for parameter, axis, column in zip(display_order, axes.flat, range(8)):
            for chain in range(traces.shape[0]):
                color = colors[chain % len(colors)]
                axis.plot(
                    traces[chain, :, column],
                    color=color,
                    linewidth=0.55,
                    alpha=0.8,
                    label=f"Chain {chain + 1}",
                )
            axis.axhline(
                truth[parameter], color="black", linestyle="--", linewidth=1.0,
                label="Generating value",
            )
            if adaptation_end is not None and int(adaptation_end) == burn:
                axis.axvline(
                    burn,
                    color="#CC79A7",
                    linewidth=1.0,
                    label="Adaptation and burn-in end",
                )
            else:
                if adaptation_end is not None:
                    axis.axvline(
                        int(adaptation_end),
                        color="#777777",
                        linestyle=":",
                        linewidth=1.0,
                        label="Adaptation end",
                    )
                axis.axvline(
                    burn, color="#CC79A7", linewidth=1.0, label="Burn-in end"
                )
            axis.set_title(labels[parameter])
            axis.grid(alpha=0.2)
        for axis in axes[-1]:
            axis.set_xlabel("Iteration")
        figure.suptitle(
            f"{payload['method_label']} - {context}",
            y=0.985,
        )
        handles, legend_labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(
            handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=min(len(legend_labels), 6),
            frameon=False,
            fontsize=8,
        )
        figure.subplots_adjust(
            left=0.07, right=0.98, bottom=0.06, top=0.89, hspace=0.32, wspace=0.15
        )
        trace_path = output / f"{filename_prefix}_gibbs_traces_{file_suffix}.pdf"
        _save_figure(figure, trace_path, save, show)

        n_post = traces.shape[1] - burn
        lag = min(int(max_lag), n_post - 1)
        if lag >= 1:
            figure, axes = plt.subplots(4, 2, figsize=(12, 10.5), sharex=True)
            lags = np.arange(lag + 1)
            reference_bound = 1.96 / np.sqrt(n_post)
            for parameter, axis, column in zip(display_order, axes.flat, range(8)):
                for chain in range(traces.shape[0]):
                    color = colors[chain % len(colors)]
                    axis.plot(
                        lags,
                        _acf(traces[chain, burn:, column], lag),
                        color=color,
                        linewidth=1.0,
                        alpha=0.9,
                        label=f"Chain {chain + 1}",
                    )
                axis.axhline(0.0, color="black", linewidth=0.7)
                axis.axhline(
                    reference_bound,
                    color="#777777",
                    linestyle=":",
                    linewidth=0.8,
                    label="White-noise reference",
                )
                axis.axhline(
                    -reference_bound,
                    color="#777777",
                    linestyle=":",
                    linewidth=0.8,
                )
                axis.set_ylim(-1.0, 1.0)
                axis.set_title(labels[parameter])
                axis.grid(alpha=0.2)
            for axis in axes[-1]:
                axis.set_xlabel("Lag (iterations)")
            figure.suptitle(
                f"ACF: {payload['method_label']} - {context}",
                y=0.985,
            )
            handles, legend_labels = axes.flat[0].get_legend_handles_labels()
            figure.legend(
                handles,
                legend_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.955),
                ncol=min(len(legend_labels), 5),
                frameon=False,
                fontsize=8,
            )
            figure.subplots_adjust(
                left=0.07,
                right=0.98,
                bottom=0.06,
                top=0.89,
                hspace=0.32,
                wspace=0.15,
            )
            acf_path = output / f"{filename_prefix}_gibbs_acf_{file_suffix}.pdf"
            _save_figure(figure, acf_path, save, show)
            saved.extend([trace_path, acf_path])

    return saved


def plot_partition_gibbs_diagnostics(
    traces, output, *, max_lag=200, save=True, show=False, truths=None,
):
    """Plot post-burn-in diagnostics for the representative partition fits."""
    representative = [
        payload for payload in traces
        if int(payload["replicate"]) == PARTITION_FIGURE_REPLICATE
    ]
    selection = [
        {"scenario": payload["scenario"], "replicate": payload["replicate"]}
        for payload in representative
    ]
    truth = {**EXPERIMENT_2_ETAS.as_dict(), "beta": EXPERIMENT_2_BETA}
    truths = {
        (payload["scenario"], int(payload["replicate"])): truth
        for payload in representative
    } | generating_parameters(representative) | (truths or {})
    return plot_accuracy_gibbs_diagnostics(
        representative,
        selection,
        output,
        max_lag=max_lag,
        save=save,
        show=show,
        truths=truths,
        filename_prefix="experiment_2",
        include_fit_role=True,
        include_replicate=True,
    )


def plot_accuracy_parameter_marginals(
    posteriors,
    selection_records,
    output,
    *,
    save=True,
    show=False,
    truths=None,
    include_replicate=False,
):
    """Compare M1--M5 marginals for each selected catalogue."""
    for selected in selection_records:
        scenario = selected["scenario"]
        replicate = int(selected["replicate"])
        available = {
            payload["method"]: payload
            for payload in posteriors
            if payload["scenario"] == scenario
            and int(payload["replicate"]) == replicate
        }
        if not available:
            continue
        reference = available.pop("m1", None)
        compared = {
            payload["method_label"]: payload["samples"]
            for payload in available.values()
        }
        if not compared and reference is None:
            continue
        if reference is None:
            first_method = next(iter(available))
            reference = available.pop(first_method)
            compared.pop(reference["method_label"], None)
        truth = (truths or {}).get((scenario, replicate)) or {
            **SCENARIOS[scenario]["etas"].as_dict(), "beta": SCENARIOS[scenario]["beta"]
        }
        figure_stem = f"experiment_1_parameter_marginals_{scenario}"
        if include_replicate:
            figure_stem += f"_rep{replicate:02d}"
        result = plot_spinh_parameter_marginals(
            compared or {reference["method_label"]: reference["samples"]},
            reference_posterior=(reference["samples"] if compared else None),
            reference_label=(reference["method_label"] if compared else "Reference"),
            true_parameters=truth,
            true_beta=truth["beta"],
            parameters=("beta", "A", "alpha", "c", "p", "d", "q", "gamma"),
            n_samples=max(
                len(values)
                for payload in posteriors
                if payload["scenario"] == scenario
                and int(payload["replicate"]) == replicate
                for values in payload["samples"].values()
            ),
            rng_seed=91_000 + replicate,
            title=f"{scenario.title()} scenario, replicate {replicate}",
            savefigure=save,
            title_savefig=figure_stem,
            output_dir=output,
            show=show,
        )
        if not show:
            plt.close(result["figure"])


def _parse_csv_value(value):
    if value == "":
        return ""
    if value in {"True", "False"}:
        return value == "True"
    try:
        number = float(value)
    except ValueError:
        return value
    return (
        int(number)
        if number.is_integer() and not any(c in value.lower() for c in (".", "e"))
        else number
    )


def _read_records(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return [
            {name: _parse_csv_value(value) for name, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def save_partition_surfaces(surfaces, output):
    """Store the actual figure grids and boundaries, not regenerated settings."""
    arrays, index = {}, []
    fields = (
        "evaluation_xy", "true_background", "fitted_background",
        "true_zones_wkt", "fit_zones_wkt",
    )
    for payload in surfaces:
        key = f"{payload['scenario']}__rep{int(payload['replicate']):02d}"
        for field in fields:
            arrays[f"{key}__{field}"] = np.asarray(payload[field])
        index.append({
            "array_key": key, "scenario": payload["scenario"],
            "replicate": payload["replicate"], "grid_size": payload["grid_size"],
        })
    np.savez_compressed(Path(output) / "experiment_2_background_intensities.npz", **arrays)
    write_records(Path(output) / "experiment_2_background_intensities_index.csv", index)


def load_partition_surfaces(output):
    output = Path(output)
    path = output / "experiment_2_background_intensities.npz"
    index = output / "experiment_2_background_intensities_index.csv"
    if not path.is_file() or not index.is_file():
        return []
    payloads = []
    with np.load(path, allow_pickle=False) as archive:
        for row in _read_records(index):
            prefix = row["array_key"] + "__"
            payloads.append({
                **row,
                **{key[len(prefix):]: archive[key] for key in archive.files if key.startswith(prefix)},
            })
    return payloads


def plot_partition(summary, output, *, save=True, show=False):
    if not summary:
        return
    metrics = (
        "delta_rel_l2_background",
        "delta_etas_parameter_log_error",
        "delta_background_brier",
    )
    titles = ("Background error", "ETAS parameter error", "Background Brier")
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for axis, metric, title in zip(axes.ravel(), metrics, titles):
        axis.bar([row["scenario"] for row in summary], [row[metric] for row in summary], color="#39706f")
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    _save_figure(
        figure,
        output / "experiment_2_partition_misspecification.pdf",
        save,
        show,
    )


def _plot_partition_boundaries(axis, zones, *, color="white", linewidth=0.8):
    for zone in zones:
        boundary = zone.boundary
        lines = [boundary] if boundary.geom_type == "LineString" else boundary.geoms
        for line in lines:
            x_values, y_values = line.xy
            axis.plot(x_values, y_values, color=color, linewidth=linewidth)


def plot_partition_surfaces(surfaces, output, *, save=True, show=False):
    """Compare generating and fitted backgrounds and their partitions."""
    if not surfaces:
        return
    payloads = {surface["scenario"]: surface for surface in surfaces}
    scenario_names = ("P0", "P1", "P2", "P3", "P4")
    available = [payloads[name] for name in scenario_names if name in payloads]
    if not available:
        return
    true_values = np.concatenate(
        [payload["true_background"] for payload in available]
    )
    upper = float(np.quantile(true_values[np.isfinite(true_values)], 0.99))
    if not np.isfinite(upper) or upper <= 0.0:
        upper = 1.0

    titles = {
        "P0": "P0: correct",
        "P1": "P1: spurious",
        "P2": "P2: displaced",
        "P3": "P3: merged",
        "P4": "P4: missing",
    }
    figure, axes = plt.subplots(
        2,
        len(scenario_names),
        figsize=(15.5, 6.4),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    image = None
    for column, scenario_name in enumerate(scenario_names):
        axes[0, column].set_title(titles[scenario_name])
        payload = payloads.get(scenario_name)
        if payload is None:
            for axis in axes[:, column]:
                axis.text(
                    0.5,
                    0.5,
                    "Unavailable",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
            continue
        grid_size = int(payload["grid_size"])
        evaluation_xy = np.asarray(payload["evaluation_xy"], dtype=float)
        x_values = evaluation_xy[:grid_size, 0]
        y_values = evaluation_xy[::grid_size, 1]
        if "true_zones_wkt" in payload and "fit_zones_wkt" in payload:
            true_zones = from_wkt(payload["true_zones_wkt"])
            fit_zones = from_wkt(payload["fit_zones_wkt"])
        else:
            # Compatibility with in-memory payloads from older versions.
            settings = _partition_scenarios()[scenario_name]
            true_zones, fit_zones = settings["true_zones"], settings["fit_zones"]
        panels = (
            (payload["true_background"], true_zones),
            (payload["fitted_background"], fit_zones),
        )
        for row, (field, zones) in enumerate(panels):
            axis = axes[row, column]
            image = axis.pcolormesh(
                x_values,
                y_values,
                np.asarray(field, dtype=float).reshape(grid_size, grid_size),
                shading="auto",
                cmap="viridis",
                vmin=0.0,
                vmax=upper,
                rasterized=True,
            )
            _plot_partition_boundaries(axis, zones)
            axis.set(
                xlim=(x_values[0] - np.diff(x_values)[0] / 2, x_values[-1] + np.diff(x_values)[-1] / 2),
                ylim=(y_values[0] - np.diff(y_values)[0] / 2, y_values[-1] + np.diff(y_values)[-1] / 2),
                aspect="equal",
            )
        axes[1, column].set_xlabel("x")
    axes[0, 0].set_ylabel("Generating\ny")
    axes[1, 0].set_ylabel("Fitted\ny")
    if image is not None:
        figure.colorbar(
            image,
            ax=axes.ravel().tolist(),
            label="Background intensity",
            shrink=0.82,
            pad=0.02,
        )
    _save_figure(
        figure,
        output / "experiment_2_partitions.pdf",
        save,
        show,
        contains_rasterized_artists=True,
    )


def _latex_metric(row, metric, digits=3):
    value = float(row.get(metric, np.nan))
    if not np.isfinite(value):
        return "--"
    return f"{value:.{digits}f}"


def _write_latex(path, columns, rows):
    alignment = "l" * len(columns)
    lines = [
        f"\\begin{{tabular}}{{{alignment}}}",
        "\\toprule",
        " & ".join(columns) + r" \\",
        "\\midrule",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_experiment_1_latex(output, accuracy_summary):
    intensity_rows = []
    parameter_rows = []
    for row in accuracy_summary:
        prefix = [row["scenario"].title(), row["method"].upper()]
        intensity_rows.append(
            prefix
            + [
                _latex_metric(row, "rel_l2_background"),
                _latex_metric(row, "mae_background"),
            ]
        )
        parameter_rows.append(
            prefix
            + [
                _latex_metric(row, "etas_parameter_log_error"),
                _latex_metric(row, "background_brier"),
                _latex_metric(row, "background_f1"),
                _latex_metric(row, "mean_true_state_probability"),
                _latex_metric(row, "candidate_recall"),
            ]
        )
    if accuracy_summary:
        _write_latex(
            output / "experiment_1_intensity_table.tex",
            (
                "Scenario",
                "Method",
                r"$e_{L_2}(\mu)$",
                r"$\operatorname{MAE}(\mu)$",
            ),
            intensity_rows,
        )
        _write_latex(
            output / "experiment_1_parameters_table.tex",
            (
                "Scenario",
                "Method",
                "ETAS log-error",
                r"$\mathrm{BS}_{\mathrm{bg}}$",
                r"$F_1$",
                r"$\pi_{z^\star}$",
                r"$\mathrm{Rec}_{\mathcal{C}}$",
            ),
            parameter_rows,
        )


def write_experiment_2_latex(output, summary):
    rows = [
        [
            row["scenario"],
            _latex_metric(row, "delta_rel_l2_background"),
            _latex_metric(row, "delta_etas_parameter_log_error"),
            _latex_metric(row, "delta_background_brier"),
            _latex_metric(row, "delta_runtime_seconds", 2),
        ]
        for row in summary
    ]
    _write_latex(
        output / "experiment_2_table.tex",
        (
            "Scenario",
            r"$\Delta e_{L_2}(\mu)$",
            r"$\Delta e_{\mathrm{ETAS}}$",
            r"$\Delta \mathrm{BS}_{\mathrm{bg}}$",
            r"$\Delta$ Time (s)",
        ),
        rows,
    )


def records_for_export(records):
    """Remove sampler-tuning diagnostics from scientific result files."""
    return [
        {
            name: value
            for name, value in record.items()
            if name not in _INTERNAL_GIBBS_FIELDS
            and not name.startswith(_INTERNAL_GIBBS_PREFIXES)
        }
        for record in records
    ]


def remove_legacy_gibbs_diagnostic_files(output):
    """Remove obsolete standalone diagnostic exports from earlier runs."""
    output = Path(output)
    for filename in _LEGACY_GIBBS_DIAGNOSTIC_FILES:
        (output / filename).unlink(missing_ok=True)
