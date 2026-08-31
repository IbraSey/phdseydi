#%%

"""Run the four simulated-data experiments described in the SSGC draft.

Examples
--------
Validate the complete pipeline quickly::

    python experiments/exp_ssgc/deliverable_simulations.py --profile smoke --n-jobs -1

Launch deliverable-quality Experiment 1::

    python experiments/exp_ssgc/deliverable_simulations.py \
        --profile full --experiment 1 --n-jobs -1
"""

from __future__ import annotations

import argparse
import sys
from numbers import Integral
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, box
from tqdm.auto import tqdm

try:
    from joblib import Parallel, delayed, effective_n_jobs, parallel_config
except ImportError:  # pragma: no cover - exercised only without the optional dependency
    Parallel = delayed = effective_n_jobs = parallel_config = None

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover - joblib environments normally provide it
    threadpool_limits = None

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from deliverable_utils import (
    CAMPAIGNS,
    EXPERIMENT_1_DURATIONS,
    EXPERIMENT_2_DURATIONS,
    INFERENCE_METHODS,
    LATENT_FIELDS,
    METHOD_LABELS,
    PLOT_METHOD_LABELS,
    PROFILES,
    RESULTS_ROOT,
    X_BOUNDS,
    Y_BOUNDS,
    area_weighted_grid,
    calibrate_model,
    configure_campaign,
    finish_figure,
    fit_intensity_method,
    kde_draws,
    make_model,
    merge_adjacent_zones,
    plot_metric_summary,
    plot_metric_boxplots,
    posterior_metrics,
    profile_partition,
    reference_intensity,
    save_partition_figures,
    save_settings_figures,
    simulate_configuration,
    summarize_records,
    write_records,
)
from data import EventCatalog
from simulation import simulate_spatial_process


# %% 
# Edit this block, then run the complete file.
INTERACTIVE_PROFILE = "smoke"   # "full"
INTERACTIVE_EXPERIMENT = "1"     # "all", "1", "2", "3" or "4"
INTERACTIVE_METHODS = tuple(INFERENCE_METHODS)      # "gibbs_exact", "gibbs_sparse", "vi_exact", "vi_sparse"
INTERACTIVE_N_JOBS = -1     # 1: sequential; -1: use all available CPU 
INTERACTIVE_SAVE_FIGURES = True
INTERACTIVE_SHOW_FIGURES = True

# A value of None keeps the selected profile's default.
INTERACTIVE_CAMPAIGN_OVERRIDES = {
    "n_replicates": None,
    "n_chains": None,
    "gibbs_iterations": None,
    "gibbs_thin": None,
    "vi_iterations": None,
    "evaluation_grid": None,
    "quadrature_grid": None,
    "posterior_draws": None,
    "duration_scale": None,
    "exact_max_events": None,
    "use_calibration": None,
}


def _record(
    *,
    experiment,
    replicate,
    method,
    model_name,
    n_events,
    diagnostics,
    metrics=None,
    **labels,
):
    record = {
        "experiment": int(experiment),
        "replicate": int(replicate),
        "method": method,
        "method_label": METHOD_LABELS.get(method, method),
        "model": model_name,
        "n_events": int(n_events),
        **labels,
        **diagnostics,
    }
    if metrics is not None:
        record.update(metrics)
    return record


def _simulation_characteristics(simulation, **labels):
    catalog = simulation.catalog
    n_domains = len(simulation.domains)
    domain_indices = simulation.domains.locate(catalog.x, catalog.y)
    domain_counts = np.bincount(
        domain_indices[domain_indices >= 0],
        minlength=n_domains,
    )
    x_coordinates = simulation.grid.x[0]
    y_coordinates = simulation.grid.y[:, 0]
    spatial_integral = float(
        np.trapezoid(
            np.trapezoid(
                simulation.grid.intensity,
                x=x_coordinates,
                axis=1,
            ),
            x=y_coordinates,
        )
    )
    area = float(simulation.domains.observation_geometry.area)
    expected_count = simulation.duration * spatial_integral
    exposure = simulation.duration * area
    return {
        **labels,
        "n_events": len(catalog),
        "expected_n_events": expected_count,
        "duration": simulation.duration,
        "area": area,
        "n_domains": n_domains,
        "observed_event_rate": len(catalog) / exposure,
        "expected_event_rate": expected_count / exposure,
        "domain_event_counts": "|".join(map(str, domain_counts.tolist())),
        "baseline_intensities": "|".join(
            f"{value:g}" for value in simulation.baseline_intensities
        ),
        "latent_min": float(np.min(simulation.grid.latent)),
        "latent_max": float(np.max(simulation.grid.latent)),
        "intensity_min": float(np.min(simulation.grid.intensity)),
        "intensity_max": float(np.max(simulation.grid.intensity)),
    }


def _print_simulation_characteristics(characteristics):
    identifiers = " ".join(
        f"{name}={characteristics[name]}"
        for name in ("experiment", "profile", "setting", "data_case", "replicate")
        if name in characteristics
    )
    tqdm.write(
        f"[SIM {identifiers}] "
        f"N={characteristics['n_events']} "
        f"E[N]~{characteristics['expected_n_events']:.1f} "
        f"T={characteristics['duration']:.3g} "
        f"area={characteristics['area']:.3g} "
        f"domains={characteristics['n_domains']}"
    )
    tqdm.write(
        "    "
        f"observed_rate={characteristics['observed_event_rate']:.3f} "
        f"expected_rate={characteristics['expected_event_rate']:.3f} "
        f"intensity=[{characteristics['intensity_min']:.3f}, "
        f"{characteristics['intensity_max']:.3f}] "
        f"latent=[{characteristics['latent_min']:.3f}, "
        f"{characteristics['latent_max']:.3f}]"
    )
    tqdm.write(
        "    "
        f"domain_counts=[{characteristics['domain_event_counts'].replace('|', ', ')}] "
        f"baselines=[{characteristics['baseline_intensities'].replace('|', ', ')}]"
    )


def _plot_simulated_catalogue(
    simulation,
    characteristics,
    filename,
    title,
    *,
    save=True,
    show=False,
):
    figure, axis = plt.subplots(figsize=(6.8, 5.8), layout="constrained")
    upper = float(np.max(simulation.grid.intensity))
    if upper <= 0.0:
        upper = 1.0
    image = axis.pcolormesh(
        simulation.grid.x,
        simulation.grid.y,
        simulation.grid.intensity,
        shading="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=upper,
        rasterized=True,
    )
    figure.colorbar(
        image,
        ax=axis,
        label=r"Generating intensity $\mu^\star(s)$",
        shrink=0.86,
    )
    domain_counts = [
        int(value) for value in characteristics["domain_event_counts"].split("|")
    ]
    for index, zone in enumerate(simulation.domains.polygons):
        boundary = zone.boundary
        lines = [boundary] if boundary.geom_type == "LineString" else boundary.geoms
        for line in lines:
            x_line, y_line = line.xy
            axis.plot(x_line, y_line, color="white", linewidth=0.8, alpha=0.9)
        center = zone.representative_point()
        axis.text(
            center.x,
            center.y,
            f"D{index + 1}: {domain_counts[index]}",
            ha="center",
            va="center",
            fontsize=7,
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72},
        )
    if len(simulation.catalog):
        marker_size = float(np.clip(1200.0 / len(simulation.catalog), 3.0, 18.0))
        axis.scatter(
            simulation.catalog.x,
            simulation.catalog.y,
            s=marker_size,
            color="#b3261e",
            alpha=0.58,
            linewidths=0,
            rasterized=True,
        )
    axis.set(
        xlim=simulation.x_bounds,
        ylim=simulation.y_bounds,
        aspect="equal",
        xlabel="x",
        ylabel="y",
        title=(
            f"{title}\n"
            f"N={characteristics['n_events']}, "
            f"E[N]~{characteristics['expected_n_events']:.1f}, "
            f"T={characteristics['duration']:.3g}"
        ),
    )
    path = finish_figure(figure, filename, save=save, show=show)
    if path is not None:
        tqdm.write(f"    simulation_map={path}")


def _plot_intensity_panel(
    points,
    truth,
    estimates,
    filename,
    title,
    *,
    save=True,
    show=False,
):
    n_panels = 1 + len(estimates)
    n_columns = min(3, n_panels)
    n_rows = int(np.ceil(n_panels / n_columns))
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(3.8 * n_columns, 3.5 * n_rows),
        squeeze=False,
        layout="constrained",
    )
    flat_axes = axes.reshape(-1)
    fields = [("Truth", truth), *estimates]
    lower = min(float(np.min(field)) for _, field in fields)
    upper = max(float(np.max(field)) for _, field in fields)
    if np.isclose(lower, upper):
        upper = lower + max(abs(lower), 1.0) * 1e-6
    images = []
    for axis, (label, field) in zip(flat_axes, fields):
        image = axis.tripcolor(
            points[:, 0],
            points[:, 1],
            field,
            shading="gouraud",
            cmap="inferno",
            vmin=lower,
            vmax=upper,
            rasterized=True,
        )
        images.append(image)
        axis.set(aspect="equal", xlim=X_BOUNDS, ylim=Y_BOUNDS, xlabel="x", ylabel="y")
        axis.set_title(label)
    for axis in flat_axes[n_panels:]:
        axis.set_visible(False)
    figure.colorbar(
        images[-1],
        ax=flat_axes[:n_panels].tolist(),
        label=r"Intensity $\mu(s)$",
        shrink=0.86,
        pad=0.02,
    )
    figure.suptitle(title)
    finish_figure(figure, f"ssgc/{filename}", save=save, show=show)


def _fit_and_score(
    model,
    catalog,
    method,
    campaign,
    seed,
    evaluation_xy,
    truth,
    weights,
):
    draws, diagnostics = fit_intensity_method(
        model,
        catalog,
        method,
        campaign,
        seed,
        evaluation_xy,
        show_progress=False,
    )
    metrics = None if draws is None else posterior_metrics(draws, truth, weights)
    return draws, diagnostics, metrics


def _validate_n_jobs(n_jobs):
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, Integral):
        raise ValueError("n_jobs must be a non-zero integer.")
    n_jobs = int(n_jobs)
    if n_jobs == 0:
        raise ValueError("n_jobs must be a non-zero integer.")
    return n_jobs


def _call_with_limited_threads(worker, task):
    if threadpool_limits is None:
        return worker(*task)
    with threadpool_limits(limits=1):
        return worker(*task)


def _run_independent_tasks(
    worker,
    tasks,
    *,
    n_jobs,
    fits_per_task,
    description,
    recycle_workers=False,
):
    """Run deterministic independent tasks with bounded process parallelism."""
    tasks = tuple(tasks)
    if not tasks:
        return []
    n_jobs = _validate_n_jobs(n_jobs)
    fits_per_task = int(fits_per_task)
    if fits_per_task < 1:
        raise ValueError("fits_per_task must be positive.")

    if n_jobs == 1 or len(tasks) == 1:
        iterator = (worker(*task) for task in tasks)
        worker_count = 1
    else:
        if Parallel is None:
            raise ImportError(
                "joblib is required when n_jobs is different from 1."
            )
        worker_count = min(int(effective_n_jobs(n_jobs)), len(tasks))
        if worker_count <= 1:
            iterator = (worker(*task) for task in tasks)
            worker_count = 1
            backend_name = "sequential"
        elif recycle_workers:
            # OpenTURNS-backed Gibbs fits retain native allocations longer than
            # a Python task. Recycling after each dataset keeps long campaigns
            # stable instead of allowing memory to accumulate in a worker.
            backend_name = "multiprocessing (recycled)"

            def recycled_iterator():
                for start in range(0, len(tasks), worker_count):
                    batch = tasks[start : start + worker_count]
                    yield from Parallel(
                        n_jobs=len(batch),
                        backend="multiprocessing",
                        batch_size=1,
                        pre_dispatch=len(batch),
                        maxtasksperchild=1,
                    )(
                        delayed(_call_with_limited_threads)(worker, task)
                        for task in batch
                    )

            iterator = recycled_iterator()
        else:
            backend_name = "loky"
            # Each task already performs substantial NumPy/OpenTURNS work. One
            # inner numerical thread per process prevents CPU oversubscription.
            def parallel_iterator():
                with parallel_config(
                    backend="loky",
                    n_jobs=worker_count,
                    inner_max_num_threads=1,
                ):
                    yield from Parallel(
                        return_as="generator",
                        batch_size=1,
                        pre_dispatch=worker_count,
                    )(delayed(worker)(*task) for task in tasks)

            iterator = parallel_iterator()
    if worker_count == 1:
        backend_name = "sequential"

    tqdm.write(
        f"[{description}] tasks={len(tasks)} workers={worker_count} "
        f"fits_per_task={fits_per_task} backend={backend_name}"
    )
    results = []
    with tqdm(
        total=len(tasks) * fits_per_task,
        desc=description,
        unit="fit",
        dynamic_ncols=True,
    ) as progress:
        for result in iterator:
            results.append(result)
            progress.update(fits_per_task)
    return results


def _experiment_1_task(profile_name, setting, replicate, campaign, methods):
    duration = (
        EXPERIMENT_1_DURATIONS[(profile_name, setting)]
        * campaign.duration_scale
    )
    seed = 10_000 + 101 * replicate + 17 * int(profile_name) + ord(setting)
    simulation = simulate_configuration(
        profile_name,
        setting,
        duration,
        seed,
        grid_res=max(40, campaign.evaluation_grid),
    )
    catalog = simulation.catalog
    zones = list(simulation.domains.polygons)
    characteristics = _simulation_characteristics(
        simulation,
        experiment=1,
        profile=profile_name,
        setting=setting,
        replicate=replicate,
        seed=seed,
    )
    evaluation_xy, weights = area_weighted_grid(
        X_BOUNDS,
        Y_BOUNDS,
        campaign.evaluation_grid,
        campaign.evaluation_grid,
        simulation.domains.observation_geometry,
    )
    truth = reference_intensity(simulation)(
        evaluation_xy[:, 0], evaluation_xy[:, 1]
    )
    base_model = make_model(zones, duration)
    base_model, calibration = calibrate_model(
        base_model, catalog, campaign, seed
    )
    model_specs = {
        "SSGC": base_model,
        "Homogeneous SGCP": make_model(
            [box(X_BOUNDS[0], Y_BOUNDS[0], X_BOUNDS[1], Y_BOUNDS[1])],
            duration,
            gp_prior=base_model.gp_prior,
        ),
    }
    records = []
    representative_estimates = []
    for model_name, model in model_specs.items():
        for method in methods:
            draws, diagnostics, metrics = _fit_and_score(
                model,
                catalog,
                method,
                campaign,
                seed + sum(map(ord, model_name + method)),
                evaluation_xy,
                truth,
                weights,
            )
            records.append(
                _record(
                    experiment=1,
                    replicate=replicate,
                    method=method,
                    model_name=model_name,
                    n_events=len(catalog),
                    diagnostics=diagnostics | calibration,
                    metrics=metrics,
                    profile=profile_name,
                    setting=setting,
                )
            )
            if (
                draws is not None
                and replicate == 0
                and profile_name == "1"
                and setting == "A"
                and len(representative_estimates) < 4
            ):
                representative_estimates.append(
                    (
                        f"{model_name}\n{METHOD_LABELS[method]}",
                        draws.mean(axis=1),
                    )
                )

    kde = kde_draws(catalog, duration, evaluation_xy)
    records.append(
        _record(
            experiment=1,
            replicate=replicate,
            method="kde",
            model_name="KDE",
            n_events=len(catalog),
            diagnostics={
                "status": "ok",
                "runtime_seconds": 0.0,
                "peak_memory_mb": 0.0,
            },
            metrics=posterior_metrics(kde, truth, weights)
            | {
                "crps": float("nan"),
                "ecp_50": float("nan"),
                "ecp_90": float("nan"),
                "mpiw_50": float("nan"),
                "mpiw_90": float("nan"),
            },
            profile=profile_name,
            setting=setting,
        )
    )
    return {
        "records": records,
        "simulation_records": [characteristics],
        "representative_grid": (
            evaluation_xy if representative_estimates else None
        ),
        "representative_truth": truth if representative_estimates else None,
        "representative_estimates": representative_estimates,
    }


def run_experiment_1(
    campaign,
    methods,
    save_figures=True,
    show_figures=False,
    n_jobs=1,
):
    """SSGC versus homogeneous SGCP and KDE across six data settings."""
    methods = tuple(methods)
    tasks = [
        (profile_name, setting, replicate, campaign, methods)
        for profile_name in PROFILES
        for setting in LATENT_FIELDS
        for replicate in range(campaign.n_replicates)
    ]
    task_results = _run_independent_tasks(
        _experiment_1_task,
        tasks,
        n_jobs=n_jobs,
        fits_per_task=2 * len(methods) + 1,
        description="Experiment 1 fits",
        recycle_workers=any(method.startswith("gibbs") for method in methods),
    )
    records = []
    simulation_records = []
    representative_estimates = []
    representative_grid = representative_truth = None
    for task_result in task_results:
        records.extend(task_result["records"])
        simulation_records.extend(task_result["simulation_records"])
        for characteristics in task_result["simulation_records"]:
            _print_simulation_characteristics(characteristics)
        if task_result["representative_estimates"]:
            representative_estimates = task_result["representative_estimates"]
            representative_grid = task_result["representative_grid"]
            representative_truth = task_result["representative_truth"]

    if save_figures or show_figures:
        for characteristics in simulation_records:
            if characteristics["replicate"] != 0:
                continue
            profile_name = characteristics["profile"]
            setting = characteristics["setting"]
            simulation = simulate_configuration(
                profile_name,
                setting,
                characteristics["duration"],
                characteristics["seed"],
                grid_res=max(40, campaign.evaluation_grid),
            )
            _plot_simulated_catalogue(
                simulation,
                characteristics,
                (
                    f"ssgc/simulated_catalogues/{campaign.name}/"
                    f"experiment_1_profile_{profile_name}_setting_{setting}"
                ),
                f"Experiment 1 | Profile {profile_name} | Setting {setting}",
                save=save_figures,
                show=show_figures,
            )

    summary = summarize_records(records, ("profile", "setting", "model", "method_label"))
    for row in summary:
        row["comparison"] = f"{row['model']} / {row['method_label']}"
    output = RESULTS_ROOT / campaign.name
    write_records(output / "experiment_1_raw.csv", records)
    write_records(output / "experiment_1_summary.csv", summary)
    write_records(output / "experiment_1_simulations.csv", simulation_records)
    if save_figures or show_figures:
        plot_records = []
        for record in records:
            plot_record = dict(record)
            if record["model"] == "KDE":
                plot_record["plot_group"] = "KDE"
                plot_record["plot_model"] = "KDE"
            else:
                model_label = "SSGC" if record["model"] == "SSGC" else "SGCP"
                plot_record["plot_group"] = (
                    f"{model_label} / {PLOT_METHOD_LABELS[record['method']]}"
                )
                plot_record["plot_model"] = model_label
            plot_record["plot_method"] = PLOT_METHOD_LABELS[record["method"]]
            plot_records.append(plot_record)
        plot_metric_boxplots(
            plot_records,
            x_field="profile",
            group_field="plot_group",
            color_field="plot_method",
            style_field="plot_model",
            panel_field="setting",
            filename="experiment_1_scores",
            title="Experiment 1: SSGC, SGCP and KDE",
            save=save_figures,
            show=show_figures,
        )
        if representative_estimates:
            _plot_intensity_panel(
                representative_grid,
                representative_truth,
                representative_estimates,
                "experiment_1_intensity_comparison",
                "Profile 1 / Setting A",
                save=save_figures,
                show=show_figures,
            )
    return records


def _subcatalog(catalog, mask):
    magnitudes = None if catalog.magnitudes is None else catalog.magnitudes[mask]
    return EventCatalog(catalog.t[mask], catalog.x[mask], catalog.y[mask], magnitudes)


def _zonewise_draws(
    zones,
    catalog,
    duration,
    method,
    campaign,
    seed,
    evaluation_xy,
):
    assembled = np.zeros((evaluation_xy.shape[0], campaign.posterior_draws))
    runtime = 0.0
    peak_memory = 0.0
    zone_records = []
    for zone_index, zone in enumerate(zones):
        event_mask = np.array(
            [zone.covers(Point(float(x), float(y))) for x, y in catalog.xy],
            dtype=bool,
        )
        grid_mask = np.array(
            [zone.covers(Point(float(x), float(y))) for x, y in evaluation_xy],
            dtype=bool,
        )
        n_events = int(event_mask.sum())
        if n_events == 0 or not np.any(grid_mask):
            return None, {
                "status": "skipped_empty_domain",
                "runtime_seconds": float("nan"),
                "peak_memory_mb": float("nan"),
            }, zone_records
        local_catalog = _subcatalog(catalog, event_mask)
        local_model = make_model([zone], duration)
        local_model, _ = calibrate_model(local_model, local_catalog, campaign, seed + zone_index)
        local_draws, diagnostics = fit_intensity_method(
            local_model,
            local_catalog,
            method,
            campaign,
            seed + 101 * zone_index,
            evaluation_xy[grid_mask],
            show_progress=False,
        )
        if local_draws is None:
            return None, diagnostics, zone_records
        assembled[grid_mask] = local_draws
        runtime += diagnostics["runtime_seconds"]
        peak_memory = max(peak_memory, diagnostics["peak_memory_mb"])
        zone_records.append(
            {
                "zone": zone_index,
                "n_events_domain": n_events,
                "runtime_seconds": diagnostics["runtime_seconds"],
            }
        )
    return assembled, {
        "status": "ok",
        "runtime_seconds": runtime,
        "peak_memory_mb": peak_memory,
        "rhat_max": float("nan"),
        "ess_min": float("nan"),
    }, zone_records


def _experiment_2_task(profile_name, setting, replicate, campaign, methods):
    duration_scale = max(
        campaign.duration_scale,
        0.10 if campaign.name == "smoke" else 1.0,
    )
    duration = (
        EXPERIMENT_2_DURATIONS[(profile_name, setting)] * duration_scale
    )
    seed = 20_000 + 109 * replicate + 19 * int(profile_name) + ord(setting)
    simulation = simulate_configuration(profile_name, setting, duration, seed)
    catalog = simulation.catalog
    zones = list(simulation.domains.polygons)
    characteristics = _simulation_characteristics(
        simulation,
        experiment=2,
        profile=profile_name,
        setting=setting,
        replicate=replicate,
        seed=seed,
    )
    evaluation_xy, weights = area_weighted_grid(
        X_BOUNDS,
        Y_BOUNDS,
        campaign.evaluation_grid,
        campaign.evaluation_grid,
        simulation.domains.observation_geometry,
    )
    truth = reference_intensity(simulation)(
        evaluation_xy[:, 0], evaluation_xy[:, 1]
    )
    joint_model = make_model(zones, duration)
    joint_model, calibration = calibrate_model(
        joint_model, catalog, campaign, seed
    )
    true_domain = simulation.domains.locate(
        evaluation_xy[:, 0], evaluation_xy[:, 1]
    )
    records = []
    domain_records = []
    for method in methods:
        _, joint_diagnostics, joint_metrics = _fit_and_score(
            joint_model,
            catalog,
            method,
            campaign,
            seed,
            evaluation_xy,
            truth,
            weights,
        )
        records.append(
            _record(
                experiment=2,
                replicate=replicate,
                method=method,
                model_name="Joint SSGC",
                n_events=len(catalog),
                diagnostics=joint_diagnostics | calibration,
                metrics=joint_metrics,
                profile=profile_name,
                setting=setting,
            )
        )
        zonewise, zone_diagnostics, zones_run = _zonewise_draws(
            zones,
            catalog,
            duration,
            method,
            campaign,
            seed,
            evaluation_xy,
        )
        zone_metrics = (
            None
            if zonewise is None
            else posterior_metrics(zonewise, truth, weights)
        )
        records.append(
            _record(
                experiment=2,
                replicate=replicate,
                method=method,
                model_name="Independent domain SGCPs",
                n_events=len(catalog),
                diagnostics=zone_diagnostics,
                metrics=zone_metrics,
                profile=profile_name,
                setting=setting,
            )
        )
        if zonewise is not None:
            for zone_index, zone_info in enumerate(zones_run):
                mask = true_domain == zone_index
                local_metrics = posterior_metrics(
                    zonewise[mask], truth[mask], weights[mask]
                )
                domain_records.append(
                    {
                        "profile": profile_name,
                        "setting": setting,
                        "replicate": replicate,
                        "method": method,
                        **zone_info,
                        **local_metrics,
                    }
                )
    return {
        "records": records,
        "domain_records": domain_records,
        "simulation_records": [characteristics],
    }


def run_experiment_2(
    campaign,
    methods,
    save_figures=True,
    show_figures=False,
    n_jobs=1,
):
    """Joint SSGC versus independently fitted domain-wise SGCPs."""
    methods = tuple(methods)
    tasks = [
        (profile_name, setting, replicate, campaign, methods)
        for profile_name in ("1", "2")
        for setting in LATENT_FIELDS
        for replicate in range(campaign.n_replicates)
    ]
    task_results = _run_independent_tasks(
        _experiment_2_task,
        tasks,
        n_jobs=n_jobs,
        fits_per_task=2 * len(methods),
        description="Experiment 2 fits",
        recycle_workers=any(method.startswith("gibbs") for method in methods),
    )
    records = []
    domain_records = []
    simulation_records = []
    for task_result in task_results:
        records.extend(task_result["records"])
        domain_records.extend(task_result["domain_records"])
        simulation_records.extend(task_result["simulation_records"])
        for characteristics in task_result["simulation_records"]:
            _print_simulation_characteristics(characteristics)

    if save_figures or show_figures:
        for characteristics in simulation_records:
            if characteristics["replicate"] != 0:
                continue
            profile_name = characteristics["profile"]
            setting = characteristics["setting"]
            simulation = simulate_configuration(
                profile_name,
                setting,
                characteristics["duration"],
                characteristics["seed"],
            )
            _plot_simulated_catalogue(
                simulation,
                characteristics,
                (
                    f"ssgc/simulated_catalogues/{campaign.name}/"
                    f"experiment_2_profile_{profile_name}_setting_{setting}"
                ),
                f"Experiment 2 | Profile {profile_name} | Setting {setting}",
                save=save_figures,
                show=show_figures,
            )

    summary = summarize_records(records, ("profile", "setting", "model", "method_label"))
    for row in summary:
        row["comparison"] = f"{row['model']} / {row['method_label']}"
    output = RESULTS_ROOT / campaign.name
    write_records(output / "experiment_2_raw.csv", records)
    write_records(output / "experiment_2_summary.csv", summary)
    write_records(output / "experiment_2_simulations.csv", simulation_records)
    if domain_records:
        write_records(output / "experiment_2_by_domain.csv", domain_records)
    if save_figures or show_figures:
        plot_records = []
        for record in records:
            plot_record = dict(record)
            model_label = (
                "Joint SSGC"
                if record["model"] == "Joint SSGC"
                else "Independent SGCPs"
            )
            plot_record["plot_group"] = (
                f"{model_label} / {PLOT_METHOD_LABELS[record['method']]}"
            )
            plot_record["plot_model"] = model_label
            plot_record["plot_method"] = PLOT_METHOD_LABELS[record["method"]]
            plot_records.append(plot_record)
        plot_metric_boxplots(
            plot_records,
            x_field="profile",
            group_field="plot_group",
            color_field="plot_method",
            style_field="plot_model",
            panel_field="setting",
            filename="experiment_2_scores",
            title="Experiment 2: joint versus domain-wise inference",
            save=save_figures,
            show=show_figures,
        )
        if domain_records:
            figure, axis = plt.subplots(
                figsize=(6.2, 4.6),
                layout="constrained",
            )
            for method in methods:
                selected = [record for record in domain_records if record["method"] == method]
                if selected:
                    axis.scatter(
                        [record["n_events_domain"] for record in selected],
                        [record["rel_l2"] for record in selected],
                        alpha=0.65,
                        label=METHOD_LABELS[method],
                    )
            axis.set(xlabel=r"Events in domain $N_j$", ylabel=r"Domain relative $L_2$ error")
            axis.grid(alpha=0.25)
            axis.legend(fontsize="small")
            finish_figure(
                figure,
                "ssgc/experiment_2_domain_error",
                save=save_figures,
                show=show_figures,
            )
    return records


def _misspecification_cases(setting, duration, seed):
    full_domain = box(X_BOUNDS[0], Y_BOUNDS[0], X_BOUNDS[1], Y_BOUNDS[1])
    oracle, _ = profile_partition("1")
    crossed, _ = profile_partition("2", seed_offset=8)
    homogeneous = simulate_spatial_process(
        X_bounds=X_BOUNDS,
        Y_bounds=Y_BOUNDS,
        T=duration,
        polygons=[full_domain],
        mus=[5.0],
        f=LATENT_FIELDS[setting],
        grid_res=80,
        rng_seed=seed,
    )
    structured = simulate_configuration("1", setting, duration, seed + 1)
    return (
        ("M1", homogeneous, (("oracle_J1", [full_domain]), ("superfluous_J6", oracle))),
        ("M2", structured, (("oracle_J6", oracle), ("crossed_J5", crossed))),
        (
            "M2b",
            structured,
            (("oracle_J6", oracle), ("merged_J4", merge_adjacent_zones(oracle, 4))),
        ),
        ("M3", structured, (("oracle_J6", oracle), ("homogeneous_J1", [full_domain]))),
    )


def _experiment_3_task(setting, replicate, campaign, methods):
    duration = 100.0 * campaign.duration_scale
    seed = 30_000 + 113 * replicate + ord(setting)
    records = []
    simulation_records = []
    reported_data_cases = set()
    for scenario, simulation, specifications in _misspecification_cases(
        setting, duration, seed
    ):
        catalog = simulation.catalog
        data_case = "homogeneous" if scenario == "M1" else "structured"
        if data_case not in reported_data_cases:
            scenarios = "M1" if data_case == "homogeneous" else "M2|M2b|M3"
            simulation_records.append(
                _simulation_characteristics(
                    simulation,
                    experiment=3,
                    setting=setting,
                    data_case=data_case,
                    scenarios=scenarios,
                    replicate=replicate,
                    seed=seed if data_case == "homogeneous" else seed + 1,
                )
            )
            reported_data_cases.add(data_case)
        evaluation_xy, weights = area_weighted_grid(
            X_BOUNDS,
            Y_BOUNDS,
            campaign.evaluation_grid,
            campaign.evaluation_grid,
            simulation.domains.observation_geometry,
        )
        truth = reference_intensity(simulation)(
            evaluation_xy[:, 0], evaluation_xy[:, 1]
        )
        for partition_name, zones in specifications:
            model = make_model(zones, duration)
            model, calibration = calibrate_model(
                model, catalog, campaign, seed
            )
            for method in methods:
                _, diagnostics, metrics = _fit_and_score(
                    model,
                    catalog,
                    method,
                    campaign,
                    seed,
                    evaluation_xy,
                    truth,
                    weights,
                )
                records.append(
                    _record(
                        experiment=3,
                        replicate=replicate,
                        method=method,
                        model_name=partition_name,
                        n_events=len(catalog),
                        diagnostics=diagnostics | calibration,
                        metrics=metrics,
                        scenario=scenario,
                        setting=setting,
                    )
                )
    return {
        "records": records,
        "simulation_records": simulation_records,
    }


def run_experiment_3(
    campaign,
    methods,
    save_figures=True,
    show_figures=False,
    n_jobs=1,
):
    """Partition misspecification under both latent-field settings."""
    methods = tuple(methods)
    tasks = [
        (setting, replicate, campaign, methods)
        for setting in LATENT_FIELDS
        for replicate in range(campaign.n_replicates)
    ]
    task_results = _run_independent_tasks(
        _experiment_3_task,
        tasks,
        n_jobs=n_jobs,
        fits_per_task=8 * len(methods),
        description="Experiment 3 fits",
        recycle_workers=any(method.startswith("gibbs") for method in methods),
    )
    records = []
    simulation_records = []
    for task_result in task_results:
        records.extend(task_result["records"])
        simulation_records.extend(task_result["simulation_records"])
        for characteristics in task_result["simulation_records"]:
            _print_simulation_characteristics(characteristics)

    if save_figures or show_figures:
        characteristics_by_case = {
            (
                item["setting"],
                item["data_case"],
                item["replicate"],
            ): item
            for item in simulation_records
        }
        duration = 100.0 * campaign.duration_scale
        for setting in LATENT_FIELDS:
            seed = 30_000 + ord(setting)
            plotted_cases = set()
            for scenario, simulation, _ in _misspecification_cases(
                setting, duration, seed
            ):
                data_case = (
                    "homogeneous" if scenario == "M1" else "structured"
                )
                if data_case in plotted_cases:
                    continue
                characteristics = characteristics_by_case[
                    (setting, data_case, 0)
                ]
                _plot_simulated_catalogue(
                    simulation,
                    characteristics,
                    (
                        f"ssgc/simulated_catalogues/{campaign.name}/"
                        f"experiment_3_{data_case}_setting_{setting}"
                    ),
                    (
                        f"Experiment 3 | {data_case.title()} data | "
                        f"Setting {setting}"
                    ),
                    save=save_figures,
                    show=show_figures,
                )
                plotted_cases.add(data_case)

    summary = summarize_records(records, ("scenario", "setting", "model", "method_label"))
    for row in summary:
        row["comparison"] = f"{row['model']} / {row['method_label']}"
    output = RESULTS_ROOT / campaign.name
    write_records(output / "experiment_3_raw.csv", records)
    write_records(output / "experiment_3_summary.csv", summary)
    write_records(output / "experiment_3_simulations.csv", simulation_records)
    if save_figures or show_figures:
        plot_metric_summary(
            summary, "scenario", "comparison", "experiment_3_scores",
            "Experiment 3: partition misspecification",
            save=save_figures,
            show=show_figures,
        )
    return records


def _plot_sensitivity(records, methods, *, save=True, show=False):
    d0_values = sorted({float(record["delta_0"]) for record in records})
    d1_values = sorted({float(record["delta_1"]) for record in records})
    for method in methods:
        selected = [
            record for record in records
            if record["method"] == method and record.get("status") == "ok"
        ]
        if not selected:
            continue
        figure, axes = plt.subplots(1, 3, figsize=(14.8, 4.8), layout="constrained")
        for axis, metric, title in zip(
            axes,
            ("rel_l2", "crps", "ecp_90"),
            (r"Relative $L_2$", "CRPS", "ECP 90%"),
        ):
            matrix = np.full((len(d0_values), len(d1_values)), np.nan)
            for record in selected:
                row = d0_values.index(float(record["delta_0"]))
                column = d1_values.index(float(record["delta_1"]))
                matrix[row, column] = record[metric]
            color_map = plt.get_cmap("viridis").with_extremes(bad="#eeeeee")
            image = axis.imshow(
                np.ma.masked_invalid(matrix),
                origin="lower",
                aspect="equal",
                cmap=color_map,
                vmin=0.0 if metric == "ecp_90" else None,
                vmax=1.0 if metric == "ecp_90" else None,
            )
            axis.set_xticks(range(len(d1_values)), [f"{value:g}" for value in d1_values])
            axis.set_yticks(range(len(d0_values)), [f"{value:g}" for value in d0_values])
            axis.set(xlabel=r"$\delta_1$", ylabel=r"$\delta_0$", title=title)
            reference = (1.0, 0.01)
            for row in range(matrix.shape[0]):
                for column in range(matrix.shape[1]):
                    if np.isfinite(matrix[row, column]):
                        axis.text(
                            column,
                            row,
                            f"{matrix[row, column]:.2f}",
                            ha="center",
                            va="center",
                            fontsize=6.5,
                            color=(
                                "white"
                                if image.norm(matrix[row, column]) < 0.58
                                else "black"
                            ),
                        )
            if reference[0] in d0_values and reference[1] in d1_values:
                axis.plot(
                    d1_values.index(reference[1]) - 0.32,
                    d0_values.index(reference[0]) + 0.32,
                    marker="*",
                    markersize=9,
                    markerfacecolor="white",
                    markeredgecolor="black",
                    markeredgewidth=0.8,
                )
            figure.colorbar(image, ax=axis, shrink=0.82)
        figure.suptitle(f"Experiment 4: {METHOD_LABELS[method]}", fontsize=13)
        finish_figure(
            figure,
            f"ssgc/experiment_4_sensitivity_{method}",
            save=save,
            show=show,
        )


def _experiment_4_task(
    delta_0,
    delta_1,
    campaign,
    methods,
    gp_prior,
    calibration,
):
    duration = (
        EXPERIMENT_1_DURATIONS[("1", "A")] * campaign.duration_scale
    )
    simulation = simulate_configuration("1", "A", duration, seed=40_000)
    catalog = simulation.catalog
    zones = list(simulation.domains.polygons)
    evaluation_xy, weights = area_weighted_grid(
        X_BOUNDS,
        Y_BOUNDS,
        campaign.evaluation_grid,
        campaign.evaluation_grid,
        simulation.domains.observation_geometry,
    )
    truth = reference_intensity(simulation)(
        evaluation_xy[:, 0], evaluation_xy[:, 1]
    )
    model = make_model(
        zones,
        duration,
        eps_prior_variance=delta_0,
        eps_prior_length_scale=delta_1,
        gp_prior=gp_prior,
    )
    records = []
    for method in methods:
        _, diagnostics, metrics = _fit_and_score(
            model,
            catalog,
            method,
            campaign,
            40_000,
            evaluation_xy,
            truth,
            weights,
        )
        records.append(
            _record(
                experiment=4,
                replicate=0,
                method=method,
                model_name="SSGC",
                n_events=len(catalog),
                diagnostics=diagnostics | calibration,
                metrics=metrics,
                delta_0=delta_0,
                delta_1=delta_1,
            )
        )
    return records


def run_experiment_4(
    campaign,
    methods,
    save_figures=True,
    show_figures=False,
    n_jobs=1,
):
    """Sensitivity to the epsilon-prior variance and correlation range."""
    full_d0 = (0.01, 0.1, 0.5, 1.0, 2.0, 5.0)
    full_d1 = (0.001, 0.01, 0.05, 0.1, 0.5, 1.0)
    pairs = tuple((d0, d1) for d0 in full_d0 for d1 in full_d1)
    duration = EXPERIMENT_1_DURATIONS[("1", "A")] * campaign.duration_scale
    simulation = simulate_configuration("1", "A", duration, seed=40_000)
    catalog = simulation.catalog
    zones = list(simulation.domains.polygons)
    characteristics = _simulation_characteristics(
        simulation,
        experiment=4,
        profile="1",
        setting="A",
        replicate=0,
        seed=40_000,
    )
    _print_simulation_characteristics(characteristics)
    if save_figures or show_figures:
        _plot_simulated_catalogue(
            simulation,
            characteristics,
            f"ssgc/simulated_catalogues/{campaign.name}/experiment_4",
            "Experiment 4 | Profile 1 | Setting A",
            save=save_figures,
            show=show_figures,
        )
    reference_model = make_model(zones, duration)
    reference_model, calibration = calibrate_model(reference_model, catalog, campaign, 40_000)
    methods = tuple(methods)
    tasks = [
        (
            delta_0,
            delta_1,
            campaign,
            methods,
            reference_model.gp_prior,
            calibration,
        )
        for delta_0, delta_1 in pairs
    ]
    task_results = _run_independent_tasks(
        _experiment_4_task,
        tasks,
        n_jobs=n_jobs,
        fits_per_task=len(methods),
        description="Experiment 4 fits",
        recycle_workers=any(method.startswith("gibbs") for method in methods),
    )
    records = [record for result in task_results for record in result]
    output = RESULTS_ROOT / campaign.name
    write_records(output / "experiment_4_raw.csv", records)
    write_records(output / "experiment_4_simulations.csv", [characteristics])
    if save_figures or show_figures:
        _plot_sensitivity(
            records,
            methods,
            save=save_figures,
            show=show_figures,
        )
    return records


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=CAMPAIGNS, default="smoke")
    parser.add_argument(
        "--experiment",
        choices=("all", "1", "2", "3", "4"),
        default="1",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=INFERENCE_METHODS,
        default=list(INFERENCE_METHODS),
    )
    parser.add_argument("--replicates", type=int, default=None)
    parser.add_argument("--n-chains", type=int, default=None)
    parser.add_argument("--gibbs-iterations", type=int, default=None)
    parser.add_argument("--gibbs-thin", type=int, default=None)
    parser.add_argument("--vi-iterations", type=int, default=None)
    parser.add_argument("--evaluation-grid", type=int, default=None)
    parser.add_argument("--quadrature-grid", type=int, default=None)
    parser.add_argument("--posterior-draws", type=int, default=None)
    parser.add_argument("--duration-scale", type=float, default=None)
    parser.add_argument("--exact-max-events", type=int, default=None)
    parser.add_argument(
        "--calibration",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable GP hyperparameter calibration.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help=(
            "Number of independent worker processes. Use 1 for sequential "
            "execution and -1 for all available CPU cores."
        ),
    )
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument(
        "--show-figures",
        action="store_true",
        help="Display figures in addition to saving them.",
    )
    return parser.parse_args(argv)


def run_campaign(
    *,
    profile="smoke",
    experiment="1",
    methods=INFERENCE_METHODS,
    n_jobs=-1,
    save_figures=True,
    show_figures=False,
    campaign_overrides=None,
):
    """Run selected experiments from Python or an interactive session."""
    if experiment not in {"all", "1", "2", "3", "4"}:
        raise ValueError("experiment must be 'all', '1', '2', '3' or '4'.")
    methods = tuple(methods)
    unknown_methods = set(methods) - set(INFERENCE_METHODS)
    if not methods or unknown_methods:
        unknown = ", ".join(sorted(unknown_methods))
        detail = f" Unknown method(s): {unknown}." if unknown else ""
        raise ValueError(f"At least one valid inference method is required.{detail}")
    n_jobs = _validate_n_jobs(n_jobs)
    campaign = configure_campaign(profile, **(campaign_overrides or {}))

    print(
        "Campaign settings: "
        f"profile={campaign.name}, replicates={campaign.n_replicates}, "
        f"chains={campaign.n_chains}, Gibbs={campaign.gibbs_iterations}, "
        f"VI={campaign.vi_iterations}, posterior_draws={campaign.posterior_draws}, "
        f"calibration={campaign.use_calibration}, n_jobs={n_jobs}"
    )
    if save_figures or show_figures:
        save_settings_figures(save=save_figures, show=show_figures)
        save_partition_figures(save=save_figures, show=show_figures)
    runners = {
        "1": run_experiment_1,
        "2": run_experiment_2,
        "3": run_experiment_3,
        "4": run_experiment_4,
    }
    selected = runners if experiment == "all" else {experiment: runners[experiment]}
    for name, runner in selected.items():
        print(
            f"Running SSGC deliverable Experiment {name} "
            f"({campaign.name}, n_jobs={n_jobs})"
        )
        records = runner(
            campaign,
            methods,
            save_figures=save_figures,
            show_figures=show_figures,
            n_jobs=n_jobs,
        )
        completed = sum(record.get("status") == "ok" for record in records)
        print(f"completed={completed}, records={len(records)}")
    print(f"Results: {RESULTS_ROOT / campaign.name}")
    if show_figures and "ipykernel" not in sys.modules:
        plt.show()


def main(argv=None):
    args = parse_args(argv)
    overrides = {
        "n_replicates": args.replicates,
        "n_chains": args.n_chains,
        "gibbs_iterations": args.gibbs_iterations,
        "gibbs_thin": args.gibbs_thin,
        "vi_iterations": args.vi_iterations,
        "evaluation_grid": args.evaluation_grid,
        "quadrature_grid": args.quadrature_grid,
        "posterior_draws": args.posterior_draws,
        "duration_scale": args.duration_scale,
        "exact_max_events": args.exact_max_events,
        "use_calibration": args.calibration,
    }
    run_campaign(
        profile=args.profile,
        experiment=args.experiment,
        methods=args.methods,
        n_jobs=args.n_jobs,
        save_figures=not args.no_figures,
        show_figures=args.show_figures,
        campaign_overrides=overrides,
    )


if __name__ == "__main__":
    if "ipykernel" in sys.modules:
        run_campaign(
            profile=INTERACTIVE_PROFILE,
            experiment=INTERACTIVE_EXPERIMENT,
            methods=INTERACTIVE_METHODS,
            n_jobs=INTERACTIVE_N_JOBS,
            save_figures=INTERACTIVE_SAVE_FIGURES,
            show_figures=INTERACTIVE_SHOW_FIGURES,
            campaign_overrides=INTERACTIVE_CAMPAIGN_OVERRIDES,
        )
    else:
        main()

# %%
