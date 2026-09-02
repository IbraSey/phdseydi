"""

"""

# %% Imports
from __future__ import annotations
import argparse
import sys
from numbers import Integral
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from shapely.ops import unary_union

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exp_spinh.test_utils import (
    CAMPAIGNS,
    EXPERIMENT_2_BETA,
    EXPERIMENT_2_DURATIONS,
    EXPERIMENT_2_ETAS,
    HIGH_CONTRAST_MUS,
    INITIAL_ETAS,
    METHODS,
    MISSPECIFIED_PARTITION_REGIONS,
    MISSPECIFIED_PARTITION_SEED,
    N_REGIONS,
    PARTITION_SEED,
    REFERENCE_MUS,
    RESULTS_ROOT,
    SCENARIOS,
    TRAIN_FRACTION,
    X_BOUNDS,
    Y_BOUNDS,
    branching_metrics,
    calibrate_gp,
    candidate_diagnostics,
    configure_campaign,
    fit_spinh_method,
    generate_partition,
    intensity_recovery_metrics,
    latent_field,
    make_model,
    merge_adjacent_zones,
    omitted_temporal_mass,
    parameter_recovery_metrics,
    posterior_parameter_draws,
    posterior_background_draws,
    predictive_log_score,
    regular_spatial_grid,
    simulate_configuration,
    subset_catalog,
    summarize_records,
    temporal_cutoff,
    validate_scientific_settings,
    write_campaign,
    write_records,
)
from experiments.exp_spinh.runner_utils import (
    checkpoint_directory,
    effective_worker_count,
    parallel_map,
    resolve_n_jobs,
)
from spatial import DomainPartition


# %% ========================================================================
# EDITOR SETTINGS
# =============================================================================
# Edit only this block for a normal "Run Python File" / "Run in Interactive
# Window" workflow.  When the script receives command-line arguments, those
# arguments are used instead.  ``None`` means: keep the selected profile's
# default value.  Start with ``smoke`` before launching ``full``.

EDITOR_PROFILE = "full"         # validate memory use before selecting "full"
EDITOR_EXPERIMENT = "1"           # "1", "2" or "all"
EDITOR_PANEL = "both"             # "accuracy", "scaling" or "both"
EDITOR_METHODS = tuple(METHODS)    # any subset of ("m1", ..., "m5")
EDITOR_N_JOBS = -1                # simultaneous fits; start with 1, test 2 later
EDITOR_SCALING_TARGETS = None      # e.g. (500, 1_000); None uses profile defaults
EDITOR_RESUME = True               # reuse completed task checkpoints
EDITOR_SAVE_FIGURES = True
EDITOR_SHOW_FIGURES = False

# Experiment 2 uses this deterministic replicate and grid for the promised
# generating-versus-fitted partition figure.
PARTITION_FIGURE_REPLICATE = 0
PARTITION_FIGURE_GRID_SIZE = 80
SMOKE_SCALING_TARGETS = (40, 80)
FULL_SCALING_METHODS = {
    500: ("m1", "m2", "m3", "m4", "m5"),
    1_000: ("m2", "m3", "m4", "m5"),
    5_000: ("m3", "m5"),
}
FULL_SCALING_TARGETS = tuple(FULL_SCALING_METHODS)

EDITOR_CAMPAIGN_OVERRIDES = {
    "n_replicates": None,
    "n_scaling_replicates": None,
    "n_partition_replicates": None,
    "n_chains": None,
    "vi_starts": None,
    "gibbs_iterations": None,
    "gibbs_thin": None,
    "vi_iterations": None,
    "evaluation_space_grid": None,
    "evaluation_time_grid": None,
    "quadrature_space_grid": None,
    "quadrature_time_grid": None,
    "posterior_draws": None,
    "max_parallel_calibrations": None,
    "duration_scale": None,
    "exact_max_events": None,
    "dense_max_events": None,
    "use_calibration": None,
}

# Generating parameters, priors, domain bounds and the truncation threshold are
# grouped in the SCIENTIFIC SETTINGS section at the top of test_utils.py.

# %% End of editor settings

ACCURACY_METRICS = (
    "rel_l2_background",
    "rel_l2_triggering",
    "rel_l2_total",
    "mae_background",
    "mae_triggering",
    "mae_total",
    "predictive_log_score",
    "parameter_log_error",
    "background_brier",
    "background_accuracy",
    "background_f1",
    "mean_true_state_probability",
    "exact_parent_accuracy",
    "candidate_recall",
    "runtime_seconds",
    "branching_update_seconds",
    "ess_per_second",
)

_CHECKPOINT_SOURCES = (
    Path(__file__),
    Path(__file__).with_name("test_utils.py"),
    Path(__file__).with_name("runner_utils.py"),
    REPO_ROOT / "package",
)


def editor_run_options():
    """Return an isolated copy of the options from the editor settings block."""
    return {
        "profile": EDITOR_PROFILE,
        "experiment": EDITOR_EXPERIMENT,
        "panel": EDITOR_PANEL,
        "methods": tuple(EDITOR_METHODS),
        "n_jobs": EDITOR_N_JOBS,
        "scaling_targets": (
            None
            if EDITOR_SCALING_TARGETS is None
            else tuple(EDITOR_SCALING_TARGETS)
        ),
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


def _validate_execution_settings(n_jobs, scaling_targets):
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, Integral) or n_jobs == 0:
        raise ValueError("n_jobs must be a non-zero integer (use -1 for all CPUs).")
    if scaling_targets is None:
        return None
    targets = []
    for value in scaling_targets:
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError("Every scaling target must be a positive integer.")
        if int(value) not in targets:
            targets.append(int(value))
    if not targets:
        raise ValueError("scaling_targets cannot be empty.")
    return tuple(targets)


def scaling_plan(profile, methods, targets=None):
    """Intersect the requested methods with the predefined size-specific plan."""
    if targets is None:
        targets = SMOKE_SCALING_TARGETS if profile == "smoke" else FULL_SCALING_TARGETS
    plan = {}
    for target in targets:
        if target in FULL_SCALING_METHODS:
            allowed = FULL_SCALING_METHODS[target]
        elif profile == "smoke" and target in SMOKE_SCALING_TARGETS:
            allowed = tuple(METHODS)
        else:
            raise ValueError(
                f"No scaling methods configured for N={target}; "
                "add this target to FULL_SCALING_METHODS first."
            )
        selected = tuple(method for method in methods if method in allowed)
        if selected:
            plan[int(target)] = selected
    if not plan:
        raise ValueError("No requested method is scheduled at the selected scaling sizes.")
    return plan


def _validate_partition_figure_settings(campaign):
    if (
        isinstance(PARTITION_FIGURE_REPLICATE, bool)
        or not isinstance(PARTITION_FIGURE_REPLICATE, Integral)
        or not 0 <= PARTITION_FIGURE_REPLICATE < campaign.n_partition_replicates
    ):
        raise ValueError(
            "PARTITION_FIGURE_REPLICATE must identify a configured partition replicate."
        )
    if (
        isinstance(PARTITION_FIGURE_GRID_SIZE, bool)
        or not isinstance(PARTITION_FIGURE_GRID_SIZE, Integral)
        or PARTITION_FIGURE_GRID_SIZE < 20
    ):
        raise ValueError("PARTITION_FIGURE_GRID_SIZE must be an integer >= 20.")


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


def _print_run_settings(campaign, experiment, panel, methods, n_jobs, targets, output):
    print("\n" + "=" * 78)
    print("SPIN-H SIMULATED-DATA TESTS")
    print("=" * 78)
    print(
        f"Profile={campaign.name} | experiment={experiment} | panel={panel} | "
        f"methods={','.join(method.upper() for method in methods)}"
    )
    print(
        f"Replicates: accuracy={campaign.n_replicates}, "
        f"scaling={campaign.n_scaling_replicates}, "
        f"partition={campaign.n_partition_replicates} | "
        f"Gibbs: {campaign.n_chains} chain(s) x {campaign.gibbs_iterations} iter. | "
        f"VI: {campaign.vi_starts} start(s) x {campaign.vi_iterations} iter."
    )
    print(
        f"Posterior draws={campaign.posterior_draws}, "
        f"GP calibration={'on' if campaign.use_calibration else 'off'}, "
        f"workers={effective_worker_count(n_jobs)} (n_jobs={n_jobs})"
    )
    if campaign.use_calibration:
        print(
            "GP calibration: complete training catalogue, "
            f"at most {campaign.max_parallel_calibrations} simultaneous fit(s)"
        )
    if experiment in {"1", "all"} and panel in {"scaling", "both"}:
        for target, selected in scaling_plan(campaign.name, methods, targets).items():
            print(f"Scaling N={target}: {', '.join(method.upper() for method in selected)}")
        print("Scaling fits run without a wall-clock time limit.")
    print(f"Output directory: {output}")
    if campaign.name == "full":
        print("Full profile selected: this campaign can require substantial compute time.")


def _print_run_summary(results, output):
    print("\n" + "-" * 78)
    print("RUN SUMMARY")
    failures = []
    if "experiment_1_accuracy" in results:
        raw, summary = results["experiment_1_accuracy"]
        failures.extend(record for record in raw if record.get("status") != "ok")
        print("Experiment 1 - accuracy")
        print(f"{'Scenario':<12} {'Method':<6} {'N':>6} {'L2 total':>10} {'Score':>10} {'Time (s)':>10}")
        for row in summary:
            completed = [
                record
                for record in raw
                if record["scenario"] == row["scenario"]
                and record["method"] == row["method"]
                and record.get("status") == "ok"
            ]
            mean_events = int(round(np.mean([record["n_events"] for record in completed])))
            print(
                f"{row['scenario']:<12} {row['method'].upper():<6} "
                f"{mean_events:>6} {row['rel_l2_total']:>10.3f} "
                f"{row['predictive_log_score']:>10.3f} {row['runtime_seconds']:>10.2f}"
            )
    if "experiment_1_scaling" in results:
        raw, summary = results["experiment_1_scaling"]
        failures.extend(record for record in raw if record.get("status") != "ok")
        print("\nExperiment 1 - scaling")
        print(
            f"{'Target':>8} {'Method':<6} {'Observed':>9} "
            f"{'Retained pairs':>15} {'Branch (s)':>11} {'Status':<22}"
        )
        for row in summary:
            print(
                f"{row['target_n_events']:>8} {row['method'].upper():<6} "
                f"{row['n_events']:>9.0f} {row['candidate_parent_count']:>15.0f} "
                f"{row['branching_update_seconds']:>11.3f} {row['status']:<22}"
            )
    if "experiment_2" in results:
        raw, paired, summary, _ = results["experiment_2"]
        failures.extend(record for record in raw if record.get("status") != "ok")
        failures.extend(record for record in paired if record.get("status") != "ok")
        print("\nExperiment 2 - paired partition effects")
        print(f"{'Case':<6} {'Delta L2 bg':>12} {'Delta L2 trig':>14} {'Delta score':>12}")
        for row in summary:
            print(
                f"{row['scenario']:<6} {row['delta_rel_l2_background']:>12.3f} "
                f"{row['delta_rel_l2_triggering']:>14.3f} "
                f"{row['delta_predictive_log_score']:>12.3f}"
            )
    if failures:
        print(f"\nWarnings: {len(failures)} run(s) were skipped or failed.")
        for record in failures[:10]:
            identifier = "/".join(
                str(record.get(name, "?"))
                for name in ("scenario", "method", "replicate")
            )
            detail = record.get("error_message", record.get("status", "unknown"))
            print(f"  {identifier}: {detail}")
    else:
        print("\nAll requested runs completed successfully.")
    print(f"Results written to: {output}")
    print("-" * 78)


def _base_record(experiment, scenario, replicate, method, simulation, cutoff):
    return {
        "experiment": int(experiment),
        "scenario": scenario,
        "replicate": int(replicate),
        "method": method,
        "method_label": METHODS[method]["label"],
        "seed": int(10_000 * experiment + 100 * replicate + sum(map(ord, scenario))),
        "n_events": len(simulation.catalog),
        "n_background": simulation.n_background,
        "n_triggered": simulation.n_triggered,
        "true_background_fraction": simulation.n_background / max(len(simulation.catalog), 1),
        "maximum_generation": int(np.max(simulation.generations)) if len(simulation.catalog) else 0,
        "parent_time_window": float(cutoff),
    }


def _fit_accuracy_method(
    method,
    scenario_name,
    replicate,
    scenario,
    simulation,
    zones,
    training_catalog,
    training_parent_indices,
    train_end,
    cutoff,
    tail_mass,
    gp_prior,
    calibration_seconds,
    calibration_succeeded,
    calibration_n_events,
    campaign,
):
    reconstruction = None
    record = _base_record(1, scenario_name, replicate, method, simulation, cutoff)
    record.update(
        {
            "n_train": len(training_catalog),
            "n_test": len(simulation.catalog) - len(training_catalog),
            "omitted_temporal_mass": tail_mass,
            "gp_variance": gp_prior.variance,
            "gp_length_scale": gp_prior.length_scale,
            "gp_calibration_seconds": calibration_seconds,
            "gp_calibration_succeeded": calibration_succeeded,
            "gp_calibration_n_events": calibration_n_events,
        }
    )
    seed = int(record["seed"] + 1009 * list(METHODS).index(method))
    try:
        model = make_model(zones, train_end, etas=INITIAL_ETAS, gp_prior=gp_prior)
        bundle, diagnostics = fit_spinh_method(
            model,
            training_catalog,
            method,
            campaign,
            seed,
            parent_time_window=cutoff,
        )
        record.update(diagnostics)
        if bundle is None:
            record["runtime_seconds"] = float("nan")
            return record, reconstruction
        record["inference_seconds"] = diagnostics["runtime_seconds"]
        record["runtime_seconds"] = diagnostics["runtime_seconds"] + calibration_seconds
        record.update(
            branching_metrics(
                bundle,
                training_parent_indices,
                training_catalog.t,
                cutoff,
            )
        )
        parameter_draws = posterior_parameter_draws(
            bundle, campaign.posterior_draws, seed=seed + 17
        )
        record.update(
            parameter_recovery_metrics(
                parameter_draws, scenario["etas"], scenario["beta"]
            )
        )
        intensity_result = intensity_recovery_metrics(
            bundle,
            simulation,
            scenario["mus"],
            scenario["field_scale"],
            scenario["etas"],
            campaign,
            seed + 31,
            return_payload=(method == "m1"),
        )
        if method == "m1":
            intensity_metrics, reconstruction = intensity_result
            reconstruction.update(
                {
                    "scenario": scenario_name,
                    "replicate": int(replicate),
                    "method": method,
                }
            )
        else:
            intensity_metrics = intensity_result
        record.update(intensity_metrics)
        record["predictive_log_score"] = predictive_log_score(
            bundle,
            simulation.catalog,
            train_end,
            campaign,
            seed + 43,
        )
        candidates = candidate_diagnostics(
            training_catalog.t, training_parent_indices, cutoff
        )
        dense_pairs = candidates["dense_candidate_count"]
        retained_pairs = (
            candidates["candidate_parent_count"]
            if METHODS[method]["truncated"]
            else dense_pairs
        )
        record.update(
            {
                "candidate_parent_count": int(retained_pairs),
                "dense_candidate_count": int(dense_pairs),
                "retained_candidate_fraction": retained_pairs / max(dense_pairs, 1),
                "mean_candidate_count": (
                    candidates["mean_candidate_count"]
                    if METHODS[method]["truncated"]
                    else dense_pairs / max(len(training_catalog), 1)
                ),
                "candidate_count_q95": candidates["candidate_count_q95"],
            }
        )
    except Exception as error:
        record.update(
            {
                "status": "error",
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )
    return record, reconstruction


def _accuracy_replicate(scenario_name, replicate, methods, campaign):
    scenario = SCENARIOS[scenario_name]
    zones, _ = generate_partition(N_REGIONS, seed=PARTITION_SEED)
    duration = scenario["duration"] * campaign.duration_scale
    seed = 11_000 + 1000 * list(SCENARIOS).index(scenario_name) + replicate
    simulation = simulate_configuration(
        zones,
        scenario["mus"],
        duration,
        scenario["field_scale"],
        scenario["etas"],
        scenario["beta"],
        seed,
        grid_res=40 if campaign.name == "smoke" else 100,
    )
    train_end = TRAIN_FRACTION * duration
    training_mask = simulation.catalog.t <= train_end
    training_catalog = subset_catalog(simulation.catalog, training_mask)
    training_parent_indices = simulation.parent_indices[training_mask]
    cutoff = temporal_cutoff(scenario["etas"])
    tail_mass = omitted_temporal_mass(scenario["etas"], cutoff, duration)
    calibration_model = make_model(zones, train_end, etas=INITIAL_ETAS)
    (
        gp_prior,
        calibration_seconds,
        calibration_succeeded,
        calibration_n_events,
    ) = calibrate_gp(
        calibration_model, training_catalog, campaign, seed + 71
    )
    return [
        _fit_accuracy_method(
            method,
            scenario_name,
            replicate,
            scenario,
            simulation,
            zones,
            training_catalog,
            training_parent_indices,
            train_end,
            cutoff,
            tail_mass,
            gp_prior,
            calibration_seconds,
            calibration_succeeded,
            calibration_n_events,
            campaign,
        )
        for method in methods
    ]


def run_accuracy_panel(
    campaign,
    methods,
    n_jobs=1,
    *,
    checkpoint_dir=None,
    resume=True,
):
    tasks = [
        (scenario, replicate, tuple(methods), campaign)
        for scenario in SCENARIOS
        for replicate in range(campaign.n_replicates)
    ]
    task_keys = [(scenario, replicate) for scenario, replicate, *_ in tasks]
    nested = parallel_map(
        _accuracy_replicate,
        tasks,
        n_jobs,
        "Experiment 1 accuracy",
        task_keys=task_keys,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        max_parallel_calibrations=campaign.max_parallel_calibrations,
    )
    fitted = [result for group in nested for result in group]
    records = [record for record, _ in fitted]
    reconstructions = [payload for _, payload in fitted if payload is not None]
    return records, reconstructions


def _prepare_scaling_replicate(target_size, replicate, campaign):
    """Simulate and calibrate once for all methods in one paired replicate."""
    scenario_name = "easy"
    scenario = SCENARIOS[scenario_name]
    zones, _ = generate_partition(N_REGIONS, seed=PARTITION_SEED)
    duration = scenario["duration"] * float(target_size) / 500.0
    seed = 31_000 + 10 * int(target_size) + replicate
    simulation = simulate_configuration(
        zones,
        scenario["mus"],
        duration,
        scenario["field_scale"],
        scenario["etas"],
        scenario["beta"],
        seed,
        grid_res=30 if campaign.name == "smoke" else 80,
    )
    cutoff = temporal_cutoff(scenario["etas"])
    calibration_model = make_model(zones, duration, etas=INITIAL_ETAS)
    gp_prior, calibration_seconds, calibration_succeeded, calibration_n_events = (
        calibrate_gp(calibration_model, simulation.catalog, campaign, seed + 71)
    )
    candidates = candidate_diagnostics(
        simulation.catalog.t, simulation.parent_indices, cutoff
    )
    return (
        scenario_name,
        scenario,
        zones,
        duration,
        seed,
        simulation.catalog,
        cutoff,
        gp_prior,
        calibration_seconds,
        calibration_succeeded,
        calibration_n_events,
        candidates,
    )


def _scaling_method_task(target_size, replicate, method, campaign, prepared):
    (
        scenario_name,
        scenario,
        zones,
        duration,
        seed,
        catalog,
        cutoff,
        gp_prior,
        calibration_seconds,
        calibration_succeeded,
        calibration_n_events,
        candidates,
    ) = prepared
    dense_pairs = candidates["dense_candidate_count"]
    retained_pairs = (
        candidates["candidate_parent_count"]
        if METHODS[method]["truncated"]
        else dense_pairs
    )
    record = {
        "experiment": 1,
        "panel": "scaling",
        "scenario": scenario_name,
        "target_n_events": int(target_size),
        "replicate": int(replicate),
        "method": method,
        "method_label": METHODS[method]["label"],
        "seed": seed,
        "n_events": len(catalog),
        "duration": duration,
        "parent_time_window": cutoff,
        "omitted_temporal_mass": omitted_temporal_mass(
            scenario["etas"], cutoff, duration
        ),
        "candidate_parent_count": int(retained_pairs),
        "dense_candidate_count": int(dense_pairs),
        "retained_candidate_fraction": retained_pairs / max(dense_pairs, 1),
        "true_parent_candidate_recall": candidates["true_parent_candidate_recall"],
        "gp_calibration_seconds": calibration_seconds,
        "gp_calibration_succeeded": calibration_succeeded,
        "gp_calibration_n_events": calibration_n_events,
    }
    try:
        model = make_model(zones, duration, etas=INITIAL_ETAS, gp_prior=gp_prior)
        bundle, diagnostics = fit_spinh_method(
            model,
            catalog,
            method,
            campaign,
            seed + 1009 * list(METHODS).index(method),
            parent_time_window=cutoff,
        )
        record.update(diagnostics)
        record["inference_seconds"] = diagnostics.get("runtime_seconds", float("nan"))
        record["runtime_seconds"] = (
            record["inference_seconds"] + calibration_seconds
            if bundle is not None else float("nan")
        )
    except Exception as error:
        record.update(
            status="error",
            runtime_seconds=float("nan"),
            error_type=type(error).__name__,
            error_message=str(error),
        )
    return record


def run_scaling_panel(
    campaign,
    methods,
    n_jobs=1,
    targets=None,
    *,
    checkpoint_dir=None,
    resume=True,
):
    plan = scaling_plan(campaign.name, methods, targets)
    preparation_tasks = [
        (target, replicate, campaign)
        for target in plan
        for replicate in range(campaign.n_scaling_replicates)
    ]
    preparation_keys = [(target, replicate) for target, replicate, _ in preparation_tasks]
    preparation_dir = None
    if checkpoint_dir is not None:
        preparation_dir = Path(checkpoint_dir) / "preparation"
        preparation_dir.mkdir(parents=True, exist_ok=True)
    prepared = parallel_map(
        _prepare_scaling_replicate,
        preparation_tasks,
        n_jobs,
        "Experiment 1 scaling preparation",
        task_keys=preparation_keys,
        checkpoint_dir=preparation_dir,
        resume=resume,
        max_parallel_calibrations=campaign.max_parallel_calibrations,
    )
    paired_inputs = dict(zip(preparation_keys, prepared))
    tasks = [
        (target, replicate, method, campaign, paired_inputs[(target, replicate)])
        for target, selected_methods in plan.items()
        for replicate in range(campaign.n_scaling_replicates)
        for method in selected_methods
    ]
    task_keys = [
        (target, replicate, method) for target, replicate, method, *_ in tasks
    ]
    records = parallel_map(
        _scaling_method_task,
        tasks,
        n_jobs,
        "Experiment 1 scaling",
        task_keys=task_keys,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        max_parallel_calibrations=campaign.max_parallel_calibrations,
    )
    return records


def _median_iqr(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return float("nan"), float("nan"), float("nan")
    median, q25, q75 = np.quantile(values, [0.5, 0.25, 0.75])
    return float(median), float(q25), float(q75)


def summarize_scaling_records(records):
    """Summarize scaling replicates by medians and interquartile ranges."""
    summaries = []
    keys = sorted(
        {
            (record["target_n_events"], record["method"], record["method_label"])
            for record in records
        }
    )
    fit_metrics = (
        "runtime_seconds",
        "branching_update_seconds",
        "ess_per_second",
        "final_elbo",
        "n_iter_run",
    )
    for target, method, label in keys:
        group = [
            record
            for record in records
            if record["target_n_events"] == target and record["method"] == method
        ]
        completed = [record for record in group if record.get("status") == "ok"]
        statuses = sorted({record.get("status", "unknown") for record in group})
        if len(completed) == len(group):
            status = "ok"
        elif completed:
            status = "incomplete"
        else:
            status = "|".join(statuses)
        row = {
            "target_n_events": target,
            "method": method,
            "method_label": label,
            "status": status,
            "n_runs": len(group),
            "n_completed": len(completed),
        }
        for metric in ("n_events", "candidate_parent_count", "retained_candidate_fraction"):
            median, q25, q75 = _median_iqr(
                [record.get(metric, np.nan) for record in group]
            )
            row[metric] = median
            row[f"{metric}_q25"] = q25
            row[f"{metric}_q75"] = q75
        for metric in fit_metrics:
            median, q25, q75 = _median_iqr(
                [record.get(metric, np.nan) for record in completed]
            )
            row[metric] = median
            row[f"{metric}_q25"] = q25
            row[f"{metric}_q75"] = q75
        summaries.append(row)
    return summaries


def summarize_scaling_maxima(records):
    """Summarize each method at its largest scheduled target, not a capacity limit."""
    rows = []
    for method, settings in METHODS.items():
        method_records = [record for record in records if record["method"] == method]
        if not method_records:
            continue
        maximum_target = max(record["target_n_events"] for record in method_records)
        maximum_records = [
            record for record in method_records
            if record["target_n_events"] == maximum_target
        ]
        completed = [record for record in maximum_records if record.get("status") == "ok"]
        row = {
            "method": method,
            "method_label": settings["label"],
            "largest_target": int(maximum_target),
            "n_runs": len(maximum_records),
            "n_completed": len(completed),
            "runtime_hours": float("nan"),
            "retained_candidate_fraction": float("nan"),
        }
        if completed:
            row["runtime_hours"] = _median_iqr(
                [record.get("runtime_seconds", np.nan) for record in completed]
            )[0] / 3600.0
            row["retained_candidate_fraction"] = _median_iqr(
                [
                    record.get("retained_candidate_fraction", np.nan)
                    for record in completed
                ]
            )[0]
        rows.append(row)
    return rows


def _partition_scenarios():
    true_six, _ = generate_partition(N_REGIONS, seed=PARTITION_SEED)
    displaced_five, _ = generate_partition(
        MISSPECIFIED_PARTITION_REGIONS,
        seed=MISSPECIFIED_PARTITION_SEED,
    )
    union = unary_union(true_six)
    return {
        "P0": {
            "true_zones": true_six,
            "fit_zones": true_six,
            "oracle_zones": true_six,
            "mus": REFERENCE_MUS,
        },
        "P1": {
            "true_zones": [union],
            "fit_zones": true_six,
            "oracle_zones": [union],
            "mus": (5.5,),
        },
        "P2": {
            "true_zones": true_six,
            "fit_zones": displaced_five,
            "oracle_zones": true_six,
            "mus": REFERENCE_MUS,
        },
        "P3": {
            "true_zones": true_six,
            "fit_zones": merge_adjacent_zones(true_six, 4),
            "oracle_zones": true_six,
            "mus": REFERENCE_MUS,
        },
        "P4": {
            "true_zones": true_six,
            "fit_zones": [union],
            "oracle_zones": true_six,
            "mus": HIGH_CONTRAST_MUS,
        },
    }


def _partition_surface_payload(
    bundle,
    scenario_name,
    replicate,
    settings,
    campaign,
    seed,
):
    """Evaluate generating and fitted backgrounds for the partition figure."""
    evaluation_xy, _ = regular_spatial_grid(PARTITION_FIGURE_GRID_SIZE)
    true_partition = DomainPartition.from_polygons(settings["true_zones"])
    true_domains = true_partition.locate(
        evaluation_xy[:, 0], evaluation_xy[:, 1]
    )
    if np.any(true_domains < 0):
        raise RuntimeError("The partition figure grid must lie inside the true domain.")
    regional_baselines = np.asarray(settings["mus"], dtype=float)
    true_background = regional_baselines[true_domains] / (
        1.0
        + np.exp(
            -latent_field(evaluation_xy[:, 0], evaluation_xy[:, 1], scale=1.0)
        )
    )
    fitted_draws = posterior_background_draws(
        bundle,
        evaluation_xy,
        campaign.posterior_draws,
        seed=seed,
    )
    return {
        "scenario": scenario_name,
        "replicate": int(replicate),
        "grid_size": int(PARTITION_FIGURE_GRID_SIZE),
        "evaluation_xy": evaluation_xy,
        "true_background": true_background,
        "fitted_background": fitted_draws.mean(axis=1),
    }


def _partition_fit(
    role,
    scenario_name,
    replicate,
    fit_zones,
    simulation,
    training_catalog,
    training_parent_indices,
    train_end,
    cutoff,
    true_settings,
    gp_prior,
    calibration_seconds,
    calibration_n_events,
    campaign,
    seed,
    capture_surface=False,
):
    model = make_model(fit_zones, train_end, etas=INITIAL_ETAS, gp_prior=gp_prior)
    bundle, diagnostics = fit_spinh_method(
        model,
        training_catalog,
        "m5",
        campaign,
        seed,
        parent_time_window=cutoff,
    )
    record = {
        "experiment": 2,
        "scenario": scenario_name,
        "replicate": replicate,
        "fit_role": role,
        "method": "m5",
        "method_label": METHODS["m5"]["label"],
        "n_events": len(simulation.catalog),
        "n_train": len(training_catalog),
        "n_fit_regions": len(fit_zones),
        "parent_time_window": cutoff,
        "gp_calibration_seconds": calibration_seconds,
        "gp_calibration_n_events": calibration_n_events,
        **diagnostics,
    }
    if bundle is None:
        return record, None
    record["inference_seconds"] = diagnostics["runtime_seconds"]
    record["runtime_seconds"] = diagnostics["runtime_seconds"] + calibration_seconds
    record.update(
        intensity_recovery_metrics(
            bundle,
            simulation,
            true_settings["mus"],
            1.0,
            EXPERIMENT_2_ETAS,
            campaign,
            seed + 31,
        )
    )
    record.update(
        branching_metrics(
            bundle,
            training_parent_indices,
            training_catalog.t,
            cutoff,
        )
    )
    record["predictive_log_score"] = predictive_log_score(
        bundle, simulation.catalog, train_end, campaign, seed + 43
    )
    surface = None
    if capture_surface:
        surface = _partition_surface_payload(
            bundle,
            scenario_name,
            replicate,
            true_settings,
            campaign,
            seed + 59,
        )
    return record, surface


def _partition_replicate(scenario_name, replicate, campaign):
    settings = _partition_scenarios()[scenario_name]
    duration = EXPERIMENT_2_DURATIONS[scenario_name] * campaign.duration_scale
    seed = 51_000 + 1000 * list(_partition_scenarios()).index(scenario_name) + replicate
    simulation = simulate_configuration(
        settings["true_zones"],
        settings["mus"],
        duration,
        1.0,
        EXPERIMENT_2_ETAS,
        EXPERIMENT_2_BETA,
        seed,
        grid_res=40 if campaign.name == "smoke" else 100,
    )
    train_end = TRAIN_FRACTION * duration
    training_mask = simulation.catalog.t <= train_end
    training_catalog = subset_catalog(simulation.catalog, training_mask)
    training_parent_indices = simulation.parent_indices[training_mask]
    cutoff = temporal_cutoff(EXPERIMENT_2_ETAS)
    calibration_zones = [unary_union(settings["true_zones"])]
    calibration_model = make_model(calibration_zones, train_end, etas=INITIAL_ETAS)
    gp_prior, calibration_seconds, _, calibration_n_events = calibrate_gp(
        calibration_model, training_catalog, campaign, seed + 71
    )
    try:
        oracle, oracle_surface = _partition_fit(
            "oracle",
            scenario_name,
            replicate,
            settings["oracle_zones"],
            simulation,
            training_catalog,
            training_parent_indices,
            train_end,
            cutoff,
            settings,
            gp_prior,
            calibration_seconds,
            calibration_n_events,
            campaign,
            seed + 101,
            capture_surface=(
                replicate == PARTITION_FIGURE_REPLICATE and scenario_name == "P0"
            ),
        )
        if scenario_name == "P0":
            misspecified = {**oracle, "fit_role": "misspecified"}
            surface = oracle_surface
        else:
            misspecified, surface = _partition_fit(
                "misspecified",
                scenario_name,
                replicate,
                settings["fit_zones"],
                simulation,
                training_catalog,
                training_parent_indices,
                train_end,
                cutoff,
                settings,
                gp_prior,
                calibration_seconds,
                calibration_n_events,
                campaign,
                seed + 101,
                capture_surface=(replicate == PARTITION_FIGURE_REPLICATE),
            )
    except Exception as error:
        failed = {
            "experiment": 2,
            "scenario": scenario_name,
            "replicate": replicate,
            "status": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        return [failed], [], None
    paired = {
        "experiment": 2,
        "scenario": scenario_name,
        "replicate": replicate,
        "status": "ok"
        if oracle.get("status") == misspecified.get("status") == "ok"
        else "incomplete",
        "n_events": len(simulation.catalog),
        "delta_rel_l2_background": misspecified.get("rel_l2_background", np.nan)
        - oracle.get("rel_l2_background", np.nan),
        "delta_rel_l2_triggering": misspecified.get("rel_l2_triggering", np.nan)
        - oracle.get("rel_l2_triggering", np.nan),
        "delta_background_brier": misspecified.get("background_brier", np.nan)
        - oracle.get("background_brier", np.nan),
        "delta_predictive_log_score": oracle.get("predictive_log_score", np.nan)
        - misspecified.get("predictive_log_score", np.nan),
        "delta_runtime_seconds": misspecified.get("runtime_seconds", np.nan)
        - oracle.get("runtime_seconds", np.nan),
    }
    return [oracle, misspecified], [paired], surface


def run_partition_experiment(
    campaign,
    n_jobs=1,
    *,
    checkpoint_dir=None,
    resume=True,
):
    tasks = [
        (scenario, replicate, campaign)
        for scenario in _partition_scenarios()
        for replicate in range(campaign.n_partition_replicates)
    ]
    task_keys = [(scenario, replicate) for scenario, replicate, _ in tasks]
    results = parallel_map(
        _partition_replicate,
        tasks,
        n_jobs,
        "Experiment 2",
        task_keys=task_keys,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        max_parallel_calibrations=campaign.max_parallel_calibrations,
    )
    raw = [record for raw_group, _, _ in results for record in raw_group]
    paired = [record for _, paired_group, _ in results for record in paired_group]
    surfaces = [surface for _, _, surface in results if surface is not None]
    return raw, paired, surfaces


def _save_figure(
    figure,
    path,
    save,
    show,
    *,
    contains_rasterized_artists=False,
):
    if save:
        path = Path(path).with_suffix(".pdf")
        path.parent.mkdir(parents=True, exist_ok=True)
        options = {
            "bbox_inches": "tight",
            "pad_inches": 0.08,
            "facecolor": figure.get_facecolor(),
            "transparent": False,
        }
        if contains_rasterized_artists:
            options["dpi"] = 300
        figure.savefig(path, **options)
    if show:
        figure.show()
    else:
        plt.close(figure)


def select_representative_reconstructions(records, reconstructions):
    """Select each scenario's M1 replicate nearest the median total error."""
    payloads = {
        (payload["scenario"], payload["replicate"]): payload
        for payload in reconstructions
    }
    selected = []
    selection_records = []
    for scenario_name in SCENARIOS:
        candidates = [
            record
            for record in records
            if record.get("scenario") == scenario_name
            and record.get("method") == "m1"
            and record.get("status") == "ok"
            and np.isfinite(record.get("rel_l2_total", np.nan))
            and (scenario_name, record["replicate"]) in payloads
        ]
        if not candidates:
            continue
        median_error = float(
            np.median([record["rel_l2_total"] for record in candidates])
        )
        representative = min(
            candidates,
            key=lambda record: (
                abs(record["rel_l2_total"] - median_error),
                record["replicate"],
            ),
        )
        selected.append(payloads[(scenario_name, representative["replicate"])])
        selection_records.append(
            {
                "scenario": scenario_name,
                "method": "m1",
                "replicate": int(representative["replicate"]),
                "median_rel_l2_total": median_error,
                "selected_rel_l2_total": float(representative["rel_l2_total"]),
            }
        )
    return selected, selection_records


def _reconstruction_surface(payload, intensity_name, estimate):
    suffix = "estimate" if estimate else "true"
    values = np.asarray(payload[f"{intensity_name}_{suffix}"], dtype=float)
    n_space = int(payload["space_grid_size"])
    if intensity_name == "background":
        return values.reshape(n_space, n_space)
    n_time = int(payload["time_grid_size"])
    return values.reshape(n_time, n_space, n_space).mean(axis=0)


def plot_accuracy_reconstruction(reconstructions, output, *, save=True, show=False):
    """Plot true and M1 posterior-mean intensities for median-error replicates."""
    if not reconstructions:
        return
    ordered = sorted(
        reconstructions,
        key=lambda payload: list(SCENARIOS).index(payload["scenario"]),
    )
    intensity_names = ("background", "triggering", "total")
    scales = {}
    for intensity_name in intensity_names:
        values = np.concatenate(
            [
                _reconstruction_surface(payload, intensity_name, estimate).ravel()
                for payload in ordered
                for estimate in (False, True)
            ]
        )
        finite = values[np.isfinite(values)]
        scales[intensity_name] = float(np.max(finite)) if finite.size else 1.0
        if scales[intensity_name] <= 0.0:
            scales[intensity_name] = 1.0

    figure, axes = plt.subplots(
        len(ordered),
        2 * len(intensity_names),
        figsize=(16.5, 3.15 * len(ordered)),
        squeeze=False,
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    images = {}
    titles = {
        "background": "Background",
        "triggering": "Mean triggering",
        "total": "Mean total",
    }
    zones, _ = generate_partition(N_REGIONS, seed=PARTITION_SEED)
    for row, payload in enumerate(ordered):
        spatial_xy = np.asarray(payload["spatial_xy"], dtype=float)
        n_space = int(payload["space_grid_size"])
        x_values = spatial_xy[:n_space, 0]
        y_values = spatial_xy[::n_space, 1]
        for intensity_index, intensity_name in enumerate(intensity_names):
            for estimate_index, estimate in enumerate((False, True)):
                column = 2 * intensity_index + estimate_index
                axis = axes[row, column]
                image = axis.pcolormesh(
                    x_values,
                    y_values,
                    _reconstruction_surface(payload, intensity_name, estimate),
                    cmap="viridis",
                    vmin=0.0,
                    vmax=scales[intensity_name],
                    shading="auto",
                    rasterized=True,
                )
                _plot_partition_boundaries(
                    axis, zones, color="white", linewidth=0.55
                )
                axis.set(xlim=X_BOUNDS, ylim=Y_BOUNDS, aspect="equal")
                if row == 0:
                    role = "M1" if estimate else "Truth"
                    axis.set_title(f"{role}\n{titles[intensity_name]}")
                if row == len(ordered) - 1:
                    axis.set_xlabel("x")
                images[intensity_name] = image
        axes[row, 0].set_ylabel(
            f"{payload['scenario'].title()} (rep. {payload['replicate']})\ny"
        )
    for intensity_index, intensity_name in enumerate(intensity_names):
        pair_axes = axes[:, 2 * intensity_index : 2 * intensity_index + 2]
        figure.colorbar(
            images[intensity_name],
            ax=pair_axes.ravel().tolist(),
            shrink=0.78,
            pad=0.015,
        )
    _save_figure(
        figure,
        output / "experiment_1_reconstruction.pdf",
        save,
        show,
        contains_rasterized_artists=True,
    )


def plot_accuracy(summary, output, *, save=True, show=False):
    if not summary:
        return
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout="constrained")
    colors = {"easy": "#2f6f6d", "difficult": "#b65b43"}
    for axis, metric, title in zip(
        axes,
        ("rel_l2_background", "rel_l2_triggering", "rel_l2_total"),
        ("Background", "Triggering", "Total intensity"),
    ):
        width = 0.38
        x = np.arange(len(METHODS))
        for offset, scenario in zip((-0.5, 0.5), SCENARIOS):
            rows = {row["method"]: row for row in summary if row["scenario"] == scenario}
            values = [rows.get(method, {}).get(metric, np.nan) for method in METHODS]
            axis.bar(x + offset * width, values, width, color=colors[scenario], label=scenario.title())
        axis.set_title(title)
        axis.set_xticks(x, [method.upper() for method in METHODS])
        axis.set_ylabel("Relative L2 error")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    _save_figure(figure, output / "experiment_1_accuracy.pdf", save, show)


def plot_scaling(summary, output, *, save=True, show=False):
    if not summary:
        return
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), layout="constrained")
    colors = ["#2f6f6d", "#5687a3", "#b65b43", "#8a6b9e", "#6d7450"]
    for method, color in zip(METHODS, colors):
        rows = sorted(
            [row for row in summary if row["method"] == method],
            key=lambda row: row["target_n_events"],
        )
        if not rows:
            continue
        completed = [row for row in rows if np.isfinite(row["runtime_seconds"])]
        if completed:
            x = np.asarray([row["n_events"] for row in completed], dtype=float)
            y = np.asarray([row["runtime_seconds"] for row in completed], dtype=float)
            q25 = np.asarray(
                [row["runtime_seconds_q25"] for row in completed], dtype=float
            )
            q75 = np.asarray(
                [row["runtime_seconds_q75"] for row in completed], dtype=float
            )
            axes[0].plot(x, y, marker="o", label=method.upper(), color=color)
            axes[0].fill_between(x, q25, q75, color=color, alpha=0.18)
        pair_rows = [
            row for row in rows if np.isfinite(row["candidate_parent_count"])
        ]
        if pair_rows:
            pair_x = np.asarray([row["n_events"] for row in pair_rows], dtype=float)
            pair_y = np.asarray(
                [row["candidate_parent_count"] for row in pair_rows], dtype=float
            )
            pair_q25 = np.asarray(
                [row["candidate_parent_count_q25"] for row in pair_rows], dtype=float
            )
            pair_q75 = np.asarray(
                [row["candidate_parent_count_q75"] for row in pair_rows], dtype=float
            )
            axes[1].plot(
                pair_x,
                pair_y,
                marker="o",
                label=method.upper(),
                color=color,
            )
            axes[1].fill_between(
                pair_x, pair_q25, pair_q75, color=color, alpha=0.18
            )
    axes[0].set(xscale="log", yscale="log", xlabel="Observed events", ylabel="Wall-clock time (s)")
    axes[1].set(xscale="log", yscale="log", xlabel="Observed events", ylabel="Retained parent-offspring pairs")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, ncol=2)
    _save_figure(figure, output / "experiment_1_scaling.pdf", save, show)


def plot_partition(summary, output, *, save=True, show=False):
    if not summary:
        return
    metrics = (
        "delta_rel_l2_background",
        "delta_rel_l2_triggering",
        "delta_background_brier",
        "delta_predictive_log_score",
    )
    titles = ("Background error", "Triggering error", "Background Brier", "Predictive score loss")
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
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
    scenarios = _partition_scenarios()
    payloads = {surface["scenario"]: surface for surface in surfaces}
    available = [payloads[name] for name in scenarios if name in payloads]
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
        len(scenarios),
        figsize=(15.5, 6.4),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    image = None
    for column, (scenario_name, settings) in enumerate(scenarios.items()):
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
        panels = (
            (payload["true_background"], settings["true_zones"]),
            (payload["fitted_background"], settings["fit_zones"]),
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
                xlim=X_BOUNDS,
                ylim=Y_BOUNDS,
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
    lower = float(row.get(f"{metric}_ci_low", np.nan))
    upper = float(row.get(f"{metric}_ci_high", np.nan))
    if np.isfinite(lower) and np.isfinite(upper):
        return f"{value:.{digits}f} [{lower:.{digits}f}, {upper:.{digits}f}]"
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


def write_experiment_1_latex(output, accuracy_summary, scaling_maxima):
    intensity_rows = []
    parameter_rows = []
    for row in accuracy_summary:
        prefix = [row["scenario"].title(), row["method"].upper()]
        intensity_rows.append(
            prefix
            + [
                _latex_metric(row, "rel_l2_background"),
                _latex_metric(row, "rel_l2_triggering"),
                _latex_metric(row, "rel_l2_total"),
                _latex_metric(row, "mae_background"),
                _latex_metric(row, "predictive_log_score"),
            ]
        )
        parameter_rows.append(
            prefix
            + [
                _latex_metric(row, "parameter_log_error"),
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
                r"$e_{L_2}(\lambda_{\mathrm{trig}})$",
                r"$e_{L_2}(\lambda)$",
                r"$\operatorname{MAE}(\mu)$",
                r"$S_{\mathrm{test}}$",
            ),
            intensity_rows,
        )
        _write_latex(
            output / "experiment_1_parameters_table.tex",
            (
                "Scenario",
                "Method",
                "Log-error",
                r"$\mathrm{BS}_{\mathrm{bg}}$",
                r"$F_1$",
                r"$\pi_{z^\star}$",
                r"$\mathrm{Rec}_{\mathcal{C}}$",
            ),
            parameter_rows,
        )
    scaling_rows = []
    for row in scaling_maxima:
        scaling_rows.append(
            [
                row["method"].upper(),
                str(row["largest_target"]),
                f"{row['n_completed']}/{row['n_runs']}",
                _latex_metric(row, "runtime_hours", 2),
                _latex_metric(row, "retained_candidate_fraction"),
            ]
        )
    if scaling_maxima:
        _write_latex(
            output / "experiment_1_scaling_table.tex",
            (
                "Method",
                "Largest target",
                "Completed",
                "Time (h)",
                "Pair fraction",
            ),
            scaling_rows,
        )


def write_experiment_2_latex(output, summary):
    rows = [
        [
            row["scenario"],
            _latex_metric(row, "delta_rel_l2_background"),
            _latex_metric(row, "delta_rel_l2_triggering"),
            _latex_metric(row, "delta_background_brier"),
            _latex_metric(row, "delta_predictive_log_score"),
            _latex_metric(row, "delta_runtime_seconds", 2),
        ]
        for row in summary
    ]
    _write_latex(
        output / "experiment_2_table.tex",
        (
            "Scenario",
            r"$\Delta e_{L_2}(\mu)$",
            r"$\Delta e_{L_2}(\lambda_{\mathrm{trig}})$",
            r"$\Delta \mathrm{BS}_{\mathrm{bg}}$",
            r"$\Delta S_{\mathrm{test}}$",
            r"$\Delta$ Time (s)",
        ),
        rows,
    )


def run(
    profile="smoke",
    experiment="all",
    panel="both",
    methods=tuple(METHODS),
    *,
    n_jobs=None,
    scaling_targets=None,
    resume=True,
    save_figures=True,
    show_figures=False,
    campaign_overrides=None,
):
    validate_scientific_settings()
    methods = tuple(methods)
    unknown = set(methods) - set(METHODS)
    if not methods or unknown:
        raise ValueError(f"At least one valid method is required; unknown={sorted(unknown)}.")
    if experiment not in {"1", "2", "all"}:
        raise ValueError("experiment must be '1', '2' or 'all'.")
    if panel not in {"accuracy", "scaling", "both"}:
        raise ValueError("panel must be 'accuracy', 'scaling' or 'both'.")
    n_jobs = resolve_n_jobs(profile, n_jobs)
    scaling_targets = _validate_execution_settings(n_jobs, scaling_targets)
    if not all(isinstance(value, bool) for value in (resume, save_figures, show_figures)):
        raise ValueError("resume, save_figures and show_figures must be boolean.")
    display_figures = _resolve_figure_display(show_figures)
    campaign = configure_campaign(profile, **(campaign_overrides or {}))
    selected_scaling_plan = (
        scaling_plan(profile, methods, scaling_targets)
        if experiment in {"1", "all"} and panel in {"scaling", "both"}
        else {}
    )
    if experiment in {"2", "all"}:
        _validate_partition_figure_settings(campaign)
    output = RESULTS_ROOT / campaign.name
    output.mkdir(parents=True, exist_ok=True)
    _print_run_settings(
        campaign,
        experiment,
        panel,
        methods,
        n_jobs,
        scaling_targets,
        output,
    )
    write_campaign(
        output / "simulation_campaign.json",
        campaign,
        {
            "methods": methods,
            "experiment": experiment,
            "panel": panel,
            "n_jobs": int(n_jobs),
            "effective_workers": effective_worker_count(n_jobs),
            "resume": resume,
            "scaling_targets": scaling_targets,
            "scaling_methods_by_target": selected_scaling_plan,
            "partition_figure_replicate": PARTITION_FIGURE_REPLICATE,
            "partition_figure_grid_size": PARTITION_FIGURE_GRID_SIZE,
            "experiment_2_durations": EXPERIMENT_2_DURATIONS,
        },
    )
    results = {}
    if experiment in {"1", "all"}:
        accuracy_summary = []
        scaling_summary = []
        scaling_maxima = []
        if panel in {"accuracy", "both"}:
            accuracy_checkpoints = checkpoint_directory(
                output,
                "simulation_accuracy",
                campaign,
                settings={"methods": methods},
                source_paths=_CHECKPOINT_SOURCES,
            )
            accuracy, reconstructions = run_accuracy_panel(
                campaign,
                methods,
                n_jobs=n_jobs,
                checkpoint_dir=accuracy_checkpoints,
                resume=resume,
            )
            accuracy_summary = summarize_records(
                accuracy, ("scenario", "method", "method_label"), ACCURACY_METRICS
            )
            accuracy_summary.sort(
                key=lambda row: (
                    list(SCENARIOS).index(row["scenario"]),
                    list(METHODS).index(row["method"]),
                )
            )
            write_records(output / "experiment_1_accuracy_raw.csv", accuracy)
            write_records(output / "experiment_1_accuracy_table.csv", accuracy_summary)
            selected, selection_records = select_representative_reconstructions(
                accuracy, reconstructions
            )
            write_records(
                output / "experiment_1_reconstruction_selection.csv",
                selection_records,
            )
            plot_accuracy(accuracy_summary, output, save=save_figures, show=display_figures)
            plot_accuracy_reconstruction(
                selected,
                output,
                save=save_figures,
                show=display_figures,
            )
            results["experiment_1_accuracy"] = (accuracy, accuracy_summary)
        if panel in {"scaling", "both"}:
            scaling_checkpoints = checkpoint_directory(
                output,
                "simulation_scaling",
                campaign,
                settings={"methods_by_target": selected_scaling_plan},
                source_paths=_CHECKPOINT_SOURCES,
            )
            scaling = run_scaling_panel(
                campaign,
                methods,
                n_jobs=n_jobs,
                targets=tuple(selected_scaling_plan),
                checkpoint_dir=scaling_checkpoints,
                resume=resume,
            )
            scaling_summary = summarize_scaling_records(scaling)
            scaling_maxima = summarize_scaling_maxima(scaling)
            write_records(output / "experiment_1_scaling_raw.csv", scaling)
            write_records(output / "experiment_1_scaling_table.csv", scaling_summary)
            write_records(
                output / "experiment_1_scaling_maxima.csv", scaling_maxima
            )
            plot_scaling(
                scaling_summary,
                output,
                save=save_figures,
                show=display_figures,
            )
            results["experiment_1_scaling"] = (scaling, scaling_summary)
        write_experiment_1_latex(output, accuracy_summary, scaling_maxima)
    if experiment in {"2", "all"}:
        partition_checkpoints = checkpoint_directory(
            output,
            "simulation_partition",
            campaign,
            settings={"scenarios": tuple(_partition_scenarios())},
            source_paths=_CHECKPOINT_SOURCES,
        )
        raw, paired, surfaces = run_partition_experiment(
            campaign,
            n_jobs=n_jobs,
            checkpoint_dir=partition_checkpoints,
            resume=resume,
        )
        paired_summary = summarize_records(
            paired,
            ("scenario",),
            (
                "delta_rel_l2_background",
                "delta_rel_l2_triggering",
                "delta_background_brier",
                "delta_predictive_log_score",
                "delta_runtime_seconds",
            ),
        )
        write_records(output / "experiment_2_fits_raw.csv", raw)
        write_records(output / "experiment_2_paired_raw.csv", paired)
        write_records(output / "experiment_2_table.csv", paired_summary)
        write_experiment_2_latex(output, paired_summary)
        plot_partition(paired_summary, output, save=save_figures, show=display_figures)
        plot_partition_surfaces(
            surfaces,
            output,
            save=save_figures,
            show=display_figures,
        )
        results["experiment_2"] = (raw, paired, paired_summary, surfaces)
    _print_run_summary(results, output)
    if display_figures and "ipykernel" not in sys.modules:
        plt.show()
    return results


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
    selection = parser.add_argument_group("experiment selection")
    selection.add_argument(
        "--profile",
        choices=CAMPAIGNS,
        default="smoke",
        help="Numerical budget profile.",
    )
    selection.add_argument(
        "--experiment",
        choices=("1", "2", "all"),
        default="all",
        help="Experiment to run.",
    )
    selection.add_argument(
        "--panel",
        choices=("accuracy", "scaling", "both"),
        default="both",
        help="Experiment 1 panel; ignored when only Experiment 2 is selected.",
    )
    selection.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
        help="Subset of the M1--M5 methods used in Experiment 1.",
    )
    selection.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Simultaneous jobs (default: 1). Test 2 before requesting more; RAM is shared.",
    )
    selection.add_argument(
        "--scaling-targets",
        nargs="+",
        type=int,
        default=None,
        help="Target catalogue sizes for the scaling panel.",
    )

    budget = parser.add_argument_group("campaign budget overrides")
    budget.add_argument("--n-replicates", type=int, default=None)
    budget.add_argument("--n-scaling-replicates", type=int, default=None)
    budget.add_argument("--n-partition-replicates", type=int, default=None)
    budget.add_argument("--n-chains", type=int, default=None)
    budget.add_argument("--vi-starts", type=int, default=None)
    budget.add_argument("--gibbs-iterations", type=int, default=None)
    budget.add_argument("--gibbs-thin", type=int, default=None)
    budget.add_argument("--vi-iterations", type=int, default=None)
    budget.add_argument("--evaluation-space-grid", type=int, default=None)
    budget.add_argument("--evaluation-time-grid", type=int, default=None)
    budget.add_argument("--quadrature-space-grid", type=int, default=None)
    budget.add_argument("--quadrature-time-grid", type=int, default=None)
    budget.add_argument("--posterior-draws", type=int, default=None)
    budget.add_argument("--max-parallel-calibrations", type=int, default=None)
    budget.add_argument("--duration-scale", type=float, default=None)
    budget.add_argument("--exact-max-events", type=int, default=None)
    budget.add_argument("--dense-max-events", type=int, default=None)
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


def _overrides_from_args(args):
    return {
        "n_replicates": args.n_replicates,
        "n_scaling_replicates": args.n_scaling_replicates,
        "n_partition_replicates": args.n_partition_replicates,
        "n_chains": args.n_chains,
        "vi_starts": args.vi_starts,
        "gibbs_iterations": args.gibbs_iterations,
        "gibbs_thin": args.gibbs_thin,
        "vi_iterations": args.vi_iterations,
        "evaluation_space_grid": args.evaluation_space_grid,
        "evaluation_time_grid": args.evaluation_time_grid,
        "quadrature_space_grid": args.quadrature_space_grid,
        "quadrature_time_grid": args.quadrature_time_grid,
        "posterior_draws": args.posterior_draws,
        "max_parallel_calibrations": args.max_parallel_calibrations,
        "duration_scale": args.duration_scale,
        "exact_max_events": args.exact_max_events,
        "dense_max_events": args.dense_max_events,
        "use_calibration": False if args.no_calibration else None,
    }


def main(argv=None):
    args = parse_args(argv)
    return run(
        profile=args.profile,
        experiment=args.experiment,
        panel=args.panel,
        methods=args.methods,
        n_jobs=args.n_jobs,
        scaling_targets=args.scaling_targets,
        resume=not args.no_resume,
        save_figures=not args.no_figures,
        show_figures=args.show_figures,
        campaign_overrides=_overrides_from_args(args),
    )


# %% Run the file
if __name__ == "__main__":
    if should_use_editor_settings():
        run_from_editor()
    else:
        main()

# %%
