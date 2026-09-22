"""Experiment 1 and 2: one reproducible catalogue per worker task."""

from __future__ import annotations

from dataclasses import replace
from numbers import Integral

from shapely.ops import unary_union
import numpy as np

from .runner_utils import parallel_map, simulation_execution_guard as execution_guard
from .simulation_settings import (
    ACCURACY_TARGET_EVENTS,
    EXPERIMENT_2_BETA,
    EXPERIMENT_2_DURATIONS,
    EXPERIMENT_2_ETAS,
    EXPERIMENT_2_METHOD,
    HIGH_CONTRAST_MUS,
    INITIAL_ETAS,
    METHODS,
    MISSPECIFIED_PARTITION_REGIONS,
    MISSPECIFIED_PARTITION_SEED,
    N_REGIONS,
    PARAMETER_NAMES,
    PARTITION_FIGURE_GRID_SIZE,
    PARTITION_FIGURE_REPLICATE,
    PARTITION_SEED,
    REFERENCE_MUS,
    SCENARIOS,
)
from .test_utils import (
    background_recovery_metrics,
    branching_metrics,
    calibrate_gp,
    candidate_diagnostics,
    fit_spinh_method,
    generate_partition,
    gibbs_parameter_traces,
    latent_field,
    make_model,
    merge_adjacent_zones,
    omitted_temporal_mass,
    parameter_recovery_metrics,
    posterior_background_draws,
    posterior_parameter_draws,
    posterior_parameter_means,
    regular_spatial_grid,
    regular_spatial_quadrature,
    simulate_configuration,
    temporal_cutoff,
)
from spatial import DomainPartition, SpatialQuadrature


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
    simulation,
    zones,
    cutoff,
    tail_mass,
    gp_prior,
    calibration_seconds,
    calibration_succeeded,
    calibration_n_events,
    campaign,
    evaluation_quadrature,
    background_quadrature,
    spatial_compensator_quadrature,
):
    """Fit one method to the complete simulated catalogue."""
    scenario = SCENARIOS[scenario_name]
    catalog = simulation.catalog
    parent_indices = simulation.parent_indices
    duration = simulation.background_simulation.duration
    reconstruction = None
    posterior = None
    record = _base_record(1, scenario_name, replicate, method, simulation, cutoff)
    record.update(
        {
            "n_fitted": len(catalog),
            "observation_duration": duration,
            "simulation_seed": 11_000 + 1000 * list(SCENARIOS).index(scenario_name) + replicate,
            "omitted_temporal_mass": tail_mass,
            "gp_variance": gp_prior.variance,
            "gp_length_scale": gp_prior.length_scale,
            "gp_calibration_seconds": calibration_seconds,
            "gp_calibration_succeeded": calibration_succeeded,
            "gp_calibration_n_events": calibration_n_events,
        }
    )
    # M4/M5 share starts and Monte Carlo draws for their paired comparison.
    seed_method = "m4" if method == "m5" else method
    seed = int(record["seed"] + 1009 * list(METHODS).index(seed_method))
    record["inference_seed"] = seed
    try:
        model = make_model(zones, duration, etas=INITIAL_ETAS, gp_prior=gp_prior)
        bundle, diagnostics = fit_spinh_method(
            model,
            catalog,
            method,
            campaign,
            seed,
            parent_time_window=cutoff,
            background_quadrature=background_quadrature,
            spatial_compensator_quadrature=spatial_compensator_quadrature,
        )
        record.update(diagnostics)
        if bundle is None:
            record["runtime_seconds"] = float("nan")
            return record, reconstruction, posterior
        record["inference_seconds"] = diagnostics["runtime_seconds"]
        record["runtime_seconds"] = diagnostics["runtime_seconds"] + calibration_seconds
        record.update(
            branching_metrics(
                bundle,
                parent_indices,
                catalog.t,
                cutoff,
            )
        )
        record.update(
            parameter_recovery_metrics(
                posterior_parameter_means(bundle), scenario["etas"], scenario["beta"]
            )
        )
        posterior = {
            "scenario": scenario_name,
            "replicate": int(replicate),
            "method": method,
            "method_label": METHODS[method]["label"],
            "samples": posterior_parameter_draws(
                bundle, campaign.parameter_draws, seed=seed + 23
            ),
        }
        if METHODS[method]["family"] == "gibbs":
            posterior.update(
                {
                    "gibbs_trace": gibbs_parameter_traces(bundle),
                    "burn_in_fraction": campaign.gibbs_burn_in,
                    "adaptation_end": bundle.fits[0].raw["proposal_steps"].get(
                        "adaptation_end"
                    ),
                }
            )
        intensity_metrics, reconstruction = background_recovery_metrics(
            bundle,
            simulation,
            scenario["mus"],
            scenario["field_scale"],
            campaign,
            seed + 31,
            quadrature=evaluation_quadrature,
            return_payload=True,
        )
        reconstruction.update(
            {
                "scenario": scenario_name,
                "replicate": int(replicate),
                "method": method,
                "method_label": METHODS[method]["label"],
                "zones_wkt": [zone.wkt for zone in zones],
            }
        )
        record.update(intensity_metrics)
        candidates = candidate_diagnostics(
            catalog.t, parent_indices, cutoff
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
                    else dense_pairs / max(len(catalog), 1)
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
    return record, reconstruction, posterior


def _accuracy_replicate(
    scenario_name,
    replicate,
    methods,
    campaign,
    evaluation_quadrature=None,
    background_quadrature=None,
    spatial_compensator_quadrature=None,
):
    if evaluation_quadrature is None:
        evaluation_quadrature = regular_spatial_quadrature(
            campaign.evaluation_space_grid
        )
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
    cutoff = temporal_cutoff(scenario["etas"], horizon=duration)
    tail_mass = omitted_temporal_mass(scenario["etas"], cutoff, duration)
    calibration_model = make_model(zones, duration, etas=INITIAL_ETAS)
    (
        gp_prior,
        calibration_seconds,
        calibration_succeeded,
        calibration_n_events,
    ) = calibrate_gp(
        calibration_model, simulation.catalog, campaign, seed + 71
    )
    return [
        _fit_accuracy_method(
            method,
            scenario_name,
            replicate,
            simulation,
            zones,
            cutoff,
            tail_mass,
            gp_prior,
            calibration_seconds,
            calibration_succeeded,
            calibration_n_events,
            campaign,
            evaluation_quadrature,
            background_quadrature,
            spatial_compensator_quadrature,
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
    worker_memory_reservation_gib=None,
    evaluation_quadrature=None,
    background_quadrature=None,
    spatial_compensator_quadrature=None,
):
    if worker_memory_reservation_gib is None:
        worker_memory_reservation_gib = execution_guard(
            "1", ACCURACY_TARGET_EVENTS
        )["memory_reservation_gib"]
    if evaluation_quadrature is None:
        evaluation_quadrature = regular_spatial_quadrature(
            campaign.evaluation_space_grid
        )
    elif not isinstance(evaluation_quadrature, SpatialQuadrature):
        raise TypeError("evaluation_quadrature must be a SpatialQuadrature instance.")
    tasks = [
        (
            scenario,
            replicate,
            tuple(methods),
            campaign,
            evaluation_quadrature,
            background_quadrature,
            spatial_compensator_quadrature,
        )
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
        isolate_tasks=campaign.name == "full",
        worker_memory_reservation_gib=worker_memory_reservation_gib,
    )
    fitted = [result for group in nested for result in group]
    records = [record for record, _, _ in fitted]
    reconstructions = [payload for _, payload, _ in fitted if payload is not None]
    posteriors = [payload for _, _, payload in fitted if payload is not None]
    return records, reconstructions, posteriors


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
    cutoff,
    true_settings,
    gp_prior,
    calibration_seconds,
    calibration_n_events,
    campaign,
    seed,
    evaluation_quadrature,
    background_quadrature,
    spatial_compensator_quadrature,
    capture_surface=False,
):
    catalog = simulation.catalog
    duration = simulation.background_simulation.duration
    model = make_model(fit_zones, duration, etas=INITIAL_ETAS, gp_prior=gp_prior)
    bundle, diagnostics = fit_spinh_method(
        model,
        catalog,
        EXPERIMENT_2_METHOD,
        campaign,
        seed,
        parent_time_window=cutoff,
        background_quadrature=background_quadrature,
        spatial_compensator_quadrature=spatial_compensator_quadrature,
    )
    record = {
        "experiment": 2,
        "scenario": scenario_name,
        "replicate": replicate,
        "fit_role": role,
        "method": EXPERIMENT_2_METHOD,
        "method_label": METHODS[EXPERIMENT_2_METHOD]["label"],
        "n_events": len(simulation.catalog),
        "n_fitted": len(catalog),
        "observation_duration": duration,
        "n_fit_regions": len(fit_zones),
        "parent_time_window": cutoff,
        "gp_calibration_seconds": calibration_seconds,
        "gp_calibration_n_events": calibration_n_events,
        **diagnostics,
    }
    if bundle is None:
        return record, None, None
    if len(bundle.fits) == 1:
        record["diagnostic_status"] = "single_chain_ess_only"
    record["inference_seconds"] = diagnostics["runtime_seconds"]
    record["runtime_seconds"] = diagnostics["runtime_seconds"] + calibration_seconds
    record.update(
        background_recovery_metrics(
            bundle,
            simulation,
            true_settings["mus"],
            1.0,
            campaign,
            seed + 31,
            quadrature=evaluation_quadrature,
        )
    )
    record.update(
        parameter_recovery_metrics(
            posterior_parameter_means(bundle),
            EXPERIMENT_2_ETAS,
            EXPERIMENT_2_BETA,
        )
    )
    record.update(
        branching_metrics(
            bundle,
            simulation.parent_indices,
            catalog.t,
            cutoff,
        )
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
        surface["true_zones_wkt"] = np.asarray([zone.wkt for zone in true_settings["true_zones"]])
        surface["fit_zones_wkt"] = np.asarray([zone.wkt for zone in fit_zones])
    trace = {
        "scenario": scenario_name,
        "replicate": int(replicate),
        "fit_role": role,
        "method": EXPERIMENT_2_METHOD,
        "method_label": METHODS[EXPERIMENT_2_METHOD]["label"],
        "gibbs_trace": gibbs_parameter_traces(bundle),
        **{f"true_{name}": record[f"true_{name}"] for name in PARAMETER_NAMES},
        "burn_in_fraction": campaign.gibbs_burn_in,
        "adaptation_end": bundle.fits[0].raw["proposal_steps"].get(
            "adaptation_end"
        ),
        "ess_min": diagnostics.get("ess_min", np.nan),
        "ess": {
            name: diagnostics.get(f"ess_bulk_{name}", np.nan)
            for name in PARAMETER_NAMES
        },
    }
    return record, surface, trace


def _partition_replicate(
    scenario_name,
    replicate,
    campaign,
    evaluation_quadrature=None,
    background_quadrature=None,
    spatial_compensator_quadrature=None,
):
    if evaluation_quadrature is None:
        evaluation_quadrature = regular_spatial_quadrature(
            campaign.evaluation_space_grid
        )
    partition_campaign = replace(
        campaign,
        n_chains=campaign.n_partition_chains,
        gibbs_burn_in=campaign.partition_gibbs_burn_in,
    )
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
    cutoff = temporal_cutoff(EXPERIMENT_2_ETAS, horizon=duration)
    calibration_zones = [unary_union(settings["true_zones"])]
    calibration_model = make_model(calibration_zones, duration, etas=INITIAL_ETAS)
    gp_prior, calibration_seconds, _, calibration_n_events = calibrate_gp(
        calibration_model, simulation.catalog, campaign, seed + 71
    )
    records, traces, surface = {}, [], None
    roles = ("oracle",) if scenario_name == "P0" else ("oracle", "misspecified")
    for role in roles:
        try:
            record, fitted_surface, trace = _partition_fit(
                role,
                scenario_name,
                replicate,
                settings["oracle_zones"] if role == "oracle" else settings["fit_zones"],
                simulation,
                cutoff,
                settings,
                gp_prior,
                calibration_seconds,
                calibration_n_events,
                partition_campaign,
                seed + 101,
                evaluation_quadrature,
                background_quadrature,
                spatial_compensator_quadrature,
                capture_surface=(
                    replicate == PARTITION_FIGURE_REPLICATE
                    and (scenario_name == "P0" or role == "misspecified")
                ),
            )
            records[role] = record
            if fitted_surface is not None:
                surface = fitted_surface
            if trace is not None:
                traces.append(trace)
        except Exception as error:
            records[role] = {
                "experiment": 2,
                "scenario": scenario_name,
                "replicate": replicate,
                "fit_role": role,
                "method": EXPERIMENT_2_METHOD,
                "method_label": METHODS[EXPERIMENT_2_METHOD]["label"],
                "n_events": len(simulation.catalog),
                "status": "error",
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
    oracle = records["oracle"]
    misspecified = (
        {**oracle, "fit_role": "misspecified"}
        if scenario_name == "P0" else records["misspecified"]
    )
    paired = {
        "experiment": 2,
        "scenario": scenario_name,
        "replicate": replicate,
        "status": "ok"
        if oracle.get("status") == misspecified.get("status") == "ok"
        else "incomplete",
        "n_events": len(simulation.catalog),
        "diagnostic_status": (
            "single_chain_ess_only"
            if oracle.get("status") == misspecified.get("status") == "ok"
            and campaign.n_partition_chains == 1
            else "ok"
            if oracle.get("diagnostic_status") == misspecified.get("diagnostic_status") == "ok"
            else "check_paired_fits"
        ),
        "delta_rel_l2_background": misspecified.get("rel_l2_background", np.nan)
        - oracle.get("rel_l2_background", np.nan),
        "delta_etas_parameter_log_error": misspecified.get(
            "etas_parameter_log_error", np.nan
        )
        - oracle.get("etas_parameter_log_error", np.nan),
        "delta_background_brier": misspecified.get("background_brier", np.nan)
        - oracle.get("background_brier", np.nan),
        "delta_runtime_seconds": misspecified.get("runtime_seconds", np.nan)
        - oracle.get("runtime_seconds", np.nan),
    }
    return [oracle, misspecified], [paired], surface, traces


def run_partition_experiment(
    campaign,
    n_jobs=1,
    *,
    checkpoint_dir=None,
    resume=True,
    worker_memory_reservation_gib=None,
    evaluation_quadrature=None,
    background_quadrature=None,
    spatial_compensator_quadrature=None,
):
    if worker_memory_reservation_gib is None:
        worker_memory_reservation_gib = execution_guard(
            "2", ACCURACY_TARGET_EVENTS
        )["memory_reservation_gib"]
    if evaluation_quadrature is None:
        evaluation_quadrature = regular_spatial_quadrature(
            campaign.evaluation_space_grid
        )
    elif not isinstance(evaluation_quadrature, SpatialQuadrature):
        raise TypeError("evaluation_quadrature must be a SpatialQuadrature instance.")
    tasks = [
        (
            scenario,
            replicate,
            campaign,
            evaluation_quadrature,
            background_quadrature,
            spatial_compensator_quadrature,
        )
        for scenario in _partition_scenarios()
        for replicate in range(campaign.n_partition_replicates)
    ]
    task_keys = [(scenario, replicate) for scenario, replicate, *_ in tasks]
    results = parallel_map(
        _partition_replicate,
        tasks,
        n_jobs,
        "Experiment 2",
        task_keys=task_keys,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        max_parallel_calibrations=campaign.max_parallel_calibrations,
        isolate_tasks=campaign.name == "full",
        worker_memory_reservation_gib=worker_memory_reservation_gib,
    )
    raw = [record for raw_group, _, _, _ in results for record in raw_group]
    paired = [record for _, paired_group, _, _ in results for record in paired_group]
    surfaces = [surface for _, _, surface, _ in results if surface is not None]
    traces = [trace for _, _, _, group in results for trace in group]
    return raw, paired, surfaces, traces


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
