"""Campaign orchestration and command-line interface."""

from __future__ import annotations

from numbers import Integral
from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

from .runner_utils import (
    checkpoint_directory,
    effective_worker_count,
    resolve_n_jobs,
    simulation_execution_guard as execution_guard,
)
from .simulation_outputs import (
    ACCURACY_METRICS,
    PARTITION_METRICS,
    _read_records,
    load_accuracy_backgrounds,
    load_accuracy_posteriors,
    generating_parameters,
    load_gibbs_traces,
    load_partition_surfaces,
    render_accuracy_outputs,
    plot_partition,
    plot_partition_gibbs_diagnostics,
    plot_partition_surfaces,
    records_for_export,
    remove_legacy_gibbs_diagnostic_files,
    save_accuracy_backgrounds,
    save_accuracy_posteriors,
    save_partition_gibbs_traces,
    save_partition_surfaces,
    select_representative_reconstructions,
    write_experiment_1_latex,
    write_experiment_2_latex,
)
from .simulation_settings import (
    ACCURACY_TARGET_EVENTS,
    CAMPAIGNS,
    ETAS_PARAMETER_NAMES,
    EXPERIMENT_2_DURATIONS,
    EXPERIMENT_2_METHOD,
    METHODS,
    PARTITION_FIGURE_GRID_SIZE,
    PARTITION_FIGURE_REPLICATE,
    REPO_ROOT,
    RESULTS_ROOT,
    SCENARIOS,
    configure_campaign,
    simulation_protocol,
    validate_scientific_settings,
)
from .simulation_studies import (
    _partition_scenarios,
    _validate_partition_figure_settings,
    run_accuracy_panel,
    run_partition_experiment,
)
from .test_utils import (
    quadrature_metadata,
    summarize_records,
    write_campaign,
    write_records,
)


_CHECKPOINT_SOURCES = (
    Path(__file__).with_name("simulation_settings.py"),
    Path(__file__).with_name("simulation_studies.py"),
    Path(__file__).with_name("test_utils.py"),
    Path(__file__).with_name("runner_utils.py"),
    REPO_ROOT / "package",
    REPO_ROOT / "data",
    REPO_ROOT / "simulation",
    REPO_ROOT / "spatial",
)


def should_use_editor_settings(arguments=None, in_ipykernel=None):
    """Select editor mode for notebooks or direct launches without CLI options."""
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    if in_ipykernel is None:
        in_ipykernel = "ipykernel" in sys.modules
    return bool(in_ipykernel) or not arguments


def _validate_execution_settings(n_jobs):
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, Integral) or n_jobs == 0:
        raise ValueError("n_jobs must be a non-zero integer (use -1 for all CPUs).")


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


def _print_run_settings(campaign, experiment, methods, n_jobs, output):
    print("\n" + "=" * 78)
    print("SPIN-H SIMULATED-DATA TESTS")
    print("=" * 78)
    if experiment == "1":
        print(
            f"Profile={campaign.name} | experiment=1 | "
            f"methods={','.join(method.upper() for method in methods)}"
        )
        print(
            f"Replicates per scenario={campaign.n_replicates} | "
            f"Gibbs: {campaign.n_chains} chain(s) x {campaign.gibbs_iterations} iter. | "
            f"VI: {campaign.vi_starts} start(s) x {campaign.vi_iterations} iter."
        )
    elif experiment == "2":
        print(
            f"Profile={campaign.name} | experiment=2 | "
            f"method={EXPERIMENT_2_METHOD.upper()}"
        )
        print(
            f"Replicates per partition scenario={campaign.n_partition_replicates} | "
            f"Gibbs: {campaign.n_partition_chains} chain(s) x "
            f"{campaign.gibbs_iterations} iter. | "
            f"burn-in={campaign.partition_gibbs_burn_in:.0%}"
        )
    else:
        print(
            f"Profile={campaign.name} | experiments=1,2 | "
            f"Experiment 1 methods={','.join(method.upper() for method in methods)} | "
            f"Experiment 2 method={EXPERIMENT_2_METHOD.upper()}"
        )
        print(
            f"Replicates: accuracy={campaign.n_replicates}, "
            f"partition={campaign.n_partition_replicates} | "
            f"Gibbs chains: accuracy={campaign.n_chains}, "
            f"partition={campaign.n_partition_chains} x "
            f"{campaign.gibbs_iterations} iter. | "
            f"VI: {campaign.vi_starts} start(s) x {campaign.vi_iterations} iter."
        )
    print(
        f"Background draws={campaign.posterior_draws}, "
        f"parameter draws={campaign.parameter_draws}, "
        f"GP calibration={'on' if campaign.use_calibration else 'off'}, "
        f"workers={effective_worker_count(n_jobs)} (n_jobs={n_jobs})"
    )
    if campaign.use_calibration:
        print(
            "GP calibration: complete observed catalogue, "
            f"at most {campaign.max_parallel_calibrations} simultaneous fit(s)"
        )
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
        print(f"{'Scenario':<12} {'Method':<6} {'N':>6} {'L2 bg':>10} {'ETAS err.':>10} {'Time (s)':>10}")
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
                f"{mean_events:>6} {row['rel_l2_background']:>10.3f} "
                f"{row['etas_parameter_log_error']:>10.3f} "
                f"{row['runtime_seconds']:>10.2f}"
            )
    if "experiment_2" in results:
        raw, paired, summary, *_ = results["experiment_2"]
        failures.extend(record for record in raw if record.get("status") != "ok")
        failures.extend(record for record in paired if record.get("status") != "ok")
        print(f"\nExperiment 2 - paired partition effects ({EXPERIMENT_2_METHOD.upper()})")
        print(f"{'Case':<6} {'Delta L2 bg':>12} {'Delta ETAS err.':>16}")
        for row in summary:
            print(
                f"{row['scenario']:<6} {row['delta_rel_l2_background']:>12.3f} "
                f"{row['delta_etas_parameter_log_error']:>16.3f}"
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
        print("\nAll requested computations completed.")
    print(f"Results written to: {output}")
    print("-" * 78)


def postprocess_accuracy_results(
    profile="full", *, save_figures=True, show_figures=False, output_dir=None,
):
    """Rebuild tables, backgrounds, marginals and traces from saved arrays."""
    output = Path(output_dir) if output_dir is not None else RESULTS_ROOT / profile
    raw_path = output / "experiment_1_accuracy_raw.csv"
    if not raw_path.is_file():
        raise FileNotFoundError(f"Missing accuracy results: {raw_path}")
    records = _read_records(raw_path)
    for record in records:
        # Older runs stored individual errors but not their ETAS-only average.
        if record.get("etas_parameter_log_error") in (None, ""):
            errors = [record.get(f"log_error_{name}") for name in ETAS_PARAMETER_NAMES]
            if all(value not in (None, "") and np.isfinite(value) for value in errors):
                record["etas_parameter_log_error"] = float(np.mean(errors))
    summary = summarize_records(
        records, ("scenario", "method", "method_label"), ACCURACY_METRICS
    )
    write_records(output / "experiment_1_accuracy_table.csv", summary)
    write_experiment_1_latex(output, summary)
    backgrounds = load_accuracy_backgrounds(output)
    posteriors = load_accuracy_posteriors(output)
    selection_path = output / "experiment_1_reconstruction_selection.csv"
    selection = _read_records(selection_path) if selection_path.is_file() else []
    render_accuracy_outputs(
        records, summary, backgrounds, posteriors, selection, output,
        save=save_figures, show=_resolve_figure_display(show_figures),
    )
    print(f"Experiment 1 outputs rebuilt without inference: {output}")
    if not backgrounds:
        print("No saved background arrays were found; reconstruction figures were not rebuilt.")
    if not posteriors:
        print("No saved posterior arrays were found; marginal and trace figures were not rebuilt.")
    return records, summary


def postprocess_partition_results(
    profile="full", *, save_figures=True, show_figures=False, output_dir=None,
):
    """Rebuild Experiment 2 outputs from saved results, without inference."""
    output = Path(output_dir) if output_dir is not None else RESULTS_ROOT / profile
    raw_path = output / "experiment_2_fits_raw.csv"
    paired_path = output / "experiment_2_paired_raw.csv"
    for path in (raw_path, paired_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing partition results: {path}")
    raw, paired = _read_records(raw_path), _read_records(paired_path)
    summary = summarize_records(paired, ("scenario",), PARTITION_METRICS)
    write_records(output / "experiment_2_table.csv", summary)
    write_experiment_2_latex(output, summary)
    surfaces = load_partition_surfaces(output)
    traces = load_gibbs_traces(output, experiment=2)
    display = _resolve_figure_display(show_figures)
    if save_figures or display:
        plot_partition(summary, output, save=save_figures, show=display)
        plot_partition_surfaces(surfaces, output, save=save_figures, show=display)
        plot_partition_gibbs_diagnostics(
            traces, output, save=save_figures, show=display,
            truths=generating_parameters(raw),
        )
    print(f"Experiment 2 outputs rebuilt without inference: {output}")
    if not surfaces:
        print("No saved background arrays were found; partition maps were not rebuilt.")
    if not traces:
        print("No saved Gibbs traces were found; trace and ACF figures were not rebuilt.")
    return raw, paired, summary, surfaces, traces


def run(
    profile="smoke",
    experiment="all",
    methods=tuple(METHODS),
    *,
    n_jobs=None,
    resume=True,
    save_figures=True,
    show_figures=False,
    campaign_overrides=None,
    output_dir=None,
    evaluation_quadrature=None,
    background_quadrature=None,
    spatial_compensator_quadrature=None,
):
    validate_scientific_settings()
    methods = tuple(methods)
    unknown = set(methods) - set(METHODS)
    if not methods or unknown or len(set(methods)) != len(methods):
        raise ValueError(f"At least one valid method is required; unknown={sorted(unknown)}.")
    if experiment not in {"1", "2", "all"}:
        raise ValueError("experiment must be '1', '2' or 'all'.")
    guard = execution_guard(experiment, ACCURACY_TARGET_EVENTS)
    n_jobs = resolve_n_jobs(
        profile,
        n_jobs,
        max_full_workers=guard["max_workers"],
        worker_memory_reservation_gib=guard["memory_reservation_gib"],
    )
    _validate_execution_settings(n_jobs)
    if not all(isinstance(value, bool) for value in (resume, save_figures, show_figures)):
        raise ValueError("resume, save_figures and show_figures must be boolean.")
    display_figures = _resolve_figure_display(show_figures)
    campaign = configure_campaign(profile, **(campaign_overrides or {}))
    quadratures = {
        "evaluation": quadrature_metadata(evaluation_quadrature),
        "vi_background": quadrature_metadata(background_quadrature),
        "etas_spatial_compensator": quadrature_metadata(
            spatial_compensator_quadrature
        ),
    }
    if experiment in {"2", "all"}:
        _validate_partition_figure_settings(campaign)
    output = Path(output_dir) if output_dir is not None else RESULTS_ROOT / campaign.name
    output.mkdir(parents=True, exist_ok=True)
    remove_legacy_gibbs_diagnostic_files(output)
    _print_run_settings(
        campaign,
        experiment,
        methods,
        n_jobs,
        output,
    )
    write_campaign(
        output / "simulation_campaign.json",
        campaign,
        {
            "methods": methods,
            "experiment_1_methods": methods,
            "experiment_2_method": EXPERIMENT_2_METHOD,
            "experiment": experiment,
            "n_jobs": int(n_jobs),
            "effective_workers": effective_worker_count(n_jobs),
            "memory_policy": guard,
            "resume": resume,
            "partition_figure_replicate": PARTITION_FIGURE_REPLICATE,
            "partition_figure_grid_size": PARTITION_FIGURE_GRID_SIZE,
            "experiment_2_durations": EXPERIMENT_2_DURATIONS,
            "scientific_protocol": simulation_protocol(),
            "custom_quadratures": quadratures,
        },
    )
    results = {}
    if experiment in {"1", "all"}:
        accuracy_checkpoints = checkpoint_directory(
            output,
            "simulation_accuracy",
            campaign,
            settings={"methods": methods, "quadratures": quadratures},
            source_paths=_CHECKPOINT_SOURCES,
        )
        accuracy, reconstructions, posteriors = run_accuracy_panel(
            campaign,
            methods,
            n_jobs=n_jobs,
            checkpoint_dir=accuracy_checkpoints,
            resume=resume,
            worker_memory_reservation_gib=guard["memory_reservation_gib"],
            evaluation_quadrature=evaluation_quadrature,
            background_quadrature=background_quadrature,
            spatial_compensator_quadrature=spatial_compensator_quadrature,
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
        write_records(
            output / "experiment_1_accuracy_raw.csv",
            records_for_export(accuracy),
        )
        write_records(output / "experiment_1_accuracy_table.csv", accuracy_summary)
        save_accuracy_posteriors(posteriors, output)
        selected, selection_records = select_representative_reconstructions(
            accuracy, reconstructions
        )
        write_records(
            output / "experiment_1_reconstruction_selection.csv",
            selection_records,
        )
        save_accuracy_backgrounds(reconstructions, output)
        render_accuracy_outputs(
            accuracy, accuracy_summary, selected, posteriors, selection_records,
            output, save=save_figures, show=display_figures,
        )
        results["experiment_1_accuracy"] = (accuracy, accuracy_summary)
        write_experiment_1_latex(output, accuracy_summary)
    if experiment in {"2", "all"}:
        partition_checkpoints = checkpoint_directory(
            output,
            "simulation_partition",
            campaign,
            settings={
                "scenarios": tuple(_partition_scenarios()),
                "quadratures": quadratures,
            },
            source_paths=_CHECKPOINT_SOURCES,
        )
        raw, paired, surfaces, traces = run_partition_experiment(
            campaign,
            n_jobs=n_jobs,
            checkpoint_dir=partition_checkpoints,
            resume=resume,
            worker_memory_reservation_gib=guard["memory_reservation_gib"],
            evaluation_quadrature=evaluation_quadrature,
            background_quadrature=background_quadrature,
            spatial_compensator_quadrature=spatial_compensator_quadrature,
        )
        paired_summary = summarize_records(
            paired,
            ("scenario",),
            PARTITION_METRICS,
        )
        write_records(output / "experiment_2_fits_raw.csv", records_for_export(raw))
        write_records(output / "experiment_2_paired_raw.csv", records_for_export(paired))
        write_records(output / "experiment_2_table.csv", paired_summary)
        write_experiment_2_latex(output, paired_summary)
        save_partition_gibbs_traces(traces, output)
        save_partition_surfaces(surfaces, output)
        if save_figures or display_figures:
            plot_partition(paired_summary, output, save=save_figures, show=display_figures)
            plot_partition_surfaces(
                surfaces, output, save=save_figures, show=display_figures,
            )
            plot_partition_gibbs_diagnostics(
                traces, output, save=save_figures, show=display_figures,
            )
        results["experiment_2"] = (
            raw, paired, paired_summary, surfaces, traces,
        )
    _print_run_summary(results, output)
    if display_figures and "ipykernel" not in sys.modules:
        plt.show()
    return results


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
        "--action",
        choices=("run", "postprocess", "marginals"),
        default="run",
        help=(
            "Run experiments or rebuild figures/tables from saved arrays. "
            "'marginals' is an alias for postprocess and never refits a model."
        ),
    )
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
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
        help="Subset of the M1--M5 methods used in Experiment 1.",
    )
    selection.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: results/<profile>).",
    )
    selection.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Requested simultaneous jobs (default: 1); full runs are capped by available RAM.",
    )
    budget = parser.add_argument_group("campaign budget overrides")
    budget.add_argument("--n-replicates", type=int, default=None)
    budget.add_argument("--n-partition-replicates", type=int, default=None)
    budget.add_argument("--n-chains", type=int, default=None)
    budget.add_argument("--n-partition-chains", type=int, default=None)
    budget.add_argument("--vi-starts", type=int, default=None)
    budget.add_argument("--gibbs-iterations", type=int, default=None)
    budget.add_argument("--gibbs-thin", type=int, default=None)
    budget.add_argument("--gibbs-burn-in", type=float, default=None)
    budget.add_argument("--partition-gibbs-burn-in", type=float, default=None)
    budget.add_argument("--gibbs-adaptation-fraction", type=float, default=None)
    budget.add_argument("--vi-iterations", type=int, default=None)
    budget.add_argument("--evaluation-space-grid", type=int, default=None)
    budget.add_argument("--quadrature-space-grid", type=int, default=None)
    budget.add_argument("--posterior-draws", type=int, default=None)
    budget.add_argument("--parameter-draws", type=int, default=None)
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
        "n_partition_replicates": args.n_partition_replicates,
        "n_chains": args.n_chains,
        "n_partition_chains": args.n_partition_chains,
        "vi_starts": args.vi_starts,
        "gibbs_iterations": args.gibbs_iterations,
        "gibbs_thin": args.gibbs_thin,
        "gibbs_burn_in": args.gibbs_burn_in,
        "partition_gibbs_burn_in": args.partition_gibbs_burn_in,
        "gibbs_adaptation_fraction": args.gibbs_adaptation_fraction,
        "vi_iterations": args.vi_iterations,
        "evaluation_space_grid": args.evaluation_space_grid,
        "quadrature_space_grid": args.quadrature_space_grid,
        "posterior_draws": args.posterior_draws,
        "parameter_draws": args.parameter_draws,
        "max_parallel_calibrations": args.max_parallel_calibrations,
        "duration_scale": args.duration_scale,
        "exact_max_events": args.exact_max_events,
        "dense_max_events": args.dense_max_events,
        "use_calibration": False if args.no_calibration else None,
    }


def execute(action="run", **options):
    """Shared dispatch for editor settings and command-line arguments."""
    if action in {"postprocess", "marginals"}:
        settings = {
            name: value for name, value in options.items()
            if name in {"profile", "save_figures", "show_figures", "output_dir"}
        }
        experiment = options.get("experiment", "all")
        if experiment not in {"1", "2", "all"}:
            raise ValueError("experiment must be '1', '2' or 'all'.")
        if experiment == "1":
            return postprocess_accuracy_results(**settings)
        if experiment == "2":
            return postprocess_partition_results(**settings)
        output = settings.get("output_dir")
        output = Path(output) if output is not None else RESULTS_ROOT / settings.get("profile", "full")
        results = {}
        if (output / "experiment_1_accuracy_raw.csv").is_file():
            results["experiment_1_accuracy"] = postprocess_accuracy_results(**settings)
        if (output / "experiment_2_fits_raw.csv").is_file():
            results["experiment_2"] = postprocess_partition_results(**settings)
        if not results:
            raise FileNotFoundError(f"No saved simulation results found in {output}")
        return results
    if action != "run":
        raise ValueError("action must be 'run', 'postprocess' or 'marginals'.")
    return run(**options)


def main(argv=None):
    args = parse_args(argv)
    return execute(
        action=args.action,
        profile=args.profile,
        experiment=args.experiment,
        methods=args.methods,
        n_jobs=args.n_jobs,
        resume=not args.no_resume,
        save_figures=not args.no_figures,
        show_figures=args.show_figures,
        output_dir=args.output_dir,
        campaign_overrides=_overrides_from_args(args),
    )
