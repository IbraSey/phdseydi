"""Run every SPIN-H numerical experiment currently promised in the draft.

Run this file directly from an editor, or use::

    python experiments/exp_spinh/test_full_campaign.py

The default full campaign runs one fit at a time, writes task checkpoints and
can be resumed by running the same command again after an interruption.
"""

from __future__ import annotations

# %% Imports
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exp_spinh import test_fcat17, test_simulations
from experiments.exp_spinh.test_utils import METHODS


# %% ========================================================================
# EDITOR SETTINGS
# =============================================================================
# This is the single block to edit before using "Run Python File".  The full
# defaults below generate every table and figure currently implemented for the
# main numerical-experiments draft.

EDITOR_PROFILE = "full"                 # use "smoke" for a short validation
EDITOR_RUN_SIMULATIONS = True
EDITOR_RUN_FCAT17 = True
EDITOR_N_JOBS = 1                        # simultaneous fits; start with 1
EDITOR_RESUME = True                     # continue from compatible checkpoints
EDITOR_SAVE_FIGURES = True
EDITOR_SHOW_FIGURES = False

# %% End of editor settings


def run(
    profile="full",
    *,
    run_simulations=True,
    run_fcat17=True,
    n_jobs=1,
    resume=True,
    save_figures=True,
    show_figures=False,
):
    """Run the complete implemented campaign and return both result bundles."""
    if not run_simulations and not run_fcat17:
        raise ValueError("At least one campaign component must be selected.")
    results = {}
    if run_simulations:
        results["simulations"] = test_simulations.run(
            profile=profile,
            experiment="all",
            methods=tuple(METHODS),
            n_jobs=n_jobs,
            resume=resume,
            save_figures=save_figures,
            show_figures=show_figures,
        )
    if run_fcat17:
        results["fcat17"] = test_fcat17.run(
            profile=profile,
            models=test_fcat17.MODELS,
            inference_method="vi_sparse",
            n_jobs=n_jobs,
            resume=resume,
            save_figures=save_figures,
            show_figures=show_figures,
        )
    return results


def run_from_editor():
    return run(
        profile=EDITOR_PROFILE,
        run_simulations=EDITOR_RUN_SIMULATIONS,
        run_fcat17=EDITOR_RUN_FCAT17,
        n_jobs=EDITOR_N_JOBS,
        resume=EDITOR_RESUME,
        save_figures=EDITOR_SAVE_FIGURES,
        show_figures=EDITOR_SHOW_FIGURES,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Simultaneous jobs (default: 1). Test 2 before requesting more; RAM is shared.",
    )
    parser.add_argument("--skip-simulations", action="store_true")
    parser.add_argument("--skip-fcat17", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--show-figures", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    return run(
        profile=args.profile,
        run_simulations=not args.skip_simulations,
        run_fcat17=not args.skip_fcat17,
        n_jobs=args.n_jobs,
        resume=not args.no_resume,
        save_figures=not args.no_figures,
        show_figures=args.show_figures,
    )


# %% Run the file
if __name__ == "__main__":
    if test_simulations.should_use_editor_settings():
        run_from_editor()
    else:
        main()

# %%
