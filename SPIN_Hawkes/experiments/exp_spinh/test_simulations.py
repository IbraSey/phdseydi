"""Launch the SPIN-H numerical studies from Python or a VS Code cell.

The default full run completes and saves Experiment 1 before starting
Experiment 2. Restart the interactive kernel after changing imported modules.
"""

# %% Imports and settings
import sys
from pathlib import Path

start = Path(globals().get("__file__", Path.cwd() / "interactive.py")).resolve().parent
search_roots = [start, *start.parents]
for candidate in search_roots + [path / "SPIN_Hawkes" for path in search_roots]:
    if (candidate / "package/models/spinh.py").is_file():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break
else:
    raise RuntimeError("Open the repository folder before running this script.")

from experiments.exp_spinh.simulation_settings import METHODS
from experiments.exp_spinh.simulation_runner import execute, main, run, should_use_editor_settings

EDITOR_ACTION = "run"             # "run" or "postprocess"
EDITOR_PROFILE = "full"           # complete production budgets
EDITOR_EXPERIMENT = "all"         # run Experiment 1, save it, then run Experiment 2
EDITOR_METHODS = tuple(METHODS)    # M1--M5
EDITOR_N_JOBS = -1                 # all safe workers, capped automatically by RAM
EDITOR_RESUME = True               # reuse completed task checkpoints
EDITOR_SAVE_FIGURES = True
EDITOR_SHOW_FIGURES = False
EDITOR_OUTPUT_DIR = None           # None: results/spinh_test/<profile>

# Optional overrides; all numerical defaults live in simulation_settings.py.
EDITOR_CAMPAIGN_OVERRIDES = {
    # Example: "gibbs_iterations": 4000,
}


def editor_run_options():
    """Return an isolated copy of the options from the editor settings block."""
    return {
        "action": EDITOR_ACTION,
        "profile": EDITOR_PROFILE,
        "experiment": EDITOR_EXPERIMENT,
        "methods": tuple(EDITOR_METHODS),
        "n_jobs": EDITOR_N_JOBS,
        "resume": EDITOR_RESUME,
        "save_figures": EDITOR_SAVE_FIGURES,
        "show_figures": EDITOR_SHOW_FIGURES,
        "output_dir": EDITOR_OUTPUT_DIR,
        "campaign_overrides": dict(EDITOR_CAMPAIGN_OVERRIDES),
    }


def run_from_editor():
    return execute(**editor_run_options())


# %% Run the selected experiment
if __name__ == "__main__":
    if should_use_editor_settings():
        results = run_from_editor()
    else:
        results = main()


# %%
