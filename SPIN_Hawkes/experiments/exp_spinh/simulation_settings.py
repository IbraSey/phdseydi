"""Scientific protocol and smoke/full numerical budgets."""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from package import ETASParameters


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results" / "spinh_test"

# Shared observation domain and magnitude range for simulated catalogues.
X_BOUNDS = (0.0, 2.0)
Y_BOUNDS = (0.0, 2.0)
MAGNITUDE_MIN = 2.0
MAGNITUDE_MAX = 6.0
TRUNCATION_RELATIVE_DENSITY = 1e-3
N_REGIONS = 6
PARTITION_SEED = 15
MISSPECIFIED_PARTITION_REGIONS = 5
MISSPECIFIED_PARTITION_SEED = 47
PARAMETER_NAMES = ("A", "alpha", "c", "p", "d", "q", "gamma", "beta")
ETAS_PARAMETER_NAMES = PARAMETER_NAMES[:-1]

# Accuracy durations were selected from 50 generation-only pilot catalogues per
# scenario, using seeds starting at 120_000 (easy) and 130_000 (difficult).
# The selected horizons generated 399.0 and 398.1 events on average.
ACCURACY_TARGET_EVENTS = 400
ACCURACY_DURATION_CALIBRATION_REPLICATES = 50
ACCURACY_DURATION_CALIBRATION_SEED = 120_000

# Common initialization and priors used by every compared inference method.
INITIAL_ETAS = ETASParameters(
    A=0.4,
    alpha=0.6,
    c=0.03,
    p=1.35,
    d=0.06,
    q=1.7,
    gamma=0.3,
)
INITIAL_BETA = 2.3
# Proposal pilots live in test_proposal_steps.py. MALA uses initial conditional
# curvature; MH standard deviations are on log parameters (log(p-1), log(q-1)).
MALA_CURVATURE_SCALE = 1.8
MH_ETAS_REFERENCE_STEP = 0.35
MH_ETAS_REFERENCE_EVENTS = 50
MH_BETA_SCALE = 2.4
VI_INITIAL_CONCENTRATION_MULTIPLIER = 10.0
VI_START_PROFILES = (
    {"name": "background_rich", "A_multiplier": 0.5, "background_fraction": 0.85},
    {"name": "balanced", "A_multiplier": 1.5, "background_fraction": 0.65},
    {"name": "triggering_rich", "A_multiplier": 3.0, "background_fraction": 0.45},
)
VI_ETAS_UPDATE_START = 5
VI_ETAS_UPDATE_EVERY = 5
VI_MAX_OPTIMIZER_ITER = 20
VI_GAMMA_QUADRATURE_NODES = 4
# Same finite-window spatial integration as Gibbs, independent of the GP grid.
ETAS_SPATIAL_QUADRATURE = 40
# Broad shared priors on A, alpha, c, p-1, d, q-1, gamma; not refitted per scenario.
THETA_PRIORS = {
    "a_A": 2.0,
    "b_A": 5.0,
    "a_alpha": 2.0,
    "b_alpha": 2.0 / 0.6,
    "a_c": 2.0,
    "b_c": 50.0,
    "a_p": 2.0,
    "b_p": 5.0,
    "a_d": 2.0,
    "b_d": 20.0,
    "a_q": 2.0,
    "b_q": 3.0,
    "a_gamma": 1.0,
    "b_gamma": 1.0 / 0.3,
}
INITIAL_GAMMA_FACTORS = {
    "A": (10.0 * INITIAL_ETAS.A, 10.0),
    "alpha": (10.0 * INITIAL_ETAS.alpha, 10.0),
    "c": (100.0 * INITIAL_ETAS.c, 100.0),
    "p_minus_1": (10.0 * (INITIAL_ETAS.p - 1.0), 10.0),
    "d": (50.0 * INITIAL_ETAS.d, 50.0),
    "q_minus_1": (10.0 * (INITIAL_ETAS.q - 1.0), 10.0),
    "gamma": (10.0 * INITIAL_ETAS.gamma, 10.0),
    "beta": (10.0 * INITIAL_BETA, 10.0),
}

# M1--M5 are kept in insertion order throughout tables and figures.
METHODS = {
    "m1": {
        "label": "M1 Exact-GP Gibbs",
        "family": "gibbs",
        "gp_backend": "exact",
        "truncated": False,
    },
    "m2": {
        "label": "M2 HSGP Gibbs",
        "family": "gibbs",
        "gp_backend": "sparse",
        "truncated": False,
    },
    "m3": {
        "label": "M3 Truncated HSGP Gibbs",
        "family": "gibbs",
        "gp_backend": "sparse",
        "truncated": True,
    },
    "m4": {
        "label": "M4 HSGP MF-VI",
        "family": "vi",
        "gp_backend": "sparse",
        "truncated": False,
    },
    "m5": {
        "label": "M5 Truncated HSGP MF-VI",
        "family": "vi",
        "gp_backend": "sparse",
        "truncated": True,
    },
}

# Experiment 1 uses one common background in both scenarios. Their difference
# is deliberately confined to the ETAS triggering process.
ACCURACY_BACKGROUND_MUS = (8.0, 1.0, 2.0, 8.0, 7.0, 2.0)

# Experiment 1 generating configurations.
SCENARIOS = {
    "easy": {
        "duration": 51.0,
        "field_scale": 1.0,
        "mus": ACCURACY_BACKGROUND_MUS,
        "etas": ETASParameters(
            A=0.40,
            alpha=0.60,
            c=0.03,
            p=1.35,
            d=0.06,
            q=1.70,
            gamma=0.30,
        ),
        "beta": 2.30,
    },
    "difficult": {
        "duration": 52.0,
        "field_scale": 1.0,
        "mus": ACCURACY_BACKGROUND_MUS,
        "etas": ETASParameters(
            A=0.50,
            alpha=0.60,
            c=0.08,
            p=1.30,
            d=0.10,
            q=1.55,
            gamma=0.30,
        ),
        "beta": 2.30,
    },
}

# Experiment 2 generating configuration and regional baselines.
EXPERIMENT_2_ETAS = ETASParameters(
    A=0.50,
    alpha=0.80,
    c=0.02,
    p=1.30,
    d=0.05,
    q=1.80,
    gamma=0.50,
)
EXPERIMENT_2_BETA = 2.30
EXPERIMENT_2_METHOD = "m3"
EXPERIMENT_2_TARGET_EVENTS = 1_000
# Validated on 20 independent generation-only catalogues per scenario, using
# seeds starting at 210_000 and an offset of 1_000 between scenarios. Mean
# catalogue sizes ranged from 977 to 1,023 events (1,002 overall).
EXPERIMENT_2_DURATIONS = {
    "P0": 90.4,
    "P1": 46.2,
    "P2": 90.0,
    "P3": 88.7,
    "P4": 59.5,
}
REFERENCE_MUS = (10.0, 1.0, 2.0, 10.0, 8.0, 2.0)
HIGH_CONTRAST_MUS = (20.0, 1.0, 1.0, 1.0, 1.0, 20.0)


# =============================================================================
# COMPUTATIONAL PROFILES
# =============================================================================


@dataclass(frozen=True)
class CampaignConfig:
    """Validated numerical budget shared by both simulated experiments."""

    name: str
    n_replicates: int
    n_partition_replicates: int
    n_chains: int
    n_partition_chains: int
    vi_starts: int
    gibbs_iterations: int
    gibbs_thin: int
    gibbs_burn_in: float
    partition_gibbs_burn_in: float
    gibbs_adaptation_fraction: float
    vi_iterations: int
    evaluation_space_grid: int
    quadrature_space_grid: int
    posterior_draws: int
    parameter_draws: int
    max_parallel_calibrations: int
    duration_scale: float
    exact_max_events: int
    dense_max_events: int
    use_calibration: bool

    def __post_init__(self):
        integer_fields = {
            "n_replicates": 1,
            "n_partition_replicates": 1,
            "n_chains": 1,
            "n_partition_chains": 1,
            "vi_starts": 1,
            "gibbs_iterations": 1,
            "gibbs_thin": 1,
            "vi_iterations": 1,
            "evaluation_space_grid": 2,
            "quadrature_space_grid": 2,
            "posterior_draws": 1,
            "parameter_draws": 2,
            "max_parallel_calibrations": 1,
            "exact_max_events": 1,
            "dense_max_events": 1,
        }
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string.")
        for name, minimum in integer_fields.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        for name in (
            "duration_scale",
            "gibbs_burn_in",
            "partition_gibbs_burn_in",
            "gibbs_adaptation_fraction",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive.")
        if not 0.0 < self.gibbs_burn_in < 1.0:
            raise ValueError("gibbs_burn_in must be in (0, 1).")
        if not 0.0 < self.partition_gibbs_burn_in < 1.0:
            raise ValueError("partition_gibbs_burn_in must be in (0, 1).")
        if not 0.0 < self.gibbs_adaptation_fraction <= self.gibbs_burn_in:
            raise ValueError(
                "gibbs_adaptation_fraction must be in (0, gibbs_burn_in]."
            )
        if self.gibbs_adaptation_fraction > self.partition_gibbs_burn_in:
            raise ValueError(
                "gibbs_adaptation_fraction must not exceed "
                "partition_gibbs_burn_in."
            )
        if not isinstance(self.use_calibration, bool):
            raise ValueError("use_calibration must be boolean.")
        if int(self.gibbs_iterations * self.gibbs_adaptation_fraction) < 1:
            raise ValueError("The adaptation period must contain at least one iteration.")


CAMPAIGNS = {
    "smoke": CampaignConfig(
        name="smoke",
        n_replicates=1,
        n_partition_replicates=1,
        n_chains=1,
        n_partition_chains=1,
        vi_starts=1,
        gibbs_iterations=50,
        gibbs_thin=2,
        gibbs_burn_in=0.5,
        partition_gibbs_burn_in=0.5,
        gibbs_adaptation_fraction=0.4,
        vi_iterations=10,
        evaluation_space_grid=5,
        quadrature_space_grid=4,
        posterior_draws=4,
        parameter_draws=50,
        max_parallel_calibrations=1,
        duration_scale=0.08,
        exact_max_events=150,
        dense_max_events=300,
        use_calibration=False,
    ),
    "full": CampaignConfig(
        name="full",
        n_replicates=5,
        n_partition_replicates=3,
        n_chains=3,
        n_partition_chains=1,
        vi_starts=3,
        gibbs_iterations=8000,
        gibbs_thin=5,
        gibbs_burn_in=0.25,
        partition_gibbs_burn_in=0.5,
        gibbs_adaptation_fraction=0.25,
        vi_iterations=500,
        evaluation_space_grid=25,
        quadrature_space_grid=12,
        posterior_draws=100,
        parameter_draws=1000,
        max_parallel_calibrations=1,
        duration_scale=1.0,
        exact_max_events=1200,
        dense_max_events=12_000,
        use_calibration=True,
    ),
}


def validate_scientific_settings():
    """Validate all user-editable protocol constants before costly work starts."""
    if (
        isinstance(ETAS_SPATIAL_QUADRATURE, bool)
        or not isinstance(ETAS_SPATIAL_QUADRATURE, Integral)
        or ETAS_SPATIAL_QUADRATURE < 2
    ):
        raise ValueError("ETAS_SPATIAL_QUADRATURE must be an integer >= 2.")
    for name, value, minimum in (
        ("VI_ETAS_UPDATE_START", VI_ETAS_UPDATE_START, 0),
        ("VI_ETAS_UPDATE_EVERY", VI_ETAS_UPDATE_EVERY, 1),
        ("VI_MAX_OPTIMIZER_ITER", VI_MAX_OPTIMIZER_ITER, 1),
        ("VI_GAMMA_QUADRATURE_NODES", VI_GAMMA_QUADRATURE_NODES, 2),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    for name, value in (
        ("MALA_CURVATURE_SCALE", MALA_CURVATURE_SCALE),
        ("MH_ETAS_REFERENCE_STEP", MH_ETAS_REFERENCE_STEP),
        ("MH_ETAS_REFERENCE_EVENTS", MH_ETAS_REFERENCE_EVENTS),
        ("MH_BETA_SCALE", MH_BETA_SCALE),
        ("VI_INITIAL_CONCENTRATION_MULTIPLIER", VI_INITIAL_CONCENTRATION_MULTIPLIER),
    ):
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    for name, bounds in (("X_BOUNDS", X_BOUNDS), ("Y_BOUNDS", Y_BOUNDS)):
        if len(bounds) != 2 or not np.all(np.isfinite(bounds)) or bounds[0] >= bounds[1]:
            raise ValueError(f"{name} must contain two finite increasing values.")
    if not np.isfinite(MAGNITUDE_MIN) or not np.isfinite(MAGNITUDE_MAX):
        raise ValueError("Magnitude bounds must be finite.")
    if MAGNITUDE_MIN >= MAGNITUDE_MAX:
        raise ValueError("MAGNITUDE_MIN must be smaller than MAGNITUDE_MAX.")
    if not 0.0 < TRUNCATION_RELATIVE_DENSITY < 1.0:
        raise ValueError("TRUNCATION_RELATIVE_DENSITY must lie in (0, 1).")
    if not VI_START_PROFILES:
        raise ValueError("VI_START_PROFILES must contain at least one profile.")
    profile_names = set()
    for profile in VI_START_PROFILES:
        if set(profile) != {"name", "A_multiplier", "background_fraction"}:
            raise ValueError("Every VI start profile must define name, A_multiplier and background_fraction.")
        if not isinstance(profile["name"], str) or not profile["name"]:
            raise ValueError("Every VI start profile name must be a non-empty string.")
        if profile["name"] in profile_names:
            raise ValueError("VI start profile names must be unique.")
        profile_names.add(profile["name"])
        for name in ("A_multiplier", "background_fraction"):
            value = profile[name]
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"VI start profile {name} must be finite and positive.")
        if profile["background_fraction"] > 1.0:
            raise ValueError("VI start background_fraction must be at most 1.")
    if isinstance(N_REGIONS, bool) or not isinstance(N_REGIONS, Integral) or N_REGIONS < 1:
        raise ValueError("N_REGIONS must be a positive integer.")
    if isinstance(PARTITION_SEED, bool) or not isinstance(PARTITION_SEED, Integral):
        raise ValueError("PARTITION_SEED must be an integer.")
    if (
        isinstance(MISSPECIFIED_PARTITION_REGIONS, bool)
        or not isinstance(MISSPECIFIED_PARTITION_REGIONS, Integral)
        or MISSPECIFIED_PARTITION_REGIONS < 1
    ):
        raise ValueError("MISSPECIFIED_PARTITION_REGIONS must be a positive integer.")
    if (
        isinstance(MISSPECIFIED_PARTITION_SEED, bool)
        or not isinstance(MISSPECIFIED_PARTITION_SEED, Integral)
    ):
        raise ValueError("MISSPECIFIED_PARTITION_SEED must be an integer.")

    required = {"duration", "field_scale", "mus", "etas", "beta"}
    if not SCENARIOS:
        raise ValueError("SCENARIOS must contain at least one configuration.")
    for scenario_name, scenario in SCENARIOS.items():
        missing = required.difference(scenario)
        if missing:
            raise ValueError(
                f"Scenario {scenario_name!r} is missing settings: {sorted(missing)}."
            )
        if len(scenario["mus"]) != N_REGIONS:
            raise ValueError(
                f"Scenario {scenario_name!r} must define {N_REGIONS} regional baselines."
            )
        positive_values = {
            "duration": scenario["duration"],
            "beta": scenario["beta"],
        }
        for setting_name, value in positive_values.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"SCENARIOS[{scenario_name!r}][{setting_name!r}] "
                    "must be finite and positive."
                )
        if not np.isfinite(scenario["field_scale"]) or scenario["field_scale"] < 0.0:
            raise ValueError(
                f"SCENARIOS[{scenario_name!r}]['field_scale'] must be finite and non-negative."
            )
        mus = np.asarray(scenario["mus"], dtype=float)
        if np.any(~np.isfinite(mus)) or np.any(mus <= 0.0):
            raise ValueError(
                f"SCENARIOS[{scenario_name!r}]['mus'] must be finite and positive."
            )
        if not isinstance(scenario["etas"], ETASParameters):
            raise TypeError(
                f"SCENARIOS[{scenario_name!r}]['etas'] must be ETASParameters."
            )

    for name, values in (
        ("REFERENCE_MUS", REFERENCE_MUS),
        ("HIGH_CONTRAST_MUS", HIGH_CONTRAST_MUS),
    ):
        values = np.asarray(values, dtype=float)
        if values.size != N_REGIONS or np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError(
                f"{name} must contain {N_REGIONS} finite positive baselines."
            )
    if not isinstance(EXPERIMENT_2_ETAS, ETASParameters):
        raise TypeError("EXPERIMENT_2_ETAS must be ETASParameters.")
    method = METHODS.get(EXPERIMENT_2_METHOD)
    if method is None or (
        method["family"], method["gp_backend"], method["truncated"]
    ) != ("gibbs", "sparse", True):
        raise ValueError(
            "EXPERIMENT_2_METHOD must select the truncated HSGP Gibbs method."
        )
    for name, value in (("EXPERIMENT_2_BETA", EXPERIMENT_2_BETA),):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    expected_scenarios = {"P0", "P1", "P2", "P3", "P4"}
    if set(EXPERIMENT_2_DURATIONS) != expected_scenarios:
        raise ValueError(
            "EXPERIMENT_2_DURATIONS must define exactly P0, P1, P2, P3 and P4."
        )
    for scenario_name, duration in EXPERIMENT_2_DURATIONS.items():
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError(
                f"EXPERIMENT_2_DURATIONS[{scenario_name!r}] must be finite and positive."
            )


def configure_campaign(profile: str, **overrides) -> CampaignConfig:
    if profile not in CAMPAIGNS:
        raise ValueError(f"Unknown profile {profile!r}; choose from {tuple(CAMPAIGNS)}.")
    allowed = set(CampaignConfig.__dataclass_fields__) - {"name"}
    unknown = set(overrides) - allowed
    if unknown:
        raise ValueError(f"Unknown campaign override(s): {sorted(unknown)}")
    updates = {name: value for name, value in overrides.items() if value is not None}
    return replace(CAMPAIGNS[profile], **updates)



PARTITION_FIGURE_REPLICATE = 0
PARTITION_FIGURE_GRID_SIZE = 80


def simulation_protocol():
    """Snapshot the scientific settings as well as the computational budget."""
    return {
        "fit_scope": "complete_catalogue",
        "predictive_score": False,
        "accuracy_target_events": ACCURACY_TARGET_EVENTS,
        "accuracy_duration_calibration": {
            "replicates_per_scenario": ACCURACY_DURATION_CALIBRATION_REPLICATES,
            "base_seed": ACCURACY_DURATION_CALIBRATION_SEED,
        },
        "experiment_2_target_events": EXPERIMENT_2_TARGET_EVENTS,
        "experiment_2_method": EXPERIMENT_2_METHOD,
        "scenarios": {
            name: {**settings, "etas": settings["etas"].as_dict()}
            for name, settings in SCENARIOS.items()
        },
        "theta_priors": dict(THETA_PRIORS),
        "initial_etas": INITIAL_ETAS.as_dict(),
        "initial_beta": INITIAL_BETA,
        "vi_initial_concentration_multiplier": VI_INITIAL_CONCENTRATION_MULTIPLIER,
        "vi_start_profiles": [dict(profile) for profile in VI_START_PROFILES],
        "vi_etas_update_start": VI_ETAS_UPDATE_START,
        "vi_etas_update_every": VI_ETAS_UPDATE_EVERY,
        "vi_max_optimizer_iter": VI_MAX_OPTIMIZER_ITER,
        "vi_gamma_quadrature_nodes": VI_GAMMA_QUADRATURE_NODES,
        "gibbs_productivity_update": "partially_collapsed",
        "x_bounds": X_BOUNDS,
        "y_bounds": Y_BOUNDS,
        "magnitude_bounds": (MAGNITUDE_MIN, MAGNITUDE_MAX),
        "partition_seed": PARTITION_SEED,
        "n_regions": N_REGIONS,
        "truncation_relative_density": TRUNCATION_RELATIVE_DENSITY,
        "truncated_compensator": True,
        "etas_spatial_quadrature": ETAS_SPATIAL_QUADRATURE,
    }
