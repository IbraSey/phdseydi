"""Shared infrastructure for the SPIN-Hawkes numerical tests.

The runners keep simulation, inference and evaluation choices explicit.  The
two public campaign profiles are deliberately different: ``smoke`` validates
the complete workflow in minutes, while ``full`` encodes the current numerical
protocol and requires a substantially larger computational budget.
"""

from __future__ import annotations

import csv
import json
import math
import platform
import subprocess
import time
import warnings
from dataclasses import asdict, dataclass, replace
from numbers import Integral, Real
from pathlib import Path

import numpy as np
from shapely.ops import unary_union

from experiments.exp_spinh.runner_utils import calibration_slot
from package import (
    ETASParameters,
    EventCatalog,
    GPParameters,
    SPINHGibbsConfig,
    SPINHModel,
    SPINHVIConfig,
    SparseGP,
    TemporalCandidateGraph,
    generate_voronoi_cells,
    simulate_hawkes_process,
)


# =============================================================================
# SCIENTIFIC SETTINGS
# =============================================================================
# Edit this section only when changing the experimental protocol itself.  The
# settings used to choose which experiment to run live at the top of each
# executable script (test_simulations.py and test_fcat17.py).

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results" / "spinh_test"

# Shared observation domain and magnitude range for simulated catalogues.
X_BOUNDS = (0.0, 2.0)
Y_BOUNDS = (0.0, 2.0)
MAGNITUDE_MIN = 2.0
MAGNITUDE_MAX = 6.0
TRUNCATION_RELATIVE_DENSITY = 1e-3
TRUNCATION_MAX_TAIL_MASS = 0.02
N_REGIONS = 6
PARTITION_SEED = 15
MISSPECIFIED_PARTITION_REGIONS = 5
MISSPECIFIED_PARTITION_SEED = 47
PARAMETER_NAMES = ("A", "alpha", "c", "p", "d", "q", "gamma", "beta")

# Accuracy durations were selected from 50 generation-only pilot catalogues per
# scenario, using seeds starting at 120_000 (easy) and 130_000 (difficult).
# The selected horizons generated 399.0 and 402.5 events on average.
ACCURACY_TARGET_EVENTS = 400
ACCURACY_DURATION_CALIBRATION_REPLICATES = 50
ACCURACY_DURATION_CALIBRATION_SEED = 120_000

# Keep the scaling design independent from the shorter accuracy catalogues.
SCALING_REFERENCE_EVENTS = 500
SCALING_REFERENCE_DURATION = 66.0

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
VI_INITIAL_LOG_SD = 0.35
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

# Experiment 1 generating configurations.
SCENARIOS = {
    "easy": {
        "duration": 51.0,
        "field_scale": 0.90,
        "mus": (8.0, 1.0, 2.0, 8.0, 7.0, 2.0),
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
        "duration": 32.0,
        "field_scale": 0.45,
        "mus": (5.0, 4.0, 4.5, 5.0, 4.0, 4.5),
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
# Calibrated from five pilot catalogues per scenario so that each full-profile
# configuration generates approximately 10,000 events on average.
EXPERIMENT_2_DURATIONS = {
    "P0": 845.0,
    "P1": 426.0,
    "P2": 844.0,
    "P3": 827.0,
    "P4": 548.0,
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
    n_scaling_replicates: int
    n_partition_replicates: int
    n_chains: int
    vi_starts: int
    gibbs_iterations: int
    gibbs_thin: int
    gibbs_burn_in: float
    gibbs_adaptation_fraction: float
    vi_iterations: int
    evaluation_space_grid: int
    evaluation_time_grid: int
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
            "n_scaling_replicates": 1,
            "n_partition_replicates": 1,
            "n_chains": 1,
            "vi_starts": 1,
            "gibbs_iterations": 1,
            "gibbs_thin": 1,
            "vi_iterations": 1,
            "evaluation_space_grid": 2,
            "evaluation_time_grid": 2,
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
        if not 0.0 < self.gibbs_adaptation_fraction <= self.gibbs_burn_in:
            raise ValueError(
                "gibbs_adaptation_fraction must be in (0, gibbs_burn_in]."
            )
        if not isinstance(self.use_calibration, bool):
            raise ValueError("use_calibration must be boolean.")


CAMPAIGNS = {
    "smoke": CampaignConfig(
        name="smoke",
        n_replicates=1,
        n_scaling_replicates=1,
        n_partition_replicates=1,
        n_chains=1,
        vi_starts=1,
        gibbs_iterations=50,
        gibbs_thin=2,
        gibbs_burn_in=0.5,
        gibbs_adaptation_fraction=0.4,
        vi_iterations=10,
        evaluation_space_grid=5,
        evaluation_time_grid=3,
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
        n_replicates=10,
        n_scaling_replicates=5,
        n_partition_replicates=3,
        n_chains=3,
        vi_starts=3,
        gibbs_iterations=6000,
        gibbs_thin=5,
        gibbs_burn_in=0.5,
        gibbs_adaptation_fraction=0.5,
        vi_iterations=500,
        evaluation_space_grid=25,
        evaluation_time_grid=4096,
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
    for name, value in (
        ("MALA_CURVATURE_SCALE", MALA_CURVATURE_SCALE),
        ("MH_ETAS_REFERENCE_STEP", MH_ETAS_REFERENCE_STEP),
        ("MH_ETAS_REFERENCE_EVENTS", MH_ETAS_REFERENCE_EVENTS),
        ("MH_BETA_SCALE", MH_BETA_SCALE),
        ("VI_INITIAL_LOG_SD", VI_INITIAL_LOG_SD),
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
    if not 0.0 < TRUNCATION_MAX_TAIL_MASS < 1.0:
        raise ValueError("TRUNCATION_MAX_TAIL_MASS must lie strictly between zero and one.")
    if not 0.0 < TRUNCATION_RELATIVE_DENSITY < 1.0:
        raise ValueError("TRUNCATION_RELATIVE_DENSITY must lie in (0, 1).")
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


def latent_field(x, y, scale=1.0):
    """Latent field from the numerical-experiment specification."""
    x_values, y_values = np.broadcast_arrays(
        np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    )
    points = np.column_stack([x_values.reshape(-1), y_values.reshape(-1)])
    centers = np.array(
        [[0.5, 0.5], [0.5, 1.5], [1.5, 0.5], [1.5, 1.5]], dtype=float
    )
    weights = float(scale) * np.array([1.5, -1.5, 3.0, -3.0])
    values = np.zeros(points.shape[0], dtype=float)
    for weight, center in zip(weights, centers):
        squared_distance = np.sum((points - center) ** 2, axis=1)
        values += weight * np.exp(-squared_distance / 0.6) / (0.6 * np.pi)
    return values.reshape(x_values.shape)


def generate_partition(n_regions=N_REGIONS, seed=PARTITION_SEED):
    cells, germs = generate_voronoi_cells(
        n_germs=int(n_regions),
        X_bounds=X_BOUNDS,
        Y_bounds=Y_BOUNDS,
        rng_seed=int(seed),
    )
    return list(cells), np.asarray(germs, dtype=float)


def merge_adjacent_zones(zones, target_count):
    """Greedily merge the pair sharing the longest boundary."""
    merged = list(zones)
    target_count = int(target_count)
    if target_count < 1 or target_count > len(merged):
        raise ValueError("target_count must be between one and len(zones).")
    while len(merged) > target_count:
        candidates = []
        for left in range(len(merged)):
            for right in range(left + 1, len(merged)):
                shared = merged[left].boundary.intersection(merged[right].boundary).length
                candidates.append((shared, left, right))
        _, left, right = max(candidates)
        replacement = unary_union([merged[left], merged[right]])
        merged = [
            zone for index, zone in enumerate(merged) if index not in {left, right}
        ] + [replacement]
    return merged


def make_model(
    zones,
    duration,
    *,
    etas=INITIAL_ETAS,
    gp_prior=GPParameters(variance=5.0, length_scale=0.2),
    x_bounds=X_BOUNDS,
    y_bounds=Y_BOUNDS,
):
    return SPINHModel.from_polygons(
        polygons=list(zones),
        duration=float(duration),
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        gp_prior=gp_prior,
        eps_prior_variance=1.0,
        eps_prior_length_scale=0.01,
        nu_prior_rate=0.5,
        jitter=1e-5,
        etas_parameters=etas,
        magnitude_min=MAGNITUDE_MIN,
        magnitude_max=MAGNITUDE_MAX,
    )


def simulate_configuration(
    zones,
    mus,
    duration,
    field_scale,
    etas,
    beta,
    seed,
    *,
    grid_res=80,
):
    return simulate_hawkes_process(
        X_bounds=X_BOUNDS,
        Y_bounds=Y_BOUNDS,
        T=float(duration),
        polygons=list(zones),
        mus=tuple(mus),
        f=lambda x, y: latent_field(x, y, field_scale),
        etas_parameters=etas,
        beta=float(beta),
        magnitude_min=MAGNITUDE_MIN,
        magnitude_max=MAGNITUDE_MAX,
        rng_seed=int(seed),
        grid_res=int(grid_res),
    )


def temporal_cutoff(parameters, relative_density=TRUNCATION_RELATIVE_DENSITY, *, horizon=None):
    """Kernel-height cutoff, widened to control omitted mass on a fixed horizon."""
    relative_density = float(relative_density)
    if not 0.0 < relative_density < 1.0:
        raise ValueError("relative_density must lie in (0, 1).")
    cutoff = float(parameters.c * (relative_density ** (-1.0 / parameters.p) - 1.0))
    if horizon is not None:
        horizon = float(horizon)
        if not np.isfinite(horizon) or horizon <= 0:
            raise ValueError("horizon must be finite and positive.")
        c, p = parameters.c, parameters.p
        horizon_tail = (c / (c + horizon)) ** (p - 1)
        allowed_tail = horizon_tail + TRUNCATION_MAX_TAIL_MASS * (1 - horizon_tail)
        mass_cutoff = c * (allowed_tail ** (-1 / (p - 1)) - 1)
        cutoff = min(horizon, max(cutoff, mass_cutoff))
    return cutoff


def omitted_temporal_mass(parameters, cutoff, horizon):
    cutoff = min(float(cutoff), float(horizon))
    horizon = float(horizon)
    c, p = parameters.c, parameters.p
    horizon_tail = (c / (c + horizon)) ** (p - 1.0)
    numerator = (c / (c + cutoff)) ** (p - 1.0) - horizon_tail
    denominator = 1.0 - horizon_tail
    return float(max(0.0, numerator / max(denominator, np.finfo(float).eps)))


def regular_spatial_grid(n_side, x_bounds=X_BOUNDS, y_bounds=Y_BOUNDS):
    x_edges = np.linspace(x_bounds[0], x_bounds[1], int(n_side) + 1)
    y_edges = np.linspace(y_bounds[0], y_bounds[1], int(n_side) + 1)
    x_mid = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_mid = 0.5 * (y_edges[:-1] + y_edges[1:])
    x_grid, y_grid = np.meshgrid(x_mid, y_mid)
    points = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    cell_area = (x_edges[1] - x_edges[0]) * (y_edges[1] - y_edges[0])
    return points, np.full(points.shape[0], cell_area)


def regular_spacetime_grid(
    n_space,
    n_time,
    time_bounds,
    x_bounds=X_BOUNDS,
    y_bounds=Y_BOUNDS,
):
    spatial, spatial_weights = regular_spatial_grid(n_space, x_bounds, y_bounds)
    time_edges = np.linspace(float(time_bounds[0]), float(time_bounds[1]), int(n_time) + 1)
    times = 0.5 * (time_edges[:-1] + time_edges[1:])
    time_step = time_edges[1] - time_edges[0]
    return (
        np.repeat(times, len(spatial)),
        np.tile(spatial, (len(times), 1)),
        np.tile(spatial_weights * time_step, len(times)),
    )


def relative_l2_and_mae(estimate, truth):
    estimate = np.asarray(estimate, dtype=float).reshape(-1)
    truth = np.asarray(truth, dtype=float).reshape(-1)
    if estimate.shape != truth.shape or not estimate.size:
        raise ValueError("estimate and truth must be aligned non-empty vectors.")
    denominator = np.sum(truth**2)
    rel_l2 = np.sqrt(np.sum((estimate - truth) ** 2) / max(denominator, 1e-15))
    return float(rel_l2), float(np.mean(np.abs(estimate - truth)))


def mcmc_diagnostics(chains):
    """ArviZ rank R-hat, bulk/tail ESS and mean MCSE, one value per coordinate."""
    import arviz as az

    chains = np.asarray(chains, dtype=float)
    if chains.ndim != 3 or min(chains.shape) < 1:
        raise ValueError("Expected non-empty (chain, draw, parameter) arrays.")
    result = {name: np.full(chains.shape[2], np.nan) for name in ("rhat", "ess_bulk", "ess_tail", "mcse_mean")}
    for k in range(chains.shape[2]):
        values = chains[:, :, k]
        if not np.isfinite(values).all() or np.any(np.ptp(values, axis=1) == 0):
            result["rhat"][k] = np.inf
            result["ess_bulk"][k] = result["ess_tail"][k] = 0.
        elif values.shape[1] >= 4:
            if len(values) >= 2:
                result["rhat"][k] = float(az.rhat(values, method="rank"))
            result["ess_bulk"][k] = float(az.ess(values, method="bulk"))
            result["ess_tail"][k] = float(az.ess(values, method="tail"))
            result["mcse_mean"][k] = float(az.mcse(values, method="mean"))
    return result


def split_rhat(chains):
    """Maximum rank-normalized split R-hat, including the folded diagnostic."""
    chains = np.asarray(chains, dtype=float)
    if chains.ndim != 3 or chains.shape[0] < 2 or chains.shape[1] < 4:
        return float("nan")
    return float(np.max(mcmc_diagnostics(chains)["rhat"]))


def effective_sample_size(chains):
    """Minimum bulk ESS, accounting for both within- and between-chain variation."""
    chains = np.asarray(chains, dtype=float)
    if chains.ndim != 3 or chains.shape[1] < 3:
        return float("nan")
    return float(np.min(mcmc_diagnostics(chains)["ess_bulk"]))


@dataclass
class FitBundle:
    method: str
    model: SPINHModel
    fits: tuple
    runtime_seconds: float
    diagnostics: dict
    burn_in: float = 0.5


def _gibbs_parameter_chains(fits, burn_in=0.5):
    chains = []
    for fit in fits:
        theta = np.asarray(
            getattr(fit, "etas_diagnostic_chain", fit.etas_chain), dtype=float
        )
        beta = np.asarray(
            getattr(fit, "beta_diagnostic_chain", fit.beta_chain), dtype=float
        ).reshape(-1, 1)
        burn = int(theta.shape[0] * burn_in)
        chains.append(np.column_stack([theta[burn:], beta[burn:]]))
    minimum = min(chain.shape[0] for chain in chains)
    return np.stack([chain[-minimum:] for chain in chains], axis=0)


def gibbs_parameter_traces(bundle):
    """Return complete unthinned scalar traces with shape (chain, draw, parameter)."""
    if METHODS[bundle.method]["family"] != "gibbs":
        raise TypeError("Complete Gibbs traces are unavailable for VI fits.")
    chains = []
    for fit in bundle.fits:
        theta = np.asarray(fit.etas_diagnostic_chain, dtype=float)
        beta = np.asarray(fit.beta_diagnostic_chain, dtype=float).reshape(-1, 1)
        if theta.shape[0] != beta.shape[0]:
            raise ValueError("ETAS and beta diagnostic traces must be aligned.")
        chains.append(np.column_stack([theta, beta]))
    minimum = min(chain.shape[0] for chain in chains)
    return np.stack([chain[-minimum:] for chain in chains], axis=0)


def epsilon_precision(model, counts):
    """Initial epsilon curvature, matching the all-background Gibbs start."""
    counts = np.asarray(counts, dtype=float).reshape(-1)
    if counts.size != model.n_domains or np.any(counts < 0) or not np.all(np.isfinite(counts)):
        raise ValueError("counts must contain one finite non-negative count per zone.")
    covariance = model.epsilon_prior_covariance() + model.jitter * np.eye(model.n_domains)
    return np.linalg.solve(covariance, np.eye(model.n_domains)) + np.diag(2 * counts)


def proposal_steps(model, catalog):
    """Catalogue-specific steps chosen before inference, with no truth leakage."""
    if len(catalog) == 0:
        raise ValueError("Proposal initialization requires observed events.")
    indices = model.validate_catalog(catalog)
    counts = np.bincount(indices, minlength=model.n_domains)
    curvature = float(np.linalg.eigvalsh(epsilon_precision(model, counts))[-1])
    return {
        "mala_step": MALA_CURVATURE_SCALE / np.sqrt(curvature),
        "sigma_mh_etas": MH_ETAS_REFERENCE_STEP * np.sqrt(MH_ETAS_REFERENCE_EVENTS / len(catalog)),
        "sigma_mh_beta": MH_BETA_SCALE / np.sqrt(len(catalog)),
    }


def initial_gamma_factors_for_start(start, seed):
    """Keep the reference start, then perturb means without changing rate bounds."""
    factors = dict(INITIAL_GAMMA_FACTORS)
    if start == 0:
        return factors
    rng = np.random.default_rng(seed)
    for name, (shape, rate) in factors.items():
        if name != "beta":
            factors[name] = (
                shape * float(np.exp(rng.normal(0, VI_INITIAL_LOG_SD))),
                rate,
            )
    return factors


def _fit_spinh_vi(model, catalog, method, settings, campaign, seed, cutoff, started):
    """Run the MF-VI starts used by M4 and M5 and retain the best ELBO."""
    candidates = []
    for start in range(campaign.vi_starts):
        start_seed = int(seed + 1009 * start)
        config = SPINHVIConfig(
            n_iter=campaign.vi_iterations,
            tolerance=1e-5,
            verbose=False,
            random_seed=start_seed,
            gp_backend=settings["gp_backend"],
            use_calibration=False,
            quadrature_nx=campaign.quadrature_space_grid,
            quadrature_ny=campaign.quadrature_space_grid,
            eps_newton_steps=8,
            spatial_compensator_grid=ETAS_SPATIAL_QUADRATURE,
            etas_update_start=min(5, max(0, campaign.vi_iterations - 1)),
            etas_update_every=5,
            max_optimizer_iter=10,
            gamma_quadrature_nodes=4,
            theta_priors=THETA_PRIORS,
            initial_gamma_factors=initial_gamma_factors_for_start(start, start_seed),
            parent_time_window=cutoff,
        )
        fit = model.vi(catalog, config=config)
        final_elbo = float(fit.elbo_trace[-1]) if fit.elbo_trace else -np.inf
        candidates.append((final_elbo, fit))
    runtime = time.perf_counter() - started
    best_start, (final_elbo, fit) = max(
        enumerate(candidates),
        key=lambda item: item[1][0] if np.isfinite(item[1][0]) else -np.inf,
    )
    diagnostics = {
        "status": "ok",
        "runtime_seconds": float(runtime),
        "rhat_max": float("nan"),
        "ess_min": float("nan"),
        "n_iter_run": int(fit.diagnostics["n_iter_run"]),
        "final_elbo": float(final_elbo),
        "vi_starts_run": campaign.vi_starts,
        "vi_best_start": int(best_start),
        "vi_start_elbos": [float(elbo) for elbo, _ in candidates],
        "converged": bool(fit.diagnostics["converged"]),
        "etas_variational_family": "mean_field_gamma",
        "diagnostic_status": (
            "ok" if fit.diagnostics["converged"] else "not_converged"
        ),
    }
    truncation = fit.diagnostics.get("branching_truncation")
    if truncation:
        diagnostics.update(truncation)
    return FitBundle(
        method, model, (fit,), runtime, diagnostics, campaign.gibbs_burn_in
    ), diagnostics


def fit_spinh_method(
    model,
    catalog,
    method,
    campaign,
    seed,
    *,
    parent_time_window,
    mala_step=None,
):
    """Fit one of M1--M5 and return a uniform result bundle."""
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}.")
    settings = METHODS[method]
    if settings["gp_backend"] == "exact" and len(catalog) > campaign.exact_max_events:
        return None, {"status": "skipped_exact_size"}
    if not settings["truncated"] and len(catalog) > campaign.dense_max_events:
        return None, {"status": "skipped_dense_size"}
    cutoff = float(parent_time_window) if settings["truncated"] else None
    started = time.perf_counter()
    if settings["family"] == "vi":
        return _fit_spinh_vi(
            model, catalog, method, settings, campaign, seed, cutoff, started
        )
    steps = proposal_steps(model, catalog)
    if mala_step is not None:
        steps["mala_step"] = float(mala_step)
    fits = []
    for chain in range(campaign.n_chains):
        sparse_gp = None
        if settings["gp_backend"] == "sparse":
            sparse_gp = SparseGP.from_bounds(
                model.x_bounds,
                model.y_bounds,
                model.gp_prior.variance,
                model.gp_prior.length_scale,
            )
        adaptation_start = min(200, max(0, campaign.gibbs_iterations // 4))
        adaptation_end = min(campaign.gibbs_iterations, max(
            adaptation_start + 1,
            int(campaign.gibbs_iterations * campaign.gibbs_adaptation_fraction),
        ))
        config = SPINHGibbsConfig(
            n_iter=campaign.gibbs_iterations,
            thin=campaign.gibbs_thin,
            **steps,
            verbose=False,
            use_calibration=False,
            beta_init=INITIAL_BETA,
            theta_priors=THETA_PRIORS,
            adaptation_start=adaptation_start,
            etas_adaptation_end=adaptation_end,
            proposal_jitter=1e-6,
            spatial_compensator_grid=ETAS_SPATIAL_QUADRATURE,
            parent_time_window=cutoff,
        )
        fits.append(
            model.gibbs(
                catalog,
                config=config,
                gp_backend=settings["gp_backend"],
                sparse_gp=sparse_gp,
                rng_seed=int(seed + 1009 * chain),
            )
        )
    runtime = time.perf_counter() - started
    parameter_chains = _gibbs_parameter_chains(
        fits, burn_in=campaign.gibbs_burn_in
    )
    parameter_diagnostics = mcmc_diagnostics(parameter_chains)
    ess = float(np.min(parameter_diagnostics["ess_bulk"]))
    background_chains = np.stack([
        np.column_stack([
            fit.eps_diagnostic_chain,
            fit.raw.get(
                "background_fraction_trace",
                (fit.branching_chain == 0).mean(axis=1),
            ),
        ])[int(fit.eps_diagnostic_chain.shape[0] * campaign.gibbs_burn_in):]
        for fit in fits
    ])
    background_diagnostics = mcmc_diagnostics(background_chains)
    diagnostics = {
        "status": "ok",
        "runtime_seconds": float(runtime),
        "rhat_max": float(np.max(parameter_diagnostics["rhat"])),
        "ess_min": ess,
        "ess_tail_min": float(np.min(parameter_diagnostics["ess_tail"])),
        "rhat_background_max": float(np.max(background_diagnostics["rhat"])),
        "ess_background_min": float(np.min(background_diagnostics["ess_bulk"])),
        "ess_background_tail_min": float(np.min(background_diagnostics["ess_tail"])),
        "mcmc_diagnostic_method": "arviz_rank_bulk_tail",
        "n_iter_run": campaign.gibbs_iterations,
        "burn_in_fraction": campaign.gibbs_burn_in,
        **steps,
    }
    if len(fits) < 2:
        diagnostics["diagnostic_status"] = "insufficient_chains"
    else:
        rhats = [diagnostics["rhat_max"], diagnostics["rhat_background_max"]]
        effective_sizes = [ess, diagnostics["ess_tail_min"], diagnostics["ess_background_min"], diagnostics["ess_background_tail_min"]]
        diagnostics["diagnostic_status"] = (
            "ok" if np.isfinite(rhats).all() and max(rhats) <= 1.01
            and np.isfinite(effective_sizes).all() and min(effective_sizes) >= 100 * len(fits)
            else "check_mixing"
        )
    for k, name in enumerate(PARAMETER_NAMES):
        for statistic, values in parameter_diagnostics.items():
            diagnostics[f"{statistic}_{name}"] = float(values[k])
    for block in fits[0].raw["acceptance_history"]:
        histories = [fit.raw["acceptance_history"][block] for fit in fits]
        for phase, start, stop in (
            ("initial", 0, config.adaptation_start + 1),
            (
                "retained",
                int(campaign.gibbs_iterations * campaign.gibbs_burn_in),
                campaign.gibbs_iterations,
            ),
        ):
            diagnostics[f"acceptance_{block}_{phase}"] = float(np.mean(
                np.concatenate([history[start:stop] for history in histories])
            ))
    if campaign.gibbs_iterations >= 200:
        problematic = []
        for block in fits[0].raw["acceptance_history"]:
            rate = diagnostics[f"acceptance_{block}_retained"]
            if rate < 0.05 or (block == "eps" and rate > 0.98):
                problematic.append(f"{block}={rate:.1%}")
        if problematic:
            message = f"{method.upper()} proposal acceptance requires inspection: " + ", ".join(problematic)
            diagnostics["proposal_warning"] = message
            warnings.warn(message, RuntimeWarning, stacklevel=2)
    truncation = fits[0].raw.get("branching_truncation")
    if truncation:
        diagnostics.update(truncation)
    for block, adaptation in fits[0].raw.get("etas_adaptation", {}).items():
        diagnostics[f"proposal_scale_{block}"] = adaptation["scale"]
    return FitBundle(
        method, model, tuple(fits), runtime, diagnostics, campaign.gibbs_burn_in
    ), diagnostics


def _vi_parameter_draws(fit, n_draws, seed):
    return fit.posterior_parameter_samples(n_samples=n_draws, rng_seed=seed)


def posterior_parameter_draws(bundle, n_draws, seed=0, burn_in=None):
    n_draws = int(n_draws)
    if METHODS[bundle.method]["family"] == "vi":
        return _vi_parameter_draws(bundle.fits[0], n_draws, seed)
    burn_in = getattr(bundle, "burn_in", 0.5) if burn_in is None else burn_in
    chains = _gibbs_parameter_chains(bundle.fits, burn_in=burn_in)
    flattened = chains.reshape(-1, chains.shape[-1])
    n_draws = min(n_draws, len(flattened))
    positions = np.linspace(0, len(flattened) - 1, n_draws).round().astype(int)
    selected = flattened[positions]
    return {name: selected[:, index] for index, name in enumerate(PARAMETER_NAMES)}


def posterior_parameter_means(bundle, burn_in=None):
    """Avoid subsampling noise when a parameter's posterior mean is available."""
    if METHODS[bundle.method]["family"] == "vi":
        fit = bundle.fits[0]
        return {**fit.etas_mean().as_dict(), "beta": float(fit.state.etas.beta_mean)}
    burn_in = getattr(bundle, "burn_in", 0.5) if burn_in is None else burn_in
    means = _gibbs_parameter_chains(bundle.fits, burn_in=burn_in).mean(axis=(0, 1))
    return {name: float(means[index]) for index, name in enumerate(PARAMETER_NAMES)}


def posterior_background_draws(bundle, xy, n_draws, seed=0, burn_in=None):
    xy = np.asarray(xy, dtype=float)
    if METHODS[bundle.method]["family"] == "vi":
        return bundle.fits[0].background_intensity_samples(
            xy[:, 0], xy[:, 1], n_samples=n_draws, rng_seed=seed
        )
    burn_in = getattr(bundle, "burn_in", 0.5) if burn_in is None else burn_in
    per_chain = int(math.ceil(n_draws / len(bundle.fits)))
    samples = [
        fit.background_intensity_samples(
            xy[:, 0], xy[:, 1], burn_in=burn_in, n_samples=per_chain
        )
        for fit in bundle.fits
    ]
    return np.concatenate(samples, axis=1)[:, :n_draws]


def triggering_on_grid(model, time_nodes, spatial_xy, history, parameters):
    """Evaluate the exact separable ETAS kernel without a (time*space)-by-N array."""
    time_nodes = np.asarray(time_nodes, dtype=float).reshape(-1)
    spatial_xy = np.asarray(spatial_xy, dtype=float)
    if spatial_xy.ndim != 2 or spatial_xy.shape[1] != 2:
        raise ValueError("spatial_xy must have shape (n_points, 2).")
    if not np.all(np.isfinite(time_nodes)) or not np.all(np.isfinite(spatial_xy)):
        raise ValueError("Evaluation times and coordinates must be finite.")
    if len(history) and parameters.marked and history.magnitudes is None:
        raise ValueError("Marked triggering intensity requires history magnitudes.")
    result = np.zeros((time_nodes.size, len(spatial_xy)))
    kernel = model.etas_kernel
    for start in range(0, len(history), 2048):
        stop = min(start + 2048, len(history))
        magnitudes = (
            history.magnitudes[start:stop] if history.magnitudes is not None
            else np.full(stop - start, model.magnitude_min)
        )
        dt = time_nodes[:, None] - history.t[None, start:stop]
        r2 = (spatial_xy[:, 0, None] - history.x[None, start:stop]) ** 2
        r2 += (spatial_xy[:, 1, None] - history.y[None, start:stop]) ** 2
        temporal = kernel.temporal.evaluate(dt, parameters)
        spatial = kernel.spatial.evaluate(r2, magnitudes[None, :], parameters, model.magnitude_min)
        productivity = kernel.productivity.evaluate(magnitudes, parameters, model.magnitude_min)
        result += temporal @ (spatial * productivity).T
    return result.reshape(-1)


def parameter_recovery_metrics(parameter_draws, true_etas, true_beta):
    truths = {**true_etas.as_dict(), "beta": float(true_beta)}
    log_error_shifts = {"p": 1.0, "q": 1.0}
    log_errors = []
    metrics = {}
    for name in PARAMETER_NAMES:
        draws = np.asarray(parameter_draws[name], dtype=float)
        estimate = float(np.mean(draws))
        truth = truths[name]
        shift = log_error_shifts.get(name, 0.0)
        transformed_estimate = estimate - shift
        transformed_truth = truth - shift
        if (
            not np.isfinite(transformed_estimate)
            or not np.isfinite(transformed_truth)
            or transformed_estimate <= 0.0
            or transformed_truth <= 0.0
        ):
            raise ValueError(
                f"The transformed values used for log_error_{name} must be positive."
            )
        log_error = abs(np.log(transformed_estimate / transformed_truth))
        metrics.update(
            {
                f"true_{name}": truth,
                f"estimate_{name}": estimate,
                f"log_error_{name}": float(log_error),
            }
        )
        log_errors.append(log_error)
    metrics["parameter_log_error"] = float(np.mean(log_errors))
    return metrics


def _binary_f1(truth, predicted):
    truth = np.asarray(truth, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    true_positive = np.sum(truth & predicted)
    denominator = 2 * true_positive + np.sum(~truth & predicted) + np.sum(truth & ~predicted)
    return float(2 * true_positive / denominator) if denominator else 1.0


def branching_metrics(bundle, true_parent_indices, event_times, cutoff):
    true_parent_indices = np.asarray(true_parent_indices, dtype=int)
    true_labels = np.where(true_parent_indices < 0, 0, true_parent_indices + 1)
    true_background = true_parent_indices < 0
    if METHODS[bundle.method]["family"] == "gibbs":
        chains = []
        for fit in bundle.fits:
            values = np.asarray(fit.branching_chain, dtype=int)
            chains.append(values[int(0.5 * len(values)) :])
        labels = np.concatenate(chains, axis=0)
        p_background = np.mean(labels == 0, axis=0)
        true_probability = np.mean(labels == true_labels[None, :], axis=0)
    else:
        probabilities = bundle.fits[0].state.branching.probabilities
        p_background = bundle.fits[0].state.branching.p_background
        if hasattr(probabilities, "tocsr"):
            probabilities = probabilities.tocsr()
            true_probability = np.array(
                [probabilities[event, label] for event, label in enumerate(true_labels)],
                dtype=float,
            )
        else:
            true_probability = probabilities[np.arange(len(true_labels)), true_labels]
    predicted_background = p_background >= 0.5
    triggered = ~true_background
    if METHODS[bundle.method]["truncated"]:
        retained = np.ones(len(true_labels), dtype=bool)
        child = np.flatnonzero(triggered)
        retained[child] = (
            np.asarray(event_times)[child] - np.asarray(event_times)[true_parent_indices[child]]
            <= float(cutoff)
        )
        candidate_recall = float(np.mean(retained[triggered])) if np.any(triggered) else float("nan")
    else:
        candidate_recall = 1.0
    return {
        "background_brier": float(np.mean((p_background - true_background) ** 2)),
        "background_accuracy": float(np.mean(predicted_background == true_background)),
        "background_f1": _binary_f1(true_background, predicted_background),
        "mean_true_state_probability": float(np.mean(true_probability)),
        "candidate_recall": candidate_recall,
        "estimated_background_fraction": float(np.mean(p_background)),
    }


def candidate_diagnostics(event_times, parent_indices, cutoff):
    times = np.asarray(event_times, dtype=float)
    graph = TemporalCandidateGraph.from_times(times, cutoff)
    parent_indices = np.asarray(parent_indices, dtype=int)
    triggered = parent_indices >= 0
    child = np.flatnonzero(triggered)
    recall = float(
        np.mean(times[child] - times[parent_indices[child]] <= cutoff)
    ) if child.size else float("nan")
    counts = np.diff(graph.indptr)
    return {
        **graph.diagnostics(),
        "mean_candidate_count": float(np.mean(counts)) if counts.size else 0.0,
        "candidate_count_q95": float(np.quantile(counts, 0.95)) if counts.size else 0.0,
        "true_parent_candidate_recall": recall,
    }


def intensity_recovery_metrics(
    bundle,
    simulation,
    true_mus,
    field_scale,
    true_etas,
    campaign,
    seed,
    *,
    return_payload=False,
):
    spatial_xy, _ = regular_spatial_grid(campaign.evaluation_space_grid)
    time_nodes = (np.arange(campaign.evaluation_time_grid) + 0.5) * (
        simulation.background_simulation.duration / campaign.evaluation_time_grid
    )
    background_draws = posterior_background_draws(
        bundle, spatial_xy, campaign.posterior_draws, seed=seed
    )
    parameter_draws = posterior_parameter_draws(
        bundle, campaign.posterior_draws, seed=seed + 17
    )
    triggering_estimate = np.zeros(len(time_nodes) * len(spatial_xy))
    for draw in range(campaign.posterior_draws):
        parameters = ETASParameters(
            **{name: parameter_draws[name][draw] for name in PARAMETER_NAMES[:-1]}
        )
        triggering_estimate += triggering_on_grid(
            bundle.model, time_nodes, spatial_xy, simulation.catalog, parameters,
        )
    triggering_estimate /= campaign.posterior_draws
    background_partition = simulation.background_simulation.domains
    spatial_domains = background_partition.locate(spatial_xy[:, 0], spatial_xy[:, 1])
    true_eps = np.log(np.asarray(true_mus, dtype=float))
    background_true = np.exp(true_eps[spatial_domains]) / (
        1.0 + np.exp(-latent_field(spatial_xy[:, 0], spatial_xy[:, 1], field_scale))
    )
    background_true_st = np.tile(background_true, len(time_nodes))
    triggering_true = triggering_on_grid(
        bundle.model, time_nodes, spatial_xy, simulation.catalog, true_etas,
    )
    background_estimate = background_draws.mean(axis=1)
    total_estimate = np.tile(background_estimate, len(time_nodes)) + triggering_estimate
    total_true = background_true_st + triggering_true
    metrics = {}
    for name, estimate, truth in (
        ("background", background_estimate, background_true),
        ("triggering", triggering_estimate, triggering_true),
        ("total", total_estimate, total_true),
    ):
        rel_l2, mae = relative_l2_and_mae(estimate, truth)
        metrics[f"rel_l2_{name}"] = rel_l2
        metrics[f"mae_{name}"] = mae
        metrics[f"truth_rms_{name}"] = float(np.sqrt(np.mean(truth ** 2)))
        metrics[f"rmse_{name}"] = float(np.sqrt(np.mean((estimate - truth) ** 2)))
    metrics["triggering_mean_ratio"] = float(np.mean(triggering_estimate) / max(np.mean(triggering_true), 1e-15))
    if not return_payload:
        return metrics
    payload = {
        "space_grid_size": int(campaign.evaluation_space_grid),
        "time_grid_size": int(campaign.evaluation_time_grid),
        "spatial_xy": spatial_xy,
        "surfaces_time_averaged": True,
        "background_true": background_true,
        "background_estimate": background_estimate,
        "triggering_true": triggering_true.reshape(len(time_nodes), -1).mean(axis=0),
        "triggering_estimate": triggering_estimate.reshape(len(time_nodes), -1).mean(axis=0),
        "total_true": total_true.reshape(len(time_nodes), -1).mean(axis=0),
        "total_estimate": total_estimate.reshape(len(time_nodes), -1).mean(axis=0),
    }
    return metrics, payload


def calibrate_gp(model, training_catalog, campaign, seed):
    if not campaign.use_calibration:
        return model.gp_prior, 0.0, False, 0
    with calibration_slot():
        started = time.perf_counter()
        try:
            prior = model.calibrate_gp_prior(
                training_catalog,
                rng_seed=seed,
                verbose=False,
            )
            return prior, time.perf_counter() - started, True, len(training_catalog)
        except Exception as error:
            warnings.warn(
                f"GP calibration failed; using the configured prior: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            return (
                model.gp_prior,
                time.perf_counter() - started,
                False,
                len(training_catalog),
            )


def software_metadata():
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        revision = "unknown"
    return {
        "git_revision": revision,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def write_campaign(path, campaign, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"campaign": asdict(campaign), "software": software_metadata()}
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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
        "scenarios": {
            name: {**settings, "etas": settings["etas"].as_dict()}
            for name, settings in SCENARIOS.items()
        },
        "theta_priors": dict(THETA_PRIORS),
        "initial_etas": INITIAL_ETAS.as_dict(),
        "initial_beta": INITIAL_BETA,
        "vi_initial_log_sd": VI_INITIAL_LOG_SD,
        "x_bounds": X_BOUNDS,
        "y_bounds": Y_BOUNDS,
        "magnitude_bounds": (MAGNITUDE_MIN, MAGNITUDE_MAX),
        "partition_seed": PARTITION_SEED,
        "n_regions": N_REGIONS,
        "truncation_relative_density": TRUNCATION_RELATIVE_DENSITY,
        "truncation_max_tail_mass": TRUNCATION_MAX_TAIL_MASS,
        "etas_spatial_quadrature": ETAS_SPATIAL_QUADRATURE,
    }


def _serializable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple, dict, np.ndarray)):
        return json.dumps(value, default=lambda item: np.asarray(item).tolist())
    return value


def write_records(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = list(records)
    if not records:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames = []
    for record in records:
        for name in record:
            if name not in fieldnames:
                fieldnames.append(name)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({name: _serializable(record.get(name, "")) for name in fieldnames})
    return path


def mean_confidence_interval(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(values))
    if values.size == 1:
        return mean, float("nan"), float("nan")
    half_width = 1.96 * float(np.std(values, ddof=1)) / np.sqrt(values.size)
    return mean, mean - half_width, mean + half_width


def summarize_records(records, group_fields, metrics):
    summaries = []
    keys = sorted({tuple(record[field] for field in group_fields) for record in records})
    for key in keys:
        group = [
            record
            for record in records
            if tuple(record[field] for field in group_fields) == key
            and record.get("status") == "ok"
        ]
        if not group:
            continue
        row = dict(zip(group_fields, key))
        row["n_completed"] = len(group)
        for metric in metrics:
            values = np.asarray(
                [record.get(metric, np.nan) for record in group], dtype=float
            )
            values = values[np.isfinite(values)]
            row[metric] = float(np.mean(values)) if values.size else float("nan")
        summaries.append(row)
    return summaries
