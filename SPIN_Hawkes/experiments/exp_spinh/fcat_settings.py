"""Numerical budgets for FCAT-17, independent of the simulated experiments."""

from dataclasses import dataclass, replace
from numbers import Integral, Real

import numpy as np


@dataclass(frozen=True)
class FCATConfig:
    name: str
    random_replicates: int = 10
    validation_seed: int = 20261006
    temporal_train_fraction: float = 0.8
    cv_n_chains: int = 1
    full_n_chains: int = 3
    cv_gibbs_iterations: int = 5000
    full_gibbs_iterations: int = 6000
    gibbs_thin: int = 3
    gibbs_burn_in: float = 0.4
    mala_initial_step: float = 0.06
    mala_adaptation_start: int = 50
    mala_target_acceptance: float = 0.574
    mala_adaptation_decay: float = 0.6
    mala_precondition: bool = True
    mala_acceptance_bounds: tuple[float, float] = (0.50, 0.64)
    vi_iterations: int = 500
    vi_tolerance: float = 1e-5
    quadrature_space_grid: int = 100
    score_posterior_draws: int = 1000
    map_posterior_draws: int = 500
    max_parallel_calibrations: int = 1
    exact_max_events: int = 1200
    use_calibration: bool = True

    def __post_init__(self):
        if self.name not in {"smoke", "full"}:
            raise ValueError("FCAT profile must be 'smoke' or 'full'.")
        for name in (
            "random_replicates",
            "cv_n_chains", "full_n_chains", "cv_gibbs_iterations",
            "full_gibbs_iterations", "gibbs_thin",
            "vi_iterations",
            "quadrature_space_grid",
            "score_posterior_draws", "map_posterior_draws",
            "max_parallel_calibrations", "exact_max_events",
        ):
            value = getattr(self, name)
            minimum = 2 if name.endswith("space_grid") else 1
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if (
            isinstance(self.validation_seed, bool)
            or not isinstance(self.validation_seed, Integral)
            or self.validation_seed < 0
        ):
            raise ValueError("validation_seed must be a non-negative integer.")
        if (
            isinstance(self.temporal_train_fraction, bool)
            or not isinstance(self.temporal_train_fraction, Real)
            or not 0.0 < self.temporal_train_fraction < 1.0
        ):
            raise ValueError("temporal_train_fraction must be in (0, 1).")
        if (
            isinstance(self.vi_tolerance, bool)
            or not isinstance(self.vi_tolerance, Real)
            or not np.isfinite(self.vi_tolerance)
            or self.vi_tolerance <= 0
        ):
            raise ValueError("vi_tolerance must be finite and positive.")
        if not 0.0 < self.gibbs_burn_in < 1.0:
            raise ValueError("gibbs_burn_in must be in (0, 1).")
        if (
            isinstance(self.mala_initial_step, bool)
            or not isinstance(self.mala_initial_step, Real)
            or not np.isfinite(self.mala_initial_step)
            or self.mala_initial_step <= 0.0
        ):
            raise ValueError("mala_initial_step must be finite and positive.")
        if (
            isinstance(self.mala_adaptation_start, bool)
            or not isinstance(self.mala_adaptation_start, Integral)
            or self.mala_adaptation_start < 0
        ):
            raise ValueError("mala_adaptation_start must be a non-negative integer.")
        adaptation_ends = (
            int(self.cv_gibbs_iterations * self.gibbs_burn_in),
            int(self.full_gibbs_iterations * self.gibbs_burn_in),
        )
        if self.mala_adaptation_start >= min(adaptation_ends):
            raise ValueError(
                "mala_adaptation_start must precede the end of Gibbs burn-in."
            )
        if (
            isinstance(self.mala_target_acceptance, bool)
            or not isinstance(self.mala_target_acceptance, Real)
            or not np.isfinite(self.mala_target_acceptance)
            or not 0.0 < self.mala_target_acceptance < 1.0
        ):
            raise ValueError("mala_target_acceptance must be in (0, 1).")
        if (
            isinstance(self.mala_adaptation_decay, bool)
            or not isinstance(self.mala_adaptation_decay, Real)
            or not np.isfinite(self.mala_adaptation_decay)
            or not 0.5 < self.mala_adaptation_decay <= 1.0
        ):
            raise ValueError("mala_adaptation_decay must be in (0.5, 1].")
        if (
            len(self.mala_acceptance_bounds) != 2
            or not np.all(np.isfinite(self.mala_acceptance_bounds))
            or not 0.0 < self.mala_acceptance_bounds[0]
            < self.mala_acceptance_bounds[1] < 1.0
        ):
            raise ValueError(
                "mala_acceptance_bounds must contain two increasing values in (0, 1)."
            )
        for name in ("use_calibration", "mala_precondition"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean.")


CAMPAIGNS = {
    "smoke": FCATConfig(
        name="smoke", random_replicates=2, cv_n_chains=1, full_n_chains=1,
        cv_gibbs_iterations=50, full_gibbs_iterations=50, gibbs_thin=2,
        mala_adaptation_start=5,
        vi_iterations=10, quadrature_space_grid=12,
        score_posterior_draws=4, map_posterior_draws=4,
        exact_max_events=150, use_calibration=False,
    ),
    "full": FCATConfig(name="full"),
}


def configure_campaign(profile, **overrides):
    if profile not in CAMPAIGNS:
        raise ValueError(f"Unknown FCAT profile {profile!r}.")
    unknown = set(overrides) - (set(FCATConfig.__dataclass_fields__) - {"name"})
    if unknown:
        raise ValueError(f"Unknown FCAT settings: {sorted(unknown)}.")
    return replace(
        CAMPAIGNS[profile],
        **{name: value for name, value in overrides.items() if value is not None},
    )
