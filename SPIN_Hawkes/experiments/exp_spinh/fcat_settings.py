"""Numerical budgets for FCAT-17, independent of the simulated experiments."""

from dataclasses import dataclass, replace
from numbers import Integral, Real

import numpy as np


@dataclass(frozen=True)
class FCATConfig:
    name: str
    n_chains: int = 3
    gibbs_iterations: int = 8000
    gibbs_thin: int = 5
    vi_iterations: int = 500
    vi_tolerance: float = 1e-5
    evaluation_space_grid: int = 25
    quadrature_space_grid: int = 70
    score_posterior_draws: int = 1000
    map_posterior_draws: int = 500
    max_parallel_calibrations: int = 1
    exact_max_events: int = 1200
    use_calibration: bool = True

    def __post_init__(self):
        if self.name not in {"smoke", "full"}:
            raise ValueError("FCAT profile must be 'smoke' or 'full'.")
        for name in (
            "n_chains", "gibbs_iterations", "gibbs_thin", "vi_iterations",
            "evaluation_space_grid", "quadrature_space_grid",
            "score_posterior_draws", "map_posterior_draws",
            "max_parallel_calibrations", "exact_max_events",
        ):
            value = getattr(self, name)
            minimum = 2 if name.endswith("space_grid") else 1
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if (
            isinstance(self.vi_tolerance, bool)
            or not isinstance(self.vi_tolerance, Real)
            or not np.isfinite(self.vi_tolerance)
            or self.vi_tolerance <= 0
        ):
            raise ValueError("vi_tolerance must be finite and positive.")
        if not isinstance(self.use_calibration, bool):
            raise ValueError("use_calibration must be boolean.")


CAMPAIGNS = {
    "smoke": FCATConfig(
        name="smoke", n_chains=1, gibbs_iterations=50, gibbs_thin=2,
        vi_iterations=10, evaluation_space_grid=5, quadrature_space_grid=4,
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
