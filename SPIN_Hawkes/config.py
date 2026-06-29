"""Configuration and parameter value objects."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GPParameters:
    variance: float = 0.3
    length_scale: float = 0.5

    def __post_init__(self):
        if self.variance <= 0 or self.length_scale <= 0:
            raise ValueError("GP variance and length scale must be positive.")


@dataclass(frozen=True)
class ETASParameters:
    A: float = 0.5
    c: float = 0.01
    p: float = 1.5
    d: float = 0.01
    q: float = 2.0
    alpha: float | None = None
    gamma: float | None = None

    def __post_init__(self):
        if self.A < 0 or self.c <= 0 or self.p <= 1 or self.d <= 0 or self.q <= 1:
            raise ValueError("ETAS requires A>=0, c>0, p>1, d>0, and q>1.")
        if self.alpha is not None and self.alpha < 0:
            raise ValueError("alpha must be non-negative.")
        if self.gamma is not None and self.gamma < 0:
            raise ValueError("gamma must be non-negative.")

    @property
    def marked(self) -> bool:
        return self.alpha is not None or self.gamma is not None

    def as_dict(self) -> dict[str, float]:
        values = {"A": self.A, "c": self.c, "p": self.p, "d": self.d, "q": self.q}
        if self.alpha is not None:
            values["alpha"] = self.alpha
        if self.gamma is not None:
            values["gamma"] = self.gamma
        return values


@dataclass(frozen=True)
class MCMCConfig:
    n_iter: int = 3000
    thin: int = 1
    mala_step: float = 0.25
    learn_nu: bool = False
    calibration_method: str = "openturns"
    use_calibration: bool = True
    verbose: bool = True
    verbose_every: int = 100
    t0_nu: int = 50
    step_nu_init: float = 0.1
    compute_emu: bool = False
    emu_every: int = 10
    grid_nx: int = 30
    grid_ny: int = 30

    def __post_init__(self):
        if self.n_iter <= 0 or self.thin <= 0:
            raise ValueError("n_iter and thin must be positive.")
        if self.mala_step <= 0 or self.verbose_every <= 0:
            raise ValueError("mala_step and verbose_every must be positive.")
        if self.calibration_method not in {"sklearn", "openturns"}:
            raise ValueError("calibration_method must be 'sklearn' or 'openturns'.")


@dataclass(frozen=True)
class ETASInferenceConfig:
    learn_beta: bool = False
    beta_init: float = 2.3
    beta_prior: dict[str, float] = field(
        default_factory=lambda: {"a_beta": 2.0, "b_beta": 1.0}
    )
    theta_priors: dict[str, float] = field(default_factory=dict)
    sigma_mh_etas: float = 0.1
    sigma_mh_beta: float = 0.1
    adaptation_start: int = 50
    proposal_jitter: float = 1e-6

