"""Configuration and parameter value objects."""

from dataclasses import dataclass, field


ETAS_PARAMETER_NAMES = ("A", "alpha", "c", "p", "d", "q", "gamma")


@dataclass(frozen=True)
class GPParameters:
    variance: float = 1.0
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
class GibbsConfig:
    """Settings shared by the SSGC and SPIN-H Gibbs samplers."""

    n_iter: int = 3000
    thin: int = 1
    mala_step: float = 0.25
    learn_nu: bool = False
    calibration_method: str = "openturns"
    calibration_target: str = "homogeneous"
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
        if self.calibration_target not in {"homogeneous", "zone_corrected"}:
            raise ValueError("calibration_target must be 'homogeneous' or 'zone_corrected'.")


@dataclass(frozen=True)
class SPINHGibbsConfig(GibbsConfig):
    """Configuration of a SPIN-H Gibbs run, including ETAS updates."""

    beta_init: float = 2.3
    fixed_beta: float | None = None
    beta_prior: dict[str, float] = field(
        default_factory=lambda: {"a_beta": 2.0, "b_beta": 1.0}
    )
    theta_priors: dict[str, float] = field(default_factory=dict)
    fixed_etas: dict[str, float] = field(default_factory=dict)
    sample_z: bool = True
    known_z: object = None
    sigma_mh_etas: float = 0.1
    sigma_mh_beta: float = 0.1
    adaptation_start: int = 50
    proposal_jitter: float = 1e-6

    def __post_init__(self):
        super().__post_init__()
        if self.beta_init <= 0:
            raise ValueError("beta_init must be positive.")
        if self.fixed_beta is not None and self.fixed_beta <= 0:
            raise ValueError("fixed_beta must be positive.")
        unknown_fixed = set(self.fixed_etas).difference(ETAS_PARAMETER_NAMES)
        if unknown_fixed:
            raise ValueError(f"Unknown fixed ETAS parameters: {sorted(unknown_fixed)}")
        for name, value in self.fixed_etas.items():
            if not isinstance(value, (int, float)):
                raise ValueError("fixed_etas values must be numeric.")
        if not isinstance(self.sample_z, bool):
            raise ValueError("sample_z must be a boolean.")
        if self.sigma_mh_etas <= 0 or self.sigma_mh_beta <= 0:
            raise ValueError("Metropolis proposal scales must be positive.")
        if self.adaptation_start < 0:
            raise ValueError("adaptation_start must be non-negative.")
        if self.proposal_jitter <= 0:
            raise ValueError("proposal_jitter must be positive.")


@dataclass(frozen=True)
class SPINHVIConfig:
    """Configuration for the SPIN-H variational inference routine."""

    n_iter: int = 200
    tolerance: float = 1e-4
    verbose: bool = True
    verbose_every: int = 10
    random_seed: int | None = None

    update_z: bool = True
    update_polya_gamma: bool = True
    update_latent_poisson: bool = True
    update_gp: bool = True
    update_eps: bool = True
    update_etas: bool = True

    fixed_etas: dict[str, float] = field(default_factory=dict)
    fixed_beta: float | None = None
    beta_init: float = 2.3
    beta_prior: dict[str, float] = field(
        default_factory=lambda: {"a_beta": 2.0, "b_beta": 1.0}
    )
    theta_priors: dict[str, float] = field(default_factory=dict)

    quadrature_nx: int = 30
    quadrature_ny: int = 30
    eps_newton_steps: int = 8
    eps_damping: float = 1.0
    eps_bounds: tuple[float, float] = (-20.0, 8.0)
    f_bounds: tuple[float, float] = (-15.0, 15.0)
    latent_poisson_damping: float = 0.5
    latent_poisson_max_multiplier: float | None = 1.5
    etas_update_start: int = 10
    etas_update_every: int = 20
    etas_initial_relative_variance: float = 0.05
    parameter_damping: float = 0.6
    max_optimizer_iter: int = 8
    etas_quadrature_nodes: int = 8
    full_gp_max_events: int = 800
    gp_backend: str = "exact"
    sparse_gp: object | None = None
    spatial_compensator_grid: int = 0
    jitter: float = 1e-6

    def __post_init__(self):
        if self.n_iter <= 0:
            raise ValueError("n_iter must be positive.")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive.")
        if self.verbose_every <= 0:
            raise ValueError("verbose_every must be positive.")
        if self.quadrature_nx <= 1 or self.quadrature_ny <= 1:
            raise ValueError("quadrature grid sizes must be greater than one.")
        if self.eps_newton_steps <= 0:
            raise ValueError("eps_newton_steps must be positive.")
        if not 0 < self.eps_damping <= 1:
            raise ValueError("eps_damping must be in (0, 1].")
        if self.eps_bounds[0] >= self.eps_bounds[1]:
            raise ValueError("eps_bounds must be increasing.")
        if self.f_bounds[0] >= self.f_bounds[1]:
            raise ValueError("f_bounds must be increasing.")
        if not 0 < self.latent_poisson_damping <= 1:
            raise ValueError("latent_poisson_damping must be in (0, 1].")
        if (
            self.latent_poisson_max_multiplier is not None
            and self.latent_poisson_max_multiplier <= 0
        ):
            raise ValueError("latent_poisson_max_multiplier must be positive or None.")
        if self.etas_update_start < 0:
            raise ValueError("etas_update_start must be non-negative.")
        if self.etas_update_every <= 0:
            raise ValueError("etas_update_every must be positive.")
        if self.etas_initial_relative_variance <= 0:
            raise ValueError("etas_initial_relative_variance must be positive.")
        if not 0 < self.parameter_damping <= 1:
            raise ValueError("parameter_damping must be in (0, 1].")
        if self.max_optimizer_iter <= 0:
            raise ValueError("max_optimizer_iter must be positive.")
        if self.etas_quadrature_nodes <= 1:
            raise ValueError("etas_quadrature_nodes must be greater than one.")
        if self.full_gp_max_events <= 0:
            raise ValueError("full_gp_max_events must be positive.")
        if str(self.gp_backend).lower() not in {"exact", "sparse"}:
            raise ValueError("gp_backend must be 'exact' or 'sparse'.")
        if self.sparse_gp is not None and str(self.gp_backend).lower() != "sparse":
            raise ValueError("sparse_gp requires gp_backend='sparse'.")
        if self.spatial_compensator_grid < 0:
            raise ValueError("spatial_compensator_grid must be non-negative.")
        if self.jitter <= 0:
            raise ValueError("jitter must be positive.")
        if self.beta_init <= 0:
            raise ValueError("beta_init must be positive.")
        if self.fixed_beta is not None and self.fixed_beta <= 0:
            raise ValueError("fixed_beta must be positive.")
        unknown = set(self.fixed_etas).difference(ETAS_PARAMETER_NAMES)
        if unknown:
            raise ValueError(f"Unknown fixed ETAS parameters: {sorted(unknown)}")
        for name, value in self.fixed_etas.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"fixed_etas[{name!r}] must be numeric.")
