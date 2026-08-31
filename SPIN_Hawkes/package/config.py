"""Configuration and parameter value objects."""

from dataclasses import dataclass, field
from math import isfinite
from numbers import Integral, Real


ETAS_PARAMETER_NAMES = ("A", "alpha", "c", "p", "d", "q", "gamma")
VI_GAMMA_FACTOR_NAMES = (
    "A",
    "alpha",
    "c",
    "p_minus_1",
    "d",
    "q_minus_1",
    "gamma",
    "beta",
)


def _require_real(name, value, *, minimum=None, strict=False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite real number.")
    value = float(value)
    if minimum is not None:
        valid = value > minimum if strict else value >= minimum
        if not valid:
            relation = ">" if strict else ">="
            raise ValueError(f"{name} must be {relation} {minimum}.")
    return value


def _require_integer(name, value, *, minimum=None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer.")
    value = int(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return value


def _require_boolean(name, value) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean.")


def _validate_etas_value(name: str, value) -> float:
    value = _require_real(f"ETAS parameter {name!r}", value)
    if name in {"A", "alpha", "gamma"} and value < 0.0:
        raise ValueError(f"ETAS parameter {name!r} must be non-negative.")
    if name in {"c", "d"} and value <= 0.0:
        raise ValueError(f"ETAS parameter {name!r} must be positive.")
    if name in {"p", "q"} and value <= 1.0:
        raise ValueError(f"ETAS parameter {name!r} must be greater than one.")
    return value


@dataclass(frozen=True)
class GPParameters:
    variance: float = 1.0
    length_scale: float = 0.5

    def __post_init__(self):
        object.__setattr__(
            self,
            "variance",
            _require_real("GP variance", self.variance, minimum=0.0, strict=True),
        )
        object.__setattr__(
            self,
            "length_scale",
            _require_real(
                "GP length_scale",
                self.length_scale,
                minimum=0.0,
                strict=True,
            ),
        )


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
        for name in ("A", "c", "p", "d", "q"):
            object.__setattr__(
                self,
                name,
                _validate_etas_value(name, getattr(self, name)),
            )
        if (self.alpha is None) != (self.gamma is None):
            raise ValueError(
                "alpha and gamma must either both be provided or both be None. "
                "Use zero to disable one magnitude effect in a marked model."
            )
        for name in ("alpha", "gamma"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _validate_etas_value(name, value))

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
    use_calibration: bool = True
    verbose: bool = True
    verbose_every: int = 100
    t0_nu: int = 50
    step_nu_init: float = 0.1
    compute_emu: bool = False
    emu_every: int = 10
    grid_nx: int = 30
    grid_ny: int = 30
    beta_init: float = 2.3
    fixed_beta: float | None = None
    beta_prior: dict[str, float] = field(
        default_factory=lambda: {"a_beta": 2.0, "b_beta": 1.0}
    )
    sigma_mh_beta: float = 0.1
    adaptation_start: int = 50
    proposal_jitter: float = 1e-6

    def __post_init__(self):
        for name in (
            "n_iter",
            "thin",
            "verbose_every",
            "emu_every",
            "grid_nx",
            "grid_ny",
        ):
            _require_integer(name, getattr(self, name), minimum=1)
        for name in ("t0_nu", "adaptation_start"):
            _require_integer(name, getattr(self, name), minimum=0)
        for name in ("learn_nu", "use_calibration", "verbose", "compute_emu"):
            _require_boolean(name, getattr(self, name))
        for name in (
            "mala_step",
            "step_nu_init",
            "beta_init",
            "sigma_mh_beta",
            "proposal_jitter",
        ):
            _require_real(name, getattr(self, name), minimum=0.0, strict=True)
        if self.fixed_beta is not None:
            _require_real("fixed_beta", self.fixed_beta, minimum=0.0, strict=True)
        if not isinstance(self.beta_prior, dict):
            raise TypeError("beta_prior must be a dictionary.")
        allowed_beta_priors = {"a_beta", "b_beta"}
        unknown = set(self.beta_prior).difference(allowed_beta_priors)
        if unknown:
            raise ValueError(f"Unknown beta prior parameters: {sorted(unknown)}")
        for name, value in self.beta_prior.items():
            _require_real(
                f"beta_prior[{name!r}]",
                value,
                minimum=0.0,
                strict=True,
            )


@dataclass(frozen=True)
class SPINHGibbsConfig(GibbsConfig):
    """Configuration of a SPIN-H Gibbs run, including ETAS updates."""

    theta_priors: dict[str, float] = field(default_factory=dict)
    fixed_etas: dict[str, float] = field(default_factory=dict)
    sample_z: bool = True
    known_z: object = None
    sigma_mh_etas: float = 0.1
    parent_time_window: float | None = None

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.fixed_etas, dict):
            raise TypeError("fixed_etas must be a dictionary.")
        unknown_fixed = set(self.fixed_etas).difference(ETAS_PARAMETER_NAMES)
        if unknown_fixed:
            raise ValueError(f"Unknown fixed ETAS parameters: {sorted(unknown_fixed)}")
        for name, value in self.fixed_etas.items():
            _validate_etas_value(name, value)
        _require_boolean("sample_z", self.sample_z)
        _require_real(
            "sigma_mh_etas",
            self.sigma_mh_etas,
            minimum=0.0,
            strict=True,
        )
        if self.parent_time_window is not None:
            _require_real(
                "parent_time_window",
                self.parent_time_window,
                minimum=0.0,
                strict=True,
            )
        if not isinstance(self.theta_priors, dict):
            raise TypeError("theta_priors must be a dictionary.")
        theta_prior_names = {
            f"{prefix}_{name}"
            for name in ETAS_PARAMETER_NAMES
            for prefix in ("a", "b")
        }
        unknown_priors = set(self.theta_priors).difference(theta_prior_names)
        if unknown_priors:
            raise ValueError(f"Unknown ETAS prior parameters: {sorted(unknown_priors)}")
        for name, value in self.theta_priors.items():
            _require_real(
                f"theta_priors[{name!r}]",
                value,
                minimum=0.0,
                strict=True,
            )


@dataclass(frozen=True)
class SSGCVIConfig:
    """Configuration shared by SSGC variational inference runs."""

    n_iter: int = 200
    tolerance: float = 1e-4
    verbose: bool = True
    verbose_every: int = 10
    elbo_every: int = 1
    random_seed: int | None = None

    update_polya_gamma: bool = True
    update_latent_poisson: bool = True
    update_gp: bool = True
    update_eps: bool = True

    fixed_beta: float | None = None
    beta_prior: dict[str, float] = field(
        default_factory=lambda: {"a_beta": 2.0, "b_beta": 1.0}
    )
    initial_gamma_factors: dict[str, tuple[float, float]] = field(
        default_factory=dict
    )

    quadrature_nx: int = 30
    quadrature_ny: int = 30
    eps_newton_steps: int = 8
    max_optimizer_iter: int = 20
    gamma_quadrature_nodes: int = 8
    gp_backend: str = "exact"
    use_calibration: bool = False
    sparse_gp: object | None = None
    jitter: float = 1e-6

    def _allowed_initial_gamma_factors(self) -> set[str]:
        return {"beta"}

    def __post_init__(self):
        for name in (
            "n_iter",
            "verbose_every",
            "elbo_every",
            "eps_newton_steps",
            "max_optimizer_iter",
        ):
            _require_integer(name, getattr(self, name), minimum=1)
        for name in ("quadrature_nx", "quadrature_ny", "gamma_quadrature_nodes"):
            _require_integer(name, getattr(self, name), minimum=2)
        for name in (
            "verbose",
            "update_polya_gamma",
            "update_latent_poisson",
            "update_gp",
            "update_eps",
            "use_calibration",
        ):
            _require_boolean(name, getattr(self, name))
        if self.random_seed is not None:
            _require_integer("random_seed", self.random_seed, minimum=0)
        _require_real("tolerance", self.tolerance, minimum=0.0, strict=True)
        if not isinstance(self.gp_backend, str) or self.gp_backend.lower() not in {
            "exact",
            "sparse",
        }:
            raise ValueError("gp_backend must be 'exact' or 'sparse'.")
        if self.sparse_gp is not None and self.gp_backend.lower() != "sparse":
            raise ValueError("sparse_gp requires gp_backend='sparse'.")
        if self.use_calibration and self.sparse_gp is not None:
            raise ValueError(
                "use_calibration=True cannot be combined with an injected sparse_gp."
            )
        _require_real("jitter", self.jitter, minimum=0.0, strict=True)
        if self.fixed_beta is not None:
            _require_real("fixed_beta", self.fixed_beta, minimum=0.0, strict=True)
        if not isinstance(self.beta_prior, dict):
            raise TypeError("beta_prior must be a dictionary.")
        beta_prior_names = {"a_beta", "b_beta"}
        unknown_beta_priors = set(self.beta_prior).difference(beta_prior_names)
        if unknown_beta_priors:
            raise ValueError(
                f"Unknown beta prior parameters: {sorted(unknown_beta_priors)}"
            )
        for name, value in self.beta_prior.items():
            _require_real(
                f"beta_prior[{name!r}]",
                value,
                minimum=0.0,
                strict=True,
            )
        if not isinstance(self.initial_gamma_factors, dict):
            raise TypeError("initial_gamma_factors must be a dictionary.")
        unknown_initial = set(self.initial_gamma_factors).difference(
            self._allowed_initial_gamma_factors()
        )
        if unknown_initial:
            raise ValueError(
                "Unknown initial Gamma factors: "
                f"{sorted(unknown_initial)}"
            )
        for name, parameters in self.initial_gamma_factors.items():
            if not isinstance(parameters, (tuple, list)) or len(parameters) != 2:
                raise ValueError(
                    f"initial_gamma_factors[{name!r}] must be a (shape, rate) pair."
                )
            shape, rate = parameters
            try:
                _require_real("shape", shape, minimum=0.0, strict=True)
                _require_real("rate", rate, minimum=0.0, strict=True)
            except ValueError as error:
                raise ValueError(
                    f"initial_gamma_factors[{name!r}] must contain positive finite values."
                ) from error
        conflicting = set()
        if self.fixed_beta is not None and "beta" in self.initial_gamma_factors:
            conflicting.add("beta")
        if conflicting:
            raise ValueError(
                "Initial Gamma factors cannot be provided for fixed parameters: "
                f"{sorted(conflicting)}"
            )


@dataclass(frozen=True)
class SPINHVIConfig(SSGCVIConfig):
    """Configuration for SPIN-H variational inference, including ETAS."""

    update_z: bool = True
    update_etas: bool = True
    fixed_etas: dict[str, float] = field(default_factory=dict)
    theta_priors: dict[str, float] = field(default_factory=dict)
    etas_update_start: int = 10
    etas_update_every: int = 20
    # Kept for backward compatibility; None uses gamma_quadrature_nodes.
    etas_quadrature_nodes: int | None = None
    spatial_compensator_grid: int = 0
    parent_time_window: float | None = None

    def _allowed_initial_gamma_factors(self) -> set[str]:
        return set(VI_GAMMA_FACTOR_NAMES)

    def __post_init__(self):
        super().__post_init__()
        _require_boolean("update_z", self.update_z)
        _require_boolean("update_etas", self.update_etas)
        _require_integer("etas_update_start", self.etas_update_start, minimum=0)
        _require_integer("etas_update_every", self.etas_update_every, minimum=1)
        if self.etas_quadrature_nodes is not None:
            _require_integer(
                "etas_quadrature_nodes",
                self.etas_quadrature_nodes,
                minimum=2,
            )
        _require_integer(
            "spatial_compensator_grid",
            self.spatial_compensator_grid,
            minimum=0,
        )
        if self.parent_time_window is not None:
            _require_real(
                "parent_time_window",
                self.parent_time_window,
                minimum=0.0,
                strict=True,
            )

        if not isinstance(self.fixed_etas, dict):
            raise TypeError("fixed_etas must be a dictionary.")
        unknown = set(self.fixed_etas).difference(ETAS_PARAMETER_NAMES)
        if unknown:
            raise ValueError(f"Unknown fixed ETAS parameters: {sorted(unknown)}")
        for name, value in self.fixed_etas.items():
            _validate_etas_value(name, value)

        if not isinstance(self.theta_priors, dict):
            raise TypeError("theta_priors must be a dictionary.")
        theta_prior_names = {
            f"{prefix}_{name}"
            for name in ETAS_PARAMETER_NAMES
            for prefix in ("a", "b")
        }
        unknown_theta_priors = set(self.theta_priors).difference(theta_prior_names)
        if unknown_theta_priors:
            raise ValueError(
                f"Unknown ETAS prior parameters: {sorted(unknown_theta_priors)}"
            )
        for name, value in self.theta_priors.items():
            _require_real(
                f"theta_priors[{name!r}]",
                value,
                minimum=0.0,
                strict=True,
            )

        shifted_factor_names = {"p": "p_minus_1", "q": "q_minus_1"}
        fixed_factor_names = {
            shifted_factor_names.get(name, name) for name in self.fixed_etas
        }
        conflicting = fixed_factor_names.intersection(self.initial_gamma_factors)
        if conflicting:
            raise ValueError(
                "Initial Gamma factors cannot be provided for fixed parameters: "
                f"{sorted(conflicting)}"
            )
