"""Simulation, fitting and evaluation helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
import csv
import json
import math
import platform
import subprocess
import time
import warnings

from shapely.ops import unary_union
import numpy as np

from .runner_utils import calibration_slot
from .simulation_settings import (
    ETAS_PARAMETER_NAMES,
    ETAS_SPATIAL_QUADRATURE,
    INITIAL_BETA,
    INITIAL_ETAS,
    INITIAL_GAMMA_FACTORS,
    MAGNITUDE_MAX,
    MAGNITUDE_MIN,
    MALA_CURVATURE_SCALE,
    METHODS,
    MH_BETA_SCALE,
    MH_ETAS_REFERENCE_EVENTS,
    MH_ETAS_REFERENCE_STEP,
    N_REGIONS,
    PARAMETER_NAMES,
    PARTITION_SEED,
    REPO_ROOT,
    THETA_PRIORS,
    TRUNCATION_RELATIVE_DENSITY,
    VI_ETAS_UPDATE_EVERY,
    VI_ETAS_UPDATE_START,
    VI_GAMMA_QUADRATURE_NODES,
    VI_INITIAL_CONCENTRATION_MULTIPLIER,
    VI_MAX_OPTIMIZER_ITER,
    VI_START_PROFILES,
    X_BOUNDS,
    Y_BOUNDS,
)
from package import (
    GPParameters,
    SPINHGibbsConfig,
    SPINHModel,
    SPINHVIConfig,
    SparseGP,
    TemporalCandidateGraph,
    generate_voronoi_cells,
    simulate_hawkes_process,
)
from spatial import SpatialQuadrature, midpoint_quadrature


_ARVIZ_MODULE = None
_ARVIZ_IMPORT_ATTEMPTED = False


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
    """Return the first lag below a chosen relative temporal-kernel height."""
    relative_density = float(relative_density)
    if not 0.0 < relative_density < 1.0:
        raise ValueError("relative_density must lie in (0, 1).")
    cutoff = float(parameters.c * (relative_density ** (-1.0 / parameters.p) - 1.0))
    if horizon is not None:
        horizon = float(horizon)
        if not np.isfinite(horizon) or horizon <= 0:
            raise ValueError("horizon must be finite and positive.")
        cutoff = min(horizon, cutoff)
    return cutoff


def omitted_temporal_mass(parameters, cutoff, horizon):
    cutoff = min(float(cutoff), float(horizon))
    horizon = float(horizon)
    c, p = parameters.c, parameters.p
    horizon_tail = (c / (c + horizon)) ** (p - 1.0)
    numerator = (c / (c + cutoff)) ** (p - 1.0) - horizon_tail
    denominator = 1.0 - horizon_tail
    return float(max(0.0, numerator / max(denominator, np.finfo(float).eps)))


def regular_spatial_quadrature(n_side, x_bounds=X_BOUNDS, y_bounds=Y_BOUNDS):
    return midpoint_quadrature(x_bounds, y_bounds, int(n_side))


def regular_spatial_grid(n_side, x_bounds=X_BOUNDS, y_bounds=Y_BOUNDS):
    """Compatibility helper returning the nodes and weights of the default rule."""
    rule = regular_spatial_quadrature(n_side, x_bounds, y_bounds)
    return rule.points, rule.weights


def quadrature_metadata(quadrature):
    if quadrature is None:
        return None
    if not isinstance(quadrature, SpatialQuadrature):
        raise TypeError("quadrature must be a SpatialQuadrature instance.")
    return {
        "n_points": int(len(quadrature.weights)),
        "weight_sum": float(np.sum(quadrature.weights)),
        "fingerprint": quadrature.fingerprint(),
    }


def relative_l2_and_mae(estimate, truth, weights=None):
    estimate = np.asarray(estimate, dtype=float).reshape(-1)
    truth = np.asarray(truth, dtype=float).reshape(-1)
    if estimate.shape != truth.shape or not estimate.size:
        raise ValueError("estimate and truth must be aligned non-empty vectors.")
    if weights is None:
        weights = np.ones(estimate.size, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if weights.shape != estimate.shape:
        raise ValueError("weights must be aligned with estimate and truth.")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("weights must be finite and positive.")
    denominator = np.sum(weights * truth**2)
    rel_l2 = np.sqrt(
        np.sum(weights * (estimate - truth) ** 2) / max(denominator, 1e-15)
    )
    mae = np.sum(weights * np.abs(estimate - truth)) / np.sum(weights)
    return float(rel_l2), float(mae)


def _load_arviz():
    """Import optional diagnostics once without invalidating completed fits."""
    global _ARVIZ_IMPORT_ATTEMPTED, _ARVIZ_MODULE
    if _ARVIZ_IMPORT_ATTEMPTED:
        return _ARVIZ_MODULE
    _ARVIZ_IMPORT_ATTEMPTED = True
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=FutureWarning, module=r"arviz(\..*)?$"
            )
            import arviz as az
    except (ImportError, OSError) as error:
        warnings.warn(
            "ArviZ diagnostics are unavailable; fitted posteriors are retained "
            f"without R-hat, ESS or MCSE ({type(error).__name__}: {error}).",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    _ARVIZ_MODULE = az
    return _ARVIZ_MODULE


def mcmc_diagnostics(chains):
    """ArviZ rank R-hat, bulk/tail ESS and mean MCSE, one value per coordinate."""
    chains = np.asarray(chains, dtype=float)
    if chains.ndim != 3 or min(chains.shape) < 1:
        raise ValueError("Expected non-empty (chain, draw, parameter) arrays.")
    result = {name: np.full(chains.shape[2], np.nan) for name in ("rhat", "ess_bulk", "ess_tail", "mcse_mean")}
    az = _load_arviz()
    for k in range(chains.shape[2]):
        values = chains[:, :, k]
        if not np.isfinite(values).all() or np.any(np.ptp(values, axis=1) == 0):
            result["rhat"][k] = np.inf
            result["ess_bulk"][k] = result["ess_tail"][k] = 0.
        elif az is not None and values.shape[1] >= 4:
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


def _gibbs_background_chains(fits, burn_in):
    """Use scalar traces without expanding the stored branching arrays."""
    chains = []
    for fit in fits:
        fraction = fit.raw.get("background_fraction_trace")
        if fraction is None:
            # Older results only have thinned allocations and epsilon samples.
            fraction = (fit.branching_chain == 0).mean(axis=1)
            epsilon = fit.eps_chain
        else:
            epsilon = fit.eps_diagnostic_chain
        values = np.column_stack([epsilon, fraction])
        chains.append(values[int(len(values) * burn_in):])
    minimum = min(len(chain) for chain in chains)
    return np.stack([chain[:minimum] for chain in chains])


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


def vi_start_profile(start):
    """Return one deterministic, truth-independent MF-VI starting profile."""
    if isinstance(start, bool) or not isinstance(start, Integral) or start < 0:
        raise ValueError("start must be a non-negative integer.")
    return dict(VI_START_PROFILES[int(start) % len(VI_START_PROFILES)])


def initial_gamma_factors_for_start(start, seed=None):
    """Concentrate initial ETAS factors and span plausible productivity modes."""
    del seed  # Retained in the public helper signature for backward compatibility.
    profile = vi_start_profile(start)
    factors = dict(INITIAL_GAMMA_FACTORS)
    for name, (shape, rate) in tuple(factors.items()):
        if name != "beta":
            factors[name] = (
                shape * VI_INITIAL_CONCENTRATION_MULTIPLIER,
                rate * VI_INITIAL_CONCENTRATION_MULTIPLIER,
            )
    A_shape, _ = factors["A"]
    A_mean = INITIAL_ETAS.A * profile["A_multiplier"]
    factors["A"] = (A_shape, A_shape / A_mean)
    return factors


def _fit_spinh_vi(
    model,
    catalog,
    method,
    settings,
    campaign,
    seed,
    cutoff,
    started,
    background_quadrature,
    spatial_compensator_quadrature,
):
    """Run the MF-VI starts used by M4 and M5 and retain the best ELBO."""
    elbos = []
    best_fit = None
    best_elbo = -np.inf
    best_start = None
    for start in range(campaign.vi_starts):
        start_seed = int(seed + 1009 * start)
        start_profile = vi_start_profile(start)
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
            etas_update_start=min(
                VI_ETAS_UPDATE_START,
                max(0, campaign.vi_iterations - 1),
            ),
            etas_update_every=VI_ETAS_UPDATE_EVERY,
            max_optimizer_iter=VI_MAX_OPTIMIZER_ITER,
            gamma_quadrature_nodes=VI_GAMMA_QUADRATURE_NODES,
            theta_priors=THETA_PRIORS,
            initial_gamma_factors=initial_gamma_factors_for_start(start, start_seed),
            initial_background_fraction=start_profile["background_fraction"],
            parent_time_window=cutoff,
        )
        fit = model.vi(
            catalog,
            config=config,
            quadrature=background_quadrature,
            spatial_compensator_quadrature=spatial_compensator_quadrature,
        )
        final_elbo = float(fit.elbo_trace[-1]) if len(fit.elbo_trace) else -np.inf
        elbos.append(final_elbo)
        if np.isfinite(final_elbo) and (best_fit is None or final_elbo > best_elbo):
            best_start, best_elbo, best_fit = start, final_elbo, fit
        # Keep only the best fitted state; HSGP and branching arrays can be large.
        del fit
    runtime = time.perf_counter() - started
    if best_fit is None:
        raise FloatingPointError("No VI start produced a finite final ELBO.")
    fit, final_elbo = best_fit, best_elbo
    diagnostics = {
        "status": "ok",
        "runtime_seconds": float(runtime),
        "rhat_max": float("nan"),
        "ess_min": float("nan"),
        "n_iter_run": int(fit.diagnostics["n_iter_run"]),
        "final_elbo": float(final_elbo),
        "vi_starts_run": campaign.vi_starts,
        "vi_best_start": int(best_start),
        "vi_start_elbos": elbos,
        "vi_start_profiles": [
            vi_start_profile(start)["name"]
            for start in range(campaign.vi_starts)
        ],
        "vi_best_start_profile": vi_start_profile(best_start)["name"],
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
    background_quadrature=None,
    spatial_compensator_quadrature=None,
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
            model,
            catalog,
            method,
            settings,
            campaign,
            seed,
            cutoff,
            started,
            background_quadrature,
            spatial_compensator_quadrature,
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
        adaptation_end = int(campaign.gibbs_iterations * campaign.gibbs_adaptation_fraction)
        adaptation_start = min(200, campaign.gibbs_iterations // 4, adaptation_end - 1)
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
                spatial_quadrature=spatial_compensator_quadrature,
            )
        )
    runtime = time.perf_counter() - started
    parameter_chains = _gibbs_parameter_chains(
        fits, burn_in=campaign.gibbs_burn_in
    )
    parameter_diagnostics = mcmc_diagnostics(parameter_chains)
    ess = float(np.min(parameter_diagnostics["ess_bulk"]))
    background_chains = _gibbs_background_chains(fits, campaign.gibbs_burn_in)
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
        "collapse_productivity": bool(fits[0].raw["collapse_productivity"]),
        **steps,
    }
    if len(fits) < 2:
        diagnostics["diagnostic_status"] = "insufficient_chains"
    else:
        rhats = [diagnostics["rhat_max"], diagnostics["rhat_background_max"]]
        effective_sizes = [ess, diagnostics["ess_tail_min"], diagnostics["ess_background_min"], diagnostics["ess_background_tail_min"]]
        if not np.isfinite(rhats + effective_sizes).any():
            diagnostics["diagnostic_status"] = "diagnostics_unavailable"
        else:
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
            if rate < 0.10 or (block == "eps" and rate > 0.95):
                problematic.append(f"{block}={rate:.1%}")
        if problematic:
            diagnostics["proposal_warning"] = (
                f"{method.upper()} proposal acceptance requires inspection: "
                + ", ".join(problematic)
            )
    truncation = fits[0].raw.get("branching_truncation")
    if truncation:
        diagnostics.update(truncation)
    for block, adaptation in fits[0].raw.get("etas_adaptation", {}).items():
        diagnostics[f"proposal_scale_{block}"] = adaptation["scale"]
    return FitBundle(
        method, model, tuple(fits), runtime, diagnostics, campaign.gibbs_burn_in
    ), diagnostics


def posterior_parameter_draws(bundle, n_draws, seed=0, burn_in=None):
    n_draws = int(n_draws)
    if METHODS[bundle.method]["family"] == "vi":
        return bundle.fits[0].posterior_parameter_samples(n_samples=n_draws, rng_seed=seed)
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


def parameter_recovery_metrics(parameter_draws, true_etas, true_beta):
    truths = {**true_etas.as_dict(), "beta": float(true_beta)}
    log_error_shifts = {"p": 1.0, "q": 1.0}
    etas_log_errors = []
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
        if name in ETAS_PARAMETER_NAMES:
            etas_log_errors.append(log_error)
    metrics["etas_parameter_log_error"] = float(np.mean(etas_log_errors))
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
        burn_in = getattr(bundle, "burn_in", 0.5)
        for fit in bundle.fits:
            values = np.asarray(fit.branching_chain, dtype=int)
            chains.append(values[int(burn_in * len(values)) :])
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


def _true_background_intensity(simulation, points, true_mus, field_scale):
    points = np.asarray(points, dtype=float)
    partition = simulation.background_simulation.domains
    domains = partition.locate(points[:, 0], points[:, 1])
    if np.any(domains < 0):
        raise ValueError("Every evaluation node must lie in the simulated domain.")
    true_eps = np.log(np.asarray(true_mus, dtype=float))
    return np.exp(true_eps[domains]) / (
        1.0
        + np.exp(-latent_field(points[:, 0], points[:, 1], field_scale))
    )


def background_recovery_metrics(
    bundle,
    simulation,
    true_mus,
    field_scale,
    campaign,
    seed,
    *,
    quadrature=None,
    return_payload=False,
):
    if quadrature is None:
        quadrature = regular_spatial_quadrature(campaign.evaluation_space_grid)
    elif not isinstance(quadrature, SpatialQuadrature):
        raise TypeError("quadrature must be a SpatialQuadrature instance.")
    spatial_xy = quadrature.points
    background_draws = posterior_background_draws(
        bundle, spatial_xy, campaign.posterior_draws, seed=seed
    )
    background_true = _true_background_intensity(
        simulation, spatial_xy, true_mus, field_scale
    )
    background_estimate = background_draws.mean(axis=1)
    rel_l2, mae = relative_l2_and_mae(
        background_estimate, background_true, quadrature.weights
    )
    metrics = {"rel_l2_background": rel_l2, "mae_background": mae}
    if not return_payload:
        return metrics
    plot_rule = regular_spatial_quadrature(campaign.evaluation_space_grid)
    if plot_rule.fingerprint() == quadrature.fingerprint():
        plot_estimate = background_estimate
        plot_true = background_true
    else:
        plot_estimate = posterior_background_draws(
            bundle,
            plot_rule.points,
            campaign.posterior_draws,
            seed=seed + 17,
        ).mean(axis=1)
        plot_true = _true_background_intensity(
            simulation, plot_rule.points, true_mus, field_scale
        )
    payload = {
        "space_grid_size": int(campaign.evaluation_space_grid),
        "spatial_xy": plot_rule.points,
        "background_true": plot_true,
        "background_estimate": plot_estimate,
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
        writer = csv.DictWriter(
            stream, fieldnames=fieldnames, lineterminator="\n"
        )
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

# Protocol re-exports retained for existing experiment callers.
from .simulation_settings import (  # noqa: F401
    ACCURACY_BACKGROUND_MUS,
    ACCURACY_DURATION_CALIBRATION_REPLICATES,
    ACCURACY_DURATION_CALIBRATION_SEED,
    ACCURACY_TARGET_EVENTS,
    CAMPAIGNS,
    CampaignConfig,
    EXPERIMENT_2_BETA,
    EXPERIMENT_2_DURATIONS,
    EXPERIMENT_2_ETAS,
    EXPERIMENT_2_METHOD,
    EXPERIMENT_2_TARGET_EVENTS,
    HIGH_CONTRAST_MUS,
    MISSPECIFIED_PARTITION_REGIONS,
    MISSPECIFIED_PARTITION_SEED,
    PARTITION_FIGURE_GRID_SIZE,
    PARTITION_FIGURE_REPLICATE,
    REFERENCE_MUS,
    RESULTS_ROOT,
    SCENARIOS,
    configure_campaign,
    simulation_protocol,
    validate_scientific_settings,
)
