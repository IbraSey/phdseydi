"""Train/test splits, Silverman KDE fitting and zonal scoring for FCAT-17."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.special import logsumexp
from scipy.stats import gaussian_kde

from data import EventCatalog


PROTOCOL_LABELS = {
    "random": "Random 50/50",
    "temporal": "Temporal 80/20",
}


@dataclass(frozen=True)
class ValidationSplit:
    protocol: str
    repeat: int
    training_mask: np.ndarray
    test_mask: np.ndarray
    training_duration: float
    test_duration: float
    training_rate_fraction: float
    test_start: float | None = None

    def __post_init__(self):
        training = np.asarray(self.training_mask, dtype=bool).copy()
        test = np.asarray(self.test_mask, dtype=bool).copy()
        if training.ndim != 1 or test.shape != training.shape:
            raise ValueError("Train/test masks must be aligned one-dimensional arrays.")
        if np.any(training & test) or not np.all(training | test):
            raise ValueError("Train and test must form a disjoint exhaustive partition.")
        if not training.any() or not test.any():
            raise ValueError("Train and test must both contain events.")
        if self.protocol not in PROTOCOL_LABELS:
            raise ValueError(f"Unknown validation protocol {self.protocol!r}.")
        if not np.all(np.isfinite([self.training_duration, self.test_duration])):
            raise ValueError("Exposures must be finite.")
        if min(self.training_duration, self.test_duration) <= 0.0:
            raise ValueError("Exposures must be positive.")
        if not np.isfinite(self.training_rate_fraction) or not 0 < self.training_rate_fraction <= 1:
            raise ValueError("The retained training-rate fraction must be in (0, 1].")
        training.setflags(write=False)
        test.setflags(write=False)
        object.__setattr__(self, "training_mask", training)
        object.__setattr__(self, "test_mask", test)

    @property
    def label(self):
        if self.protocol == "random":
            return PROTOCOL_LABELS[self.protocol]
        fraction = self.training_duration / (self.training_duration + self.test_duration)
        return f"Temporal {100 * fraction:g}/{100 * (1 - fraction):g}"


def validation_splits(data, campaign, protocols=("random", "temporal")):
    protocols = tuple(protocols)
    if not protocols or len(set(protocols)) != len(protocols):
        raise ValueError("Select unique validation protocols.")
    if set(protocols) - set(PROTOCOL_LABELS):
        raise ValueError("Only random and temporal validation are supported.")
    catalog = data["catalog"]
    splits = []
    if "random" in protocols:
        for repeat in range(campaign.random_replicates):
            rng = np.random.default_rng(
                np.random.SeedSequence([campaign.validation_seed, repeat])
            )
            training = rng.random(len(catalog)) < 0.5
            splits.append(ValidationSplit(
                "random", repeat, training, ~training,
                data["duration"], data["duration"], 0.5,
            ))
    if "temporal" in protocols:
        cutoff = campaign.temporal_train_fraction * data["duration"]
        training = catalog.t < cutoff
        splits.append(ValidationSplit(
            "temporal", 0, training, ~training,
            cutoff, data["duration"] - cutoff, 1.0,
            data["observation_start_year"] + cutoff,
        ))
    return splits


def training_catalog(catalog, split):
    mask = split.training_mask
    # Random thinning retains the original exposure and hence the original times.
    return EventCatalog(
        t=catalog.t[mask], x=catalog.x[mask], y=catalog.y[mask],
        magnitudes=None if catalog.magnitudes is None else catalog.magnitudes[mask],
    )


def zonal_quadrature(data, grid_size, rule=None):
    """Integrate each original zone using the same global grid cell boundaries."""
    if rule is None:
        from experiments.exp_ssgc.deliverable_utils import area_weighted_grid
        rule = area_weighted_grid
    points, weights, indices = [], [], []
    for index, zone in enumerate(data["zones"]):
        xy, area = rule(
            data["x_bounds"], data["y_bounds"], grid_size, grid_size, zone
        )
        if not len(xy) or not np.isclose(np.sum(area), zone.area, rtol=1e-10):
            raise ValueError("Quadrature must cover the exact area of each zone.")
        points.append(xy)
        weights.append(area)
        indices.append(np.full(len(area), index, dtype=int))
    return np.vstack(points), np.concatenate(weights), np.concatenate(indices)


def predictive_scores(
    event_log_draws, quadrature_draws, weights, duration,
    event_zones, quadrature_zones, n_zones,
):
    """Global log-mean likelihood and additive expected-log zonal diagnostics."""
    logs = np.asarray(event_log_draws, dtype=float)
    intensity = np.asarray(quadrature_draws, dtype=float)
    weights = np.asarray(weights, dtype=float)
    event_zones = np.asarray(event_zones, dtype=int)
    quadrature_zones = np.asarray(quadrature_zones, dtype=int)
    if isinstance(n_zones, bool) or not isinstance(n_zones, Integral) or n_zones < 1:
        raise ValueError("n_zones must be a positive integer.")
    if logs.ndim != 2 or intensity.ndim != 2 or logs.shape[1] != intensity.shape[1]:
        raise ValueError("Event and quadrature draws must have aligned sample columns.")
    if logs.shape[0] == 0 or logs.shape[1] == 0:
        raise ValueError("At least one test event and posterior draw are required.")
    if len(intensity) == 0:
        raise ValueError("Quadrature cannot be empty.")
    if weights.shape != (len(intensity),) or event_zones.shape != (len(logs),):
        raise ValueError("Event zones and quadrature weights must align with draws.")
    if quadrature_zones.shape != weights.shape:
        raise ValueError("Quadrature zones must align with quadrature weights.")
    if np.any(weights <= 0) or not np.all(np.isfinite(weights)):
        raise ValueError("Quadrature weights must be positive and finite.")
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("Test exposure must be finite and positive.")
    if not np.all(np.isfinite(logs)) or not np.all(np.isfinite(intensity)):
        raise FloatingPointError("All event logs and quadrature intensities must be finite.")
    if np.any(intensity < 0):
        raise ValueError("Intensities cannot be negative.")
    if any(np.any((zones < 0) | (zones >= n_zones))
           for zones in (event_zones, quadrature_zones)):
        raise ValueError("Every point must belong to an original source zone.")

    zone_logs = np.zeros((n_zones, logs.shape[1]), dtype=float)
    counts = np.bincount(event_zones, minlength=n_zones)
    predicted = np.zeros_like(zone_logs)
    for index in range(n_zones):
        mask = quadrature_zones == index
        predicted[index] = duration * (weights[mask] @ intensity[mask])
        zone_logs[index] = logs[event_zones == index].sum(axis=0) - predicted[index]
    likelihoods = zone_logs.sum(axis=0)
    if not np.isfinite(likelihoods).all():
        raise FloatingPointError("The integrated test log likelihood must be finite.")
    log_mean = float(logsumexp(likelihoods) - np.log(len(likelihoods)))
    normalized = np.exp(likelihoods - logsumexp(likelihoods))
    return {
        "predictive_log_score_per_event": log_mean / len(logs),
        "expected_log_score_per_event": float(likelihoods.mean() / len(logs)),
        "predictive_draw_weight_ess": float(1.0 / np.sum(normalized**2)),
        "n_posterior_draws": len(likelihoods),
    }, {
        "observed_count": counts,
        "predicted_count": predicted.mean(axis=1),
        "expected_log_score": zone_logs.mean(axis=1),
    }


class ExposureNormalizedKDE:
    """Gaussian KDE with a fixed Silverman bandwidth.

    The covariance and bandwidth are estimated from the training points only.
    The fitted density is normalized over the retained observation domain and
    scaled by the training exposure to obtain a point-process intensity.
    """

    def __init__(self, catalog, duration, quadrature_xy, quadrature_weights):
        points = np.asarray(catalog.xy, dtype=float)
        quadrature_xy = np.asarray(quadrature_xy, dtype=float)
        quadrature_weights = np.asarray(quadrature_weights, dtype=float)
        if not len(points):
            raise ValueError("KDE requires at least one training event.")
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("KDE training exposure must be finite and positive.")
        if quadrature_xy.ndim != 2 or quadrature_xy.shape[1] != 2 or not len(quadrature_xy):
            raise ValueError("KDE quadrature points must be a non-empty (n, 2) array.")
        if (
            quadrature_weights.shape != (len(quadrature_xy),)
            or not np.isfinite(quadrature_xy).all()
            or not np.isfinite(quadrature_weights).all()
            or np.any(quadrature_weights <= 0)
        ):
            raise ValueError(
                "KDE quadrature must have aligned finite points and positive weights."
            )
        self.n_events = len(points)
        self.duration = float(duration)
        self.area = float(np.sum(quadrature_weights))
        self.bandwidth_rule = "silverman"
        self.bandwidth_factor = np.nan
        self.kde = None
        self.retained_mass = 1.0
        if self.n_events < 3 or np.linalg.matrix_rank(np.cov(points.T)) < 2:
            return
        self.kde = gaussian_kde(points.T, bw_method=self.bandwidth_rule)
        self.bandwidth_factor = float(self.kde.factor)
        self.retained_mass = float(
            quadrature_weights @ self.kde(quadrature_xy.T)
        )
        if not np.isfinite(self.retained_mass) or self.retained_mass <= 0:
            raise FloatingPointError("KDE normalization on the training domain is invalid.")

    def log_intensity(self, xy):
        if self.kde is None:
            return np.full(len(xy), np.log(self.n_events / (self.duration * self.area)))
        scale = self.n_events / (self.duration * self.retained_mass)
        return np.log(scale) + self.kde.logpdf(np.asarray(xy).T)

    def diagnostics(self):
        return {
            "kde_uniform_fallback": self.kde is None,
            "kde_bandwidth_rule": self.bandwidth_rule,
            "kde_bandwidth_factor": self.bandwidth_factor,
            "kde_retained_mass": self.retained_mass,
        }
