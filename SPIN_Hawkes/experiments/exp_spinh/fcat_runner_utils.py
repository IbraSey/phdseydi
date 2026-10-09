"""Small persistence and MCMC-diagnostic helpers for the FCAT-17 runner."""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
_ARVIZ_MODULE = None
_ARVIZ_IMPORT_ATTEMPTED = False


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
    """Return rank R-hat, bulk/tail ESS and mean MCSE per coordinate."""
    chains = np.asarray(chains, dtype=float)
    if chains.ndim != 3 or min(chains.shape) < 1:
        raise ValueError("Expected non-empty (chain, draw, parameter) arrays.")
    result = {
        name: np.full(chains.shape[2], np.nan)
        for name in ("rhat", "ess_bulk", "ess_tail", "mcse_mean")
    }
    az = _load_arviz()
    for coordinate in range(chains.shape[2]):
        values = chains[:, :, coordinate]
        if not np.isfinite(values).all() or np.any(np.ptp(values, axis=1) == 0):
            result["rhat"][coordinate] = np.inf
            result["ess_bulk"][coordinate] = 0.0
            result["ess_tail"][coordinate] = 0.0
        elif az is not None and values.shape[1] >= 4:
            if len(values) >= 2:
                result["rhat"][coordinate] = float(az.rhat(values, method="rank"))
            result["ess_bulk"][coordinate] = float(az.ess(values, method="bulk"))
            result["ess_tail"][coordinate] = float(
                az.ess(values, method="tail", prob=(0.05, 0.95))
            )
            result["mcse_mean"][coordinate] = float(
                az.mcse(values, method="mean")
            )
    return result


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
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow(
                {name: _serializable(record.get(name, "")) for name in fieldnames}
            )
    return path
