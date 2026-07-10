#%%

import csv
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from package import EventCatalog, GPParameters, GibbsConfig, SSGCModel


# =============================================================================
# Data selection
# =============================================================================
PACKAGE_DIR = Path(__file__).resolve().parent
USE_CASE_DIR = PACKAGE_DIR / "use_case"
if not USE_CASE_DIR.is_dir():
    USE_CASE_DIR = PACKAGE_DIR.parent / "use_case"
DOMAINS_PATH = USE_CASE_DIR / "domaines_xy.csv"
CATALOG_PATH = USE_CASE_DIR / "catalog.csv"

START_YEAR = 1970
END_YEAR = 2017
MIN_MAGNITUDE = 2.5


# Read one ordered polygon boundary per CODE_GTR.
coordinates_by_domain = defaultdict(list)
with DOMAINS_PATH.open(newline="", encoding="utf-8") as stream:
    for row in csv.DictReader(stream):
        coordinates_by_domain[row["CODE_GTR"]].append(
            (float(row["X"]), float(row["Y"]))
        )

domain_names = []
domains = []
occupied = None
for name, coordinates in coordinates_by_domain.items():
    polygon = Polygon(coordinates).buffer(0)
    if occupied is not None:
        # Remove the tiny numerical overlaps present in the source boundaries.
        polygon = polygon.difference(occupied)
    if polygon.geom_type != "Polygon" or polygon.is_empty:
        raise ValueError(f"Domain {name} is not a valid single polygon.")
    domain_names.append(name)
    domains.append(polygon)
    occupied = unary_union(domains)

X_BOUNDS = (float(occupied.bounds[0]), float(occupied.bounds[2]))
Y_BOUNDS = (float(occupied.bounds[1]), float(occupied.bounds[3]))


# Convert calendar dates to elapsed years and retain events inside the domains.
start_date = date(START_YEAR, 1, 1)
end_date = date(END_YEAR, 1, 1)
DURATION = (end_date - start_date).days / 365.25
events = []
with CATALOG_PATH.open(newline="", encoding="utf-8") as stream:
    for row in csv.DictReader(stream):
        year = int(row["year"])
        magnitude = float(row["magnitude"])
        if not START_YEAR <= year < END_YEAR or magnitude < MIN_MAGNITUDE:
            continue

        longitude = float(row["longitude"])
        latitude = float(row["latitude"])
        if not occupied.covers(Point(longitude, latitude)):
            continue

        event_date = date(year, int(row["month"]), int(row["day"]))
        elapsed_years = (event_date - start_date).days / 365.25
        events.append((elapsed_years, longitude, latitude))

events.sort(key=lambda event: event[0])
if not events:
    raise ValueError("No catalog event remains after filtering.")

t_arr, x_arr, y_arr = map(np.asarray, zip(*events))
print(
    f"Loaded {len(events)} events in {len(domains)} domains "
    f"from {START_YEAR} to {END_YEAR - 1} with M >= {MIN_MAGNITUDE}."
)


# =============================================================================
# Model and Gibbs configuration
# =============================================================================
NU_INIT = (5.0, 0.2)
LAMBDA_NU = 0.5
DELTA = (1.5, 0.1)
JITTER = 1e-5

MALA_STEP = 0.035
LEARN_NU = False
USE_CALIBRATION = True
T0_NU = 50
STEP_NU_INIT = 0.0009

N_ITER = 2000
THIN = 3
BURN_IN = 0.5
NX, NY = 30, 30
NX_POST, NY_POST = 50, 50

SEED = 42
VERBOSE = True
VERBOSE_EVERY = 100
SAVE_FIGURE = False
USE_SPARSEGP = True


catalog = EventCatalog(t=t_arr, x=x_arr, y=y_arr)

model = SSGCModel.from_polygons(
    polygons=domains,
    duration=DURATION,
    x_bounds=X_BOUNDS,
    y_bounds=Y_BOUNDS,
    initial_log_intensities=0.0,
    gp_prior=GPParameters(
        variance=NU_INIT[0],
        length_scale=NU_INIT[1],
    ),
    eps_prior_variance=DELTA[0],
    eps_prior_length_scale=DELTA[1],
    nu_prior_rate=LAMBDA_NU,
    jitter=JITTER,
)

config = GibbsConfig(
    n_iter=N_ITER,
    thin=THIN,
    mala_step=MALA_STEP,
    learn_nu=LEARN_NU,
    use_calibration=USE_CALIBRATION,
    t0_nu=T0_NU,
    step_nu_init=STEP_NU_INIT,
    verbose=VERBOSE,
    verbose_every=VERBOSE_EVERY,
    grid_nx=NX,
    grid_ny=NY,
    compute_emu=False,
)

fit = model.gibbs(
    catalog,
    config=config,
    gp_backend="sparse" if USE_SPARSEGP else "exact",
    rng_seed=SEED,
)

summary = fit.summary(burn_in=BURN_IN)
print("\nPosterior means")
for name, estimate in zip(domain_names, summary["eps_hat"]):
    print(f"{name:<6} eps = {estimate:.4f}")
print(f"nu  = {summary['nu_hat']}")
print(f"acceptance rates = {fit.acceptance_rates}")

fit.plot_traces(
    burn_in=BURN_IN,
    savefigure=SAVE_FIGURE,
    title_savefig="ssgc/template/traces",
)
fit.plot_acf(
    burn_in=BURN_IN,
    savefigure=SAVE_FIGURE,
    title_savefig="ssgc/template/acf",
)
fit.posterior_intensity(
    nx=NX_POST,
    ny=NY_POST,
    burn_in=BURN_IN,
    cmap="viridis",
    n_mc=200,
    savefigure=SAVE_FIGURE,
    title_savefig="ssgc/use_case/posterior_intensity",
)

# %%
