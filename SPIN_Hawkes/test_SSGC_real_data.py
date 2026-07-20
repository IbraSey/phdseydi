# %% 
# ===========================================
# ================= IMPORTS =================
# ===========================================
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import seaborn as sns
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from package import EventCatalog, GPParameters, GibbsConfig, SSGCModel, SparseGP


# ===========================================
# ================= HELPERS =================
# ===========================================
def resolve_use_case_path():
    script_dir = Path(__file__).resolve().parent
    for candidate in (script_dir / "use_case", script_dir.parent / "use_case"):
        if (candidate / "catalog.csv").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find use_case/catalog.csv next to the repository."
    )


def project_coordinates(longitude, latitude):
    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", "EPSG:2154", always_xy=True
    )
    x, y = transformer.transform(longitude, latitude)
    return np.asarray(x, dtype=float) * 1e-3, np.asarray(y, dtype=float) * 1e-3


def load_coastlines(path):
    coordinates = np.loadtxt(path)
    separators = np.where(np.any(~np.isfinite(coordinates), axis=1))[0]
    coastlines = []
    start = 0
    for stop in np.append(separators, len(coordinates)):
        segment = coordinates[start:stop]
        if len(segment):
            x, y = project_coordinates(segment[:, 0], segment[:, 1])
            coastlines.append(np.vstack((x, y)))
        start = stop + 1
    return coastlines



# ==========================================================
# ========================== MAIN ==========================
# ==========================================================

# Load the bundled French seismicity case

YEAR = 1965
M_C = 3.0
USE_CASE_PATH = resolve_use_case_path()
catalog_df = pd.read_csv(USE_CASE_PATH / "catalog.csv")
catalog_df = catalog_df[
    (catalog_df["year"] >= YEAR) & (catalog_df["magnitude"] >= M_C)
].copy()
catalog_x, catalog_y = project_coordinates(
    catalog_df["longitude"].to_numpy(),
    catalog_df["latitude"].to_numpy(),
)
catalog_df["X"] = catalog_x
catalog_df["Y"] = catalog_y

domain_df = pd.read_csv(USE_CASE_PATH / "domaines_xy.csv")
zones = []
zone_names = []
for zone_name, frame in domain_df.groupby("CODE_GTR", sort=False):
    x, y = project_coordinates(frame["X"].to_numpy(), frame["Y"].to_numpy())
    polygon = Polygon(np.column_stack((x, y)))
    zones.append(polygon if polygon.is_valid else polygon.buffer(0))
    zone_names.append(str(zone_name))

coastlines = load_coastlines(USE_CASE_PATH / "coastlines_france.txt")
D = catalog_df[["X", "Y", "magnitude"]].to_numpy()
T = float(catalog_df["year"].max() - catalog_df["year"].min())


# -------------------------------------------------------------------------
###########################
# Define Prior and Bounds #
###########################

# Bornes spatiales deduites de l'union des zones
coords = [np.array(z.exterior.coords) for z in zones]
X_BOUNDS = (min(c[:, 0].min() for c in coords), max(c[:, 0].max() for c in coords))
Y_BOUNDS = (min(c[:, 1].min() for c in coords), max(c[:, 1].max() for c in coords))
DURATION = T
#print(X_BOUNDS, Y_BOUNDS)

# Parametres du modele
NU_INIT = (2.0, 0.5)
LAMBDA_NU = 0.5
DELTA = (10.0, 03.0)

# Parametres du Gibbs
MALA_STEP = 0.055
LEARN_NU = False
USE_CALIBRATION = True
T0_NU = 50
STEP_NU_INIT = 0.0009

N_ITER = 10000
THIN = 5
BURN_IN = 0.5
NX_POST, NY_POST = 200, 200

SEED = 42
VERBOSE = True
VERBOSE_EVERY = int(N_ITER/10)
SAVE_FIGURE = True
USE_SPARSEGP = True
MAKE_PLOTS = True
N_POSTERIOR_DRAWS = 500
JITTER = 1e-5

#%%
###########################
# Build Catalog and Model #
###########################

# Filtre spatial : on garde uniquement les points dans l'union des domaines
domain_union = unary_union(zones)
select = [domain_union.covers(Point(xy)) for xy in D[:, :2]]
D_select = D[select]

catalog = EventCatalog(
    t=np.zeros(len(D_select)),
    x=D_select[:, 0],
    y=D_select[:, 1],
)

# Slight erosion prevents numerical overlaps along shared polygon boundaries.
zones_shrunk = [z.buffer(-1e-5) for z in zones]

model = SSGCModel.from_polygons(
    polygons=zones_shrunk,
    duration=DURATION,
    x_bounds=X_BOUNDS,
    y_bounds=Y_BOUNDS,
    gp_prior=GPParameters(variance=NU_INIT[0], length_scale=NU_INIT[1]),
    eps_prior_variance=DELTA[0],
    eps_prior_length_scale=DELTA[1],
    nu_prior_rate=LAMBDA_NU,
    jitter=JITTER,
)

gp_backend = "sparse" if USE_SPARSEGP else "exact"
sparse_gp = None
if USE_SPARSEGP and not USE_CALIBRATION:
    sparse_gp = SparseGP.from_bounds(
        X_BOUNDS, Y_BOUNDS,
        variance=NU_INIT[0],
        length_scale=NU_INIT[1],
    )
    print(f"Sparse GP basis: m={sparse_gp.m}")

config = GibbsConfig(
    n_iter=N_ITER,
    thin=THIN,
    mala_step=MALA_STEP,
    learn_nu=LEARN_NU,
    use_calibration=USE_CALIBRATION,
    verbose=VERBOSE,
    verbose_every=VERBOSE_EVERY,
)


#############
# Run Gibbs #
#############

fit = model.gibbs(
    catalog,
    config=config,
    gp_backend=gp_backend,
    sparse_gp=sparse_gp,
    reference_intensity=None,
    rng_seed=SEED,
)

summary = fit.summary(burn_in=BURN_IN)
print("\nPosterior means")
print(f"eps = {summary['eps_hat']}")
print(f"nu = {summary['nu_hat']}")
print(f"acceptance rates = {fit.acceptance_rates}")

burn = int(BURN_IN * fit.eps_chain.shape[0])
posterior = pd.DataFrame(
    {
        "N_Pi": fit.latent_point_counts[burn:],
        **{
            f"lambda_{name}": np.exp(fit.eps_chain[burn:, index])
            for index, name in enumerate(zone_names)
        },
    }
)
posterior.to_csv("posterior_sample.csv", index=False)

if MAKE_PLOTS:
    fit.plot_traces(
        burn_in=BURN_IN,
        savefigure=SAVE_FIGURE,
        title_savefig="ssgc/template_v1/traces",
    )
    fit.plot_acf(
        burn_in=BURN_IN,
        max_lag=int(burn/THIN),
        savefigure=SAVE_FIGURE,
        title_savefig="ssgc/template_v1/acf",
    )
    sns.pairplot(posterior)
    plt.savefig("pairplot.png", dpi=150, bbox_inches="tight")
    plt.close()


##########################################
# Posterior intensity on prediction grid #
##########################################

gridsize_x, gridsize_y = NX_POST, NY_POST
xx, yy = np.meshgrid(
    np.linspace(X_BOUNDS[0], X_BOUNDS[1], gridsize_x),
    np.linspace(Y_BOUNDS[0], Y_BOUNDS[1], gridsize_y),
)
available_draws = fit.eps_chain.shape[0] - burn
n_draws = min(N_POSTERIOR_DRAWS, available_draws)
intensity_samples = fit.background_intensity_samples(
    xx.ravel(),
    yy.ravel(),
    burn_in=BURN_IN,
    n_samples=n_draws,
)
intensity_mean = intensity_samples.mean(axis=1).reshape(yy.shape) * T
intensity_std = intensity_samples.std(axis=1).reshape(yy.shape) * T

levels_joint = np.linspace(
    min(intensity_mean.min(), intensity_std.min()),
    max(intensity_mean.max(), intensity_std.max()),
    40,
)


# Plot - Posterior mean
if MAKE_PLOTS:
    fig = plt.figure(figsize=(10, 10))
    plt.contourf(xx, yy, intensity_mean, levels_joint)
    plt.colorbar()
    plt.scatter(
        D_select[:, 0], D_select[:, 1],
        s=np.sqrt(D_select[:, 2]),
        c="r", marker="o", alpha=0.35,
    )
    for line in coastlines:
        plt.plot(line[0], line[1], "w", linewidth=1.0)
    plt.title("Seismic intensity - posterior mean", fontsize=20)
    plt.tight_layout()
    plt.savefig("intensity_post_mean.png", dpi=150)
    plt.close()


# Plot - Posterior std
if MAKE_PLOTS:
    fig = plt.figure(figsize=(10, 10))
    plt.contourf(xx, yy, intensity_std, levels_joint)
    plt.colorbar()
    plt.scatter(
        D_select[:, 0], D_select[:, 1],
        s=np.sqrt(D_select[:, 2]),
        c="r", marker="o", alpha=0.35,
    )
    for line in coastlines:
        plt.plot(line[0], line[1], "w", linewidth=1.0)
    plt.title("Seismic intensity - posterior standard deviation", fontsize=20)
    plt.tight_layout()
    plt.savefig("intensity_post_mean_std.png", dpi=150)
    plt.close()


# %%
