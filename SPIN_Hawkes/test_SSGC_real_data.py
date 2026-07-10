#%% Imports

import os
import sys
import numpy as np
import pandas as pd
import openturns as ot
import matplotlib.pyplot as plt
import seaborn as sns
import shapely
from pathlib import Path

from package import EventCatalog, GPParameters, GibbsConfig, SSGCModel

# Résolution du chemin vers phebus
file_path = Path(__file__).resolve()
ROOT = file_path.parent.parent
phebus_path = ROOT / "lib_py"
sys.path.insert(0, str(phebus_path))

from phebus.pybus.frclass import FrenchDomainsSourceModel


#%%
#############
# Load Case #
#############

demo_path = phebus_path / "phebus" / "demos" / "FrenchDomainsAnalysis"

SM = FrenchDomainsSourceModel(
    Mmin=3.,
    PWD=demo_path,
    FILE_DOMAINS=demo_path / "data" / "domains" / "domaines_xy.csv",
)

catalog_df = SM.catalog[SM.catalog.year >= 1965]

D = np.vstack((catalog_df.X, catalog_df.Y, catalog_df.magnitude)).T

T = max(catalog_df.year) - min(catalog_df.year)

# Zones / domaines
zones     = [zone.get_polygon_xy() for zone in SM.zones]
areas     = np.array([zone.get_area_km2() for zone in SM.zones])
zone_names = [zone.name for zone in SM.zones]

# Visualisation des domaines avec les superficies
SM.plot_values_map(
    areas,
    FIGURE_PATH=os.getcwd(),
    FIGURE_NAME="lambdas_and_domains",
    catalog=catalog_df,
    coastline=SM.coastlines,
    scale=5.,
    xticks=zone_names,
)


#%%
####################################
# Define Prior and Bounds          #
####################################

# Bornes spatiales déduites de l'union des zones
coords   = [np.array(z.exterior.coords) for z in zones]
X_BOUNDS = (min(c[:, 0].min() for c in coords), max(c[:, 0].max() for c in coords))
Y_BOUNDS = (min(c[:, 1].min() for c in coords), max(c[:, 1].max() for c in coords))
DURATION = T

# Paramètres du modèle
NU_INIT   = (5.0, 0.2)
LAMBDA_NU = 0.5
DELTA     = (1.5, 0.01)
JITTER    = 1e-5

# Paramètres du Gibbs
MALA_STEP    = 0.095
LEARN_NU     = False
USE_CALIBRATION = True
T0_NU        = 50
STEP_NU_INIT = 0.0009

N_ITER        = 2000
THIN          = 2
BURN_IN       = 0.5
NX, NY        = 20, 20
NX_POST, NY_POST = 60, 60

SEED          = 42
VERBOSE       = True
VERBOSE_EVERY = 100
SAVE_FIGURE   = False
USE_SPARSEGP  = True


#%%
####################################
# Build Catalog and Model          #
####################################

# Filtre spatial : on garde uniquement les points dans l'union des domaines
# zones sont déjà des Polygon shapely (retournés par get_polygon_xy())
domain_union = shapely.unary_union(zones)
select   = [domain_union.contains(shapely.Point(xy)) for xy in D[:, :2]]
D_select = D[select]

catalog = EventCatalog(
    t=np.zeros(len(D_select)),   # pas de dimension temporelle ici (PP spatial pur)
    x=D_select[:, 0],
    y=D_select[:, 1],
)

# Légère érosion des polygones pour éviter les intersections numériques de bord
# (les zones adjacentes partagent des arêtes communes dont l'intersection a une
#  aire > 0 en virgule flottante, ce qui déclenche le check anti-overlap du modèle)
zones_shrunk = [z.buffer(-1e-5) for z in zones]

model = SSGCModel.from_polygons(
    polygons=zones_shrunk,
    duration=DURATION,
    x_bounds=X_BOUNDS,
    y_bounds=Y_BOUNDS,
    initial_log_intensities=0.0,
    gp_prior=GPParameters(variance=NU_INIT[0], length_scale=NU_INIT[1]),
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


#%%
####################################
# Run Gibbs (with save/load cache) #
####################################

savefile = "posterior_sample.csv"

if os.path.exists(savefile):
    print(f"File {savefile} already exists — loading cached sample.")
    sample = pd.read_csv(savefile, index_col=0).to_numpy()
else:
    fit = model.gibbs(
        catalog,
        config=config,
        gp_backend="sparse" if USE_SPARSEGP else "exact",
        rng_seed=SEED,
    )

    summary = fit.summary(burn_in=BURN_IN)
    print("\nPosterior means")
    print(f"  eps = {summary['eps_hat']}")
    print(f"  nu  = {summary['nu_hat']}")
    print(f"  acceptance rates = {fit.acceptance_rates}")

    # Récupération du sample brut (post burn-in) depuis fit
    # Adapter le nom d'attribut si nécessaire (fit.samples, fit.chain, ...)
    sample = fit.get_sample(burn_in=BURN_IN)   # shape (S, param_dim)
    pd.DataFrame(data=sample).to_csv(savefile)

    # Diagnostics de convergence via les outils de la nouvelle API
    fit.plot_traces(
        burn_in=BURN_IN,
        savefigure=SAVE_FIGURE,
        title_savefig="ssgc/template_v1/traces",
    )
    fit.plot_acf(
        burn_in=BURN_IN,
        savefigure=SAVE_FIGURE,
        title_savefig="ssgc/template_v1/acf",
    )


#%%
############################
# Seaborn pairplot         #
############################

# Indices des composantes à afficher : N_tot + lambda_j (à adapter selon gibbs_indices)
# Si le fit expose un attribut gibbs_indices compatible avec l'ancienne API :
#   components = [fit.gibbs_indices.Pi_indices[-1]] + fit.gibbs_indices.lambda_indices
#   names = [r"$N_{tot}$"] + [r"$\lambda_{%s}$" % j for j in range(1, len(zones) + 1)]
# Sinon utiliser les indices bruts ci-dessous (dernières J+1 colonnes par convention) :

J          = len(zones)
components = list(range(-(J + 1), 0))   # à ajuster selon la structure réelle de sample
names      = [r"$N_{tot}$"] + [r"$\lambda_{%s}$" % j for j in range(1, J + 1)]

sns.pairplot(pd.DataFrame(data=sample[:, components], columns=names))
plt.savefig("pairplot.png")
plt.close()


#%%
###############################################################
# Posterior intensity on prediction grid                      #
###############################################################

gridsize = 500
xx, yy = np.meshgrid(
    np.linspace(X_BOUNDS[0], X_BOUNDS[1], gridsize),
    np.linspace(Y_BOUNDS[0], Y_BOUNDS[1], gridsize),
)
XY_new = np.vstack((xx.ravel(), yy.ravel())).T

# Accès aux composantes du modèle ajusté
# (adapter les noms d'attributs si l'API diffère de l'ancienne SSGC_Gibbs)
sparse_gp    = fit.model.sparse_gp      # ou fit.sparse_gp selon l'API
gibbs_indices = fit.gibbs_indices        # ou fit.model.gibbs_indices

M     = np.array(sparse_gp.regressorOT(XY_new))   # (N_grid, M_basis)
U_new = np.array(fit.model.U_OT(XY_new))           # (N_grid, J) membership weights

Z_new         = np.zeros((len(sample), len(XY_new)))
intensity_new = np.zeros((len(sample), len(XY_new)))

for i in range(len(sample)):
    sample_i  = sample[i]
    epsilon_i = sample_i[gibbs_indices.epsilon_indices]
    lambda_i  = sample_i[gibbs_indices.lambda_indices]

    Z_new[i]         = np.dot(M, epsilon_i).ravel()
    sigm_i           = np.array(sigmoid(ot.Sample(Z_new[i].reshape(-1, 1)))).ravel()
    Lambda_i         = (lambda_i * U_new).sum(axis=1)
    intensity_new[i] = sigm_i * Lambda_i

intensity_mean = intensity_new.mean(axis=0).reshape(gridsize, gridsize) * T
intensity_std  = intensity_new.std(axis=0).reshape(gridsize, gridsize) * T

levels_joint = np.linspace(
    min(intensity_mean.min(), intensity_std.min()),
    max(intensity_mean.max(), intensity_std.max()),
    gridsize,
)


#%% Plot — Posterior mean

fig = plt.figure(figsize=(10, 10))
plt.contourf(xx, yy, intensity_mean, levels_joint)
plt.colorbar()
plt.scatter(
    D_select[:, 0], D_select[:, 1],
    s=np.sqrt(D_select[:, 2]),
    c='r', marker='o',
    alpha=(1. / D_select[:, 2]) / max(1. / D_select[:, 2]),
)
for line in SM.coastlines:
    plt.plot(line[0], line[1], 'w', linewidth=1.5)
plt.title("Seismic intensity — Posterior mean", fontsize=20)
plt.tight_layout()
plt.savefig("intensity_post_mean.png")
plt.close()


#%% Plot — Posterior std

fig = plt.figure(figsize=(10, 10))
plt.contourf(xx, yy, intensity_std, levels_joint)
plt.colorbar()
plt.scatter(
    D_select[:, 0], D_select[:, 1],
    s=np.sqrt(D_select[:, 2]),
    c='r', marker='o',
    alpha=(1. / D_select[:, 2]) / max(1. / D_select[:, 2]),
)
for line in SM.coastlines:
    plt.plot(line[0], line[1], 'w', linewidth=1.5)
plt.title("Seismic intensity — Posterior std", fontsize=20)
plt.tight_layout()
plt.savefig("intensity_post_mean_std.png")
plt.close()


# %%
