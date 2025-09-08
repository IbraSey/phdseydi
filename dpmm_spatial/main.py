#%% 
# **************************************************************************************************************************
# ************************************************ AFFICHAGE f0 et f0_tilde ************************************************
# **************************************************************************************************************************
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import openturns as ot
from dpmm.prior_utils import (
    define_zonage_grid,
    compute_zone_gaussian_parameters,
    sample_from_f0,
    sample_from_f0tilde,
    compute_f0_density,
    compute_f0tilde_density 
)
from visualizations.plot import (
    plot_density_heatmap,
    plot_contour_levels,
    plot_sampling
)

# -----------------------------
# 1. Définition du zonage
# -----------------------------
n_rows, n_cols = 3, 3
zones, areas = define_zonage_grid(n_rows, n_cols)
n_zones = len(zones)
zone_weights = [0.05, 0.05, 0.05, 0.10, 0.10, 0.10, 0.15, 0.15, 0.25]

# Grille 
x = np.linspace(0, 2, 300)
y = np.linspace(0, 2, 300)
X, Y = np.meshgrid(x, y)

# Densité réelle f₀ 
Z_f0 = compute_f0_density(X, Y, zones, zone_weights, areas)

# -----------------------------
# 2. Échantillonnage depuis f0
# -----------------------------
X_samples = sample_from_f0(n_samples=5000, zones=zones, weights=zone_weights, areas=areas)

# -----------------------------
# 3. Apprentissage GMM pour approximer f0 par f0tilde
# -----------------------------
mus, covs, w_gmm, gmm = compute_zone_gaussian_parameters(X_samples, n_components=n_zones)

# -----------------------------
# 4. Approximation f0tilde (mélange de gaussiennes)
# -----------------------------
Z_f0tilde = compute_f0tilde_density(X, Y, mus, covs, w_gmm)

# -----------------------------
# 5. Échantillonnage depuis f0tilde
# -----------------------------
X_tilde_samples = sample_from_f0tilde(n_samples=5000, mus=mus, covariances=covs, weights=w_gmm)

# -----------------------------
# 6. Visus
# -----------------------------
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

# Densité f0 (zonage)
plot_density_heatmap(Z_f0, title="f0 (zonage)", ax=axs[0, 0])
plot_sampling(X_samples, title="Échantillons f0", ax=axs[0, 1])
axs[0, 2].axis("off")  # Emplacement vide pour équilibre visuel

# Densité f0tilde (GMM)
plot_density_heatmap(Z_f0tilde, title="f0tilde (approx. GMM)", ax=axs[1, 0])
plot_contour_levels(X, Y, Z_f0tilde, title="f0tilde - Lignes de niveau", ax=axs[1, 2])
plot_sampling(X_tilde_samples, title="Échantillons f0tilde", ax=axs[1, 1])

plt.suptitle("Visualisation d'un zonage (f0) et de son approximation GMM (f0tilde)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
#fig.savefig("visualizations/figures/figure_f0_f0tilde_regulier.png")



#%% 
# ***************************************************************************************************************************
# *********************************** AFFICHAGE de la moyenne empirique de f0 et f0_tilde ***********************************
# ***************************************************************************************************************************
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import openturns as ot

from dpmm.prior_utils import (
    define_zonage_grid,
    compute_zone_gaussian_parameters,
    sample_from_f0,
    sample_from_f0tilde,
    compute_f0_density,
    compute_f0tilde_density 
)

from visualizations.plot import (
    plot_density_heatmap,
    plot_contour_levels
)


n_rows, n_cols = 3, 3
zone_weights = [0.05, 0.05, 0.05, 0.10, 0.10, 0.10, 0.15, 0.15, 0.25]
N = 50      # Nombre de tirages à moyenner
n_samples = 5000
x = np.linspace(0, 2, 300)
y = np.linspace(0, 2, 300)
X, Y = np.meshgrid(x, y)

zones, areas = define_zonage_grid(n_rows, n_cols)
n_zones = len(zones)

Z_f0_sum = np.zeros_like(X)
for _ in range(N):
    Z_f0 = compute_f0_density(X, Y, zones, zone_weights, areas)
    Z_f0_sum += Z_f0
Z_f0_mean = Z_f0_sum / N


Z_f0tilde_sum = np.zeros_like(X)
for _ in range(N):
    X_samples = sample_from_f0(n_samples=n_samples, zones=zones, weights=zone_weights, areas=areas)
    mus, covs, weights_gmm, _ = compute_zone_gaussian_parameters(X_samples, n_components=n_zones)
    Z_f0tilde = compute_f0tilde_density(X, Y, mus, covs, weights_gmm)
    Z_f0tilde_sum += Z_f0tilde
Z_f0tilde_mean = Z_f0tilde_sum / N


# ====================================== Visualisations ======================================
fig, axs = plt.subplots(2, 2, figsize=(11, 10))

plot_density_heatmap(Z_f0_mean, title=f"f0 (zonage) – moyenne sur {N} tirages", ax=axs[0, 0])
axs[0, 1].axis("off")

plot_density_heatmap(Z_f0tilde_mean, title=f"\nf0tilde (GMM) – moyenne sur {N} tirages\n", ax=axs[1, 0])
plot_contour_levels(X, Y, Z_f0tilde_mean, title="\nf0tilde – lignes de niveau\n", ax=axs[1, 1])

plt.suptitle(f"\nMoyenne empirique sur {N} tirages de f0 et f0tilde", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()
#fig.savefig("visualizations/figures/figure_moyenne_empirique_f0_f0tilde_regulier.png")



# %% 
# ************************************************************************************************************************
# ************************************* AFFICHAGE f DPMM informatif / non-informatif *************************************
# ************************************************************************************************************************
import openturns as ot
import matplotlib.pyplot as plt
import numpy as np
from dpmm.dpmm import DirichletProcessMixtureModel, sample_niw, sample_mixture_niw, stick_breaking

# Grille pour affichage
x = np.linspace(0, 2, 300)
y = np.linspace(0, 2, 300)
X, Y = np.meshgrid(x, y)

# Prior NON-informatif : G0 simple
dpmm_noninf = DirichletProcessMixtureModel(
    alpha=20.0,
    tau=1e-3,
    G0_sampler=sample_niw,
    G0_kwargs={
        "mu_0": ot.Point([1.0, 1.0]),
        "lambda_0": 0.5,
        "Psi_0": ot.CovarianceMatrix([[0.5, 0.0], [0.0, 0.5]]),
        "nu_0": 8
    }
)

# Prior INFORMATIF : via zonage régulier
zone_weights = [0.05, 0.05, 0.05, 0.10, 0.10, 0.10, 0.15, 0.15, 0.25]
dpmm_inf = DirichletProcessMixtureModel.from_regular_zonage(
    alpha=20.0,
    tau=1e-3,
    n_rows=3,
    n_cols=3,
    lambda_0=50.0,
    nu_0=5,
    zone_weights=zone_weights
)

# Visualisation
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

dpmm_noninf.plot_density(X, Y, ax=axs[0, 0], title="Prior non-informatif - densité")
dpmm_noninf.plot_samples(5000, ax=axs[0, 1], title="Prior non-informatif - échantillons")

dpmm_inf.plot_density(X, Y, ax=axs[1, 0], title="Prior informatif - densité")
dpmm_inf.plot_samples(5000, ax=axs[1, 1], title="Prior informatif - échantillons")

centers = np.array(dpmm_inf.gaussian_centroids)
axs[1, 1].scatter(centers[:, 0], centers[:, 1], c='red', s=100, marker='x', label='Centroïdes f̃₀')

plt.tight_layout()
plt.show()
#fig.savefig("visualizations/figures/figure_f_dpmm_inf_noninf_regulier.png")

print("== Prior non-informatif ==")
print(dpmm_noninf.get_prior())

print("\n== Prior informatif ==")
print(dpmm_inf.get_prior())



#%%
# *****************************************************************************************************************
# ****************************** Impact du paramètre alpha sur le nombre composantes ******************************
# *****************************************************************************************************************
fig, axs = plt.subplots(1, 4, figsize=(15, 5))
alphas = [0.1, 1.0, 10.0, 100.0]

for ax, a in zip(axs, alphas):
    weights = stick_breaking(alpha=a, tau=1e-4)
    ax.stem(weights)
    ax.set_title(f"Stick-breaking (alpha = {a})")
    ax.set_xlabel("Composante")
    ax.set_ylabel("Poids")
    ax.grid(True)
    print(f"Nombre de composantes générées : {len(weights)}")
plt.tight_layout()
plt.show()
#fig.savefig("visualizations/figures/figure_impact_alpha_vs_nbcompo.png")



#%%
# *****************************************************************************************************************************
# *************************** AFFICHAGE de la moyenne empirique du DPMM informatif / non informatif ***************************
# *****************************************************************************************************************************
import openturns as ot
import numpy as np
import matplotlib.pyplot as plt

from dpmm.dpmm import (
    DirichletProcessMixtureModel,
    sample_niw,
    sample_mixture_niw,
    compute_empirical_mean_density
)

from visualizations.plot import (
    plot_density_heatmap,
    plot_contour_levels
)

# Grille d’évaluation
x = np.linspace(0, 2, 300)
y = np.linspace(0, 2, 300)
X, Y = np.meshgrid(x, y)

# Paramètres
alpha = 20.0
tau = 1e-3
N = 50  # Nombre de tirages pour moyenne empirique
n_rows=3
n_cols=3
zone_weights = [0.05, 0.05, 0.05, 0.10, 0.10, 0.10, 0.15, 0.15, 0.25]
lambda_0=50.0
nu_0=5
G0_kwargs={
    "mu_0": ot.Point([1.0, 1.0]),
    "lambda_0": 0.5,
    "Psi_0": ot.CovarianceMatrix([[0.5, 0.0], [0.0, 0.5]]),
    "nu_0": 8
    }


# Fonctions de constructions qui permettent utilisation de la fonction 'compute_empirical_mean_density'
def build_dpmm_noninf():
    return DirichletProcessMixtureModel(
        alpha=alpha,
        tau=tau,
        G0_sampler=sample_niw,
        G0_kwargs=G0_kwargs
    )

def build_dpmm_inf():
    return DirichletProcessMixtureModel.from_regular_zonage(
        alpha=alpha,
        tau=tau,
        n_rows=n_rows,
        n_cols=n_cols,
        lambda_0=lambda_0,
        nu_0=nu_0,
        zone_weights=zone_weights
    )

# ESTIMATION MOYENNE EMPIRIQUE 
Z_mean_noninf = compute_empirical_mean_density(build_dpmm_noninf, N, X, Y)
Z_mean_inf = compute_empirical_mean_density(build_dpmm_inf, N, X, Y)


# ===================== VISUALISATION =====================
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

plot_density_heatmap(Z_mean_noninf, title=f"Non-informatif – moyenne sur {N} DPMM", ax=axs[0, 0], cmap="viridis")
plot_contour_levels(X, Y, Z_mean_noninf, title="Non-informatif – lignes de niveau", ax=axs[0, 1], cmap="viridis")

plot_density_heatmap(Z_mean_inf, title=f"Informatif – moyenne sur {N} DPMM", ax=axs[1, 0], cmap="viridis")
plot_contour_levels(X, Y, Z_mean_inf, title="Informatif – lignes de niveau", ax=axs[1, 1], cmap="viridis")

plt.tight_layout()
plt.show()
#fig.savefig("visualizations/figures/figure_moyenne_empirique_dpmm_inf_et_noninf_regulier.png")



#%%
# ****************************************************************************************************************************
# ******************************** Paramètres sweep - alpha/lambda_0 - Moyenne empirique DPMM ********************************
# ****************************************************************************************************************************
import numpy as np
import matplotlib.pyplot as plt
import openturns as ot

from dpmm.dpmm import DirichletProcessMixtureModel, sample_mixture_niw
from dpmm.prior_utils import (
    define_zonage_grid,
    sample_from_f0,
    compute_zone_gaussian_parameters,
    compute_f0_density,
    compute_f0tilde_density
)
from visualizations.plot import plot_density_heatmap
from experiments.compute_l2 import eval_l2_dist_vs_two_params_avg_dpmm_inf


n_rows, n_cols = 3, 3
zones, areas = define_zonage_grid(n_rows, n_cols)
n_zones = len(zones)
zone_weights = [0.05, 0.05, 0.05, 0.10, 0.10, 0.10, 0.15, 0.15, 0.25]
X_samples = sample_from_f0(n_samples=10000, zones=zones, weights=zone_weights, areas=areas)
mus, covariances, weights_base, _ = compute_zone_gaussian_parameters(X_samples, n_components=n_zones)

x = np.linspace(0, 2, 300)
y = np.linspace(0, 2, 300)
X, Y = np.meshgrid(x, y)

Z_f0_ref = compute_f0_density(X, Y, zones, zone_weights, areas)
Z_f0tilde_ref = compute_f0tilde_density(X, Y, mus, covariances, weights_base)

def informative_dpmm_factory(alpha, lambda_0):
    nu_0 = 5
    Psi_0 = []
    for Sigma in covariances:
        Sigma_reg = Sigma + 1e-6 * np.eye(2)
        Psi = ot.CovarianceMatrix(Sigma_reg * (nu_0 - 3))
        Psi_0.append(Psi)

    return lambda: DirichletProcessMixtureModel(
        alpha=alpha,
        tau=1e-3,
        G0_sampler=sample_mixture_niw,
        G0_kwargs={
            "means_base": [ot.Point(mu) for mu in mus],
            "weights_base": weights_base.tolist(),
            "lambda_0": lambda_0,
            "Psi_0": Psi_0,
            "nu_0": nu_0
        }
    )

alphas = np.linspace(0.1, 10, 20)
lambdas = np.linspace(0.1, 10, 20)

Z_f0 = eval_l2_dist_vs_two_params_avg_dpmm_inf(
    param1_values=alphas,
    param2_values=lambdas,
    param1_name="alpha",
    param2_name="lambda_0",
    reference_density_array=Z_f0_ref,
    grid_x=x,
    grid_y=y,
    N=20,
    dpmm_factory_fn=informative_dpmm_factory
)
Z_f0tilde = eval_l2_dist_vs_two_params_avg_dpmm_inf(
    param1_values=alphas,
    param2_values=lambdas,
    param1_name="alpha",
    param2_name="lambda_0",
    reference_density_array=Z_f0tilde_ref,
    grid_x=x,
    grid_y=y,
    N=20,
    dpmm_factory_fn=informative_dpmm_factory
)


# =========================== Affichage ===========================
extent = (lambdas[0], lambdas[-1], alphas[0], alphas[-1])
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# f0
plot_density_heatmap(
    Z=Z_f0,
    title="Distance L² entre f0 et DPMM (alpha, lambda_0)",
    extent=extent,
    cmap='viridis',
    ax=axs[0]
)
axs[0].set_xlabel(r"$\lambda_0$")
axs[0].set_ylabel(r"$\alpha$")

min_idx_f0 = np.unravel_index(np.nanargmin(Z_f0), Z_f0.shape)
min_alpha_f0 = alphas[min_idx_f0[0]]
min_lambda_f0 = lambdas[min_idx_f0[1]]
min_val_f0 = Z_f0[min_idx_f0]
axs[0].plot(min_lambda_f0, min_alpha_f0, 'ro')
axs[0].annotate(f"{min_val_f0:.4f}", (min_lambda_f0, min_alpha_f0), color='white',
                xytext=(5, 5), textcoords='offset points', fontsize=10, weight='bold')

# f0tilde
plot_density_heatmap(
    Z=Z_f0tilde,
    title="Distance L² entre f0tilde et DPMM (alpha, lambda_0)",
    extent=extent,
    cmap='viridis',
    ax=axs[1]
)
axs[1].set_xlabel(r"$\lambda_0$")
axs[1].set_ylabel(r"$\alpha$")

min_idx_f0tilde = np.unravel_index(np.nanargmin(Z_f0tilde), Z_f0tilde.shape)
min_alpha_f0tilde = alphas[min_idx_f0tilde[0]]
min_lambda_f0tilde = lambdas[min_idx_f0tilde[1]]
min_val_f0tilde = Z_f0tilde[min_idx_f0tilde]
axs[1].plot(min_lambda_f0tilde, min_alpha_f0tilde, 'ro')
axs[1].annotate(f"{min_val_f0tilde:.4f}", (min_lambda_f0tilde, min_alpha_f0tilde), color='white',
                xytext=(5, 5), textcoords='offset points', fontsize=10, weight='bold')

plt.tight_layout()
plt.show()
#fig.savefig("visualizations/figures/figure_moyenne_empirique_dpmm_alpha_lambda0_sweep.png")



# %%
# ****************************************************************************************************************************
# ******************************************** Construction zonage pour cas jouet ********************************************
# ****************************************************************************************************************************
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
from matplotlib.colors import Normalize

# Polygones dans [0,2]²
polygons = [
    Polygon([(0.5,0.2), (0.55,0.2), (0.51,0.22), (0.39,0.48), (0.25,0.57), (0.13,0.68), (0.15,0.4), (0.35,0.28)]),
    Polygon([(0.4,0.1), (1.15,0.1), (1.25,0.3), (1.7,0.4), (1.7,0.6), (1.2,0.5), (1.1,0.47), (0.9,0.4), (0.55,0.2)]),
    Polygon([(0.55,0.2), (0.9,0.4), (0.85,0.58), (0.62,0.6), (0.5,0.6), (0.39,0.48), (0.51,0.22), (0.55,0.2)]),
    Polygon([(0.13,0.68), (0.25,0.57), (0.39,0.48), (0.5,0.6), (0.62,0.6), (0.58,1.0), (0.4,1.25), (0.37,1.4), 
             (0.55,1.6), (0.75,1.59), (1.0,1.05), (1.35,1.2), (1.55,1.5), (1.35,1.6), (1.1, 1.7),
             (0.6,1.7), (0.23,1.4), (0.05,1.0)]),
    Polygon([(0.62,0.6), (0.85,0.58), (0.9,0.4), (1.1,0.47), (1.2,0.5), (1.5,1.25), (1.35,1.2), (1.0,1.05), (0.58,1.0)]),
    Polygon([(0.58,1.0), (1.0,1.05), (0.75,1.59), (0.55,1.6), (0.37,1.4), (0.4,1.25)]),
    Polygon([(0.23,1.4), (0.35,1.9), (1.45,1.85), (1.35,1.6), (1.1, 1.7), (0.6,1.7), (0.23,1.4)]),
    Polygon([(1.45,1.85), (1.35,1.6), (1.55,1.5), (1.35,1.2), (1.5,1.25), (1.75,1.4), (1.75,1.6), (1.7,1.77), (1.65,1.80)]),
    Polygon([(1.5,1.25), (1.2,0.5), (1.7,0.6), (1.7,0.4), (1.78,0.5), (1.82,0.75), (1.83,1.0), (1.82,1.15), (1.75,1.4)])
]

# Intensités (poids de zone)
zone_weights = np.array([0.05, 0.05, 0.25, 0.01, 0.1, 0.14, 0.14, 0.01, 0.25]) 
cmap = plt.cm.plasma
norm = Normalize(vmin=zone_weights.min(), vmax=zone_weights.max())

# Calcul des aires
areas = np.array([poly.area for poly in polygons])
for i, area in enumerate(areas, 1):
    print(f"Aiire du polygone P{i} : {area:.4f}")


# =========================================== Affichage ===========================================
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0, 2)
ax.set_ylim(0, 2)
ax.set_aspect('equal')

for i, (poly, intensity) in enumerate(zip(polygons, zone_weights), start=1):
    color = cmap(norm(intensity))
    patch = plt.Polygon(list(poly.exterior.coords), facecolor=color, edgecolor='black')
    ax.add_patch(patch)

    # Position du label
    if i == 4:
        label_x, label_y = 0.32, 0.9  # Coordonnées choisies manuellement pour P4
    else:
        centroid = poly.centroid
        label_x, label_y = centroid.x, centroid.y

    ax.text(label_x, label_y, f'P{i}', color='black', weight='bold',
            ha='center', va='center', fontsize=10,
            bbox=dict(facecolor='white', alpha=0.3, edgecolor='none'))

# Colorbar et finalisation
plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, label="Intensité")
plt.title("\nZonage sismotectonique - Cas jouet\n")
plt.grid(True)
plt.show()
#fig.savefig("visualizations/figures/figure_zonage_cas_jouet.png")



# %%
# *********************************************************************************************************************************
# *************************************** AFFICHAGE f0 et f0_tilde (Cas jouet zonage sismo) ***************************************
# *********************************************************************************************************************************
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from matplotlib.colors import Normalize
from dpmm.dpmm import DirichletProcessMixtureModel
from dpmm.prior_utils import *
from visualizations.plot import *

# -----------------------------------
# 1. Définition du zonage irrégulier
# -----------------------------------
polygons = [
    Polygon([(0.5,0.2), (0.55,0.2), (0.51,0.22), (0.39,0.48), (0.25,0.57), (0.13,0.68), (0.15,0.4), (0.35,0.28)]),
    Polygon([(0.4,0.1), (1.15,0.1), (1.25,0.3), (1.7,0.4), (1.7,0.6), (1.2,0.5), (1.1,0.47), (0.9,0.4), (0.55,0.2)]),
    Polygon([(0.55,0.2), (0.9,0.4), (0.85,0.58), (0.62,0.6), (0.5,0.6), (0.39,0.48), (0.51,0.22), (0.55,0.2)]),
    Polygon([(0.13,0.68), (0.25,0.57), (0.39,0.48), (0.5,0.6), (0.62,0.6), (0.58,1.0), (0.4,1.25), (0.37,1.4), 
             (0.55,1.6), (0.75,1.59), (1.0,1.05), (1.35,1.2), (1.55,1.5), (1.35,1.6), (1.1, 1.7),
             (0.6,1.7), (0.23,1.4), (0.05,1.0)]),
    Polygon([(0.62,0.6), (0.85,0.58), (0.9,0.4), (1.1,0.47), (1.2,0.5), (1.5,1.25), (1.35,1.2), (1.0,1.05), (0.58,1.0)]),
    Polygon([(0.58,1.0), (1.0,1.05), (0.75,1.59), (0.55,1.6), (0.37,1.4), (0.4,1.25)]),
    Polygon([(0.23,1.4), (0.35,1.9), (1.45,1.85), (1.35,1.6), (1.1, 1.7), (0.6,1.7), (0.23,1.4)]),
    Polygon([(1.45,1.85), (1.35,1.6), (1.55,1.5), (1.35,1.2), (1.5,1.25), (1.75,1.4), (1.75,1.6), (1.7,1.77), (1.65,1.80)]),
    Polygon([(1.5,1.25), (1.2,0.5), (1.7,0.6), (1.7,0.4), (1.78,0.5), (1.82,0.75), (1.83,1.0), (1.82,1.15), (1.75,1.4)])
]
zone_weights = np.array([0.05, 0.05, 0.25, 0.01, 0.1, 0.14, 0.14, 0.01, 0.25])
areas = np.array([poly.area for poly in polygons])
n_zones = len(polygons)
cmap = plt.cm.plasma
norm = Normalize(vmin=zone_weights.min(), vmax=zone_weights.max())

# Grille pour visualisation
x = np.linspace(0, 2, 300)
y = np.linspace(0, 2, 300)
X, Y = np.meshgrid(x, y)

# -------------------------------------------
# 2. Construction DPMM par zonage irrégulier
# -------------------------------------------
lambda_0 = 5
nu_0 = 5
dpmm = DirichletProcessMixtureModel.from_irregular_zonage(
    alpha=1,
    tau=1e-4,
    zones=polygons,
    zone_weights=zone_weights,
    lambda_0=lambda_0,
    nu_0=nu_0
)

# -------------------------------
# 3. Approximation f0tilde (GMM)
# -------------------------------
mus = dpmm.gaussian_centroids
covs = dpmm.gaussian_covariances
w_gmm = dpmm.weights_base
Z_f0tilde = compute_f0tilde_density(X, Y, mus, covs, w_gmm)

# -------------------
# 4. Échantillonnage
# -------------------
X_samples = sample_from_f0(n_samples=5000, zones=polygons, weights=zone_weights, areas=areas, irregular=True)
X_tilde_samples = sample_from_f0tilde(n_samples=5000, mus=mus, covariances=covs, weights=w_gmm)

# -----------------
# 5. Visualisation
# -----------------
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

# f0 (zonage)
plot_sampling(X_samples, title="Échantillons f0", ax=axs[0, 1])
for i, (poly, intensity) in enumerate(zip(polygons, zone_weights), start=1):
    color = cmap(norm(intensity))
    patch = plt.Polygon(list(poly.exterior.coords), facecolor=color, edgecolor='black')
    axs[0,0].add_patch(patch)
    axs[0,0].set_title("Zonage sismo - Cas jouet")
    axs[0,0].set_xlim(0, 2)
    axs[0,0].set_ylim(0, 2)
    axs[0,0].set_aspect('equal')
    axs[0,0].grid()
axs[0, 2].axis("off")

# f0tilde (GMM)
plot_density_heatmap(Z_f0tilde, title="f0tilde (approx. GMM)", ax=axs[1, 0], cmap='plasma')
plot_sampling(X_tilde_samples, title="Échantillons f0tilde", ax=axs[1, 1])
plot_contour_levels(X, Y, Z_f0tilde, title="f0tilde - Lignes de niveau", ax=axs[1, 2], inline=False)

plt.suptitle("\nZonage irrégulier (f0) et approximation par GMM (f0tilde)\n", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 1])
plt.show()
#fig.savefig("visualizations/figures/figure_f0_f0tilde_cas_jouet_zonage_sismo.png")



#%% 
# *********************************************************************************************************************************
# ******************************** AFFICHAGE de la moyenne empirique de f0 et f0_tilde (Cas jouet) ********************************
# *********************************************************************************************************************************
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import openturns as ot

N = 10
n_samples = 5000
lambda_0, nu_0 = 5, 5
x = np.linspace(0, 2, 300)
y = np.linspace(0, 2, 300)
X, Y = np.meshgrid(x, y)

Z_f0tilde_sum = np.zeros_like(X)
for _ in range(N):
    X_samples = sample_from_f0(n_samples=n_samples, zones=polygons, weights=zone_weights, areas=areas, irregular=True)
    mus, covs, weights_gmm, _ = compute_zone_gaussian_parameters(X_samples, n_components=n_zones)
    Z_f0tilde = compute_f0tilde_density(X, Y, mus, covs, weights_gmm)
    Z_f0tilde_sum += Z_f0tilde
Z_f0tilde_mean = Z_f0tilde_sum / N


# ============================================== Visualisation ==============================================
fig, axs = plt.subplots(2, 2, figsize=(11, 10))

for i, (poly, intensity) in enumerate(zip(polygons, zone_weights), start=1):
    color = cmap(norm(intensity))
    patch = plt.Polygon(list(poly.exterior.coords), facecolor=color, edgecolor='black')
    axs[0,0].add_patch(patch)
    axs[0,0].set_title("Zonage sismo - Cas jouet")
    axs[0,0].set_xlim(0, 2)
    axs[0,0].set_ylim(0, 2)
    axs[0,0].set_aspect('equal')
    axs[0,0].grid()
axs[0, 1].axis("off")

plot_density_heatmap(Z_f0tilde_mean, title=f"\nf0tilde (GMM) – moyenne sur {N} tirages", cmap='viridis', ax=axs[1, 0])
plot_contour_levels(X, Y, Z_f0tilde_mean, title="\nf0tilde – lignes de niveau", ax=axs[1, 1])

plt.suptitle(f"\nMoyenne empirique sur {N} tirages de f0 et f0tilde (zonage irrégulier)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()
#fig.savefig("visualizations/figures/figure_moyenne_empirique_f0_f0tilde_irregulier.png")



# %% 
# ************************************************************************************************************************************
# **************************************** AFFICHAGE f DPMM informatif avec moyenne empirique ****************************************
# ************************************************************************************************************************************
import numpy as np
import matplotlib.pyplot as plt
import openturns as ot
from shapely.geometry import Polygon

from dpmm.dpmm import DirichletProcessMixtureModel, compute_empirical_mean_density, sample_niw
from visualizations.plot import plot_density_heatmap, plot_contour_levels

x = np.linspace(0, 2, 300)
y = np.linspace(0, 2, 300)
X, Y = np.meshgrid(x, y)

alpha = 20.0
tau = 1e-3
N = 10
lambda_0 = 50.0
nu_0 = 5
G0_kwargs_noninf = {
    "mu_0": ot.Point([1.0, 1.0]),
    "lambda_0": 0.5,
    "Psi_0": ot.CovarianceMatrix([[0.5, 0.0], [0.0, 0.5]]),
    "nu_0": 8
}

# -----------------------------
# Création des deux DPMM
# -----------------------------
dpmm_noninf = DirichletProcessMixtureModel(
    alpha=alpha,
    tau=tau,
    G0_sampler=sample_niw,
    G0_kwargs=G0_kwargs_noninf
)

dpmm_inf = DirichletProcessMixtureModel.from_irregular_zonage(
    alpha=alpha,
    tau=tau,
    zones=polygons,
    zone_weights=zone_weights,
    lambda_0=lambda_0,
    nu_0=nu_0
)

# -----------------
# Visualisation 
# -----------------
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

dpmm_noninf.plot_density(X, Y, ax=axs[0, 0], title="Prior non-informatif - densité")
dpmm_noninf.plot_samples(5000, ax=axs[0, 1], title="Prior non-informatif - échantillons")

dpmm_inf.plot_density(X, Y, ax=axs[1, 0], title="Prior informatif - densité (irrégulier)")
dpmm_inf.plot_samples(5000, ax=axs[1, 1], title="Prior informatif - échantillons")

centers = np.array(dpmm_inf.gaussian_centroids)
axs[1, 1].scatter(centers[:, 0], centers[:, 1], c='red', s=100, marker='x', label='Centroïdes f̃₀')

plt.tight_layout()
plt.show()
#fig.savefig("visualizations/figures/figure_f_dpmm_inf_cas_jouet_zonage_sismo.png")

print("== Prior non-informatif ==")
print(dpmm_noninf.get_prior())
print("\n== Prior informatif (irrégulier) ==")
print(dpmm_inf.get_prior())

# ------------------------------------------------
# Fonctions de construction pour moyenne empirique
# ------------------------------------------------
def build_dpmm_noninf():
    return DirichletProcessMixtureModel(
        alpha=alpha,
        tau=tau,
        G0_sampler=sample_niw,
        G0_kwargs=G0_kwargs_noninf
    )

def build_dpmm_inf():
    return DirichletProcessMixtureModel.from_irregular_zonage(
        alpha=alpha,
        tau=tau,
        zones=polygons,
        zone_weights=zone_weights,
        lambda_0=lambda_0,
        nu_0=nu_0
    )

# -----------------------------
# Moyenne empirique sur N DPMM
# -----------------------------
Z_mean_noninf = compute_empirical_mean_density(build_dpmm_noninf, N, X, Y)
Z_mean_inf = compute_empirical_mean_density(build_dpmm_inf, N, X, Y)

# -----------------------------
# Visualisation des moyennes
# -----------------------------
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

plot_density_heatmap(Z_mean_noninf, title=f"Non-informatif – moyenne sur {N} DPMM", ax=axs[0, 0], cmap="viridis")
plot_contour_levels(X, Y, Z_mean_noninf, title="Non-informatif – lignes de niveau", ax=axs[0, 1], cmap="viridis")

plot_density_heatmap(Z_mean_inf, title=f"Informatif (irrégulier) – moyenne sur {N} DPMM", ax=axs[1, 0], cmap="viridis")
plot_contour_levels(X, Y, Z_mean_inf, title="Informatif – lignes de niveau", ax=axs[1, 1], cmap="viridis")

plt.tight_layout()
plt.show()
#fig.savefig("visualizations/figures/figure_moyenne_empirique_f_dpmm_inf_cas_jouet_zonage_sismo.png")



#%%
# *****************************************************************************************************************************************
# **************************** Paramètres sweep - alpha/lambda_0 - Moyenne empirique DPMM sur zonage cas jouet ****************************
# *****************************************************************************************************************************************
import numpy as np
import matplotlib.pyplot as plt
import openturns as ot
from shapely.geometry import Polygon

from dpmm.dpmm import DirichletProcessMixtureModel, sample_mixture_niw
from dpmm.prior_utils import (
    sample_from_f0,
    compute_zone_gaussian_parameters,
    compute_f0_density,
    compute_f0tilde_density
)
from visualizations.plot import plot_density_heatmap
from experiments.compute_l2 import eval_l2_dist_vs_two_params_avg_dpmm_inf

polygons = [
    Polygon([(0.5,0.2), (0.55,0.2), (0.51,0.22), (0.39,0.48), (0.25,0.57), (0.13,0.68), (0.15,0.4), (0.35,0.28)]),
    Polygon([(0.4,0.1), (1.15,0.1), (1.25,0.3), (1.7,0.4), (1.7,0.6), (1.2,0.5), (1.1,0.47), (0.9,0.4), (0.55,0.2)]),
    Polygon([(0.55,0.2), (0.9,0.4), (0.85,0.58), (0.62,0.6), (0.5,0.6), (0.39,0.48), (0.51,0.22), (0.55,0.2)]),
    Polygon([(0.13,0.68), (0.25,0.57), (0.39,0.48), (0.5,0.6), (0.62,0.6), (0.58,1.0), (0.4,1.25), (0.37,1.4), 
             (0.55,1.6), (0.75,1.59), (1.0,1.05), (1.35,1.2), (1.55,1.5), (1.35,1.6), (1.1, 1.7),
             (0.6,1.7), (0.23,1.4), (0.05,1.0)]),
    Polygon([(0.62,0.6), (0.85,0.58), (0.9,0.4), (1.1,0.47), (1.2,0.5), (1.5,1.25), (1.35,1.2), (1.0,1.05), (0.58,1.0)]),
    Polygon([(0.58,1.0), (1.0,1.05), (0.75,1.59), (0.55,1.6), (0.37,1.4), (0.4,1.25)]),
    Polygon([(0.23,1.4), (0.35,1.9), (1.45,1.85), (1.35,1.6), (1.1, 1.7), (0.6,1.7), (0.23,1.4)]),
    Polygon([(1.45,1.85), (1.35,1.6), (1.55,1.5), (1.35,1.2), (1.5,1.25), (1.75,1.4), (1.75,1.6), (1.7,1.77), (1.65,1.80)]),
    Polygon([(1.5,1.25), (1.2,0.5), (1.7,0.6), (1.7,0.4), (1.78,0.5), (1.82,0.75), (1.83,1.0), (1.82,1.15), (1.75,1.4)])
]
zone_weights = np.array([0.05, 0.05, 0.25, 0.01, 0.1, 0.14, 0.14, 0.01, 0.25]) 
areas = np.array([poly.area for poly in polygons])

X_samples = sample_from_f0(n_samples=10000, zones=polygons, weights=zone_weights, areas=areas, irregular=True)
mus, covariances, weights_base, _ = compute_zone_gaussian_parameters(X_samples, n_components=len(polygons))

x = np.linspace(0, 2, 300)
y = np.linspace(0, 2, 300)
X, Y = np.meshgrid(x, y)

Z_f0_ref = compute_f0_density(X, Y, zones=polygons, weights=zone_weights, areas=areas, irregular=True)
Z_f0tilde_ref = compute_f0tilde_density(X, Y, mus, covariances, weights_base)

def informative_dpmm_factory(alpha, lambda_0):
    nu_0 = 5
    Psi_0 = []
    for Sigma in covariances:
        Sigma_reg = Sigma + 1e-6 * np.eye(2)
        Psi = ot.CovarianceMatrix(Sigma_reg * (nu_0 - 3))
        Psi_0.append(Psi)

    return lambda: DirichletProcessMixtureModel(
        alpha=alpha,
        tau=1e-3,
        G0_sampler=sample_mixture_niw,
        G0_kwargs={
            "means_base": [ot.Point(mu) for mu in mus],
            "weights_base": weights_base.tolist(),
            "lambda_0": lambda_0,
            "Psi_0": Psi_0,
            "nu_0": nu_0
        }
    )

alphas = np.linspace(0.1, 15, 20)
lambdas = np.linspace(0.1, 15, 20)

Z_f0 = eval_l2_dist_vs_two_params_avg_dpmm_inf(
    param1_values=alphas,
    param2_values=lambdas,
    param1_name="alpha",
    param2_name="lambda_0",
    reference_density_array=Z_f0_ref,
    grid_x=x,
    grid_y=y,
    N=5,
    dpmm_factory_fn=informative_dpmm_factory
)
Z_f0tilde = eval_l2_dist_vs_two_params_avg_dpmm_inf(
    param1_values=alphas,
    param2_values=lambdas,
    param1_name="alpha",
    param2_name="lambda_0",
    reference_density_array=Z_f0tilde_ref,
    grid_x=x,
    grid_y=y,
    N=5,
    dpmm_factory_fn=informative_dpmm_factory
)


# ======================================== Affichage ========================================
extent = (lambdas[0], lambdas[-1], alphas[0], alphas[-1])
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# f0
plot_density_heatmap(
    Z=Z_f0,
    title="Distance L² entre f0 et DPMM (alpha, lambda_0)",
    extent=extent,
    cmap='viridis',
    ax=axs[0]
)
axs[0].set_xlabel(r"$\lambda_0$")
axs[0].set_ylabel(r"$\alpha$")
min_idx_f0 = np.unravel_index(np.nanargmin(Z_f0), Z_f0.shape)
axs[0].plot(lambdas[min_idx_f0[1]], alphas[min_idx_f0[0]], 'ro')
axs[0].annotate(f"{Z_f0[min_idx_f0]:.4f}", (lambdas[min_idx_f0[1]], alphas[min_idx_f0[0]]),
                color='white', xytext=(5, 5), textcoords='offset points', fontsize=10, weight='bold')

# f0tilde
plot_density_heatmap(
    Z=Z_f0tilde,
    title="Distance L² entre f0tilde et DPMM (alpha, lambda_0)",
    extent=extent,
    cmap='viridis',
    ax=axs[1]
)
axs[1].set_xlabel(r"$\lambda_0$")
axs[1].set_ylabel(r"$\alpha$")
min_idx_f0tilde = np.unravel_index(np.nanargmin(Z_f0tilde), Z_f0tilde.shape)
axs[1].plot(lambdas[min_idx_f0tilde[1]], alphas[min_idx_f0tilde[0]], 'ro')
axs[1].annotate(f"{Z_f0tilde[min_idx_f0tilde]:.4f}", (lambdas[min_idx_f0tilde[1]], alphas[min_idx_f0tilde[0]]),
                color='white', xytext=(5, 5), textcoords='offset points', fontsize=10, weight='bold')

plt.tight_layout()
plt.show()
#fig.savefig("visualizations/figures/figure_moyenne_emp_dpmm_alpha_lambda0_sweep_zonage_cas_jouet.png")



# %%






# %%





