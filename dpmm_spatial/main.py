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
#fig.savefig("visualizations/figures/figure_f0_f0tilde.png")



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

# Prior INFORMATIF : via zonage
zone_weights = [0.05, 0.05, 0.05, 0.10, 0.10, 0.10, 0.15, 0.15, 0.25]
dpmm_inf = DirichletProcessMixtureModel.from_zonage(
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
#fig.savefig("visualizations/figures/figure_f_dpmm_inf_noninf.png")

print("== Prior non-informatif ==")
print(dpmm_noninf.get_prior())

print("\n== Prior informatif ==")
print(dpmm_inf.get_prior())



#%%
# *************************************************************************************************************************
# ********************************** Impact du paramètre alpha sur le nombre composantes **********************************
# *************************************************************************************************************************
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
# **************************************************************************************************************************
# ********************************** AFFICHAGE de la moyenne empirique du DPMM informatif **********************************
# **************************************************************************************************************************

import numpy as np
import matplotlib.pyplot as plt
import openturns as ot
from dpmm.density import informative_dpmm_density
from dpmm.sampling import stick_breaking, sample_mixture_niw, sample_from_informative_dpmm
from visualizations.plot_density import plot_density_heatmap
from visualizations.plot_sampling import plot_sampling
from visualizations.plot_density import plot_contour_levels

# === Paramètres DPMM informatif ===
alpha = 50
tau = 1e-2
means_base = [[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]]
weights_base = np.array([2.0, 1.0, 0.5, 0.1])
weights_base /= weights_base.sum()
lambda_0 = 30.0
nu_0 = 4
Psi_0 = ot.CovarianceMatrix([[0.26, 0.0], [0.0, 0.26]])

# === Grille ===
x = np.linspace(0, 2, 200)
y = np.linspace(0, 2, 200)
X, Y = np.meshgrid(x, y)

# === Moyenne empirique de N densités ===
N = 50
Z_sum = np.zeros_like(X)

for _ in range(N):
    f_density = informative_dpmm_density(
        alpha=alpha,
        tau=tau,
        means_base=means_base,
        weights_base=weights_base,
        lambda_0=lambda_0,
        Psi_0=Psi_0,
        nu_0=nu_0
    )
    Z_sum += f_density(X, Y)

Z_mean = Z_sum / N


# ============================= Affichage =============================
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

plot_density_heatmap(Z_mean, title="Heatmap de la densité moyenne", ax=axs[0])
plot_contour_levels(X, Y, Z_mean, levels=20, title="Lignes de niveau de la densité moyenne", ax=axs[1])

plt.tight_layout()
plt.show()
fig.savefig("figure_MC_f_dpmm_inf.png")




#%%
# *************************************************************************************************************************
# ************************************************** (1) Paramètre sweep **************************************************
# *************************************************************************************************************************
import numpy as np
import matplotlib.pyplot as plt
import openturns as ot
from dpmm.density import (define_zonage_grid, compute_f0_density, 
                          compute_zone_gaussian_parameters, compute_f0tilde_density, 
                          informative_dpmm_density)
from experiments.compute_l2 import compute_l2_distance, evaluate_l2_distance_vs_param

# === Paramètres f0, f0_tilde ===
n_rows, n_cols = 2, 2
weights_base = np.array([2.0, 1.0, 0.5, 0.1])
weights_base /= weights_base.sum()
zones, x_bounds, y_bounds = define_zonage_grid(n_rows, n_cols)
area = (x_bounds[1] - x_bounds[0]) * (y_bounds[1] - y_bounds[0])
areas = np.full(len(weights_base), area)
mus, covariances = compute_zone_gaussian_parameters(zones)

# === Paramètres DPMM informatif ===
alpha = 50
tau = 1e-2
means_base = [[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]]
weights_base = np.array([2.0, 1.0, 0.5, 0.1])
weights_base /= weights_base.sum()
lambda_0 = 30.0
nu_0 = 4
Psi_0 = ot.CovarianceMatrix([[0.26, 0.0], [0.0, 0.26]])

alphas = np.linspace(0.1, 50, 30)
grid_x = np.linspace(0, 2, 200)
grid_y = np.linspace(0, 2, 200)

params = {
    "tau": tau,
    "means_base": means_base,
    "weights_base": weights_base,
    "lambda_0": lambda_0,
    "Psi_0": Psi_0,
    "nu_0": nu_0
}

ref_f0_args = {"zones": zones, "weights": weights_base, "areas": areas}
ref_f0tilde_args={"mus": mus, "covariances": covariances, "weights": weights_base}

alphas, distances_f0 = evaluate_l2_distance_vs_param(
    param_values=alphas,
    reference_density=compute_f0_density,
    dpmm_density_constructor=informative_dpmm_density,
    grid_x=np.linspace(0, 2, 200),
    grid_y=np.linspace(0, 2, 200),
    param_name="alpha",
    constructor_kwargs=params,
    ref_args=ref_f0_args
)

alphas, distances_f0tilde = evaluate_l2_distance_vs_param(
    param_values=alphas,
    reference_density=compute_f0tilde_density,
    dpmm_density_constructor=informative_dpmm_density,
    grid_x=np.linspace(0, 2, 200),
    grid_y=np.linspace(0, 2, 200),
    param_name="alpha",
    constructor_kwargs=params, 
    ref_args=ref_f0tilde_args
)

# === Affichage des deux courbes L²(alpha) ===
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Plot pour f₀
axs[0].plot(alphas, distances_f0, label=r'$L^2(f_0, \hat{f})$', color='C0')
axs[0].set_title(r'Distance $L^2$ entre $f_0$ (zonage) et $\hat{f}$')
axs[0].set_xlabel(r'$\alpha$')
axs[0].set_ylabel(r'Distance $L^2$')
axs[0].grid(True)
axs[0].legend()

# Plot pour f₀̃
axs[1].plot(alphas, distances_f0tilde, label=r'$L^2(\tilde{f}_0, \hat{f})$', color='C1')
axs[1].set_title(r'Distance $L^2$ entre $\tilde{f}_0$ (mélange gaussien) et $\hat{f}$')
axs[1].set_xlabel(r'$\alpha$')
axs[1].set_ylabel(r'Distance $L^2$')
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()
fig.savefig("figure_alpha_sweep.png")



#%%
# *************************************************************************************************************************
# ***************************************** (2) Paramètres sweep - alpha/lambda_0 *****************************************
# *************************************************************************************************************************
import numpy as np
import matplotlib.pyplot as plt
import openturns as ot
from dpmm.density import (define_zonage_grid, compute_f0_density, 
                          compute_zone_gaussian_parameters, compute_f0tilde_density, 
                          informative_dpmm_density)
from experiments.compute_l2 import compute_l2_distance, evaluate_l2_distance_vs_param, evaluate_l2_distance_vs_two_params
from visualizations.plot_density import plot_density_heatmap

# === Paramètres f0, f0_tilde ===
n_rows, n_cols = 2, 2
weights_base = np.array([2.0, 1.0, 0.5, 0.1])
weights_base /= weights_base.sum()
zones, x_bounds, y_bounds = define_zonage_grid(n_rows, n_cols)
area = (x_bounds[1] - x_bounds[0]) * (y_bounds[1] - y_bounds[0])
areas = np.full(len(weights_base), area)
mus, covariances = compute_zone_gaussian_parameters(zones)

# === Paramètres DPMM informatif ===
alpha = 50
tau = 1e-2
means_base = [[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]]
weights_base = np.array([2.0, 1.0, 0.5, 0.1])
weights_base /= weights_base.sum()
lambda_0 = 30.0
nu_0 = 4
Psi_0 = ot.CovarianceMatrix([[0.26, 0.0], [0.0, 0.26]])

grid_x = np.linspace(0, 2, 200)
grid_y = np.linspace(0, 2, 200)

alphas = np.linspace(0.1, 40, 30)
lambdas = np.linspace(0.1, 40, 30)

params = {"tau": tau, "means_base": means_base, "weights_base": weights_base, "nu_0": nu_0, "Psi_0": Psi_0}
ref_f0_args = {"zones": zones, "weights": weights_base, "areas": areas}
ref_f0tilde_args={"mus": mus, "covariances": covariances, "weights": weights_base}

alpha_grid, lambda_grid, L2_matrix_f0 = evaluate_l2_distance_vs_two_params(
    param1_values=alphas,
    param2_values=lambdas,
    param1_name="alpha",
    param2_name="lambda_0",
    reference_density=compute_f0_density,
    dpmm_density_constructor=informative_dpmm_density,
    grid_x=grid_x,
    grid_y=grid_y,
    constructor_kwargs_base=params,
    ref_args=ref_f0_args
)

alpha_grid, lambda_grid, L2_matrix_f0tilde = evaluate_l2_distance_vs_two_params(
    param1_values=alphas,
    param2_values=lambdas,
    param1_name="alpha",
    param2_name="lambda_0",
    reference_density=compute_f0tilde_density,
    dpmm_density_constructor=informative_dpmm_density,
    grid_x=grid_x,
    grid_y=grid_y,
    constructor_kwargs_base=params,
    ref_args= ref_f0tilde_args,
    verbose=False
)

# ============================== Affichage ==============================
extent = (lambdas[0], lambdas[-1], alphas[0], alphas[-1])
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

plot_density_heatmap(
    Z=L2_matrix_f0,
    title=r"Distance L² entre $f_0$ et DPMM",
    extent=extent,
    cmap='viridis',
    ax=axs[0]
)
axs[0].set_xlabel(r"$\lambda_0$")
axs[0].set_ylabel(r"$\alpha$")

# Trouver le minimum et l'annoter
min_idx_f0 = np.unravel_index(np.nanargmin(L2_matrix_f0), L2_matrix_f0.shape)
min_alpha_f0 = alphas[min_idx_f0[0]]
min_lambda_f0 = lambdas[min_idx_f0[1]]
min_val_f0 = L2_matrix_f0[min_idx_f0]
axs[0].plot(min_lambda_f0, min_alpha_f0, 'ro')
axs[0].annotate(f"{min_val_f0:.4f}", (min_lambda_f0, min_alpha_f0), color='white',
                xytext=(5, 5), textcoords='offset points', fontsize=10, weight='bold')


plot_density_heatmap(
    Z=L2_matrix_f0tilde,
    title=r"Distance L² entre $\tilde{f}_0$ et DPMM",
    extent=extent,
    cmap='viridis',
    ax=axs[1]
)
axs[1].set_xlabel(r"$\lambda_0$")
axs[1].set_ylabel(r"$\alpha$")

# Trouver le minimum et l'annoter
min_idx_f0tilde = np.unravel_index(np.nanargmin(L2_matrix_f0tilde), L2_matrix_f0tilde.shape)
min_alpha_f0tilde = alphas[min_idx_f0tilde[0]]
min_lambda_f0tilde = lambdas[min_idx_f0tilde[1]]
min_val_f0tilde = L2_matrix_f0tilde[min_idx_f0tilde]
axs[1].plot(min_lambda_f0tilde, min_alpha_f0tilde, 'ro')
axs[1].annotate(f"{min_val_f0tilde:.4f}", (min_lambda_f0tilde, min_alpha_f0tilde), color='white',
                xytext=(5, 5), textcoords='offset points', fontsize=10, weight='bold')

plt.tight_layout()
plt.show()
fig.savefig("figure_alpha_lambda_sweep.png")




#%%
# ****************************************************************************************************************************
# ****************************** (2) Paramètres sweep - alpha/lambda_0 - Moyenne empirique DPMM ******************************
# ****************************************************************************************************************************
import numpy as np
import matplotlib.pyplot as plt
import openturns as ot
from dpmm.density import (define_zonage_grid, compute_f0_density, 
                          compute_zone_gaussian_parameters, compute_f0tilde_density, 
                          informative_dpmm_density)
from experiments.compute_l2 import (compute_l2_distance, evaluate_l2_distance_vs_param, 
                                    evaluate_l2_distance_vs_two_params, eval_l2_dist_vs_two_params_avg_dpmm_inf)
from visualizations.plot_density import plot_density_heatmap

# === Paramètres f0, f0_tilde ===
n_rows, n_cols = 2, 2
weights_base = np.array([2.0, 1.0, 0.5, 0.1])
weights_base /= weights_base.sum()
zones, x_bounds, y_bounds = define_zonage_grid(n_rows, n_cols)
area = (x_bounds[1] - x_bounds[0]) * (y_bounds[1] - y_bounds[0])
areas = np.full(len(weights_base), area)
mus, covariances = compute_zone_gaussian_parameters(zones)

# === Paramètres DPMM informatif ===
alpha = 50
tau = 1e-2
means_base = [[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]]
weights_base = np.array([2.0, 1.0, 0.5, 0.1])
weights_base /= weights_base.sum()
lambda_0 = 30.0
nu_0 = 4
Psi_0 = ot.CovarianceMatrix([[0.26, 0.0], [0.0, 0.26]])


# Grille
grid_x = np.linspace(0, 2, 200)
grid_y = np.linspace(0, 2, 200)
X, Y = np.meshgrid(grid_x, grid_y)

# Référence : f0_tilde
Z_f0_ref = compute_f0_density(X, Y, zones, weights_base, areas)
Z_f0tilde_ref = compute_f0tilde_density(X, Y, mus, covariances, weights_base)

# Paramètres à faire varier
alphas = np.linspace(0.1, 10, 15)
lambdas = np.linspace(0.1, 10, 15)

# Paramètres fixes
kwargs_base = {
    "tau": tau,
    "means_base": means_base,
    "weights_base": weights_base,
    "nu_0": nu_0,
    "Psi_0": Psi_0
}

# Évaluation
Z_f0 = eval_l2_dist_vs_two_params_avg_dpmm_inf(
    param1_values=alphas,
    param2_values=lambdas,
    param1_name="alpha",
    param2_name="lambda_0",
    reference_density_array=Z_f0_ref,
    grid_x=grid_x,
    grid_y=grid_y,
    N=10,
    dpmm_density_fn=informative_dpmm_density,
    constructor_kwargs_base=kwargs_base,
    verbose=True
)

Z_f0tilde = eval_l2_dist_vs_two_params_avg_dpmm_inf(
    param1_values=alphas,
    param2_values=lambdas,
    param1_name="alpha",
    param2_name="lambda_0",
    reference_density_array=Z_f0tilde_ref,
    grid_x=grid_x,
    grid_y=grid_y,
    N=10,
    dpmm_density_fn=informative_dpmm_density,
    constructor_kwargs_base=kwargs_base,
    verbose=True
)


# ============================== Affichage ==============================
extent = (lambdas[0], lambdas[-1], alphas[0], alphas[-1])
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

plot_density_heatmap(
    Z=Z_f0,
    title="Distance L² entre f0 et DPMM (en fonction de α et λ₀)",
    extent=extent,
    cmap='viridis',
    ax=axs[0]
)
axs[0].set_xlabel(r"$\lambda_0$")
axs[0].set_ylabel(r"$\alpha$")

# Trouver le minimum et l'annoter
min_idx_f0 = np.unravel_index(np.nanargmin(Z_f0), Z_f0.shape)
min_alpha_f0 = alphas[min_idx_f0[0]]
min_lambda_f0 = lambdas[min_idx_f0[1]]
min_val_f0 = Z_f0[min_idx_f0]
axs[0].plot(min_lambda_f0, min_alpha_f0, 'ro')
axs[0].annotate(f"{min_val_f0:.4f}", (min_lambda_f0, min_alpha_f0), color='white',
                xytext=(5, 5), textcoords='offset points', fontsize=10, weight='bold')


plot_density_heatmap(
    Z=Z_f0tilde,
    title="Distance L² entre f0_tilde et DPMM (en fonction de α et λ₀)",
    extent=extent,
    cmap='viridis',
    ax=axs[1]
)
axs[1].set_xlabel(r"$\lambda_0$")
axs[1].set_ylabel(r"$\alpha$")

# Trouver le minimum et l'annoter
min_idx_f0tilde = np.unravel_index(np.nanargmin(Z_f0tilde), Z_f0tilde.shape)
min_alpha_f0tilde = alphas[min_idx_f0tilde[0]]
min_lambda_f0tilde = lambdas[min_idx_f0tilde[1]]
min_val_f0tilde = Z_f0tilde[min_idx_f0tilde]
axs[1].plot(min_lambda_f0tilde, min_alpha_f0tilde, 'ro')
axs[1].annotate(f"{min_val_f0tilde:.4f}", (min_lambda_f0tilde, min_alpha_f0tilde), color='white',
                xytext=(5, 5), textcoords='offset points', fontsize=10, weight='bold')

plt.tight_layout()
plt.show()
fig.savefig("figure_alpha_lambda_sweep_MC.png")



#%%
# *******************************************************************************************************
# ******************************************* (EN CHANTIER !) *******************************************
# *******************************************************************************************************
import numpy as np
import matplotlib.pyplot as plt
import openturns as ot
from dpmm.density import (define_zonage_grid, compute_f0_density, 
                          compute_zone_gaussian_parameters, compute_f0tilde_density, 
                          informative_dpmm_density, noninformative_dpmm_density)
from experiments.compute_l2 import (evaluate_l2_distance_vs_param, evaluate_l2_distance_vs_two_params, 
                                    eval_l2_dist_vs_two_params_avg_dpmm_inf)
from visualizations.plot_density import plot_density_heatmap
from visualizations.plot_sampling import plot_sampling
from dpmm.sampling import sample_from_noninformative_dpmm


# === Paramètres du DPMM non-informatif ===
n_samples = 10000
alpha = 20.0
tau = 1e-2

mu_0 = ot.Point([1.0, 1.0])
lambda_0 = 100.0
nu_0 = 4.0
Psi_0 = ot.CovarianceMatrix([[1.0 * (nu_0 - 3), 0.0], [0.0, 1.0 * (nu_0 - 3)]])

samples_f_noninf = sample_from_noninformative_dpmm(
    n_samples=n_samples,
    alpha=alpha,
    tau=tau,
    mu_0=mu_0,
    lambda_0=lambda_0,
    Psi_0=Psi_0,
    nu_0=nu_0
)

f_noninf_density = noninformative_dpmm_density(
    alpha=alpha,
    tau=tau,
    mu_0=mu_0,
    lambda_0=lambda_0,
    Psi_0=Psi_0,
    nu_0=nu_0
)

# Grille    
x = np.linspace(0, 2, 200)
y = np.linspace(0, 2, 200)
X, Y = np.meshgrid(x, y)

Z_noninf = f_noninf_density(X, Y)


# ============================= Affichage =============================
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

plot_density_heatmap(Z_noninf, title="Densité du DPMM non-informatif", ax=axs[0])
plot_sampling(samples_f_noninf, title="Échantillons selon $f$", ax=axs[1])

plt.tight_layout()
plt.show()
fig.savefig("figure_f_noninf.png")




#%%
# *******************************************************************************************************
# ******************************************* (EN CHANTIER !) *******************************************
# *******************************************************************************************************
import numpy as np
import matplotlib.pyplot as plt
import openturns as ot
from dpmm.density import (define_zonage_grid, compute_f0_density, 
                          compute_zone_gaussian_parameters, compute_f0tilde_density, 
                          informative_dpmm_density, noninformative_dpmm_density)
from experiments.compute_l2 import (evaluate_l2_distance_vs_param, evaluate_l2_distance_vs_two_params, 
                                    eval_l2_dist_vs_two_params_avg_dpmm_inf)
from visualizations.plot_density import plot_density_heatmap
from visualizations.plot_sampling import plot_sampling
from dpmm.sampling import sample_from_noninformative_dpmm

# === Paramètres du DPMM non-informatif ===
N = 20 
alpha = 20.0
tau = 1e-2

mu_0 = ot.Point([1.0, 1.0])
lambda_0 = 100.0
nu_0 = 4.0
Psi_0 = ot.CovarianceMatrix([[1.0 * (nu_0 - 3), 0.0], [0.0, 1.0 * (nu_0 - 3)]])

# Grille    
x = np.linspace(0, 2, 200)
y = np.linspace(0, 2, 200)
X, Y = np.meshgrid(x, y)

# === Moyenne empirique de N densités DPMM non-informatives ===
Z_sum = np.zeros_like(X)

for _ in range(N):
    mixture = noninformative_dpmm_density(
        alpha=alpha,
        tau=tau,
        mu_0=mu_0,
        lambda_0=lambda_0,
        Psi_0=Psi_0,
        nu_0=nu_0
    )
    Z = f_noninf_density(X, Y)
    Z_sum += Z

Z_mean = Z_sum / N

# ============================= Affichage =============================
fig, ax = plt.subplots(figsize=(7, 6))
plot_density_heatmap(Z_mean, title=f"Moyenne empirique ({N} densités DPMM non-informatives)", ax=ax)
plt.tight_layout()
plt.show()
fig.savefig("figure_MC_f_noninf.png")































#%%
# *******************************************************************************************************
# ******************************************* (EN CHANTIER !) *******************************************
# *******************************************************************************************************

import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# -----------------------------
# GMM avec l'algorithme EM
# -----------------------------
def gmm_em(X, G, mu_init, sigma_init=None, max_iter=100, tol=1e-6):
    n, d = X.shape
    mu = mu_init.copy()
    pi_k = np.full(G, 1.0 / G)

    # Initialisation des matrices de covariance
    if sigma_init is None:
        cov_global = np.cov(X, rowvar=False)
        sigma = np.stack([cov_global.copy() for _ in range(G)])
    else:
        assert sigma_init.shape == (G, d, d)
        sigma = sigma_init.copy()

    loglik = []
    loglik_prev = -np.inf

    for iteration in range(max_iter):
        # ----- E-Step -----
        gamma = np.zeros((n, G))
        for k in range(G):
            mvn = multivariate_normal(mean=mu[k], cov=sigma[k])
            gamma[:, k] = pi_k[k] * mvn.pdf(X)
        gamma /= gamma.sum(axis=1, keepdims=True)

        # ----- M-Step -----
        Nk = gamma.sum(axis=0)
        pi_k = Nk / n

        for k in range(G):
            mu[k] = (gamma[:, k][:, np.newaxis] * X).sum(axis=0) / Nk[k]
            X_centered = X - mu[k]
            sigma_k = (gamma[:, k][:, np.newaxis] * X_centered).T @ X_centered / Nk[k]
            sigma[k] = sigma_k + np.eye(d) * 1e-6  # stabilité numérique

        # ----- Log-vraisemblance -----
        log_prob = np.zeros((n, G))
        for k in range(G):
            mvn = multivariate_normal(mean=mu[k], cov=sigma[k])
            log_prob[:, k] = pi_k[k] * mvn.pdf(X)
        log_likelihood = np.sum(np.log(log_prob.sum(axis=1)))
        loglik.append(log_likelihood)

        if abs(log_likelihood - loglik_prev) < tol:
            break
        loglik_prev = log_likelihood

    return {
        'G': G,
        'pro': pi_k,
        'mean': mu,
        'sigma': sigma,
        'loglik': loglik
    }

# -----------------------------
# Affectation finale aux clusters
# -----------------------------
def assign_clusters(X, result):
    G = result['G']
    probs = np.array([
        result['pro'][k] * multivariate_normal(mean=result['mean'][k], cov=result['sigma'][k]).pdf(X)
        for k in range(G)
    ]).T
    return np.argmax(probs, axis=1)

# -----------------------------
# Génération de données
# -----------------------------
np.random.seed(0)
X1 = np.random.multivariate_normal([0, 0], [[0.05, 0], [0, 0.05]], size=100)
X2 = np.random.multivariate_normal([1, 1], [[0.05, 0], [0, 0.05]], size=100)
X3 = np.random.multivariate_normal([0, 1.5], [[0.05, 0], [0, 0.05]], size=100)
X = np.vstack([X1, X2, X3])

# Initialisation aléatoire des centres
G = 3
d = X.shape[1]
mu_init = X[np.random.choice(X.shape[0], G, replace=False)]

# -----------------------------
# Lancement de l'algorithme EM
# -----------------------------
result = gmm_em(X, G, mu_init)

# -----------------------------
# Visualisation
# -----------------------------
labels = assign_clusters(X, result)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=10)
plt.scatter(result['mean'][:, 0], result['mean'][:, 1], c='red', marker='x', s=100, label='Centres')
plt.title("Clustering GMM (EM)")
plt.legend()
plt.grid(True)
plt.show()




# %%









#%%
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from scipy.stats import multivariate_normal

# 1. Échantillonnage dans un polygone
def sample_points_in_polygon(polygon, n):
    minx, miny, maxx, maxy = polygon.bounds
    points = []
    while len(points) < n:
        candidates = np.random.uniform([minx, miny], [maxx, maxy], size=(n, 2))
        accepted = [Point(p).within(polygon) for p in candidates]
        accepted_points = candidates[accepted]
        points.extend(accepted_points)
    return np.array(points[:n])

# 2. Densité analytique (mélange d'uniformes)
def polygon_density_mixture(xy, polygons, weights):
    density = np.zeros(xy.shape[0])
    for poly, w in zip(polygons, weights):
        inside = np.array([poly.contains(Point(p)) for p in xy])
        density[inside] += w / poly.area
    return density

# 3. Apprentissage GMM avec moyennes et poids fixés
def fit_covariances_with_fixed_centroids(X, G, mu_fixed, pi_fixed, max_iter=100, tol=1e-6):
    n, d = X.shape
    mu = mu_fixed.copy()
    pi_k = pi_fixed.copy()
    sigma = np.stack([np.cov(X, rowvar=False) for _ in range(G)])

    loglik = []
    loglik_prev = -np.inf

    for iteration in range(max_iter):
        gamma = np.zeros((n, G))
        for k in range(G):
            mvn = multivariate_normal(mean=mu[k], cov=sigma[k])
            gamma[:, k] = pi_k[k] * mvn.pdf(X)
        gamma /= gamma.sum(axis=1, keepdims=True)

        for k in range(G):
            Nk = gamma[:, k].sum()
            X_centered = X - mu[k]
            sigma_k = (gamma[:, k][:, np.newaxis] * X_centered).T @ X_centered / Nk
            sigma[k] = sigma_k + np.eye(d) * 1e-6

        log_prob = np.zeros((n, G))
        for k in range(G):
            mvn = multivariate_normal(mean=mu[k], cov=sigma[k])
            log_prob[:, k] = pi_k[k] * mvn.pdf(X)
        log_likelihood = np.sum(np.log(log_prob.sum(axis=1)))
        loglik.append(log_likelihood)

        if abs(log_likelihood - loglik_prev) < tol:
            break
        loglik_prev = log_likelihood

    return {'G': G, 'pro': pi_k, 'mean': mu, 'sigma': sigma, 'loglik': loglik}

# 4. Évaluation de la densité GMM
def evaluate_gmm_density(xy, result):
    density = np.zeros(xy.shape[0])
    for k in range(result['G']):
        mvn = multivariate_normal(mean=result['mean'][k], cov=result['sigma'][k])
        density += result['pro'][k] * mvn.pdf(xy)
    return density

# 5. Définition des 6 zones polygonales
zones = [
    Polygon([(0.2, 0.2), (0.4, 0.3), (0.3, 0.6), (0.1, 0.5)]),
    Polygon([(1.3, 0.4), (1.6, 0.4), (1.5, 0.7), (1.2, 0.6)]),
    Polygon([(0.7, 1.2), (1.3, 1.1), (1.2, 1.7), (0.6, 1.6)]),
    Polygon([(0.8, 0.2), (1.0, 0.3), (0.9, 0.6), (0.7, 0.5)]),
    Polygon([(1.5, 1.3), (1.7, 1.4), (1.6, 1.7), (1.4, 1.6)]),
    Polygon([(0.2, 1.3), (0.4, 1.4), (0.3, 1.7), (0.1, 1.6)])
]
weights = [0.2, 0.15, 0.15, 0.2, 0.15, 0.15]

# 6. Échantillonnage
n_samples = 6000
n_per_zone = np.random.multinomial(n_samples, weights)
X = np.vstack([
    sample_points_in_polygon(polygon, n) for polygon, n in zip(zones, n_per_zone)
])

# 7. Centroïdes et poids
mu_fixed = np.array([np.array(p.centroid.coords[0]) for p in zones])
pi_fixed = np.array(weights)

# 8. Apprentissage
result = fit_covariances_with_fixed_centroids(X, G=6, mu_fixed=mu_fixed, pi_fixed=pi_fixed)

# 9. Grille d’évaluation
grid_x, grid_y = np.meshgrid(np.linspace(0, 2, 200), np.linspace(0, 2, 200))
grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

true_density = polygon_density_mixture(grid_points, zones, weights).reshape(200, 200)
gmm_density = evaluate_gmm_density(grid_points, result).reshape(200, 200)

# 10. Heatmaps comparatives
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

im0 = axs[0].imshow(true_density, origin='lower', extent=(0, 2, 0, 2), cmap='viridis')
axs[0].set_title("Densité réelle (6 zones polygonales)")
plt.colorbar(im0, ax=axs[0])

im1 = axs[1].imshow(gmm_density, origin='lower', extent=(0, 2, 0, 2), cmap='viridis')
axs[1].set_title("Densité GMM approchée")
plt.colorbar(im1, ax=axs[1])

plt.tight_layout()
plt.show()






#%%










#%%












#%%










# %%












#%%


