#%% 
# **************************************************************************************************************************
# ************************************************ AFFICHAGE f0 et f0_tilde ************************************************
# **************************************************************************************************************************

import numpy as np
import matplotlib.pyplot as plt
from dpmm.density import (define_zonage_grid, compute_f0_density, 
                          compute_zone_gaussian_parameters, compute_f0tilde_density)
from dpmm.sampling import sample_from_f0, sample_from_f0tilde
from visualizations.plot_density import plot_density_heatmap
from visualizations.plot_sampling import plot_sampling

# === Paramètres ===
n_rows, n_cols = 2, 2
weights_base = np.array([2.0, 1.0, 0.5, 0.1])
weights_base /= weights_base.sum()
zones, x_bounds, y_bounds = define_zonage_grid(n_rows, n_cols)
area = (x_bounds[1] - x_bounds[0]) * (y_bounds[1] - y_bounds[0])
areas = np.full(len(weights_base), area)
mus, covariances = compute_zone_gaussian_parameters(zones)

# Grille
x = np.linspace(0, 2, 200)
y = np.linspace(0, 2, 200)
X, Y = np.meshgrid(x, y)

Z_f0 = compute_f0_density(X, Y, zones, weights_base, areas)
Z_f0_tilde = compute_f0tilde_density(X, Y, mus, covariances, weights_base)

samples_f0 = sample_from_f0(10000, zones, weights_base, areas)
samples_f0_tilde = sample_from_f0tilde(10000, mus, covariances, weights_base, areas)


# ============================= Affichage =============================
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Densité f0
plot_density_heatmap(Z_f0, title="Densité $f_0$", cmap='gray_r', ax=axs[0, 0])
# Échantillons f0
plot_sampling(samples_f0, title="Échantillons selon $f_0$", ax=axs[0, 1])

# Densité f0_tilde
plot_density_heatmap(Z_f0_tilde, title="Densité $\\tilde{f}_0$", cmap='grey_r', ax=axs[1, 0])
# Échantillons f0_tilde
plot_sampling(samples_f0_tilde, title="Échantillons selon $\\tilde{f}_0$", ax=axs[1, 1])

plt.tight_layout()
plt.show()


# %% 
# *************************************************************************************************************************
# ********************************************** AFFICHAGE f DPMM informatif **********************************************
# *************************************************************************************************************************

import numpy as np
import matplotlib.pyplot as plt
import openturns as ot
from dpmm.density import informative_dpmm_density
from dpmm.sampling import stick_breaking, sample_mixture_niw, sample_from_informative_dpmm
from visualizations.plot_density import plot_density_heatmap
from visualizations.plot_sampling import plot_sampling

# === Paramètres DPMM informatif ===
alpha = 50
tau = 1e-2
means_base = [[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]]
weights_base = np.array([2.0, 1.0, 0.5, 0.1])
weights_base /= weights_base.sum()
lambda_0 = 30.0
nu_0 = 4
Psi_0 = ot.CovarianceMatrix([[0.26, 0.0], [0.0, 0.26]])

f_density = informative_dpmm_density(
    alpha=alpha,
    tau=tau,
    means_base=means_base,
    weights_base=weights_base,
    lambda_0=lambda_0,
    Psi_0=Psi_0,
    nu_0=nu_0
)

# Grille    
x = np.linspace(0, 2, 200)
y = np.linspace(0, 2, 200)
X, Y = np.meshgrid(x, y)

Z_f = f_density(X, Y)
samples_f = sample_from_informative_dpmm(
    n_samples=10000,
    alpha=50,
    tau=1e-2,
    means_base=[[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]],
    weights_base=weights_base,
    lambda_0=30.0,
    Psi_0=ot.CovarianceMatrix([[0.26, 0.0], [0.0, 0.26]]),
    nu_0=4
)


# ============================= Affichage =============================
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

plot_density_heatmap(Z_f, title="Densité du DPMM informatif", ax=axs[0])
plot_sampling(samples_f, title="Échantillons selon $f$", ax=axs[1])

plt.tight_layout()
plt.show()



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
    N=5,
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
    N=5,
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



#%%
# *******************************************************************************************************
# ******************************************* (EN CHANTIER !) *******************************************
# *******************************************************************************************************


lambdas













#%%
# *******************************************************************************************************
# ******************************************* (EN CHANTIER !) *******************************************
# *******************************************************************************************************










