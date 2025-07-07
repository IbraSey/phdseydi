#%% 
# **************************************************************************************************************************
# ************************************************ AFFICHAGE f0 et f0_tilde ************************************************
# **************************************************************************************************************************

import numpy as np
import matplotlib.pyplot as plt
from dpmm.density import (define_zonage_grid, compute_f0_density, compute_f0_density, 
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

Z_f0 = np.vectorize(lambda x, y: compute_f0_density(x, y, zones, weights_base, areas))(X, Y)
Z_f0_tilde = np.vectorize(lambda x, y: compute_f0tilde_density(x, y, mus, covariances, weights_base))(X, Y)

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
from dpmm.density import normalized_informative_dpmm_density
from dpmm.sampling import stick_breaking, sample_mixture_niw, sample_from_informative_dpmm
from visualizations.plot_density import plot_density_heatmap
from visualizations.plot_sampling import plot_sampling

weights_base = np.array([2.0, 1.0, 0.5, 0.1])
weights_base /= weights_base.sum()

f_density = normalized_informative_dpmm_density(
    alpha=50,
    tau=1e-2,
    means_base=[[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]],
    weights_base=weights_base,
    lambda_0=30.0,
    Psi_0=ot.CovarianceMatrix([[0.26, 0.0], [0.0, 0.26]]),
    nu_0=4
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
# *********************************************************************************************************************
# ************************************ AFFICHAGE du Monte Carlo du DPMM informatif ************************************
# *********************************************************************************************************************

import numpy as np
import matplotlib.pyplot as plt
import openturns as ot
from dpmm.density import normalized_informative_dpmm_density
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
    f_density = normalized_informative_dpmm_density(
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
plot_contour_levels(X, Y, Z_mean, levels=20, title="Lignes de niveau", ax=axs[1])

plt.tight_layout()
plt.show()



#%%
# *******************************************************************************************************
# ******************************************* (EN CHANTIER !) *******************************************
# *******************************************************************************************************







#%%
# *******************************************************************************************************
# ******************************************* (EN CHANTIER !) *******************************************
# *******************************************************************************************************






#%%
# *******************************************************************************************************
# ******************************************* (EN CHANTIER !) *******************************************
# *******************************************************************************************************








#%%
# *******************************************************************************************************
# ******************************************* (EN CHANTIER !) *******************************************
# *******************************************************************************************************










#%%
# *******************************************************************************************************
# ******************************************* (EN CHANTIER !) *******************************************
# *******************************************************************************************************
