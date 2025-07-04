#%% 
# *******************************************************************************************************
# ******************************************* (EN CHANTIER !) *******************************************
# *******************************************************************************************************

import numpy as np
import openturns as ot
from dpmm.density import normalized_informative_dpmm_density

alpha = 50
tau = 1e-2
means_base = [[0.5, 0.5], 
              [1.5, 0.5], 
              [0.5, 1.5], 
              [1.5, 1.5]]
weights_base = np.array([2.0, 1.0, 0.5, 0.1])
weights_base /= weights_base.sum()
lambda_0 = 50.0
nu_0 = 4
Psi_0 = ot.CovarianceMatrix([[0.26, 0.00], [0.00, 0.26]])

mixture = normalized_informative_dpmm_density(alpha, tau, means_base, weights_base, lambda_0, nu_0, Psi_0)
print("C'est OK !")


# %% 
# *******************************************************************************************************
# ******************************************* (EN CHANTIER !) *******************************************
# *******************************************************************************************************

import numpy as np
import matplotlib.pyplot as plt
import openturns as ot
from dpmm.density import normalized_informative_dpmm_density
from dpmm.sampling import stick_breaking, sample_mixture_niw

# ---------------- Paramètres du modèle ----------------
alpha = 50
tau = 1e-2
lambda_0 = 30.0
nu_0 = 4
Psi_0 = ot.CovarianceMatrix([[0.26, 0.00], [0.00, 0.26]])

means_base = [
    [0.5, 0.5],
    [1.5, 0.5],
    [0.5, 1.5],
    [1.5, 1.5]
]

weights_base = np.array([2.0, 1.0, 0.5, 0.1])
weights_base /= weights_base.sum()

# ---------------- Construction de la densité DPMM normalisée ----------------
f_density = normalized_informative_dpmm_density(
    alpha=alpha,
    tau=tau,
    means_base=means_base,
    weights_base=weights_base,
    lambda_0=lambda_0,
    nu_0=nu_0,
    Psi_0=Psi_0
)

# ---------------- Évaluation sur une grille ----------------
x_vals = np.linspace(0, 2, 200)
y_vals = np.linspace(0, 2, 200)
X, Y = np.meshgrid(x_vals, y_vals)
Z = f_density(X, Y)

# ---------------- Affichage ----------------
plt.figure(figsize=(8, 6))
plt.imshow(Z, extent=[0, 2, 0, 2], origin='lower', cmap='viridis', aspect='equal')
plt.title("Densité DPMM normalisée")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar(label="Densité")
plt.grid(True)
plt.show()


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
weights = np.array([2.0, 1.0, 0.5, 0.1])
weights /= weights.sum()
zones, x_bounds, y_bounds = define_zonage_grid(n_rows, n_cols)
area = (x_bounds[1] - x_bounds[0]) * (y_bounds[1] - y_bounds[0])
areas = np.full(len(weights), area)
mus, covariances = compute_zone_gaussian_parameters(zones)

# Grille d'évaluation
x = np.linspace(0, 2, 200)
y = np.linspace(0, 2, 200)
X, Y = np.meshgrid(x, y)

Z_f0 = np.vectorize(lambda x, y: compute_f0_density(x, y, zones, weights, areas))(X, Y)
Z_f0_tilde = np.vectorize(lambda x, y: compute_f0tilde_density(x, y, mus, covariances, weights))(X, Y)

samples_f0 = sample_from_f0(10000, zones, weights, areas)
samples_f0_tilde = sample_from_f0tilde(10000, mus, covariances, weights, areas)

# Affichage avec les fonctions utilitaires
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


#%%
# *******************************************************************************************************
# ******************************************* (EN CHANTIER !) *******************************************
# *******************************************************************************************************

# === Paramètres ===





#%%
# *******************************************************************************************************
# ******************************************* (EN CHANTIER !) *******************************************
# *******************************************************************************************************







#%%
# *******************************************************************************************************
# ******************************************* (EN CHANTIER !) *******************************************
# *******************************************************************************************************





