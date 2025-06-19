#%%
import openturns as ot
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  
import pandas as pd
ot.RandomGenerator.SetSeed(42)



# =============================================================================================================
# ============================================== Stick Breacking ==============================================
# =============================================================================================================
def dp_by_stick_breaking(alpha=1.0, tau=1e-3, base_measure_sampler=None):
    weights = []
    thetas = []
    r = 1.0 

    beta_dist = ot.Beta(1.0, alpha, 0.0, 1.0) 
    while r > tau:
        v_k = beta_dist.getRealization()[0]
        w_k = v_k * r
        weights.append(w_k)
        r *= (1.0 - v_k)

        theta_k = base_measure_sampler.getRealization()[0]
        thetas.append(theta_k)

    return weights, thetas

dim = 2
def sample_from_normal(d=1):
    return ot.Normal(d)

weights, thetas = dp_by_stick_breaking(alpha=1.0, tau=1e-3, base_measure_sampler=sample_from_normal(dim))

for i, (w, theta) in enumerate(zip(weights, thetas)):
    print(f"Composante {i} : poids = {w} ; theta = {theta}")

print("Nombre de composantes :", len(weights))
print("Somme des poid (proche de 1):", sum(weights))



# =============================================================================================================
# =============================== DPMM avec prior informatif (mélange de 4 NIW) ===============================
# =============================================================================================================
# %%
dim = 2
n_samples = 2000
alpha = 15.0
tau = 1e-2  # Seuil pour arrêt du SB

means_base = [
    [0.5, 0.5],
    [1.5, 0.5],
    [0.5, 1.5],
    [1.5, 1.5]
]
weights_base = np.array([2.0, 1.0, 0.5, 0.1])
weights_base /= weights_base.sum()

lambda_0 = 4.0
nu_0 = 5
Psi_0 = ot.CovarianceMatrix([[0.03, 0.00], [0.00, 0.01]])

def sample_mixture_niw():
    base_idx = np.random.choice(len(means_base), p=weights_base)
    mu_0 = ot.Point(means_base[base_idx])

    Sigma = ot.InverseWishart(Psi_0, nu_0).getRealizationAsMatrix()
    mu = ot.Normal(mu_0, ot.CovarianceMatrix(Sigma / lambda_0)).getRealization()

    return mu, Sigma

def stick_breaking(alpha, tau=1e-3):
    weights = []
    r = 1.0
    while r > tau:
        v = ot.Beta(1.0, alpha, 0.0, 1.0).getRealization()[0]
        w = v * r
        weights.append(w)
        r *= (1 - v)
    print(weights)
    return np.array(weights) / np.sum(weights)

def sample_dpmm(n_samples=1000):
    weights = stick_breaking(alpha, tau)
    print(weights.sum())
    n_prior = len(weights)
    prior = [sample_mixture_niw() for _ in range(n_prior)]

    data = []
    for _ in range(n_samples):
        k = np.random.choice(n_prior, p=weights)
        mu_k, sigma_k = prior[k]
        point = ot.Normal(mu_k, sigma_k).getRealization()
        data.append(list(point))

    return np.array(data)

# ===== Visualisation ====
samples = sample_dpmm(n_samples)

plt.figure(figsize=(6, 6))
plt.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.5)
plt.xlim(0, 2)
plt.ylim(0, 2)
plt.title("Simu DPMM avec prior informatif")
plt.grid(True)
plt.gca().set_aspect('equal')
plt.show()



# =============================================================================================================
# =========================== Étude caractéristiques prior Normale Inverse Wishart ============================
# =============================================================================================================
# %%
dim = 2  # dimension
n_sample = 500

nu_0 = dim + 5  # degrés de liberté (> dim)
Psi_0 = ot.CovarianceMatrix([[1.0, 0.3], [0.3, 1.0]])  # matrice d’échelle (dim x dim)
mu_0 = ot.Point([0.0, 0.0])  # moyenne de la normale conditionnelle
lambda_0 = 1.0  # "force" de la moyenne

Sigma = ot.InverseWishart(Psi_0, nu_0).getRealizationAsMatrix()
mu = ot.Normal(mu_0, ot.CovarianceMatrix(Sigma / lambda_0)).getRealization()

sample = ot.Normal(mu, Sigma).getSample(n_sample)

# Affichage
plt.figure(figsize=(7, 5))
plt.scatter(sample[:, 0], sample[:, 1], alpha=0.5, label='Réalisations')
plt.scatter(mu[0], mu[1], c='red', label='Moyenne', marker='x', s=100)
plt.title('Réalisations d’une loi normale avec paramètres tirés de la NIW')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()









# %%
