# %%
# =================================================================================================
# -------------------------------------------- IMPORTS --------------------------------------------
# =================================================================================================
from pathlib import Path
import os
import openturns as ot
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import math
from polyagamma import random_polyagamma
from shapely.geometry import Polygon, Point as ShapelyPoint
from shapely.prepared import prep
# Use package-relative imports for local modules
from .visualizations.plot import plot_field, plot_poisson_zones_data
from .gp.gibbs_sampler import SGCP_GibbsSampler
from .gp.data_generation import generate_data

ot.RandomGenerator.SetSeed(42)


# %%
# ========================================================================================================
# ------------------------------------------- DONNÉES SIMULÉES -------------------------------------------
# ========================================================================================================

def f(x,y):
    """

    """
    #mu1 = ot.Point([0.5, 1.5])
    #mu2 = ot.Point([1.5, 1.5])
    #mu3 = ot.Point([1.5, 1.5])
    mu4 = ot.Point([1.0, 1.0])
    #mu5 = ot.Point([1.5, 0.5])

    #Sigma1 = ot.CovarianceMatrix(2, [0.03, 0.0, 0.0, 0.03])
    #Sigma2 = ot.CovarianceMatrix(2, [0.08, -0.072, -0.072, 0.08])
    #Sigma3 = ot.CovarianceMatrix(2, [0.08, 0.072, 0.072, 0.08])
    #Sigma2 = ot.CovarianceMatrix(2, [0.05, -0.045, -0.045, 0.05])
    #Sigma3 = ot.CovarianceMatrix(2, [0.05, 0.045, 0.045, 0.05])
    Sigma4 = ot.CovarianceMatrix(2, [0.1, 0.00, 0.00, 0.1])
    #Sigma5 = ot.CovarianceMatrix(2, [0.08, 0.002, 0.004, 0.01])
    
    #d1 = ot.Normal(mu1, Sigma1)
    #d2 = ot.Normal(mu2, Sigma2)
    #d3 = ot.Normal(mu3, Sigma3)
    d4 = ot.Normal(mu4, Sigma4)
    #d5 = ot.Normal(mu5, Sigma5)

    collDist = [d4]
    weights = [1.0]
    myMixture = ot.Mixture(collDist, weights)

    data_np = np.column_stack((x, y))
    sample = ot.Sample(data_np)
    pdf_vals = np.array(myMixture.computePDF(sample)).flatten()
    
    return 10.0 * pdf_vals - 3.0


def f1(x,y):
    mu2 = ot.Point([0.33, 1.5])
    Sigma2 = ot.CovarianceMatrix(2, [0.03, 0.027, 0.027, 0.03])
    d2 = ot.Normal(mu2, Sigma2)
    collDist = [d2]
    weights = [1.0]
    myMixture = ot.Mixture(collDist, weights)
    data_np = np.column_stack((x, y))
    sample = ot.Sample(data_np)
    pdf_vals = np.array(myMixture.computePDF(sample)).flatten()
    
    return 10.0 * pdf_vals - 5.0

def f2(x,y):
    mu4 = ot.Point([1.0, 1.0])
    Sigma4 = ot.CovarianceMatrix(2, [0.015, 0.002, 0.002, 0.015])
    d4 = ot.Normal(mu4, Sigma4)
    collDist = [d4]
    weights = [1.0]
    myMixture = ot.Mixture(collDist, weights)
    data_np = np.column_stack((x, y))
    sample = ot.Sample(data_np)
    pdf_vals = np.array(myMixture.computePDF(sample)).flatten()
    
    return 10.0 * pdf_vals - 5.0

def f3(x,y):
    mu3 = ot.Point([1.65, 0.5])
    Sigma3 = ot.CovarianceMatrix(2, [0.03, 0.027, 0.027, 0.03])
    d3 = ot.Normal(mu3, Sigma3)
    collDist = [d3]
    weights = [1.0]
    myMixture = ot.Mixture(collDist, weights)
    data_np = np.column_stack((x, y))
    sample = ot.Sample(data_np)
    pdf_vals = np.array(myMixture.computePDF(sample)).flatten()

    return 10.0 * pdf_vals - 5.0

X_data, zones_gen, Xb, Yb, T_out = generate_data(X_bounds=(0.0, 2.0), Y_bounds=(0.0, 2.0), T=15.0,
    n_cols=3, n_rows=1, mus=10.0, f=[f1, f2, f3], rng_seed=13)

plot_poisson_zones_data(X_data, zones_gen, Xb, Yb, [10.0,10.0,10.0], title="Données simulées", savefigure=False)




# %%
# =========================================================================================================
# --------------------------------------------- Test du Gibbs ---------------------------------------------
# =========================================================================================================

X_bounds = (0.0, 2.0)
Y_bounds = (0.0, 2.0)
T_sim = 15.0   
mus_gen = [10.0,10.0,10.0]
nu_init = [5.0, 0.2]    # Init apprentissage & nu_1 fixé
step_nu_init = 0.01
lambda_nu = 0.5        
delta = [0.1, 0.1]
mutilde_init = 10.0 
jitter = 1e-5
burn_in = 0.3
n_iter = 1000
verbose = True
verbose_every = 100
seed = 13
f = f

X_data, zones_gen, Xb, Yb, T_out = generate_data(X_bounds=X_bounds, Y_bounds=Y_bounds, T=T_sim,
    n_cols=3, n_rows=1, mus=mus_gen, f=[f1, f2, f3], rng_seed=seed)

# Extraction données
N = X_data.getSize()
x_pt = ot.Point([float(X_data[i, 0]) for i in range(N)])
y_pt = ot.Point([float(X_data[i, 1]) for i in range(N)])
t_pt = ot.Point([float(X_data[i, 2]) for i in range(N)])

plot_poisson_zones_data(X_data, zones_gen, Xb, Yb, title="Données simulées")

### Alternative pour apprendre avec zonage différent 
# _, zones_inf, _, _, _ = generate_data(
#     X_bounds, Y_bounds, T_sim, n_cols=1, n_rows=1, mus=10.0, f=f, rng_seed=seed
# )
# polygons = zones_inf
polygons = zones_gen
zones_prep = [prep(p) for p in polygons]
Areas = [(zp, 0.0) for zp in zones_prep]





sampler = SGCP_GibbsSampler(
    X_bounds=Xb, Y_bounds=Yb, T=T_sim, Areas=Areas, polygons=polygons,
    lambda_nu=lambda_nu, nu=nu_init, a_mu=10.0, b_mu=1.0, delta=delta,
    jitter=1e-5, rng_seed=seed
)

eps_init = np.zeros(sampler.J).tolist()


if __name__ == "__main__":

    results = sampler.run(
        t=t_pt, x=x_pt, y=y_pt,
        eps_init=eps_init, mutilde_init=mutilde_init,
        n_iter=n_iter, 
        step_nu_init=step_nu_init,
        verbose=verbose, verbose_every=verbose_every
    )

    print("\n" + "="*40)
    print(f"Taux d'acceptation MH pour \nu : {results['acceptance_nu']*100}%")
    print("="*40)

    # Analyse chaines
    sampler.plot_chains(results)
    sampler.plot_acf(
        results,
        burn_in=0.3,
        max_lag=50
    )
    ess_vals = sampler.plot_ess_arviz(results, burn_in=0.3, figsize=None)
    print("ESS :")
    for k, v in ess_vals.items():
        print(f"{k:8s} : {v:.1f}")


    # Posterior intensity plot
    out = sampler.plot_posterior_intensity(
        x=np.asarray(x_pt, dtype=float),
        y=np.asarray(y_pt, dtype=float),
        t=np.asarray(t_pt, dtype=float),
        results=results,
        nx=80,
        ny=80,
        burn_in=burn_in
    )

    post = sampler.posterior_summary(results, burn_in=burn_in)
    print("\n--- Posterior summary ---")
    print(f"mutilde_hat = {post['mutilde_hat']}")
    print(f"eps_hat     = {np.asarray(post['eps_hat'])}")
    print(f"mean mu_hat      = {out['mu_hat'].mean()}")
    print(f"mean squared_mu_hat      = {out['squared_mu_hat'].mean()}")
    print(f"mean nu_hat      = {post['nu_hat']}")

    # Calcul biais et variance
    sq_e = (out['mu_hat'].mean())**2
    print(f"Biais pour le moddèle avec prior informatif : {out['mu_hat'].mean() - mus_gen}")
    print(f"Variance pour le moddèle avec prior informatif : {out['squared_mu_hat'].mean() - sq_e}")





    # %%






    # %%








    # %%









    # %%






