
from SIGMA_GP_PP_Gibbs import *

import pandas as pd

#%%
#############
# Load Data #
#############

data_path = os.path.join( os.pardir, "use_case" )

# Seismic catalog
catalog = pd.read_csv( os.path.join( data_path, "catalog.csv" ) )
# restrict to complete observation period
catalog = catalog[catalog.year >= 1962]
catalog = catalog[catalog.magnitude >= 3.0]
D = np.vstack([catalog.longitude, catalog.latitude]).T


# Plot the data
fig = plt.figure()
plt.scatter( D[:,0], D[:,1])
# plt.show()
plt.savefig("Data.png")
#plt.close()

T = max(catalog.year) - min(catalog.year)
# Domains
domains = pd.read_csv( os.path.join( data_path, "domaines_xy.csv" ) )
domain_names = domains['CODE_GTR'].unique()
domain_polygons = []
for name in domain_names:
    coords = domains[['X', 'Y']].loc[domains['CODE_GTR']==name]
    polygon = shapely.geometry.Polygon(coords.values)
    domain_polygons.append(polygon)

# merge into the six final domains (which we name zones)
domain_short_names = np.array([name[:3] for name in domain_names])
zones = []

unique, inverse = np.unique(domain_short_names, return_inverse=True)

for i in range(len(unique)):
    # break
    index = np.argwhere(inverse == i).ravel()
    zone = shapely.union_all([domain_polygons[i] for i in index])
    zones.append(zone)

areas = np.array([zone.area for zone in zones])

#%%
####################################
# Define prior and create instance #
####################################

# prior on zone effects
T0 = 0.1*T
lambda0 = 10.
a = T0 * areas * lambda0
b = T0 * areas

# Upper bound on size of augmented Poisson process
Nmax = 10000#int(ot.Poisson(LambdaMax*T).computeQuantile(1-1e-10)[0])*3

gibbs = SSGC_Gibbs(zones, T, a, b, Nmax)

#%%
#####################
# Perform inference #
#####################

gibbs.setD(D)
gibbs.run(sampleSize=50, blockSize=10,ninits=3)

#%%
############(####################
# MCMC Convergence diagnostics #
################################

burnin = 5

# components = [j for j in range(paramDim-1-J,paramDim)] 
components = [gibbs.gibbs_indices.Pi_indices[-1]] + gibbs.gibbs_indices.lambda_indices 
names = [r"$N_{tot}$"] + [r"$\lambda_{%s}$"%j for j in range(1,gibbs.J+1)]
true_values = [Ntot] + LambdaTrue.ravel().tolist()

cv_diag_mcmc = ConvergenceDiagnosticsMCMC([sample[:,components] for sample in gibbs.samples], burnin, names, true_values)    

cv_diag_mcmc.run()

#%%     
###############################################################
# Predict SIGMA-GP with zone effects throughout search domain #
###############################################################

sample = np.vstack([ s[burnin:] for s in gibbs.samples ])

import seaborn as sns
import pandas as pd

sns.pairplot( pd.DataFrame( data=sample[:,components], columns=names) )
plt.savefig("pairplot.png")

# # Plot Real sparse GP trajectory on meshgrid over search domain
gridsize = 20
xx, yy = np.meshgrid( np.linspace(0, 1, gridsize), np.linspace(0, 1, gridsize) )
XY_new = np.vstack(( xx.ravel(), yy.ravel() )).T

Z_new = np.zeros((len(sample), len(XY_new))) 
U_new = np.array(gibbs.U_OT(XY_new))
intensity_new = np.zeros((len(sample), len(XY_new))) 
M = np.array(gibbs.sparse_gp.regressorOT( XY_new ))
for i in range(len(sample)):
    # break
    sample_i =sample[i]
    # GP conditional on values at augmented Poisson process
    epsilon_i = sample_i[gibbs.gibbs_indices.epsilon_indices]
    # Ntot_i = sample_i[gibbs_indice.Pi_indices[-1]]
    # Pi_i = np.array(sample_i[gibbs_indice.Pi_indices[:-1]]).reshape(-1,2)
    Z_new[i] = np.dot( M, epsilon_i ).ravel()
    # PyRV_Pi.setParameter(py_link_function_Pi(sample[i], Nmax, J))
    # Z_new[i] = PyRV_Pi.SimulateSigmaGP( XY_new )
    # sigmoid transformation
    sigm_i = np.array(sigmoid(ot.Sample(Z_new[i].reshape(-1,1)))).ravel()
    Lambda_i = (sample_i[gibbs.gibbs_indices.lambda_indices] * U_new).sum(axis=1)
    intensity_new[i] = sigm_i * Lambda_i
    
intensity_mean = intensity_new.mean(axis=0).reshape(gridsize, gridsize) * T
levels_mean = np.linspace( intensity_mean.min(), intensity_mean.max(), gridsize )

intensity_std = intensity_new.std(axis=0).reshape(gridsize, gridsize) * T
levels_std = np.linspace( intensity_std.min(), intensity_std.max(), gridsize)

fig = plt.figure()
plt.contourf(xx, yy, intensity_mean, levels_mean)
plt.colorbar()
plt.scatter( D[:,0], D[:,1], s=100, c='r', marker='+' )
plt.title("Poisson intensity posterior mean vs Data")
plt.savefig("intensity_post_mean.png")
#plt.close()

fig = plt.figure()
plt.contourf(xx, yy, intensity_std, levels_std)
plt.colorbar()
plt.scatter( D[:,0], D[:,1], s=100, c='r', marker='+' )
plt.title("Poisson intensity Posterior std vs Data")
plt.savefig("intensity_post_std.png")
#plt.close()


# %%
