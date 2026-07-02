#%% Imports

from SIGMA_GP_PP_Gibbs import *

import os
import sys

phebus_path = os.getenv("PHEBUS_PATH")
if not phebus_path or not os.path.isdir(phebus_path):
    raise EnvironmentError(
        "PHEBUS_PATH must be set to the directory containing the phebus package."
    )

sys.path.insert(0, phebus_path)

try:
    import phebus
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        f"Could not import phebus from PHEBUS_PATH={phebus_path!r}. "
        "Ensure that directory contains a `phebus` package."
    ) from exc




# import sys, os
# sys.path.append( os.getenv("PHEBUS_PATH")) # for importing phebus
import phebus
from phebus.pybus.frclass import FrenchDomainsSourceModel

import pandas as pd


#%%
#############
# Load Case #
#############

import os
from phebus.pybus.frclass import FrenchDomainsSourceModel

# phebus_root = "/home/g80884/Documents/phebus"
# phebus_root = os.path.join( phebus_path, "phebus" )
demo_path = os.path.join( phebus_path, "phebus", "demos", "FrenchDomainsAnalysis")

SM = FrenchDomainsSourceModel(
    Mmin=3.,
    PWD=demo_path,
    FILE_DOMAINS=os.path.join(demo_path, "data", "domains", "domaines_xy.csv")
)



catalog = SM.catalog[SM.catalog.year >= 1965]

D = np.vstack((catalog.X, catalog.Y, catalog.magnitude)).T

T = max(catalog.year) - min(catalog.year)

# Domains
zones = [zone.get_polygon_xy() for zone in SM.zones]
areas = np.array([zone.get_area_km2() for zone in SM.zones])
zone_names = [zone.name for zone in SM.zones]

# SM.reduce_catalog(SM.catalog, SM.magnitudes, SM.first_years, SM.last_year)

# props = np.array([zone.reduced_catalog.counts.sum() for zone in SM.zones])

# values = props / (T * areas) * 1e6

# values = np.arange(len(zone_names))**2 + 1

values = areas
SM.plot_values_map( values, FIGURE_PATH=os.getcwd(),  FIGURE_NAME="lambdas_and_domains", catalog=catalog, coastline=SM.coastlines, scale=5., xticks= zone_names)


#%%
####################################
# Define prior and create instance #
####################################

# prior on zone effects
T0 = 0.1*T
lambda0 = 10.
a = T0 * lambda0 / areas
b = T0 / areas

# Upper bound on size of augmented Poisson process
Nmax = 15000#int(ot.Poisson(LambdaMax*T).computeQuantile(1-1e-10)[0])*3

gibbs = SSGC_Gibbs(zones, T, Nmax=Nmax)

# Restrict to data inside domain
select = [gibbs.Domain.contains(shapely.Point(x)) for x in D[:,:2]]

D_select = D[select]

# Check colors on this graph, not coherent with barplot
# SM.plot_values_map( values, FIGURE_PATH=os.getcwd(),  FIGURE_NAME="data_and_domains", catalog=pd.DataFrame(data=D_select, columns=["X", "Y", "magnitude"]), coastline=SM.coastlines, scale=5., xticks= zone_names)
# plt.show()

#%% Build posterior sample
gibbs.setD(D_select[:,:2])
gibbs.calibrate_GP_and_set_prior()

savefile = "posterior_sample.csv"
# Check first if inference has been run
if os.path.exists(savefile):
    print(f"File {savefile} already exists!")
    sample = pd.read_csv(savefile).to_numpy()[:,1:]
else:
    #####################
    # Perform inference #
    #####################
    gibbs.run(sampleSize=300, blockSize=10,ninits=3)
    sample = np.vstack([ s[burnin:] for s in gibbs.samples ])
    pd.DataFrame( data=sample ).to_csv(savefile)
    ############(###################
    # MCMC Convergence diagnostics #
    ################################

    burnin = 15

    # components = [j for j in range(paramDim-1-J,paramDim)] 
    components = [gibbs.gibbs_indices.Pi_indices[-1]] + gibbs.gibbs_indices.lambda_indices 
    names = [r"$N_{tot}$"] + [r"$\lambda_{%s}$"%j for j in range(1,gibbs.J+1)]

    cv_diag_mcmc = ConvergenceDiagnosticsMCMC([sample[:,components] for sample in gibbs.samples], burnin, names )    

    cv_diag_mcmc.run()


#%% seaborn pairplot
import seaborn as sns
import pandas as pd

sns.pairplot( pd.DataFrame( data=sample[:,components], columns=names) )
plt.savefig("pairplot.png")

#%%     
###############################################################
# Predict SIGMA-GP with zone effects throughout search domain #
###############################################################*
gridsize = 500
xx, yy = np.meshgrid( np.linspace(gibbs.lower[0], gibbs.upper[0], gridsize), np.linspace(gibbs.lower[1], gibbs.upper[1], gridsize) )
XY_new = np.vstack(( xx.ravel(), yy.ravel() )).T

Z_new = np.zeros((len(sample), len(XY_new))) 
U_new = np.array(gibbs.U_OT(XY_new))
intensity_new = np.zeros((len(sample), len(XY_new))) 
M = np.array(gibbs.sparse_gp.regressorOT( XY_new ))
for i in range(len(sample)):
    # break
    sample_i =sample[i]
    # sparse GP conditional on augmented Poisson process
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
# levels_mean = np.linspace( intensity_mean.min(), intensity_mean.max(), gridsize )

intensity_std = intensity_new.std(axis=0).reshape(gridsize, gridsize) * T
# levels_std = np.linspace( intensity_std.min(), intensity_std.max(), gridsize)

levels_joint = np.linspace( min(intensity_mean.min(), intensity_std.min()), max(intensity_mean.max(), intensity_std.max()), gridsize)

#%% Plot background intensity posterior mean  

fig = plt.figure(figsize=(10, 10))
# plt.subplot(2,1,1)
# plt.contourf(xx, yy, intensity_mean, levels_mean)
plt.contourf(xx, yy, intensity_mean, levels_joint)
plt.colorbar()
plt.scatter( D_select[:,0], D_select[:,1], s=np.sqrt(D_select[:,2]), c='r', marker='o', alpha=(1./D_select[:,2])/max(1./D_select[:,2]) )
for line in SM.coastlines:
    plt.plot(line[0], line[1], 'w', linewidth = 1.5)
plt.title("Seismic intensity Posterior mean", fontsize=20)
plt.tight_layout()
plt.savefig("intensity_post_mean.png")
#plt.close()


#%% Plot background intensity posterior std
fig = plt.figure(figsize=(10, 10))
# plt.subplot(2,1,2)
plt.contourf(xx, yy, intensity_std, levels_joint)
plt.colorbar()
plt.scatter( D_select[:,0], D_select[:,1], s=np.sqrt(D_select[:,2]), c='r', marker='o', alpha=(1./D_select[:,2])/max(1./D_select[:,2]) )
for line in SM.coastlines:
    plt.plot(line[0], line[1], 'w', linewidth = 1.5)
plt.title("Seismic intensity Posterior std", fontsize=20)
plt.tight_layout()
plt.savefig("intensity_post_mean_std.png")
#plt.close()


# %%
