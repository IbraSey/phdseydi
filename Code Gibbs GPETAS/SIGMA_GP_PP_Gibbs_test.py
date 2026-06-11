
from SIGMA_GP_PP_Gibbs import *

#%%
#######################
# Use case definition #
#######################

T = 500
cube = np.array([[0,0], [0,1], [1,1], [1,0], [0,0]])
zone00 = shapely.Polygon(cube*0.5)
zone11 = shapely.Polygon((1+cube)*0.5)
zone0 = shapely.union_all((zone00, zone11))
zone01 = shapely.Polygon(cube*0.5 + np.array([[0.5, 0]]))
zone10 = shapely.Polygon(cube*0.5 + np.array([[0, 0.5]]))
zone1 = shapely.union_all((zone01, zone10))
zones = [zone0, zone1]

PoissonScales = T * np.array([0.5, 0.5]) # lambda parameters for Poisson process in each zone
LambdaTrue = np.array([[0.2],[4.]]) # true zones effects  
LambdaMax = LambdaTrue.max() 

# prior on zone effects
a = 2*PoissonScales 
b = PoissonScales 

# GP model specification (True values)
l = 0.5
l1, l2, nu = l, l, 0.5**2
covarianceModel = ot.SquaredExponential([l1, l2], [np.sqrt(nu)])

m = ot.PythonFunction(2, 1, lambda x:[0])
    
# Upper bound on size of augmented Poisson process
Nmax = 10000#int(ot.Poisson(LambdaMax*T).computeQuantile(1-1e-10)[0])*3

GPscaleFactor = 0.5

gibbs = SSGC_Gibbs(zones, PoissonScales, a, b, Nmax)

#%%
###################
# Data generation #
###################

# Simulate according to homogogeneous Poisson process over search domain
N_star = int( ot.Poisson(LambdaMax*T).getRealization()[0] )
# myUniform = ot.ComposedDistribution([ot.Uniform(0, 1)]*2)

XY_star = gibbs.Uniform.getSample(N_star)
mesh = ot.Mesh(XY_star)

# apply trend function to mesh and create Gaussian process
mTrend = ot.TrendTransform(m, mesh)
Ftot = ot.GaussianProcess(mTrend, covarianceModel, mesh)

# Sigma GP process
field_function = ot.PythonFieldFunction(mesh, 1, mesh, 1, sigmoid)
process = ot.CompositeProcess(field_function, Ftot) 

field_f = process.getRealization()

# Renormalized Intensity
U_star = np.array( gibbs.U_OT(XY_star) )
p_accept = np.array( field_f.getValues() ) * np.dot( U_star, LambdaTrue ) / LambdaMax

# Use thinning    
accepted = np.array( ot.Uniform(0, 1).getSample(N_star) ) <= p_accept 
accepted = accepted.ravel()
N = accepted.sum()
Ntot = N_star
NPi = Ntot - N

# Assemble Augmented (Obs + Latent) Poisson process
# /!\ Zero-padded to reach Nmax length
D = np.array( XY_star )[accepted]

# Plot the data
fig = plt.figure()
plt.scatter( D[:,0], D[:,1])
plt.colorbar()
# plt.show()
plt.savefig("Data.png")
#plt.close()

#%%
#####################
# Perform inference #
#####################

gibbs.setD(D)
samples, randinits = gibbs.run()

#%%
################################
# MCMC Convergence diagnostics #
################################

# components = [j for j in range(paramDim-1-J,paramDim)] 
components = [gibbs.gibbs_indices.Pi_indices[-1]] + gibbs.gibbs_indices.lambda_indices 
names = [r"$N_{tot}$"] + [r"$\lambda_{%s}$"%j for j in range(1,gibbs.J+1)]
true_values = [Ntot] + LambdaTrue.ravel().tolist()

cv_diag_mcmc = ConvergenceDiagnosticsMCMC([sample[:,components] for sample in samples], burnin, names, true_values)    

cv_diag_mcmc.run()

#%%     
###############################################################
# Predict SIGMA-GP with zone effects throughout search domain #
###############################################################

# # Plot Real sparse GP trajectory on meshgrid over search domain
gridsize = 20
xx, yy = np.meshgrid( np.linspace(0, 1, gridsize), np.linspace(0, 1, gridsize) )
XY_new = np.vstack(( xx.ravel(), yy.ravel() )).T


Z_new = np.zeros((len(sample), len(XY_new))) 
U_new = np.array(case.U_OT(XY_new))
intensity_new = np.zeros((len(sample), len(XY_new))) 
M = np.array(case.sparse_gp.regressorOT( XY_new ))
for i in range(len(sample)):
    # break
    sample_i =sample[i]
    # GP conditional on values at augmented Poisson process
    epsilon_i = sample_i[gibbs..epsilon_indices]
    # Ntot_i = sample_i[gibbs_indice.Pi_indices[-1]]
    # Pi_i = np.array(sample_i[gibbs_indice.Pi_indices[:-1]]).reshape(-1,2)
    Z_new[i] = np.dot( M, epsilon_i ).ravel()
    # PyRV_Pi.setParameter(py_link_function_Pi(sample[i], Nmax, J))
    # Z_new[i] = PyRV_Pi.SimulateSigmaGP( XY_new )
    # sigmoid transformation
    sigm_i = np.array(sigmoid(ot.Sample(Z_new[i].reshape(-1,1)))).ravel()
    Lambda_i = (sample_i[gibbs..lambda_indices] * U_new).sum(axis=1)
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

