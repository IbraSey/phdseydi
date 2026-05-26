#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 23:12:13 2025

@author: H01971
"""
#%%

##########################
#    Necessary imports   #
##########################

import openturns as ot
import openturns.experimental as otexp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
import scipy.stats as st
import statsmodels.tsa.stattools as stattools
from polyagamma import random_polyagamma
from sparseGP import sparseGP

ot.RandomGenerator.SetSeed(0) # Make results reproducible by freezing Open TURNS's random generator's seed
np.random.seed(0) # Make results reproducible by freezing Numpy's random generator's seed

sigmoid = ot.SymbolicFunction(['z'], ['1/(1+exp(-z))'])

sigmoid_inv = ot.SymbolicFunction(['q'], ['ln( q/(1-q) )'])


#%%

######################################
# Computing indices for Gibbs blocks # 
######################################

class GibbsIndices:    
    def __init__(self, m, Nmax, N, J):
        self.m = m
        self.Nmax = Nmax
        self.N = N
        self.J = J
        self.epsilon_indices = list(range(m))
        self.Omega_indices = list(range(m, m + Nmax))
        self.Pi_indices = list(range(m + Nmax, m + 3*Nmax - 2*N + 1))
        self.lambda_indices = list(range(m + 3*Nmax - 2*N + 1, m + 3*Nmax - 2*N + 1 + J))
        self.chain_dim = m + 3*Nmax - 2*N + 1 + J

###################################
# Basic Gaussian simulation class # 
###################################

class NormalCholesky(ot.PythonRandomVector):
    """
    Generate normal vector given cholesky decomposition of precision matrix
    """
    def __init__(self, mu, Chol, Ntot):
        """
        Parameters
        ----------
        mu: array
            expected value
        Chol: TriangularMatrix
            Cholesky decomposition of 
            the precision matrix
        Ntot: int
            Total size of random vector
        
        Notes: 
            - Nmax := len(mu) must be >= Ntot
            - /!\ realizations are zero-padded to reach size Nmax
            - total parameter size: Nmax*(Nmax+1)+1
            - these are flattened and concatenated in a single list,
              in the above order
        """
        Nmax = len(np.array(Chol))
        if len( np.array(mu).ravel() ) != Nmax or Ntot > Nmax:
            print("Incompatible dimensions for mean, Cholesky and / or Nmax")
            raise ValueError
        super(NormalCholesky, self).__init__(Nmax)
        self.mu = np.array(mu).reshape(-1,1)
        self.Chol = ot.Matrix(Chol)
        self.Ntot = int(Ntot)
        self.Nmax = Nmax

    def setParameter(self, parameter):
        """

        Parameters
        ----------
        parameter : list
            parameter values
            Size : Nmax*(Nmax+1)+1

        Returns
        -------
        self.mu, self.Chol and self.Ntot            
        
        """
        Nmax=int(self.Nmax)
        self.Ntot = int(parameter[-1])
        Ntot=int(self.Ntot)
        self.mu = np.array(parameter[:Nmax]).reshape(-1,1)
        self.Chol = np.zeros((Nmax, Nmax))
        self.Chol[:Ntot,:Ntot] = np.array(parameter[Nmax:Nmax+Ntot*Ntot]).reshape(Ntot, Ntot)
    
    def getParameter(self):
        """
        Returns
        -------
        parameter : list
            Current parameter values
            Size : Nmax*(Nmax+1)+1

        """
        Nmax=int(self.Nmax)
        Ntot=int(self.Ntot)
        parameter = [0]*int(Nmax*(Nmax+1)+1)
        parameter[:Ntot] = self.mu[:Ntot].ravel()
        parameter[Nmax:Nmax+Ntot*Ntot] = np.array( self.Chol[:Ntot,:Ntot] ).ravel()
        parameter[-1]=Ntot
        return parameter
    
    def getRealization(self):
        """
        Simulates one realization of the Gaussian vector
        with expected value self.mu and precision matrix
        Cholesky decomposition given by self.Chol

        Returns
        -------
        array
            Simulated Gaussian vector value
            Size: Nmax
        """
        Nmax=int(self.Nmax)
        Ntot=int(self.Ntot)
        Z = ot.Normal().getSample(Ntot)
        # output = np.zeros(Nmax)
        output = [0]*Nmax
        # indices = [i for i in range(int(self.Ntot))]
        output[:Ntot] = np.array(ot.Matrix(self.Chol[:Ntot,:Ntot])*Z + self.mu[:Ntot]).ravel()
        # output[indices] = np.array(self.Chol[indices][:,indices]*Z + self.mu[indices]).ravel()
        return output


##################################
# Latent Gaussian process update # 
##################################


def py_link_function_eps(x, sparse_gp, gibbs_indices, regressorD):
    """
    Given the current state of the MCMC chain,
    output parameters of the conditional density of
    the truncated GP, as required by the NormalCholesky class

    Parameters
    ----------
    x : array / list
        Current MCMC chain state
    sparse_gp : sparseGP
        class providing sparse GP design matrix
    gibbs_indices : GibbsIndices
        provides parameter indices within Markov chain
    regressorD : (N, m)
        regressor functions evaluated over dataset
        must be pre-computed using sparseGP.regressorOT()
        
    Returns
    -------
    param : list
        Mean + Cholesky precision matrix + Ntot value
        in the order required by NormalCholesky class
        Size : m*(m+1)+1
        with m the sparse GP truncation order 
    
    """
    # Check that gibbs_indices has same "m" attribute as sparse_gp
    if gibbs_indices.m != sparse_gp.m:
        raise ValueError
    m = sparse_gp.m
    # Extract current state of conditioning variables
    N = gibbs_indices.N
    Ntot = int(x[gibbs_indices.Pi_indices[-1]])
    Pi = np.array(x[gibbs_indices.Pi_indices[:2*(Ntot-N)]]).reshape(-1,2)
    Omega = np.array(x[gibbs_indices.Omega_indices[:Ntot]]).reshape(-1,1)
    # # total (augmented) data
    # Dtot = ot.Sample(Ntot, 2)
    # Dtot[:N] = D
    # Dtot[N:] = Pi
    u = ot.Sample(np.array([[0.5]]*N + [[-0.5]]*(Ntot-N)))
    M = np.vstack([ regressorD, sparse_gp.regressorOT( Pi ) ])
    # precision matrix
    Q = np.eye(m) + np.dot( M.T, Omega*M )
    # invert 
    K = ot.CovarianceMatrix(Q).computeCholesky()
    Kinv = K.inverse()
    V = Kinv.transpose()*Kinv
    # # posterior mean 
    mean = V*ot.Matrix(M.T)*u
    # extract parameters in correct order (coherent with getParameter() method of RV_epsilon)
    parameter = [0]*( m*(m+1)+1 )
    parameter[:m] = np.array(mean).ravel()
    parameter[m:m*(m+1)] = np.array(Kinv).ravel()
    parameter[-1] = m
    return parameter


#%%

#########################
# Latent Poisson update # 
#########################

# Uniform_RV = ot.RandomVector(ot.Uniform())

class PoissonProcess(ot.PythonRandomVector):
    """
    Given current states of epsilon
    Generates an updated set of latent points 
    """
    def __init__( self, epsilon, Lambda, U, PoissonScale, Uniform, sparse_gp, gibbs_indices):
        """
        Parameters
        ----------
        epsilon : vector
            sparse GP coefficients 
            Size: m
        Lambda : (J,1)
            current value of zones effects
        U : Open TURNS function
            zone indicator functions
            given a point (x,y), outputs J 0-1 indicators,
            summing to 1
        PoissonScale: Scale factor for homogeneous Poisson distribution
            This is equal to observation period T times search domain area
        Uniform : OpenTURNS distribution
            uniform distribution over the search domain
        sparse_gp : (sparseGP)
            sparse GP design matrix calculation
        gibbs_indices : GibbsIndices
            provides parameter indices within Markov chain
        
        Notes
        -----
        - epsilon and Lambda are flattened and concatenated in above order
        - parameter list size: m + J
        - Simulated variables dimension: 2*(Nmax-N)+1 (Pi + Ntot)
        - Nmax must be >= Ntot (or else a ValueError is raised)
        - when Nmax > Ntot, realizations are zero-padded
        """
        Nmax = gibbs_indices.Nmax
        N = gibbs_indices.N
        super(PoissonProcess, self).__init__(int(2*(Nmax-N))+1)
        # Internal parameters (numpy arrays)
        self.epsilon = np.array(epsilon).reshape(-1,1)
        self.Lambda = Lambda
        self.J = len(Lambda)
        self.m = sparse_gp.m
        if self.m != gibbs_indices.m:
            print("sparseGP and gibbsIndices objects have different m values")
            raise ValueError
        self.U = U
        self.PoissonScale = PoissonScale
        self.Uniform = Uniform
        self.sparse_gp = sparse_gp
        self.gibbs_indices = gibbs_indices
    
    def setParameter(self, parameter):
        """

        Parameters
        ----------
        parameter : list
            concatenates current values of :
                - epsilon (m,)
                - Lambda (J,)
            in this order
            Size: m+J

        Returns
        -------
        Sets internal parameters (numpy arrays)

        """
        self.epsilon = np.array(parameter[:self.m])
        self.Lambda = np.array(parameter[-self.J:])
    
    def getParameter(self):
        # Nmax=int(self.gibbs_indices.Nmax)
        # Ntot=int(self.Ntot)
        m = self.m
        parameter = np.zeros(m + self.gibbs_indices.J)
        parameter[:m] = self.epsilon.ravel()
        parameter[-self.J:] = self.Lambda.ravel()
        return parameter.tolist()
        
        # print(gpr_result)
    def getRealization(self):
        """
        Simulates one realization of the latent Poisson process

        Returns
        -------
        list
            simulated variables are flattened 
            and concatenated in the following order:
            - New Poisson process values (shape : 2*(Nmax-N))
            - New Ntot value (size: 1)
            total size : 2*Nmax-2*N+1
        
        Notes:
            - There is no guaranty that New Ntot <= Nmax
            - New Ntot > Nmax may cause a crash
        """
        Nmax=int(self.gibbs_indices.Nmax)
        N = self.gibbs_indices.N
        # Step 2: Generate candidate points uniformly over search domain
        LambdaMax = self.Lambda.max()
        N_star = int(ot.Poisson(self.PoissonScale*LambdaMax).getRealization()[0]) # Poisson candidate number
        XY_star =  self.Uniform.getSample(N_star) # Uniformly sampled candidates
        # Step 3: Simulate GP trajectories
        M = sparse_gp.regressorOT(XY_star)
        epsilon = np.array( ot.Normal(self.m).getRealization() ).reshape(-1,1)
        f_star = np.dot( M, epsilon )
        # Step 4: Thinning
        U_star = np.array( self.U(XY_star) )
        Lambda_star = np.dot( U_star, self.Lambda).reshape(-1,1)
        p_accept = np.array( sigmoid(-f_star.reshape(-1,1)) * Lambda_star / LambdaMax )
        accept = np.array( ot.Uniform().getSample(N_star) ) <= p_accept 
        NPi_new = np.array(accept).sum()
        Ntot_new = N + NPi_new
        if Ntot_new > Nmax:
            print("Maximum size %s exceeded by simulated data size %s"%(Nmax, Ntot_new))
            raise ValueError
        # Assemble final output
        results = np.zeros(2*Nmax-2*N+1)
        results[-1] = Ntot_new
        results[:2*(Ntot_new-N)] = np.array(XY_star)[accept.ravel()].ravel()
        return results

def py_link_function_Pi(x, gibbs_indices):
    """
    Given the current state of the MCMC chain,
    output parameters of the conditional density of
    the Latent Poisson process, as required by the 
    PoissonProcess class.

    Parameters
    ----------
    x : array / list
        Current MCMC chain state
    gibbs_indices : GibbsIndices
        provides parameter indices within Markov chain
    
    Returns
    -------
    param : list
        in the order required by 
        the PoissonProcess class
        Size : m+J)
    """
    epsilon = np.array(x)[:gibbs_indices.m]
    Lambda = np.array(x)[-gibbs_indices.J:]
    return np.concatenate([epsilon, Lambda])

#%%

#############################
# Latent Polya-Gamma update # 
#############################

class PolyaGammaProcess(ot.PythonRandomVector):
    """
    Given current states of MCMC chain 
    Generates an updated set of Polya-Gamma values
    """
    def __init__( self, epsilon, Pi, Ntot, gibbs_indices, sparse_gp, regressorD ):
        """
        Parameters
        ----------
        epsilon : vector
            sparse GP coefficients 
            Size: m
        Pi : (Nmax, 2) array
            latent PP
        Ntot : int
            current value of total data size
        gibbs_indices : GibbsIndices
            provides parameter indices within Markov chain
        sparse_gp : sparseGP
            provides sparse GP approximation
        regressorD : (N, m)
            regressor functions evaluated over dataset,
            must be pre-computed using sparseGP.regressorOT() 
        """
        # Check that gibbs_indices has same "m" attribute as sparse_gp
        if gibbs_indices.m != sparse_gp.m:
            raise ValueError
        self.m = sparse_gp.m
        super(PolyaGammaProcess, self).__init__(gibbs_indices.Nmax)
        self.epsilon = epsilon.reshape(-1,1)
        self.Pi = Pi
        self.Ntot = Ntot
        self.gibbs_indices = gibbs_indices
        self.sparse_gp = sparse_gp
        self.regressorD = regressorD
    
    def setParameter(self, parameter):
        m = self.sparse_gp.m
        Nmax = self.gibbs_indices.Nmax
        N = self.gibbs_indices.N
        self.epsilon = np.array(parameter[:m]).reshape(-1,1)
        self.Pi = np.array(parameter[m:m+2*(Nmax-N)]).reshape(-1,2)
        self.Ntot = parameter[-1]
    
    def getParameter(self):
        return np.concatenate([self.epsilon.ravel(), self.Pi.ravel(), [self.Ntot]])
    
    def getRealization(self):
        """
        Simulates one realization of the latent Polya-Gamma process

        Returns
        -------
        list
            New Polya-Gamma process values
            Size : Nmax
        gibbs_indices : GibbsIndices
            provides parameter indices within Markov chain
        """     
        Nmax = self.gibbs_indices.Nmax
        Ntot = int(self.Ntot)
        w = np.zeros(Nmax)
        M = np.vstack([ regressorD, sparse_gp.regressorOT( self.Pi[:Ntot-N] ) ])
        ftot = np.dot( M, self.epsilon )
        w[:Ntot] = np.abs( random_polyagamma(z=np.array(ftot)[:,0]) )
        return w

def py_link_function_w(x, gibbs_indices):
    """
    Given the current state of the MCMC chain,
    output parameters of the conditional Polya
    Gamma process, as required by the 
    PolyaGammaProcess class.

    Parameters
    ----------
    x : array / list
        Current MCMC chain state

    Returns
    -------
    param : list
        epsilon, Pi and Ntot values
        as required by 
        the PolyaGammaProcess class
        Size : m+2*(Nmax-N)+1
    """
    epsilon = np.array(x)[gibbs_indices.epsilon_indices]
    Ntot = np.array(x)[gibbs_indices.Pi_indices[-1]]
    Pi = np.array(x)[gibbs_indices.Pi_indices[:-1]]
    return np.hstack(( epsilon, Pi, [Ntot] ))

#%%

###############################
# Latent zones effects update # 
###############################


def py_link_function_Lambda(x, D, U, PoissonScales, a, b, gibbs_indices):
    """
    Given the current state of the MCMC chain,
    output parameters of the conditional density of
    Lambda, as required by the RV_Lambda class

    Parameters
    ----------
    x : array / list
        Current MCMC chain state
        Size : 4*Nmax - 2*N + J + 1
    D : (N,2)
        Observed Poisson process
    U : Open TURNS function
        zone indicator functions
        given a point (x,y), outputs J 0-1 indicators,
        summing to 1
    PoissonScales: (J,)
        list of J scale factors for Poisson distribution
    a : (J,)
        prior shape parameters for Gamma distribution of Lambdas
    b : (J,)
        prior rate parameters for Gamma distribution of Lambdas
    gibbs_indices : GibbsIndices
        provides parameter indices within Markov chain

    Returns
    -------
    param : list
        posterior shape and rate parameters for Gamma distribution of Lambdas
        in the order required by the RV_Lambdas class
        Size : J*3
    """
    # Extract current state of conditioning variables
    Nmax = gibbs_indices.Nmax
    N = gibbs_indices.N
    J = gibbs_indices.J
    Ntot = int(x[gibbs_indices.Pi_indices[-1]])
    Pi = np.array(x[gibbs_indices.Pi_indices[:2*(Ntot-N)]]).reshape(-1,2)
    # total (augmented) data
    Dtot = ot.Sample(Ntot, 2)
    Dtot[:N] = D
    Dtot[N:] = Pi
    # Compute number of points in each zone
    U_Dtot = np.array(U_OT(Dtot))
    Nk = U_Dtot.sum(axis=0).reshape(-1,1)
    # extract parameters in correct order (coherent with getParameter() method of RV_Lambdas)
    parameter = np.zeros( J*3 )
    parameter[::3] = a + Nk.ravel() # shape parameters
    parameter[1::3] = b + PoissonScales  # rate parameters
    return parameter


#%%

if __name__ == "__main__":


    #%%
    ####################
    # Generative model #
    ####################

    # Assuming square domain [0,1]*[0,1] (surface 1)
    # and null trend

    T = 50

    def U(xy):
        u = [0, 0]
        u[0] = (xy[0]>0.5)*(xy[1]>0.5) + (xy[0]<=0.5)*(xy[1]<=0.5)
        u[1] = 1 - u[0]
        return u

    U_OT = ot.PythonFunction( 2, 2, U )
    J = 2
    PoissonScales = T * np.array([0.5, 0.5]) # lambda parameters for Poisson process in each zone
    LambdaTrue = np.array([[0.2],[4.]]) # true zones effects   

    # prior on zone effects
    a = PoissonScales * 1.
    b = PoissonScales

    # GP model specification
    l1, l2, nu = 0.1, 0.1, 0.5**2
    covarianceModel = ot.SquaredExponential([l1, l2], [np.sqrt(nu)])
    m = ot.PythonFunction(2, 1, lambda x:[0])
        
    LambdaMax = LambdaTrue.max()
    # Upper bound on size of augmented Poisson process
    Nmax = int(ot.Poisson(LambdaMax*T).computeQuantile(1-1e-6)[0])*2
    # this is a very crude upper bound, but it allows to avoid crashes due to size issues during MCMC updates. 
    # It can be set to a lower value to speed up computations, at the risk of crashes.    

    #%%


    ###################
    # Data generation #
    ###################

    # Simulate according to homogogeneous Poisson process over search domain
    N_star = int( ot.Poisson(LambdaMax*T).getRealization()[0] )
    myUniform = ot.ComposedDistribution([ot.Uniform(0, 1)]*2)

    XY_star = myUniform.getSample(N_star)
    mesh = ot.Mesh(XY_star)

    # apply trend function to mesh and create Gaussian process
    mTrend = ot.TrendTransform(m, mesh)
    Ftot = ot.GaussianProcess(mTrend, covarianceModel, mesh)

    # Sigma GP process
    field_function = ot.PythonFieldFunction(mesh, 1, mesh, 1, sigmoid)
    process = ot.CompositeProcess(field_function, Ftot) 

    field_f = process.getRealization()
        
    # Use thinning
    U_star = np.array( U_OT(XY_star) )
    p_accept = np.array( field_f.getValues() ) * np.dot( U_star, LambdaTrue ) / LambdaMax
    accepted = np.array( ot.Uniform(0, 1).getSample(N_star) ) <= p_accept 
    accepted = accepted.ravel()
    N = accepted.sum()
    Ntot = len(accepted)
    NPi = Ntot - N

    # Assemble Augmented (Obs + Latent) Poisson process
    # /!\ Zero-padded to reach Nmax length
    D = np.array( XY_star )[accepted]
    Pi = np.array( XY_star )[accepted==False]

    Dtot = np.vstack((D,Pi,[[0,0]]*(Nmax-Ntot)))

    # Assemble Augmented (Obs + Latent) Gaussian process
    # /!\ Zero-padded to reach Nmax length
    fD = np.array(field_f)[accepted]
    fPi = np.array(field_f)[accepted==False]
    ftot = np.vstack((fD,fPi,[[0]]*(Nmax-Ntot)))

    #%%


    #######################
    # TEST ON TOY DATASET #
    #######################

    # Plot the data
    fig = plt.figure()
    plt.scatter( D[:,0], D[:,1])
    plt.colorbar()
    # plt.show()
    plt.savefig("Data.png")
    #plt.close()

    ###################
    # MCMC parameters # 
    ###################

    sampleSize=1000
    blockSize=50 # Display convergence messages after every block of iterations with size: blockSize
    ninits = 3 # Number of chains run for Gelman-Rubin convergence diagnostic
    
    c1, c2, S1, S2 = 0.5, 0.5, 0.5, 0.5
    
    hypers = l1, l2, c1, c2, S1, S2, nu
    sparse_gp = sparseGP(hypers)
    regressorD = sparse_gp.regressorOT(D)
    
    gibbs_indices = GibbsIndices(sparse_gp.m, Nmax, N, J)
    
    # Augmented Gaussian Process update
    RV_eps = ot.RandomVector(NormalCholesky(mu=np.zeros(sparse_gp.m), Chol=np.diag([1]*sparse_gp.m), Ntot=sparse_gp.m))
    ot_link_function_eps = ot.PythonFunction(gibbs_indices.chain_dim, sparse_gp.m*(sparse_gp.m+1)+1, lambda x:py_link_function_eps(np.array(x), sparse_gp, gibbs_indices, regressorD))

    # Latent Poisson Process update
    PyRV_Pi = PoissonProcess(epsilon=np.zeros(sparse_gp.m), Lambda=LambdaTrue, U=U_OT, PoissonScale=T, Uniform=myUniform, sparse_gp=sparse_gp, gibbs_indices=gibbs_indices )
    RV_Pi = ot.RandomVector(PyRV_Pi)
    ot_link_function_Pi = ot.PythonFunction(gibbs_indices.chain_dim, sparse_gp.m+J, lambda x:py_link_function_Pi(x,gibbs_indices))

    # Latent Polya Gamma Process update
    PyRV_w = PolyaGammaProcess(epsilon=np.zeros(sparse_gp.m), Pi=np.zeros((Nmax-N,2)), Ntot=Ntot, gibbs_indices=gibbs_indices, sparse_gp=sparse_gp, regressorD=regressorD )
    RV_w = ot.RandomVector( PyRV_w )
    ot_link_function_w = ot.PythonFunction(gibbs_indices.chain_dim, sparse_gp.m + 2*(Nmax-N)+1, lambda x:py_link_function_w(x, gibbs_indices=gibbs_indices))

    # Latent zone effects update
    RV_Lambda = ot.RandomVector(ot.JointDistribution([ot.Gamma()]*J))
    ot_link_function_Lambda = ot.PythonFunction(gibbs_indices.chain_dim, J*3, lambda x:py_link_function_Lambda(np.array(x), D=D, U=U_OT, PoissonScales=PoissonScales, a=a, b=b, gibbs_indices=gibbs_indices))
    # TEST latent GP update
    RV_eps.getRealization()
    RV_eps.getParameter()
    # TEST latent Poisson + GP update
    RV_Pi.getRealization()
    RV_Pi.getParameter()
    # TEST latent Polya-Gamma
    RV_w.getRealization()
    RV_w.getParameter()
    # TEST latent Zone effects
    RV_Lambda.getRealization()
    RV_Lambda.getParameter()

    # Learn sparse GP on observed data
     
    
    # Plot Real sparse GP trajectory on meshgrid over search domain
    gridsize = 20
    xx, yy = np.meshgrid( np.linspace(0, 1, gridsize), np.linspace(0, 1, gridsize) )
    XY_new = np.vstack(( xx.ravel(), yy.ravel() )).T

    # # Z_True = PyRV_Pi.SimulateSigmaGP( XY_new )
    # M_new = sparse_gp.regressorOT( XY_new )
    # eps_new = np.array( ot.Normal().getSample(sparse_gp.m) ).reshape(-1,1)
    # Z_True = np.dot( M_new, eps_new )
    # # Z_True = m( XY_new )
        
    # Z_True = np.array(Z_True).reshape(gridsize, gridsize) * T
    # levels = np.linspace( Z_True.min(), Z_True.max(), gridsize )

    # fig = plt.figure()
    # plt.contourf(xx, yy, Z_True, levels)
    # plt.colorbar()
    # plt.scatter( D[:,0], D[:,1], c='r')
    # # plt.show()
    # plt.savefig('True_GP_trend.png')
    #plt.close()
        
    #%%


    ###############
    # Launch MCMC #
    ###############

    samples = []
    randinits = []

    for i in range(ninits):
        # break
        # Random initialization
        randinit = np.zeros(gibbs_indices.chain_dim)
        Ntot_init = 0
        while Ntot_init < N:
            Ntot_init = int(ot.Poisson(LambdaMax * T).getRealization()[0])
        randinit[gibbs_indices.Pi_indices[-1]] = Ntot_init
        NPi_init = int(Ntot_init - N)
        Pi_init = np.zeros(( Nmax - N, 2 ))
        Pi_init[:NPi_init] = np.array(myUniform.getSample(NPi_init))
        randinit[gibbs_indices.Pi_indices[:-1]] = Pi_init.ravel()
        randinit[gibbs_indices.Omega_indices[:Ntot_init]] = random_polyagamma(size=Ntot_init)
        # check whether useful:
        randinit[gibbs_indices.lambda_indices] = [ot.Gamma(a[j], b[j]).getRealization()[0] for j in range(J)]
        randinits.append(randinit)
        # Assemble Gibbs sampler
        print("random init %s out of %s: %s"%(str(i+1),str(ninits),str(randinits[i])))
        eps_sampler = ot.RandomVectorMetropolisHastings( RV_eps, randinits[i], gibbs_indices.epsilon_indices, ot_link_function_eps )
        Pi_sampler = ot.RandomVectorMetropolisHastings( RV_Pi, randinits[i], gibbs_indices.Pi_indices, ot_link_function_Pi ) 
        w_sampler = ot.RandomVectorMetropolisHastings( RV_w, randinits[i], gibbs_indices.Omega_indices, ot_link_function_w )
        Lambda_sampler = ot.RandomVectorMetropolisHastings( RV_Lambda, randinits[i], gibbs_indices.lambda_indices, ot_link_function_Lambda )
        Gibbs_sampler = ot.Gibbs([eps_sampler, Pi_sampler, w_sampler, Lambda_sampler])
        t1=time.time()
        sample = np.zeros((0,gibbs_indices.chain_dim))
        # Main loop
        for j in range((sampleSize)// blockSize):
            newsample = Gibbs_sampler.getSample(blockSize)
            sample = np.vstack((sample, np.array(newsample)))
            t2=time.time()
            print("%s iterations performed in %s seconds"%( (j+1)*blockSize, np.round(t2-t1)))   
            rate = (sample[1:] != sample[:-1]).mean(axis=0) 
            print("componentwise acceptance rate so far: %s"%rate)        
            print("Current state: %s"%sample[-1])
        t2=time.time()
        print("Whole MCMC run took %s seconds"%(t2-t1))    
        samples.append( sample )

    #%%


    ################################
    # MCMC Convergence diagnostics #
    ################################

    colors = list(mcolors.BASE_COLORS)[:ninits]
    burnin=0
    paramDim = sample.shape[1]
    # plotDim = 1

    # components = [j for j in range(paramDim-1-J,paramDim)] 
    components = [gibbs_indices.Pi_indices[-1]] + gibbs_indices.lambda_indices 
    names = [r"$N_{tot}$"] + [r"$\lambda_{%s}$"%j for j in range(1,J+1)]
    true_values = [Ntot] + LambdaTrue.ravel().tolist()

    # MCMC convergence plots
    fig = plt.figure( figsize=(5*J,5) )
    for i, X, c in zip( range(ninits), samples, colors ):
        # break
        for j in range(len(components)):
            # break
            plt.subplot(1, len(components), j+1)
            plt.plot(X[burnin:,components[j]], c=c)
            if i == 0:
                plt.ylabel(names[j], fontsize=16)
                plt.xlabel("Iterations", fontsize=16)    
            plt.axhline(true_values[j], lw=2, c="k")
    plt.tight_layout()
    plt.savefig("traceplots.png")
    #plt.close()

    #%%


    # ACF (MCMC autocorrelation) plot 
    fig = plt.figure( figsize=(5*J, 5))
    for i, X, c in zip( range(ninits), samples, colors ):
        for j in range(len(components)):
            plt.subplot(1, len(components), j+1)
            plt.plot(stattools.acf(X[burnin:,components[j]], nlags=600), c=c)    
            if i == 0:
                plt.ylabel(names[j], fontsize=16)
                plt.xlabel("Iterations", fontsize=16)  
    plt.tight_layout()
    plt.savefig("ACF.png")
    #plt.close()


    #%%


    # Gelman-Rubin

    def iterative_mean(X):
        length = X.shape[1]
        # on prend les moyennes cumulées suivant le deuxième axe (une par composante de la chaîne)
        return X.cumsum(axis=1) / np.linspace(1, length, length).reshape(1,-1)

    def iterative_var(X):
        length = X.shape[1]
        # on prend les variances cumulées suivant le deuxième axe (une par composante de la chaîne)
        return np.square(X).cumsum(axis=1) / np.linspace(1, length, length).reshape(1,-1) - iterative_mean(X)**2

    fig = plt.figure( figsize=(5*J, 5))
    for j in range(len(components)):
        # remarque : on enlève la première valeur des moyennes / variances cumulés
        # pour éviter des valeurs de variance égales à zéro...
        sample_means = np.array([iterative_mean(chain)[:,components[j]] for chain in samples])
        sample_vars = np.array([iterative_var(chain)[:,components[j]] for chain in samples])
        
        B = sampleSize / (ninits - 1) * sample_means.var(axis=0)
        W = sample_vars.mean(axis=0)
        V = (sampleSize - 1) / sampleSize * W + (ninits + 1) / (sampleSize * ninits) * B
        
        R = V/W
        
        print("Gelman-Rubin convergence diagnostic for %s: %s"%(names[j], V/W))
        
        # on enlève les premières iterations qui correspondent au temps de chauffe
        plt.subplot( 1, len(components), j+1)
        plt.plot(R[10:])
        
        plt.xlabel("Iterations")
        plt.ylabel(r"$\widehat R$")
    plt.tight_layout()
    plt.savefig("Gelman_Rubin.png")
    #plt.close()

    #%%

    # Pool chains
    sample = np.vstack([sample[burnin:] for sample in samples])

    # Posterior marginals (pooling from both chains)
    fig = plt.figure( figsize=(5*J, 5))
    for j in range(len(components)):
        plt.subplot( 1, len(components), j+1)
        X = sample[burnin:,components[j]]
        plt.hist(X, int(np.sqrt(len(X))))
        plt.xlabel(names[j], fontsize=16)
        plt.axvline(true_values[j], c='r')
        # plt.xlim(st.mstats.mquantiles(X,.01)[0], st.mstats.mquantiles(X,.99)[0])
        # plt.xlim(0, 14)
        print(X.mean())
        for p in [0.50, .025, .975]:
            print(st.mstats.mquantiles(X, p)[0])

    plt.tight_layout()
    plt.savefig("post_density.png")
    #plt.close()

    #%%


    #######################################
    # Predict GP throughout search domain #
    #######################################

    Z_new = np.zeros((len(sample), len(XY_new)))    
    M = np.array(sparse_gp.regressorOT( XY_new ))
    for i in range(len(sample)):
        # break
        sample_i =sample[i]
        # GP conditional on values at augmented Poisson process
        epsilon_i = sample_i[gibbs_indices.epsilon_indices]
        # Ntot_i = sample_i[gibbs_indice.Pi_indices[-1]]
        # Pi_i = np.array(sample_i[gibbs_indice.Pi_indices[:-1]]).reshape(-1,2)
        Z_new[i] = np.dot( M, epsilon_i ).ravel()
        # PyRV_Pi.setParameter(py_link_function_Pi(sample[i], Nmax, J))
        # Z_new[i] = PyRV_Pi.SimulateSigmaGP( XY_new )
        
    Z_mean = Z_new.mean(axis=0).reshape(gridsize, gridsize) * T
    levels_mean = np.linspace( Z_mean.min(), Z_mean.max(), gridsize )

    Z_std = Z_new.std(axis=0).reshape(gridsize, gridsize) * T
    levels_std = np.linspace( Z_std.min(), Z_std.max(), gridsize)

    fig = plt.figure()
    plt.contourf(xx, yy, Z_mean, levels_mean)
    plt.colorbar()
    plt.scatter( D[:,0], D[:,1], s=100, c='r', marker='+' )
    plt.title("Poisson intensity posterior mean vs Data")
    plt.savefig("f_post_mean.png")
    #plt.close()

    fig = plt.figure()
    plt.contourf(xx, yy, Z_std, levels_std)
    plt.colorbar()
    plt.scatter( D[:,0], D[:,1], s=100, c='r', marker='+' )
    plt.title("Poisson intensiety Posterior std vs Data")
    plt.savefig("f_post_std.png")
    #plt.close()















