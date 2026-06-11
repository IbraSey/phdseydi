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
import shapely 
import sys, os
sys.path.append( os.path.join( os.pardir, "spatial_density_estimation", "gp_spatial" ) )
from gp.gibbs_sampler import SSGC_GibbsSampler
from shapely.prepared import prep

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

########################################################################
# Use-Case Class, collecting all necessary inputs for SGPPI case study # 
########################################################################

class SSGC_Gibbs():
    """Use-Case Class, collecting all necessary inputs for SGPPI case study
    """
    def __init__(self, zones, T, a, b, Nmax, l=None, nu=None, D=None ):
        """Specifying data and priors for 2D SGPP model

        Args:
            zones (list): list of J shapely.Polygons
            T (float): time horizon
            a (J,): prior gamma shapes
            b (J,): prior gamma rates
            Nmax (int): Max allowed space to represent latent PP
            l (2,) (optional):  correlation lengths
            nu (float) (optional): marginal GP variance
            D (N,2) (optional): dataset (seismic catalog)
            
        Notes
            - Nmax must be >= Ntot (Size of latent PP) or an exception is raised
            - /!\ realizations are zero-padded to reach size Nmax
        
        """
        self.zones = zones
        self.T = T
        self.PoissonScales = T * np.array([zone.area for zone in zones])
        self.a = a
        self.b = b
        self.J = len(zones)
        self.Nmax = Nmax
        if not l is None:
            self.l = l
        if not nu is None:
            self.nu = nu
        self.U_OT = ot.MemoizeFunction( ot.PythonFunction( 2, self.J, self.U ) )     
        # bounding box for uniform sampling of the data
        self.Domain = shapely.union_all(self.zones).buffer(1e-6)
        coords = np.array(self.Domain.boundary.coords)
        self.lower = coords.min(axis=0)
        self.upper = coords.max(axis=0)
        self.Uniform = ot.ComposedDistribution([ot.Uniform(self.lower[j], self.upper[j]) for j in range(2)])
        if not D is None:
            self.setD(D)
        if not l is None:
            self.setSparseGP(l, nu)
    
    def setSparseGP(self, l, nu):
        """Define sparse GP approximation

        Args:
            l (2,):  correlation lengths
            nu (float): marginal GP variance
        
        Returns:
            self.sparseGP object
        """
        # Define GP and sparse GP hyperparams
        c1, c2 = 0.5*(self.lower + self.upper)
        s1, s2 = 0.5*(self.upper - self.lower)
        l1, l2 = l, l
        hypers = (l1, l2, c1, c2, s1, s2, nu)
        self.sparse_gp = sparseGP(hypers)
        if hasattr(self, "D"):
            self.regressorD = self.sparse_gp.regressorOT(self.D)
            self.gibbs_indices = GibbsIndices(self.sparse_gp.m, self.Nmax, len(self.D), self.J)
    
    def setD(self, D):
        """Set dataset for case-study

        Args:
            D (N,2) : dataset (seismic catalog)
        """
        self.D = D
        if hasattr(self, "sparse_gp"):
            self.regressorD = self.sparse_gp.regressorOT(D)
            self.gibbs_indices = GibbsIndices(self.sparse_gp.m, Nmax, len(D), self.J)
    
    ################################
    # Estimate correlation lengths #   
    ################################
    def calibrate_GP(self, D=None):
        """_summary_

        Args:
            D (N,2) (optional): Dataset. Defaults to None.

        Returns:
            list: correlations lengths + marginal GP variance
        """
        
        if D is None:
            try:
                D = self.D
            except:
                print("No data for calibration!")
                raise ValueError
        
        zones_prep = [prep(p) for p in self.zones]
        Areas      = [(zp, 0.0) for zp in zones_prep]

        sampler = SSGC_GibbsSampler(
            X_bounds  = (0, 1),
            Y_bounds  = (0, 1),
            T         = self.T,
            Areas     = Areas,
            polygons  = self.zones,
            lambda_nu = 1.,
            nu        = [0.5, 0.5],
            delta     = [0.5, 0.5],
            jitter    = 1e-5,
            rng_seed  = 15,
        )

        v, l_ot, eps_mle = sampler.calibrate_nu( D[:,0], D[:,1] )
        return [l_ot, v**2]
    
    def U(self, x):
        """zones indicators

        Args:
            x (2,) 2D point
        
        Returns:
            (J,): binary vector (sums to 1)
        """
        u = np.zeros(self.J, int)
        j = 0
        inzone = False
        while not inzone:
            u[j] = self.zones[j].contains(shapely.Point(x))
            inzone = u[j]
            j += 1
            if j == self.J: break
        return u

    ##################################
    # Latent Gaussian process update # 
    ##################################

    def py_link_function_eps(self, x):
        """
        Given the current state of the MCMC chain,
        output parameters of the conditional density of
        the truncated GP, as required by the NormalCholesky class

        Parameters
        ----------
        x : array / list
            Current MCMC chain state
            
        Returns
        -------
        param : list
            Mean + Cholesky precision matrix + Ntot value
            in the order required by NormalCholesky class
            Size : m*(m+1)+1
            with m the sparse GP truncation order 
        
        """
        gibbs_indices = self.gibbs_indices
        sparse_gp = self.sparse_gp
        regressorD = self.regressorD
        # Check that gibbs_indices has same "m" attribute as sparse_gp
        if gibbs_indices.m != sparse_gp.m:
            raise ValueError
        m = sparse_gp.m
        # Extract current state of conditioning variables
        N = gibbs_indices.N
        Ntot = int(x[gibbs_indices.Pi_indices[-1]])
        Pi = np.array(x[gibbs_indices.Pi_indices[:2*(Ntot-N)]]).reshape(-1,2)
        Omega = np.array(x[gibbs_indices.Omega_indices[:Ntot]]).reshape(-1,1)
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

    ###############################
    # Latent zones effects update # 
    ###############################

    def py_link_function_Lambda(self, x):
        """
        Given the current state of the MCMC chain,
        output parameters of the conditional density of
        Lambda, as required by the RV_Lambda class

        Parameters
        ----------
        case : SGPPI
            contains data and priors for 2D SGPP model
        x : array / list
            Current MCMC chain state
            Size : 4*Nmax - 2*N + J + 1

        Returns
        -------
        param : list
            posterior shape and rate parameters for Gamma distribution of Lambdas
            in the order required by the RV_Lambdas class
            Size : J*3
        """
        D = self.D
        U = self.U_OT 
        PoissonScales = self.PoissonScales
        a, b = self.a, self.b
        gibbs_indices = self.gibbs_indices
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
        U_Dtot = np.array(self.U_OT(Dtot))
        Nk = U_Dtot.sum(axis=0).reshape(-1,1)
        # extract parameters in correct order (coherent with getParameter() method of RV_Lambdas)
        parameter = np.zeros( J*3 )
        parameter[::3] = a + Nk.ravel() # shape parameters
        parameter[1::3] = b + PoissonScales  # rate parameters
        return parameter
    
    def Gibbs(self, sampleSize=300, blockSize=50, ninits=3):
        """Posterior sampling using blocked Gibbs sampler

        Args:
            sampleSize (int, optional): number of Gibbs iterations. Defaults to 300.
            blockSize (int, optional): block size for cv messages. Defaults to 50.
            ninits (int, optional): Number of chains run for Gelman-Rubin convergence diagnostic. Defaults to 3.

        Returns:
            samples (list of arrays): resulting MCMC chains
            randinits (list of vectors): initial points
        """
        ###################
        # MCMC parameters # 
        ###################
        
        sparse_gp = self.sparse_gp
        regressorD = self.regressorD
        gibbs_indices = self.gibbs_indices
        
        # Sparse Gaussian Process update
        RV_eps = ot.RandomVector(NormalCholesky(mu=np.zeros(self.sparse_gp.m), Chol=np.diag([1]*sparse_gp.m), Ntot=sparse_gp.m))
        ot_link_function_eps = ot.PythonFunction(gibbs_indices.chain_dim, len(RV_eps.getParameter()), lambda x:self.py_link_function_eps(np.array(x)))

        # Default chain state
        x = np.zeros(self.gibbs_indices.chain_dim)
        x[self.gibbs_indices.Pi_indices[-1]] = int(0.5*(self.gibbs_indices.Nmax + self.gibbs_indices.N))
        x[self.gibbs_indices.lambda_indices] = 1
        
        
        # Latent Poisson Process update
        PyRV_Pi = PoissonProcess(self, x)
        RV_Pi = ot.RandomVector(PyRV_Pi)
        ot_link_function_Pi = ot.PythonFunction(gibbs_indices.chain_dim, len(RV_Pi.getParameter()), PyRV_Pi.py_link_function_Pi)

        # Latent Polya Gamma Process update
        PyRV_w = PolyaGammaProcess(self, x)
        RV_w = ot.RandomVector( PyRV_w )
        ot_link_function_w = ot.PythonFunction(gibbs_indices.chain_dim, len(RV_w.getParameter()), PyRV_w.py_link_function_w)

        # Latent zone effects update
        RV_Lambda = ot.RandomVector(ot.JointDistribution([ot.Gamma()]*self.J))
        # ot_link_function_Lambda = ot.PythonFunction(gibbs_indices.chain_dim, J*3, lambda x:py_link_function_Lambda(self, np.array(x)))
        ot_link_function_Lambda = ot.PythonFunction(gibbs_indices.chain_dim, len(RV_Lambda.getParameter()), lambda x:self.py_link_function_Lambda(np.array(x)))
        
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
        
        ###############
        # Launch MCMC #
        ###############

        samples = []
        randinits = []
        
        N = self.gibbs_indices.N

        for i in range(ninits):
            # break
            # Random initialization
            randinit = np.zeros(gibbs_indices.chain_dim)
            randinit[gibbs_indices.lambda_indices] = [ot.Gamma(self.a[j], self.b[j]).getRealization()[0] for j in range(self.J)]
            LambdaMax = max(randinit[gibbs_indices.lambda_indices])
            Ntot_init = 0
            while Ntot_init <= N:
                Ntot_init = int(ot.Poisson(LambdaMax * self.T).getRealization()[0])
            randinit[gibbs_indices.Pi_indices[-1]] = Ntot_init
            NPi_init = int(Ntot_init - N)
            Pi_init = np.zeros(( self.Nmax - N, 2 ))
            Pi_init[:NPi_init] = np.array(self.Uniform.getSample(NPi_init))
            randinit[gibbs_indices.Pi_indices[:-1]] = Pi_init.ravel()
            randinit[gibbs_indices.Omega_indices[:Ntot_init]] = random_polyagamma(size=Ntot_init)
            # # check whether useful:
            randinit[gibbs_indices.lambda_indices] = [ot.Gamma(self.a[j], self.b[j]).getRealization()[0] for j in range(self.J)]
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

        return samples, randinits

    def run(self, D=None, sampleSize=300, blockSize=50, ninits=3):
        """perform Bayesian inference of SGPPI model given dataset

        Args:
            D (N,2) (optional): Dataset. Defaults to None
            sampleSize (int, optional): number of Gibbs iterations. Defaults to 300.
            blockSize (int, optional): block size for cv messages. Defaults to 50.
            ninits (int, optional): Number of chains run for Gelman-Rubin convergence diagnostic. Defaults to 3.

        returns:
            samples (list of arrays): resulting MCMC chains
            randinits (list of vectors): initial points
        """
        if not D is None:
            self.setD(D)
        elif not hasattr(self, "D"):
            print("No data for inference!")
            raise ValueError
        # calibrate GP
        l_opt, v_opt = self.calibrate_GP()
        self.setSparseGP(l_opt, v_opt)
        # Launch Gibbs 
        self.samples, _ = self.Gibbs(sampleSize=sampleSize, blockSize=blockSize,ninits=ninits)






#%%

#########################
# Latent Poisson update # 
#########################

class PoissonProcess(ot.PythonRandomVector):
    """
    Given current states of epsilon and Lambda,
    Generates an updated set of latent points 
    """
    def __init__( self, case, x=None):
        """
        Parameters
        ----------
        case : SGPPI
            contains data and priors for 2D SGPP model
        x : array / list, optional
            Current MCMC chain state
        Notes
        -----
        - parameter list size: m + J
        - Simulated variables dimension: 2*(Nmax-N)+1
        - Nmax must be >= Ntot (or else a ValueError is raised)
        - when Nmax > Ntot, realizations are zero-padded
        """
        Nmax = case.gibbs_indices.Nmax
        N = case.gibbs_indices.N
        super(PoissonProcess, self).__init__(int(2*(Nmax-N))+1)
        # Internal parameters (numpy arrays)
        self.J = case.gibbs_indices.J
        self.m = case.sparse_gp.m
        if self.m != case.gibbs_indices.m:
            print("sparseGP and gibbsIndices objects have different m values")
            raise ValueError
        self.U = case.U_OT
        self.PoissonScales = case.PoissonScales
        self.Uniform = case.Uniform
        self.sparse_gp = case.sparse_gp
        self.gibbs_indices = case.gibbs_indices
        self.case = case
        if not x is None:
            self.epsilon = x[self.gibbs_indices.epsilon_indices]
            self.Lambda = x[self.gibbs_indices.lambda_indices]
    
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
        N_star = int(ot.Poisson(self.PoissonScales.sum()*LambdaMax).getRealization()[0]) # Poisson candidate number
        XY_star =  self.Uniform.getSample(N_star) # Uniformly sampled candidates
        # Step 3: Simulate GP trajectories
        M = self.sparse_gp.regressorOT(XY_star)
        # epsilon = np.array( ot.Normal(self.m).getRealization() ).reshape(-1,1)
        f_star = np.dot( M, self.epsilon )
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

    def py_link_function_Pi(self, x):
        """
        Given the current state of the MCMC chain,
        output parameters of the conditional density of
        the Latent Poisson process, as required by the 
        PoissonProcess class.

        Parameters
        ----------
        x : array / list
            Current MCMC chain state
        
        Returns
        -------
        param : list
            in the order required by 
            the PoissonProcess class
            Size : m+J)
        """
        epsilon = np.array(x)[:self.gibbs_indices.m]
        Lambda = np.array(x)[-self.gibbs_indices.J:]
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
    def __init__( self, case, x=None ):
        """
        Parameters
        ----------
        case : SGPPI
            contains data and priors for 2D SGPP model
        x : array / list, optional
            Current MCMC chain state
        """
        self.gibbs_indices = case.gibbs_indices
        self.sparse_gp = case.sparse_gp
        # Check that gibbs_indices has same "m" attribute as sparse_gp
        if self.gibbs_indices.m != self.sparse_gp.m:
            raise ValueError
        super(PolyaGammaProcess, self).__init__(self.gibbs_indices.Nmax)
        self.regressorD = case.regressorD
        self.case = case
        self.m = self.sparse_gp.m
        if not x is None:
            self.epsilon = np.array(x[self.gibbs_indices.epsilon_indices]).reshape(-1,1)
            self.Pi = np.array(x[self.gibbs_indices.Pi_indices[:-1]]).reshape(-1,2)
            self.Ntot = x[self.gibbs_indices.Pi_indices[-1]]
    
    def setParameter(self, parameter):
        m = self.m
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
        N = self.gibbs_indices.N
        Nmax = self.gibbs_indices.Nmax
        Ntot = int(self.Ntot)
        w = np.zeros(Nmax)
        M = np.vstack([ self.regressorD, self.sparse_gp.regressorOT( self.Pi[:Ntot-N] ) ])
        ftot = np.dot( M, self.epsilon )
        w[:Ntot] = np.abs( random_polyagamma(z=np.array(ftot)[:,0]) )
        return w

    def py_link_function_w(self, x):
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
        
        Notes
        -----
        Ntot corresponds to the last component of Pi
        """
        epsilon = np.array(x)[self.gibbs_indices.epsilon_indices]
        Pi = np.array(x)[self.gibbs_indices.Pi_indices]
        return np.hstack(( epsilon, Pi ))

#%%

######################################
# MCMC Convergence diagnostics Class #
######################################



# Helper functions
def iterative_mean(X):
    length = X.shape[1]
    # on prend les moyennes cumulées suivant le deuxième axe (une par composante de la chaîne)
    return X.cumsum(axis=1) / np.linspace(1, length, length).reshape(1,-1)

def iterative_var(X):
    length = X.shape[1]
    # on prend les variances cumulées suivant le deuxième axe (une par composante de la chaîne)
    return np.square(X).cumsum(axis=1) / np.linspace(1, length, length).reshape(1,-1) - iterative_mean(X)**2

class ConvergenceDiagnosticsMCMC:
    
    def __init__(self, samples, burnin=0, names=None, trueValues=None):
        """Convergence plots for MCMC output

        Args:
            samples (list of arrays): list of MCMC chains
                chains must all have the same shape
            burnin (int, optional): Remove first iterations prior to analysis. Defaults to 0.
            names (list of strs, optional): Names attached to each component. Defaults to None.
            trueValues (list of floats, optional): true values for each component. Defaults to None.

        """
        self.samples = samples
        self.burnin = burnin
        self.ninits = len(samples)
        self.names = names
        self.trueValues = trueValues
        self.sampleSize, self.paramDim  = samples[0].shape
        self.colors = list(mcolors.BASE_COLORS)[:self.ninits]
    
    # MCMC convergence plots
    def convergencePlot(self):
        """trace plots of MCMC chains
        """
        burnin = self.burnin
        names = self.names
        fig = plt.figure( figsize=(5*self.paramDim, 5) )
        for i, X, c in zip( range(self.ninits), self.samples, self.colors ):
            # break
            for j in range(self.paramDim):
                # break
                plt.subplot(1, self.paramDim, j+1)
                plt.plot(X[burnin:,j], c=c)
                if i == 0:
                    plt.ylabel(names[j], fontsize=16)
                    plt.xlabel("Iterations", fontsize=16)    
                if not self.trueValues is None:
                    plt.axhline(self.trueValues[j], lw=2, c="k")
        plt.tight_layout()
        plt.savefig("traceplots.png")
        #plt.close()

    def autocorrelationPlot(self, nlags=600):
        """ACF (MCMC autocorrelation) plot

        Args:
            nlags (int, optional): maximum lag considered. Defaults to 600.
        """
        #  
        burnin = self.burnin
        names = self.names
        colors = self.colors
        fig = plt.figure( figsize=(5*self.paramDim, 5) )
        for i, X, c in zip( range(self.ninits), self.samples, colors ):
            # break
            for j in range(self.paramDim):
                # break
                plt.subplot(1, self.paramDim, j+1)
                plt.plot(stattools.acf(X[burnin:,j], nlags=nlags), c=c)    
                if i == 0:
                    plt.ylabel(names[j], fontsize=16)
                    plt.xlabel("Iterations", fontsize=16)  
        plt.tight_layout()
        plt.savefig("ACF.png")
        #plt.close()
    
    def GelmanRubinPlot(self):
        """Evolution of Gelman-Rubin cv statistic
        """
        names = self.names
        sampleSize = self.sampleSize
        fig = plt.figure( figsize=(5*self.paramDim, 5) )
        for j in range(self.paramDim):
            # remarque : on enlève la première valeur des moyennes / variances cumulés
            # pour éviter des valeurs de variance égales à zéro...
            sample_means = np.array([iterative_mean(chain)[:,j] for chain in self.samples])
            sample_vars = np.array([iterative_var(chain)[:,j] for chain in self.samples])
            
            B = sampleSize / (self.ninits - 1) * sample_means.var(axis=0)
            W = sample_vars.mean(axis=0)
            V = (sampleSize - 1) / sampleSize * W + (self.ninits + 1) / (sampleSize * self.ninits) * B
            
            R = V/W
            
            print("Gelman-Rubin convergence diagnostic for %s: %s"%(names[j], V/W))
            
            plt.subplot( 1, self.paramDim, j+1)
            plt.plot(R[10:])
            
            plt.xlabel("Iterations")
            plt.ylabel(r"$\widehat R$")
        plt.tight_layout()
        plt.savefig("Gelman_Rubin.png")
        #plt.close()        

    def poolChains(self):
        """final pool chains, with histograms

        Returns:
            self.sample (array): pooled samples, with burnin removed
        """
        # Pool chains
        burnin = self.burnin
        names = self.names
        self.sample = np.vstack([sample[burnin:] for sample in self.samples])

        # Posterior marginals (pooling from both chains)
        fig = plt.figure( figsize=(5*self.paramDim, 5))
        for j in range(self.paramDim):
            plt.subplot( 1, self.paramDim, j+1)
            X = self.sample[burnin:,j]
            plt.hist(X, int(np.sqrt(len(X))))
            plt.xlabel(names[j], fontsize=16)
            if not self.trueValues is None:
                plt.axvline(self.trueValues[j], c='r')
            # plt.xlim(st.mstats.mquantiles(X,.01)[0], st.mstats.mquantiles(X,.99)[0])
            # plt.xlim(0, 14)
            print(X.mean())
            for p in [0.50, .025, .975]:
                print(st.mstats.mquantiles(X, p)[0])

        plt.tight_layout()
        plt.savefig("post_density.png")
        #plt.close()
    
    def run (self, **kwargs):
        """Perform complete MCMC convergence analysis 
        """
        self.convergencePlot()
        self.autocorrelationPlot(**kwargs)
        self.GelmanRubinPlot()
        self.poolChains()
        
