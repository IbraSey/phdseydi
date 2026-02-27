#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 23:12:13 2025

@author: H01971
"""

##########################
#    Necessary imports   #
##########################

import os
import openturns as ot
import openturns.experimental as otexp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
import scipy.optimize as so
import scipy.stats as st
import statsmodels.tsa.stattools as stattools
from polyagamma import random_polyagamma

ot.RandomGenerator.SetSeed(0) # Make results reproducible by freezing Open TURNS's random generator's seed
np.random.seed(0) # Make results reproducible by freezing Numpy's random generator's seed

sigmoid = ot.SymbolicFunction(['z'], ['1/(1+exp(-z))'])

sigmoid_inv = ot.SymbolicFunction(['q'], ['ln( q/(1-q) )'])


##################################
# Latent Gaussian process update # 
##################################

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


def py_link_function_f(x, Nmax, D, U, covarianceModel):
    """
    Given the current state of the MCMC chain,
    output parameters of the conditional density of
    the GP, as required by the NormalCholesky class

    Parameters
    ----------
    x : array / list
        Current MCMC chain state
        Size : 4*Nmax - 2*N + 1 + J
    Nmax : int
        Max of Ntot (augmented Poisson process size)
    D : (N,2)
        Observed Poisson process
    U : Open TURNS function
        zone indicator functions
        given a point (x,y), outputs J 0-1 indicators,
        summing to 1
    covarianceModel : Open TURNS covariance model
        GP cov model
    
    Returns
    -------
    param : list
        Mean + Cholesky precision matrix + Ntot value
        in the order required by NormalCholesky class
        Size : Nmax*(Nmax+1)+1
    """
    # Extract cuurent state of conditioning variables
    N=len(D)
    J = U.getOutputDimension()
    Ntot = int(x[-J-1])
    Pi = np.array(x)[2*Nmax:2*Nmax+2*(Ntot-N)].reshape(-1,2)
    Omega = np.array(x)[Nmax:Nmax+Ntot]    
    Eps = np.array(x)[-J:].reshape(-1,1)
    # total (augmented) data
    Dtot = ot.Sample(Ntot, 2)
    Dtot[:N] = D
    Dtot[N:] = Pi
    u = ot.Sample(np.array([[0.5]]*N + [[-0.5]]*(Ntot-N)))
    # precision matrix
    K = covarianceModel.computeCrossCovariance(Dtot,Dtot)
    K = ot.CovarianceMatrix(K)
    L = K.computeCholesky()
    Linv = L.inverse()
    Kinv = Linv.transpose()*Linv
    # add Omega to precision matrix diagonal
    Diag = np.array(Kinv.getDiagonal())[:,0] + Omega
    Diag = Diag.tolist()
    Kinv.setDiagonal( Diag )
    # invert 
    L = ot.CovarianceMatrix(Kinv).computeCholesky()
    Linv = L.inverse()
    V = Linv.transpose()*Linv
    # prior to posterior total mean 
    m_Dtot = np.dot( np.array(U(Dtot)), Eps )
    mean = ot.Sample(m_Dtot)
    mean = Kinv * mean
    mean = mean + u
    mean = V*mean
    # extract parameters in correct order (coherent with getParameter() method of RV_f)
    parameter = [0]*( Nmax*(Nmax+1)+1 )
    parameter[:Ntot] = np.array(mean).ravel()
    parameter[Nmax:Nmax+Ntot*Ntot] = np.array(Linv).ravel()
    parameter[-1] = Ntot
    return parameter


#################################################
# Latent Poisson... and Gaussian process update # 
#################################################

class PoissonGaussianProcess(ot.PythonRandomVector):
    """
    Given current states of GP values at observed and latent points Pi (and the latter)
    Generates an updated set of latent points and associated GP Values
    """
    def __init__( self, ftot, Pi, Ntot, Eps, D, U, covarianceModel, Poisson, myUniform):
        """
        Parameters
        ----------
        ftot : vector
            current GP values at observed and latent points. 
            Size: Nmax
        Pi : array
            Current latent points. 
            shape: (Nmax-N,2)
        Ntot: int
            current total size of observed and latent process
            Ntot=N+NPi
        Eps : (J,1)
            current value of zones regressors
        D : (N,2)
            Observed Poisson process
        U : Open TURNS function
            zone indicator functions
            given a point (x,y), outputs J 0-1 indicators,
            summing to 1
        covarianceModel : Open TURNS covariance model
            GP cov model
        Poisson : Open TURNS distribution
            Poisson law of homogeneous process size
        myUniform : Open TURNS distribution
            Uniform law of homogeneous Poisson process

        Notes
        -----
        - above parameters are flattened and concatenated
          in above order
        - parameter list size: 3*Nmax-2*N+1
        - Simulated variables dimension: 3*Nmax-3*N+1
          Difference with parameter list size is the former
          doesn't account for D (size N)
        - Nmax must be >= Ntot (or else a ValueError is raised)
        - if Nmax > Ntot, realizations are zero-padded
        """
        Nmax = len(ftot)
        N = len(D)
        super(PoissonGaussianProcess, self).__init__(int(3*Nmax-3*N+1))
        # not converting to int results in weird error message:
        # super(PoissonGaussianProcess, self).__init__(3*Nmax-3*N+1)
        # if Nmax < Ntot: 
        #     print("Inputs have incompatible sizes")
        #     raise ValueError
        # Internal parameters (numpy arrays)
        self.ftot = np.array(ftot).reshape(-1,1)
        self.Pi = np.array(Pi).reshape(-1,2)
        self.Ntot = int(Ntot)
        self.Eps = Eps
        self.Nmax = Nmax
        self.D = D
        self.U = U
        self.covarianceModel = covarianceModel
        self.Poisson = Poisson
        self.myUniform = myUniform
        self.J = len(Eps)
    
    def setParameter(self, parameter):
        """

        Parameters
        ----------
        parameter : list
            concatenates current values of :
                - ftot (Nmax,)
                - Pi (Nmax-N,2)
                - Ntot (1,)
                - Eps (J,)
            in this order
            Size: 3*Nmax-2*N+J+1

        Returns
        -------
        Sets internal parameters (numpy arrays)

        """
        Nmax=int(self.Nmax)
        self.ftot = np.array(parameter[:Nmax]).reshape(-1,1)
        self.Pi = np.array(parameter[Nmax:-self.J-1]).reshape(-1,2)
        self.Ntot = int(parameter[-self.J-1])
        self.Eps = np.array(parameter[-self.J:]).reshape(-1,1)
    
    def getParameter(self):
        Nmax=int(self.Nmax)
        Ntot=int(self.Ntot)
        N = Nmax-len(self.Pi)
        parameter = np.zeros(3*Nmax-2*N+self.J+1)
        parameter[:Ntot] = self.ftot[:Ntot].ravel()
        parameter[Nmax:Nmax+2*(Ntot-N)] = self.Pi[:Ntot-N].ravel()
        parameter[-self.J-1] = Ntot
        parameter[-self.J:] = self.Eps.ravel()
        return parameter.tolist()
    
    def getGaussianProcessRegression(self):
        """
        Update Gaussian process with augmented (observed and latent) values

        Returns
        -------
        gpr_result: GaussianProcessFitter result

        """
        # Step 0: Extract GP training sample
        Nmax=int(self.Nmax)
        Ntot=int(self.Ntot)
        N = Nmax-len(self.Pi)
        inputSample = np.vstack((self.D, self.Pi[:Ntot-N]))
        outputSample = self.ftot[:Ntot].copy()
        # remove zones effect
        zone_effect = np.dot( np.array(self.U(inputSample)), self.Eps )    
        outputSample -= zone_effect
        # Step 1: Fit GP regression model to Sample
        fitter = otexp.GaussianProcessFitter(inputSample, outputSample, self.covarianceModel, ot.Basis(0))
        fitter.setOptimizeParameters(False)
        fitter.run()
        fitter_result = fitter.getResult()
        # print(fitter_result)
        # Step 2: Deduce GP law conditional on Sample
        algo = otexp.GaussianProcessRegression(fitter_result)
        algo.run()
        gpr_result = algo.getResult()        
        # print(gpr_result)
        return gpr_result
    
    def getRealization(self):
        """
        Simulates one realization of the latent Poisson process
        and the associated GP

        Returns
        -------
        list
            simulated variables are flattened 
            and concatenated in the following order:
            - New GP process values (size : Nmax-N)
            - New Poisson process values (shape : (Nmax-N,2))
            - New Ntot value (size: 1)
            total size : 3*Nmax-3*N+1        
        
        Notes:
            - There is no guaranty that New Ntot <= Nmax
            - New Ntot > Nmax may cause a crash
        """
        Nmax=int(self.Nmax)
        N = len(self.D)
        # Step 1: Update Gaussian process 
        gpr_result = self.getGaussianProcessRegression()        
        # Step 2: Generate candidate points
        N_star = int(self.Poisson.getRealization()[0]) # Poisson candidate number
        XY_star = np.array( self.myUniform.getSample(N_star) )# Uniformly sampled candidates
        # Step 3: predict GP at candidates
        process = otexp.ConditionedGaussianProcess(gpr_result, ot.Mesh(XY_star))
        f_star = np.array(process.getRealization())
        # Step 4: Thinning
        p_accept = np.array( sigmoid(-f_star.reshape(-1,1)) )
        accept = np.array( ot.Uniform().getSample(N_star) ) <= p_accept 
        NPi_new = np.array(accept).sum()
        Ntot_new = N + NPi_new
        if Ntot_new > Nmax:
            print("Maximum size %s exceeded by simulated data size %s"%(Nmax, Ntot_new))
            raise ValueError
        # Assemble final output
        f_new = np.zeros(Nmax - N)
        f_new[:NPi_new] = f_star[accept]
        Pi_new = np.zeros((Nmax - N, 2))
        Pi_new[:NPi_new] = XY_star[accept.ravel()]
        # add zones effect
        zone_effects = np.dot( np.array(self.U(Pi_new[:NPi_new])), self.Eps ).ravel()
        f_new[:NPi_new] = f_new[:NPi_new] + zone_effects
        return np.concatenate([f_new.ravel(), Pi_new.ravel(), [Ntot_new]])
        
    def SimulateSigmaGP( self, XY_new ):
        """
        Simulate from conditional Logistic Gaussian Process 
        given current values

        Parameters
        ----------
        XY_new : (N_new, 2) array
            Points 

        Returns
        -------
        f_simu : (size, N_new) array
            independent realizations of the conditioned GP
            after logistic transform
        """
        # Step 1: Update Gaussian process 
        gpr_result = self.getGaussianProcessRegression()        
        # Step 3: predict GP at new points
        process = otexp.ConditionedGaussianProcess(gpr_result, ot.Mesh(XY_new))
        f_simu = np.array(process.getRealization()).reshape(-1,1)
        # Add zone effects
        f_simu += np.dot( np.array(self.U(XY_new)), self.Eps )
        return np.array(sigmoid(f_simu)).ravel()
            


def py_link_function_Pi( x, Nmax, N, J ):
    """
    Given the current state of the MCMC chain,
    output parameters of the conditional density of
    the Latent Poisson process, and the associated 
    GP process, as required by the 
    PoissonGaussianProcess class.

    Parameters
    ----------
    x : array / list
        Current MCMC chain state
        Size : 4*Nmax-2*N+1
    Nmax : int
        Max of Ntot (augmented Poisson process size)
    J : int
        number of zones

    Returns
    -------
    param : list
        Mean + Cholesky precision matrix
        in the order required by 
        the PoissonGaussianProcess class
        Size : 3*Nmax-2*N+J+1)
    """
    ftot = np.array(x)[:Nmax].reshape(-1,1)
    Pi = np.array(x)[2*Nmax:-J-1].reshape(-1,2)
    Ntot = int(x[-J-1])
    Eps = np.array(x)[-J:].reshape(-1,1)
    return np.concatenate([ftot.ravel(), Pi.ravel(), [Ntot], Eps.ravel()])
   
#############################
# Latent Polya-Gamma update # 
#############################

class PolyaGammaProcess(ot.PythonRandomVector):
    """
    Given current states of GP values 
    Generates an updated set of Polya-Gamma values
    """
    def __init__( self, ftot, Ntot ):
        """
        Parameters
        ----------
        ftot : vector
            current GP values at observed and latent points. 
            Size: Nmax
        Ntot : int
            current value of total data size
        """
        Nmax=len(np.array(ftot).ravel())
        super(PolyaGammaProcess, self).__init__(Nmax)
        self.ftot = np.array(ftot).reshape(-1,1)   
        self.Ntot = int(Ntot)
        self.Nmax = Nmax
        
    def setParameter(self, parameter):
        self.ftot = np.array(parameter[:-1]).reshape(-1,1)
        self.Ntot = int(parameter[-1])
    
    def getParameter(self):
        return np.concatenate([self.ftot.ravel(), [int(self.Ntot)]])
    
    def getRealization(self):
        """
        Simulates one realization of the latent Polya-Gamma process

        Returns
        -------ug 
        list
            New Polya-Gamma process values
            Size : Nmax
        """     
        Nmax=int(self.Nmax)
        Ntot=int(self.Ntot)
        w = np.zeros(Nmax)
        w[:Ntot] = np.abs( random_polyagamma(z=np.array(self.ftot[:Ntot])[:,0]) )
        return w


def py_link_function_w(x, Nmax, J):
    """
    Given the current state of the MCMC chain,
    output parameters of the conditional Polya
    Gamma process, as required by the 
    PolyaGammaProcess class.

    Parameters
    ----------
    x : array / list
        Current MCMC chain state
        Size : 4*Nmax-2*N+1
    

    Returns
    -------
    param : list
        GP values and Ntot
        in the order required by 
        the PolyaGammaProcess class
        Size : Nmax+1
    """
    return np.hstack(( np.array(x)[:Nmax], [x[-J-1]] ))

###############################
# Latent zones effects update # 
###############################

def py_link_function_Eps(x, Nmax, D, U, PrecEps, covarianceModel):
    """
    Given the current state of the MCMC chain,
    output parameters of the conditional density of
    Eps, as required by the NormalCholesky class

    Parameters
    ----------
    x : array / list
        Current MCMC chain state
        Size : 4*Nmax - 2*N + 1 + J
    Nmax : int
        Max of Ntot (augmented Poisson process size)
    D : (N,2)
        Observed Poisson process
    U : Open TURNS function
        zone indicator functions
        given a point (x,y), outputs J 0-1 indicators,
        summing to 1
    PrecEps : (J,J) 
        Eps prior precision matrix
    covarianceModel : OpenTURNS covariance model
        covariance kernel for the latent GP
    
    Returns
    -------
    param : list
        Mean + Cholesky precision matrix + Ntot value
        in the order required by NormalCholesky class
        Size : Nmax*(Nmax+1)+1
    """
    # Extract cuurent state of conditioning variables
    J = PrecEps.getDimension()
    N=len(D)
    Ntot = int(x[-J-1])
    ftot = ot.Matrix(np.array(x)[:Ntot].reshape(-1,1))
    Pi = np.array(x)[2*Nmax:2*Nmax+2*(Ntot-N)].reshape(-1,2)
    # total (augmented) data
    Dtot = ot.Sample(Ntot, 2)
    Dtot[:N] = D
    Dtot[N:] = Pi
    # precision matrix
    K = covarianceModel.computeCrossCovariance(Dtot,Dtot)
    K = ot.CovarianceMatrix(K)
    L = K.computeCholesky()
    Linv = L.inverse()
    # Kinv = Linv.transpose()*Linv   
    Utot = U(Dtot)    
    LU = ot.Matrix( Linv * Utot ) 
    Q = LU.transpose() * LU + PrecEps
    # invert 
    M = ot.CovarianceMatrix(Q).computeCholesky()
    Minv = M.inverse()
    V = Minv.transpose() * Minv
    mean = V * LU.transpose() * (Linv * ftot)
    # extract parameters in correct order (coherent with getParameter() method of RV_f)
    parameter = [0]*( J*(J+1)+1 )
    parameter[:J] = np.array(mean).ravel()
    parameter[J:-1] = np.array(Minv).ravel()
    parameter[-1] = J
    return parameter

if __name__ == "__main__":
    
    ####################
    # Generative model #
    ####################
    
    # Assuming square domain [0,1]*[0,1] (surface 1)
    # and null trend
    
    lambdaBar = 10
    T = 50
    
    def U(xy):
        u = [0, 0]
        u[0] = (xy[0]>0.5)*(xy[1]>0.5) + (xy[0]<=0.5)*(xy[1]<=0.5)
        u[1] = 1 - u[0]
        return u

    U_OT = ot.PythonFunction( 2, 2, U )
    Sigma_eps = ot.CovarianceMatrix( np.eye(2)*1E-1 ) 
    PrecEps = Sigma_eps.inverse()
    J = PrecEps.getDimension()
    
    # Add piecewise constant trend
    EpsTrue = np.array( ot.Normal( ot.Point(J), Sigma_eps ).getRealization() ).reshape(-1,1)*100
    # Utot = np.array( U_OT( XY_star ) )
    # mTot = np.dot( Utot, EpsTrue )

    def trend(X, eps=EpsTrue):
        Utot = np.array( U_OT( X ) )
        mTot = np.dot( Utot, eps )
        return mTot
    
    # GP model specification
    covarianceModel = ot.SquaredExponential([0.5, 0.5], [10.0])
    m = ot.PythonFunction(2, 1, trend)
    
    # Homogeneous augmented Poisson process Size
    Poisson = ot.Poisson(lambdaBar * T)
    
    # Upper bound on size of augmented Poisson process
    Nmax = int(Poisson.computeQuantile(1-1e-20)[0])*2
    
    # where to save results (figures)
    savedir = os.path.join( os.environ['HOME'], "sigma_gp_results")
    if not os.path.exists(savedir): os.mkdir( savedir )

    
    # Zoning covariables : two zones with a four-sided intersection point
    J = len(EpsTrue) # number of zones


    ###################
    # Data generation #
    ###################
    
    # Simulate according to homogogeneous Poisson process
    N_star = int(Poisson.getRealization()[0])
    myUniform = ot.ComposedDistribution([ot.Uniform(0, 1)]*2)
    XY_star = myUniform.getSample(N_star)
    mesh = ot.Mesh(XY_star)
    
    # apply trend function to mesh and create Gaussian process
    mTrend = ot.TrendTransform(m, mesh)
    Ftot = ot.GaussianProcess(mTrend, covarianceModel, mesh)
    
    # # Sigma GP process
    field_function = ot.PythonFieldFunction(mesh, 1, mesh, 1, sigmoid)
    process = ot.CompositeProcess(field_function, Ftot) 
    field_f = process.getRealization()
     
    # Use thinning
    p_accept = np.array( field_f.getValues() ).ravel()
    accepted = np.array( ot.Uniform(0, 1).getSample(N_star) ).ravel() <= p_accept 
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
    fD = np.array(sigmoid_inv(field_f))[accepted]
    fPi = np.array(sigmoid_inv(field_f))[accepted==False]
    ftot = np.vstack((fD,fPi,[[0]]*(Nmax-Ntot)))
    
    
    #######################
    # TEST ON TOY DATASET #
    #######################

    # Plot the data
    fig = plt.figure()
    plt.scatter( D[:,0], D[:,1], c="r", marker="+", s=100 )
    # plt.show()
    plt.savefig(os.path.join(savedir, "Data.png"))
    plt.close()
    
    ###################
    # MCMC parameters # 
    ###################
    
    sampleSize=100#0
    blockSize=10#0 # Display convergence messages after every block of iterations with size: blockSize
    ninits = 3 # Number of chains run for Gelman-Rubin convergence diagnostic    

    f_indices = [i for i in range(Nmax)]
    # Augmented Gaussian Process update
    RV_f = ot.RandomVector(NormalCholesky(mu=np.zeros(Nmax), Chol=np.diag([1]*N+[0]*(Nmax-N)), Ntot=Ntot))
    ot_link_function_f = ot.PythonFunction(int(4*Nmax-2*N+J+1), int(Nmax*(Nmax+1)+1), lambda x:py_link_function_f(x,Nmax=Nmax, D=D, U=U_OT, covarianceModel=covarianceModel))

    # Latent Poisson and Gaussian Process update
    Pi_indices = [i for i in range(N,Nmax)]+[i for i in range(2*Nmax,4*Nmax-2*N+1)]
    PyRV_Pi = PoissonGaussianProcess(ftot=ftot, Pi=Dtot[N:], Ntot=Ntot, Eps=EpsTrue, D=D, U=U_OT, covarianceModel=covarianceModel, Poisson=Poisson, myUniform=myUniform )
    RV_Pi = ot.RandomVector(PyRV_Pi)
    ot_link_function_Pi = ot.PythonFunction(int(4*Nmax-2*N+J+1), int(3*Nmax-2*N+J+1), lambda x:py_link_function_Pi(x,Nmax=Nmax,N=N, J=J))
    
    # Latent Polya Gamma Process update
    w_indices = [i for i in range(Nmax,2*Nmax)]
    RV_w = ot.RandomVector(PolyaGammaProcess(ftot=np.concatenate([np.array(field_f).ravel(), np.zeros(Nmax-Ntot)]), Ntot=Ntot))
    ot_link_function_w = ot.PythonFunction(4*Nmax-2*N+J+1, Nmax+1, lambda k:py_link_function_w(k,Nmax=Nmax, J=J))
    
    # Latent zone effects update
    Eps_indices = [i for i in range(4*Nmax-2*N+1,4*Nmax-2*N+J+1)]
    RV_Eps = ot.RandomVector(NormalCholesky(mu=np.zeros(J), Chol=np.eye(J), Ntot=J))
    ot_link_function_Eps = ot.PythonFunction(4*Nmax-2*N+J+1, J*(J+1)+1, lambda x:py_link_function_Eps(x,Nmax=Nmax, D=D, U=U_OT, PrecEps=PrecEps, covarianceModel=covarianceModel))
    
    # PLOT Real GP trajectory on meshgrid over search domain
    gridsize = 20
    xx, yy = np.meshgrid( np.linspace(0, 1, gridsize), np.linspace(0, 1, gridsize) )
    XY_new = np.vstack(( xx.ravel(), yy.ravel() )).T
    Z_True = PyRV_Pi.SimulateSigmaGP( XY_new )
    Z_True = np.array(Z_True).reshape(gridsize, gridsize) * lambdaBar * T
    levels = np.linspace( Z_True.min(), Z_True.max(), gridsize )
    fig = plt.figure()
    plt.contourf(xx, yy, Z_True, levels)
    plt.colorbar()
    plt.scatter( D[:,0], D[:,1], c="r", marker="+", s=100 )
    # plt.show()
    plt.savefig(os.path.join(savedir, "True_GP_trend.png"))
    plt.close()
    
    # TEST latent GP update
    RV_f.getRealization()
    RV_f.getParameter()
    # TEST latent Poisson + GP update
    RV_Pi.getRealization()
    RV_Pi.getParameter()
    # TEST latent Polya-Gamma
    RV_w.getRealization()
    RV_w.getParameter()
    # TEST latent Zone effects
    RV_Eps.getRealization()
    RV_Eps.getParameter()
            
    ###############
    # Launch MCMC #
    ###############
    
    samples = []
    randinits = []
    
    for i in range(ninits):
        # break
        # Random initialization
        randinit = np.zeros(4*Nmax-2*N+1+J)
        Ntot_init = 0
        while Ntot_init < N:
            Ntot_init = int(ot.Poisson(lambdaBar * T).getRealization()[0])
        randinit[-1-J] = Ntot_init
        NPi_init = int(Ntot_init - N)
        Pi_init = np.array(myUniform.getSample(NPi_init)).ravel()
        randinit[2*Nmax:2*Nmax+2*NPi_init] = Pi_init 
        randinit[Nmax:Nmax+Ntot_init] = random_polyagamma(size=Ntot_init)
        # check whether useful:
        randinit[-J:] = np.array( ot.Normal( ot.Point(J), Sigma_eps ).getRealization() )
        randinits.append(randinit)
        # Assemble Gibbs sampler
        print("random init %s out of %s: %s"%(str(i+1),str(ninits),str(randinits[i])))
        f_sampler = ot.RandomVectorMetropolisHastings( RV_f, randinits[i], f_indices, ot_link_function_f )
        Pi_sampler = ot.RandomVectorMetropolisHastings( RV_Pi, randinits[i], Pi_indices, ot_link_function_Pi ) 
        w_sampler = ot.RandomVectorMetropolisHastings( RV_w, randinits[i], w_indices, ot_link_function_w )
        Eps_sampler = ot.RandomVectorMetropolisHastings( RV_Eps, randinits[i], Eps_indices, ot_link_function_Eps )
        Gibbs_sampler = ot.Gibbs([f_sampler, Pi_sampler, w_sampler, Eps_sampler])
        # test samplers 
        f_sampler.getSample(blockSize)
        Pi_sampler.getSample(blockSize)
        w_sampler.getSample(blockSize)
        Eps_sampler.getSample(blockSize)
        Gibbs_sampler.getSample(blockSize)
        t1=time.time()
        sample = np.zeros((0,4*Nmax-2*N+J+1))
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
    
    ################################
    # MCMC Convergence diagnostics #
    ################################
    
    colors = list(mcolors.BASE_COLORS)[:ninits]
    burnin=0
    paramDim = sample.shape[1]
    # plotDim = 1
    
    components = [j for j in range(paramDim-1-J,paramDim)] 
    names = [r"$N_{tot}$"] + [r"$\epsilon_{%s}$"%j for j in range(1,J+1)]
    true_values = [Ntot] + EpsTrue.ravel().tolist()
    
    # MCMC convergence plot for Ntot
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
    plt.savefig(os.path.join(savedir, "traceplots.png"))
    plt.close()
    
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
    plt.savefig(os.path.join(savedir, "ACF.png"))
    plt.close()
    
    
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
    plt.savefig(os.path.join(savedir, "Gelman_Rubin.png"))
    plt.close()
    
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
    plt.savefig(os.path.join(savedir, "Ntot_post_density.png"))
    plt.close()
    
    #######################################
    # Predict GP throughout search domain #
    #######################################
    
    Z_new = np.zeros((len(sample), len(XY_new)))    
    for i in range(len(sample)):
        # break
        # GP conditional on values at augmented Poisson process
        PyRV_Pi.setParameter(py_link_function_Pi(sample[i], Nmax, N))
        Z_new[i] = PyRV_Pi.SimulateSigmaGP( XY_new )
        
    Z_mean = Z_new.mean(axis=0).reshape(gridsize, gridsize) * lambdaBar * T
    levels_mean = np.linspace( Z_mean.min(), Z_mean.max(), gridsize )
    
    Z_std = Z_new.std(axis=0).reshape(gridsize, gridsize) * lambdaBar * T
    levels_std = np.linspace( Z_std.min(), Z_std.max(), gridsize)
    
    fig = plt.figure()
    plt.contourf(xx, yy, Z_mean, levels_mean)
    plt.colorbar()
    plt.scatter( D[:,0], D[:,1], s=100, c='r', marker='+' )
    plt.title("Poisson intensity posterior mean vs Data")
    plt.savefig(os.path.join(savedir, "f_post_mean.png"))
    plt.close()
    
    fig = plt.figure()
    plt.contourf(xx, yy, Z_std, levels_std)
    plt.colorbar()
    plt.scatter( D[:,0], D[:,1], s=100, c='r', marker='+' )
    plt.title("Poisson intensity Posterior std vs Data")
    plt.savefig(os.path.join(savedir, "f_post_std.png"))
    plt.close()
    
    
    
    
    
    
    
    






