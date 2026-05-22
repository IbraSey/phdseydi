##%

import numpy as np
import openturns as ot

def phi(x, j, L):
    """
    Fourier decomposition of square exponential kernel

    Args:
        x (array): evaluation points
        j (int): Fourier mode index
        L (float): domain radius

    Returns:
        float: j-th Fourier mode evaluated at x
    
    Note:
        x must be between -L and +L 
    """
    return np.sin( np.pi*j*(x+L) / (2*L) ) / np.sqrt(L)

class sparseGP:
    """Fourier decomposition of 2D GP with squared exponential kernel
    """
    
    def __init__(self, hypers):
        """sparseGP class constructor

        Args: hypers (7,)
            Hyperparameters defining the GP model:
            in the following order:
            * l1 (float): x correlation length 
            * l2 (float): y correlation length 
            * c1 (float): centroid x-coord
            * c2 (float): centroid y-coord
            * S1 (float): x dataset radius
            * S2 (float): y dataset radius
            * nu (float): marginal variance
        """
        l1, l2, c1, c2, S1, S2, nu = hypers
        self.L1 = int( np.ceil( max(3.2*l1, 1.2*S1) ) )
        self.L2 = int( np.ceil( max(3.2*l2, 1.2*S2) ) )
        self.l1 = l1
        self.l2 = l2
        self.c1 = c1
        self.c2 = c2
        self.m1 = int( np.ceil( 1.75*self.L1/l1 ))
        self.m2 = int( np.ceil( 1.75*self.L2/l2 ))
        self.m = self.m1*self.m2
        self.S = np.zeros((2, self.m), int)
        self.S[0] = np.repeat( np.arange(self.m1), self.m2 )
        self.S[1] = list(range(self.m2))*self.m1
        self.Delta = 2*np.pi*nu*l1*l2*np.exp(-0.125*np.pi**2*((self.S[0]*l1/self.L1)**2 + (self.S[1]*l2/self.L2)**2 ))
        self.sqrt_Delta = np.sqrt(self.Delta).reshape(1, -1) 
        self.regressorOT = ot.MemoizeFunction( ot.PythonFunction( 2, self.m, self.regressorPy ) )
        self.evaluateOT = ot.MemoizeFunction( ot.PythonFunction( self.m+2, 1, self.evaluatePy ) )
            
    def regressorPy(self, x):
        """ sparseGP regressor value,
        evaluated at point x 
        
        Args:
            x (2,): query point
            
        Returns:
            (1, m): regressor
        """
        Phi_x = phi(x[0]-c1, self.S[0], self.L1) * phi(x[1]-c2, self.S[1], self.L2)
        return (Phi_x * self.sqrt_Delta)[0]
    
    
    def evaluatePy(self, x):
        """sparseGP pointwise evaluation

        Args:
            x : (m+2,) concatenation of m regression coefficients and 2D point coords
        
        Returns:
            (1,): regression mean at query point
        """
        beta = np.array(x[:self.m]).reshape(-1,1)
        pt = x[-2:]
        M = self.regressorMemo(pt)
        return np.linalg(M, M, beta)
    
            


if __name__ == "self":
    
    import matplotlib.pyplot as plt
    
    ##%
    
    #############################################
    #############################################
    # Simulate GP using exact and sparse method #
    #############################################
    #############################################
        
    hypers = [0.5, 5, 0, 0, 5, 5, 1.]
    l1, l2, c1, c2, S1, S2, nu = hypers
    
    my_sGP = sparseGP( hypers )
    L1, L2, m = my_sGP.L1, my_sGP.L2, my_sGP.m
    
    ###############################################
    # Compare exact and sparse covariance kernels #
    ###############################################
    
    kernel = ot.SquaredExponential([l1,l2], [np.sqrt(nu)])
    
    def sparse_kernel(x):
        """compute covariance kernel approximation

        Args:
            x : 2D point or sample of 2D points
        """
        m = my_sGP.m
        return( ot.Matrix( my_sGP.regressorOT(x) ) * (ot.Matrix( m, 1, [1]*m ) ) )
        
    
    x = np.linspace( -1, +1, 50 ) * my_sGP.L1
    y = np.linspace( -1, +1, 50 ) * my_sGP.L2
    
    X, Y = np.meshgrid( x, y )
    DOE = ot.Sample( np.vstack([X.ravel(), Y.ravel()]) .T )
    
    # exact kernel 
    K = np.array([ kernel( DOE[i] ) for i in range(len(DOE)) ]).ravel()
    sK = np.array( sparse_kernel( DOE ) )
    
    plt.figure(figsize=(2*L1, 4*L2))
    
    plt.subplot(121)
    
    plt.contourf( X, Y, K.reshape(X.shape), levels=10 )
    plt.colorbar()
    plt.title("Exact kernel")
    
    plt.subplot(122)
    
    plt.contourf( X, Y, K.reshape(X.shape), levels=10 )
    plt.colorbar()
    plt.title("Approached (sparse) kernel")
    
    plt.show()
    
    #####################################################
    # Simulate 1D trajectories from exact and sparse GP #
    #####################################################
    
    N = 5
    mesh = ot.Mesh(DOE)
    GP = ot.GaussianProcess( kernel, mesh )
    GP_sample = np.array( GP.getSample( N ) )[:,:,0]
    
    M = ot.Matrix( my_sGP.regressorOT(DOE) )
    Beta = ot.Matrix( ot.Normal(N).getSample(m) )
    
    sGP_sample = np.array(M*Beta).T
    
    plt.figure(figsize=(N*10, 10))
    
    for j in range(N):
        plt.subplot(2, N, 1+j)
        plt.contourf( X, Y, GP_sample[j].reshape(X.shape), levels=10 )
        if j == 0: plt.ylabel("Exact GP", fontsize=16)
        
        plt.subplot(2, N, N+1+j)
        plt.contourf( X, Y, sGP_sample[j].reshape(X.shape), levels=10 )
        if j == 0: plt.ylabel("Sparse GP", fontsize=16)
    
    plt.tight_layout()
    plt.show()
    
    
