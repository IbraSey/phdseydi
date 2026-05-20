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
    
    def __init__(self, l1, l2, c1, c2, S1, S2, nu, D):
        """sparseGP class constructor

        Args:
            l1 (float): x correlation length 
            l2 (float): y correlation length 
            c1 (float): centroid x-coord
            c2 (float): centroid y-coord
            S1 (float): x data radius
            S2 (float): y data radius
            nu (float): marginal variance
            D (N,2): Evaluation points
        """
        self.L1 = int( np.ceil( max(3.2*l1, 1.2*S1) ) )
        self.L2 = int( np.ceil( max(3.2*l2, 1.2*S2) ) )
        self.c1 = c1
        self.c2 = c2
        self.m1 = int( np.ceil( 1.75*L1/l1 ))
        self.m2 = int( np.ceil( 1.75*L2/l2 ))
        self.m = self.m1*self.m2
        self.S = np.zeros((2, self.m), int)
        self.S[0] = np.repeat( np.arange(self.m1), self.m2 )
        self.S[1] = list(range(self.m2))*self.m1
        self.Delta = 2*np.pi*nu*l1*l2*np.exp(-0.125*np.pi**2*((S[0]*l1/self.L1)**2 + (S[1]*l2/self.L2)**2 ))
        self.sqrt_Delta = np.sqrt(self.Delta).reshape(1, -1)
        self.D = D
        self.Phi_D = phi(D[:,0:1]-c1, self.S[0:1], self.L1) * phi(D[:,1:]-c2, self.S[1:], self.L2)
        self.design_D = self.PhiD*self.sqrtDelta
            
    def designNew(self, New):
        """Design matrix (regressor values)
        for sparseGP, evaluated at new points
        
        Args:
            New (n_New, 2): new points
            
        Returns:
            (n_New, m): design matrix
        """
        Phi_New = phi(New[:,0:1]-c1, self.S[0:1], self.L1) * phi(New[:,1:]-c2, self.S[1:], self.L2)
        return Phi_New * self.sqrt_Delta