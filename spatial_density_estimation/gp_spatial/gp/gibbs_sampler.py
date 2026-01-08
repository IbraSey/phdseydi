# %%
# =================================================================================================
# -------------------------------------------- IMPORTS --------------------------------------------
# =================================================================================================
from pathlib import Path
import os, sys
ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT))
import openturns as ot
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import math
from polyagamma import random_polyagamma
from shapely.geometry import Polygon, Point as ShapelyPoint
from shapely.prepared import prep
from visualizations.plot import plot_field
ot.RandomGenerator.SetSeed(42)


# %%
# =========================================================================================================
# -------------------------------------------- GIBBS POUR SGCP --------------------------------------------
# =========================================================================================================

class SGCP_GibbsSampler:
    """
    
    """
    def __init__(
        self,
        X_bounds,
        Y_bounds,
        T,
        Areas,
        nu,
        a_mu,
        b_mu,
        jitter=1e-5,          # Paramètre sensible : trop grand peut biaiser, trop petit pas significatif
        rng_seed=None,
    ):
        self.X_bounds = tuple(X_bounds)
        self.Y_bounds = tuple(Y_bounds)
        self.T = T
        self.Areas = Areas
        self.nu = ot.Point(nu)
        self.a_mu = a_mu
        self.b_mu = b_mu
        self.jitter = jitter

        if rng_seed is not None:
            ot.RandomGenerator.SetSeed(rng_seed)
            self.rng_state = ot.RandomGenerator.GetState()

        self.areas = [a[0] for a in self.Areas]
        self.epsilons = [a[1] for a in self.Areas]
        self.J = len(self.areas)


    # ==================================
    # ----------- Outillage ------------
    # ==================================
    @staticmethod
    def sigma(z):
        """
        
        """
        z_array = np.array(z)
        return ot.Point(1.0 / (1.0 + np.exp(-z_array)))

    def compute_kernel(self, XY_data, XY_new=None):
        """
        
        """
        nu0, nu1, nu2 = map(float, self.nu)

        if not isinstance(XY_data, ot.Sample):
            XY_data = ot.Sample(np.asarray(XY_data, dtype=float).tolist())
        N_data = XY_data.getSize()

        kernel = ot.SquaredExponential([nu1, nu2], [nu0])

        if XY_new is None:
            K = kernel.discretize(XY_data)
            return ot.CovarianceMatrix(np.array(K, dtype=float).tolist())

        if not isinstance(XY_new, ot.Sample):
            XY_new = ot.Sample(np.asarray(XY_new, dtype=float).tolist())
        N_new = XY_new.getSize()

        XY_all = ot.Sample(N_data + N_new, 2)
        for i in range(N_data):
            XY_all[i, 0] = XY_data[i, 0]
            XY_all[i, 1] = XY_data[i, 1]
        for i in range(N_new):
            XY_all[N_data + i, 0] = XY_new[i, 0]
            XY_all[N_data + i, 1] = XY_new[i, 1]
        K_all = kernel.discretize(XY_all)

        K_dd = ot.CovarianceMatrix(N_data)
        for i in range(N_data):
            for j in range(i, N_data):
                K_dd[i, j] = K_all[i, j]

        K_new_data = ot.Matrix(N_new, N_data)
        for i in range(N_new):
            for j in range(N_data):
                K_new_data[i, j] = K_all[N_data + i, j]

        K_new_new = ot.CovarianceMatrix(N_new)
        for i in range(N_new):
            for j in range(i, N_new):
                K_new_new[i, j] = K_all[N_data + i, N_data + j]

        return K_dd, K_new_data, K_new_new

    def compute_U_from_areas(self, D_xy):
        """
        
        """
        if not isinstance(D_xy, ot.Sample):
            D_xy_sample = ot.Sample(np.asarray(D_xy, dtype=float).tolist())
        else:
            D_xy_sample = D_xy
            
        n = D_xy_sample.getSize()
        U = ot.Sample(n, self.J)
        
        for k in range(n):
            pt = D_xy_sample[k]
            pt_shapely = ShapelyPoint(pt[0], pt[1])
            for j, P in enumerate(self.areas):
                if P.covers(pt_shapely):
                    U[k, j] = 1.0
                    break 
        return ot.Matrix(U)

    def sample_candidats(self, N):
        """
        
        """
        xmin, xmax = self.X_bounds
        ymin, ymax = self.Y_bounds
        
        marginal_x = ot.Uniform(xmin, xmax)
        marginal_y = ot.Uniform(ymin, ymax)
        distribution = ot.ComposedDistribution([marginal_x, marginal_y])
        
        return distribution.getSample(int(N))
    

    # =========================================
    # ------ Posteriors conditionnelles -------
    # =========================================
    def update_mu_tilde(self, Z, Pi_S):
        """
        
        """
        xmin, xmax = self.X_bounds
        ymin, ymax = self.Y_bounds
        
        N_0 = sum(1 for z in Z if z == 0.0)
        N_Pi = Pi_S.getSize()
        a_post = self.a_mu + N_0 + N_Pi
        b_post = self.b_mu + self.T * (xmax - xmin) * (ymax - ymin)
        mu_tilde = ot.Gamma(a_post, b_post, 0.0).getRealization()[0]
        
        return float(mu_tilde)
    
    def update_epsilons(self, f_Df, K_ff, U):
        """
        
        """
        n = K_ff.getDimension()

        K_cov_reg = ot.CovarianceMatrix(K_ff)
        for i in range(n):
            K_cov_reg[i, i] += float(self.jitter)    # Régularisation
        K_inv = K_cov_reg.inverse()

        U_mat = ot.Matrix(U)
        U_T = U_mat.transpose()

        A_mat = U_T * K_inv * U_mat + ot.IdentityMatrix(self.J)    # A = U^T K^{-1} U + I
        A_array = np.array(A_mat)
        A_array = 0.5 * (A_array + A_array.T)    # Symétrisation, passage par numpy peut être pas nécessaire
        A_array += float(self.jitter) * np.eye(self.J)
        A_cov = ot.CovarianceMatrix(A_array.tolist())
        Sigma_sym = A_cov.inverse()
        Sigma = ot.CovarianceMatrix(np.array(Sigma_sym).tolist())
        
        B = U_T * (K_inv * f_Df) 
        mu = Sigma * B
        
        epsilons = ot.Normal(mu, Sigma).getRealization()
        
        return epsilons

    def update_f(self, x, y, eps, Z, omega_D0, Pi_S):
        """
        
        """
        idx = [i for i in range(len(Z)) if Z[i] == 0.0]    # Indices des points D_0
        N_0 = len(idx)

        if N_0 == 0:
            raise ValueError("N_0 = 0 ; Pas possible pour SGCP")

        # 1) D_0 -> (x,y) + omega_D_0
        D_0 = ot.Sample(N_0, 2)
        omega_D_0 = ot.Point(N_0)
        for k, i in enumerate(idx):
            D_0[k, 0] = x[i]
            D_0[k, 1] = y[i]
            omega_D_0[k] = omega_D0[i]

        # 2) Pi_S -> PiS(x,y) + omega_Pi
        N_Pi = Pi_S.getSize()
        if N_Pi > 0:
            PiS = ot.Sample(N_Pi, 2)
            omega_Pi = ot.Point(N_Pi)
            for i in range(N_Pi):
                PiS[i, 0] = Pi_S[i, 0]
                PiS[i, 1] = Pi_S[i, 1]
                omega_Pi[i] = Pi_S[i, 2]
        else:
            PiS = ot.Sample(0, 2)
            omega_Pi = ot.Point(0)

        # 3) D_f = D_0 U Pi_S
        N_f = N_0 + N_Pi
        D_f = ot.Sample(N_f, 2)
        for i in range(N_0):
            D_f[i, 0] = D_0[i, 0]
            D_f[i, 1] = D_0[i, 1]
        for i in range(N_Pi):
            D_f[N_0 + i, 0] = PiS[i, 0]
            D_f[N_0 + i, 1] = PiS[i, 1]

        # 4) U et m_f = U * eps
        U = self.compute_U_from_areas(D_f)     
        eps_mat = ot.Matrix([[float(eps[j])] for j in range(self.J)])    
        m_f_mat = U * eps_mat    
        m_f = ot.Point([float(m_f_mat[i, 0]) for i in range(N_f)])

        # 5) K_ff
        K_ff = self.compute_kernel(D_f)
        for i in range(N_f):
            K_ff[i, i] += float(self.jitter)    # Régularisation
        K_inv = K_ff.inverse()

        # 6) Omega 
        omega_diag = ot.Point(N_f)
        for i in range(N_0):
            omega_diag[i] = omega_D_0[i]
        for i in range(N_Pi):
            omega_diag[N_0 + i] = omega_Pi[i]

        Omega = ot.CovarianceMatrix(N_f)
        for i in range(N_f):
            Omega[i, i] = omega_diag[i]

        # 7) u = [0.5,...,0.5, -0.5,...,-0.5]
        u = ot.Point(N_f)
        for i in range(N_0):
            u[i] = 0.5
        for i in range(N_Pi):
            u[N_0 + i] = -0.5

        # 8) Sigma_f et mu_f
        A_mat = Omega + K_inv
        A_array = np.array(A_mat)
        A_array = 0.5 * (A_array + A_array.T)    # Symétrisation, passage par numpy peut être pas nécessaire
        A_array += float(self.jitter) * np.eye(N_f)
        A = ot.CovarianceMatrix(A_array.tolist())
        Sigma_f_sym = A.inverse()
        Sigma_f = ot.CovarianceMatrix(np.array(Sigma_f_sym).tolist())

        temp = K_inv * m_f + u
        mu_f = Sigma_f * temp

        f_new = ot.Normal(mu_f, Sigma_f).getRealization()
        
        return f_new, D_f, U, K_ff, m_f
    
    def sample_Pi_S(self, mu_tilde, X_data, Y_data, f_data, eps):
        """
        
        """
        N = X_data.getSize()
        xmin, xmax = self.X_bounds
        ymin, ymax = self.Y_bounds
        area = (xmax - xmin) * (ymax - ymin)

        # 1) Candidats et données observées
        mean_cand = float(mu_tilde * self.T * area)
        #mean_cand = min(mean_cand, 200)          # Critère pour éviter explosion du nombre de candidats
        N_cand = int(ot.Poisson(mean_cand).getRealization()[0])
        #N_cand = min(N_cand, 200)           # Critère pour éviter explosion du nombre de candidats
        if N_cand == 0:
            return ot.Sample(0, 3)

        XY_cand = self.sample_candidats(N_cand)
        XY_data = ot.Sample([[X_data[i], Y_data[i]] for i in range(N)])

        # 2) GP conditionnel avec calcul moyennes et calcul kernels (cf écriture Merlin) 
        U_data_mat = self.compute_U_from_areas(XY_data)
        U_cand_mat = self.compute_U_from_areas(XY_cand)
        eps_col = ot.Matrix([[float(eps[j])] for j in range(self.J)])
        m_data_mat = U_data_mat * eps_col
        m_cand_mat = U_cand_mat * eps_col
        m_data = ot.Point([float(m_data_mat[i, 0]) for i in range(N)])
        m_cand = ot.Point([float(m_cand_mat[i, 0]) for i in range(N_cand)])

        K_dd, K_star_d, K_star_star = self.compute_kernel(XY_data, XY_cand)
        K_dd_reg = ot.CovarianceMatrix(K_dd)
        for i in range(N):
            K_dd_reg[i, i] += float(self.jitter)    # Régularisation
        K_inv = K_dd_reg.inverse()

        delta = f_data - m_data
        mu_star = m_cand + K_star_d * (K_inv * delta)

        Sigma_star_mat = K_star_star - K_star_d * (K_inv * K_star_d.transpose())
        Sigma_array = np.array(Sigma_star_mat)
        Sigma_array = 0.5 * (Sigma_array + Sigma_array.T)    # Symétrisation, passage par numpy peut être pas nécessaire
        Sigma_array += float(self.jitter) * np.eye(N_cand)    # Régularisation
        Sigma_star = ot.CovarianceMatrix(Sigma_array.tolist())
        
        f_star = ot.Normal(mu_star, Sigma_star).getRealization()

        # 3) Phase de Thinning
        accept_probs = self.sigma(-f_star)
        Uu = ot.Uniform(0.0, 1.0).getSample(N_cand)
        mask = [i for i in range(N_cand) if Uu[i, 0] < accept_probs[i]]
        if len(mask) == 0:
            return ot.Sample(0, 3)

        XY_acc = ot.Sample(len(mask), 2)
        f_acc = np.zeros(len(mask))
        for k, i in enumerate(mask):
            XY_acc[k, 0] = XY_cand[i, 0]
            XY_acc[k, 1] = XY_cand[i, 1]
            f_acc[k] = f_star[i]

        # 4) Construction de Pi_S
        omega_acc = random_polyagamma(1.0, f_acc)
        n_acc = len(omega_acc)
        Pi_S = ot.Sample(n_acc, 3)
        for i in range(n_acc):
            Pi_S[i, 0] = XY_acc[i, 0]
            Pi_S[i, 1] = XY_acc[i, 1]
            Pi_S[i, 2] = omega_acc[i]

        return Pi_S


    # =========================================
    # ------------- Run du Gibbs --------------
    # =========================================
    def run(self, t, x, y, eps_init, mu_init, n_iter=1000, verbose=True, verbose_every=100):
        """
        
        """
        N = len(t)

        # Seulelent évènements de fond : ETAS = 0
        Z = ot.Point([0.0] * N)
        
        # Initialisations
        eps = ot.Point(eps_init)
        mu_tilde = float(mu_init)
        XY_data = ot.Sample([[x[i], y[i]] for i in range(N)])
        U_data = self.compute_U_from_areas(XY_data)
        f_data = ot.Point(U_data * eps)      # f_data initialisé à m_f
        
        # Stockage
        mu_chain = np.zeros(n_iter)
        eps_chain = np.zeros((n_iter, self.J))
        nPi_chain = np.zeros(n_iter, dtype=int)
        fdata_chain = np.zeros((n_iter, N)) 
        
        if verbose:
            print("="*100)
            print("-"*30 + f" Démarrage Gibbs: {n_iter} itérations, N={N} " + "-"*30)
            print("="*100 + "\n")
        
        for it in range(n_iter):
            try:
                # 1) omega_D0 | ...
                omega_D0 = ot.Point(random_polyagamma(1.0, f_data))
                
                # 2) Pi_S | ...
                Pi_S = self.sample_Pi_S(mu_tilde, x, y, f_data, eps)

                # 3) f_Df | ...
                f_Df, D_f_xy, U_Df, K_ff, m_f = self.update_f(x, y, eps, Z, omega_D0, Pi_S)
                f_data = ot.Point([f_Df[i] for i in range(N)])
                
                # 4) eps | ...
                eps = self.update_epsilons(f_Df, K_ff, U_Df)

                # 5) mu_tilde | ...
                mu_tilde = self.update_mu_tilde(Z, Pi_S)
                
                # Affichage
                if verbose and (it % verbose_every == 0 or it == n_iter - 1):
                    eps_arr = np.array(eps)
                    n_pi = Pi_S.getSize()
                    
                    print(
                        f"[Gibbs iteration {it}] "
                        f"mu = {mu_tilde:4f} | "
                        f"|Pi| = {n_pi} | "
                        f"eps = {eps_arr} "
                    )
                
                # Stockage
                mu_chain[it] = mu_tilde
                eps_chain[it, :] = np.array(eps)
                nPi_chain[it] = Pi_S.getSize()
                fdata_chain[it, :] = np.array(f_data)
                    
            except Exception as e:
                print(f"\nErreur iteration {it} : {e}")
                raise
        
        if verbose:
            print("\n" + "="*100)
            print("-"*41 + " Gibbs terminé !! " + "-"*41)
            print("="*100)
        
        return {
            "mu_tilde": mu_chain,
            "eps": eps_chain,
            "nPi": nPi_chain,
            "f_data": fdata_chain,
            "last_state": {
                "mu_tilde": mu_tilde,
                "eps": np.array(eps),
                "nu": list(self.nu)
            },
        }
    

    # =====================================================================================
    # ---------------------------- Analyse postérieure ------------------------------------
    # =====================================================================================
    def posterior_summary(self, results, burn_in=0.3):
        """
        
        """
        mutilde_chain = np.asarray(results["mu_tilde"])
        eps_chain = np.asarray(results["eps"])
        f_chain = np.asarray(results["f_data"])
        burn = int(len(mutilde_chain) * burn_in)

        return {
            "mutilde_hat": float(mutilde_chain[burn:].mean()),
            "eps_hat": eps_chain[burn:].mean(axis=0),
            "f_data_hat": f_chain[burn:].mean(axis=0),
        }
    
    def posterior_gp(self, XY_data, f_data_hat, mesh, eps_hat):
        """
        
        """
        # Extraction des vertices du mesh
        XY_grid = mesh.getVertices()
        
        # Moyennes (U * eps)
        U_data = self.compute_U_from_areas(XY_data)
        U_grid = self.compute_U_from_areas(XY_grid)
        eps_col = ot.Matrix([[float(eps_hat[j])] for j in range(self.J)])
        m_data_mat = U_data * eps_col
        m_grid_mat = U_grid * eps_col
        m_data = ot.Point([float(m_data_mat[i, 0]) for i in range(U_data.getNbRows())])
        m_grid = ot.Point([float(m_grid_mat[i, 0]) for i in range(U_grid.getNbRows())])
        
        # Kernels
        N = XY_data.getSize()
        M = XY_grid.getSize()
        K_dd, K_gd, K_gg = self.compute_kernel(XY_data, XY_grid)
        
        K_dd_reg = ot.CovarianceMatrix(K_dd)
        for i in range(N):
            K_dd_reg[i, i] += float(self.jitter)    # Régularisation
        K_inv = K_dd_reg.inverse()

        # Moyenne postérieure : mu_post = m_grid + K_gd * K_dd^{-1} * (f_data - m_data)
        delta = f_data_hat - m_data
        mu_post = m_grid + K_gd * (K_inv * delta)

        # Covariance postérieure : Sigma_post = K_gg - K_gd * K_dd^{-1} * K_dg
        Sigma_post_mat = ot.Matrix(K_gg) - K_gd * (K_inv * K_gd.transpose())
        Sigma_post_np = np.array(Sigma_post_mat, dtype=float)
        Sigma_post_np = 0.5 * (Sigma_post_np + Sigma_post_np.T)   # Symétrisation, passage par numpy peut être pas nécessaire
        Sigma_post_np += float(self.jitter) * np.eye(M)     # Régularisation
        Sigma_post = ot.CovarianceMatrix(Sigma_post_np.tolist())
        
        return mu_post, Sigma_post

    def plot_posterior_intensity(
        self,
        x,
        y,
        t,
        results,
        nx=60,
        ny=60,
        burn_in=0.3
    ):
        """
        
        """
        post_sum = self.posterior_summary(results, burn_in)
        mutilde_hat = post_sum["mutilde_hat"]
        eps_hat = post_sum["eps_hat"]
        f_data_hat = post_sum["f_data_hat"]
        N = len(t)
        XY_data = ot.Sample([[x[i], y[i]] for i in range(N)])
        
        # Création du mesh 
        xmin, xmax = self.X_bounds
        ymin, ymax = self.Y_bounds
        interval = ot.Interval([xmin, ymin], [xmax, ymax])
        mesher = ot.IntervalMesher([nx - 1, ny - 1])      # nb d'arêtes
        mesh = mesher.build(interval)

        M = mesh.getVertices().getSize()
        if M > 10000:            # Critère pour éviter maillage trop grand (question de compléxité)
            raise ValueError(f"Maillage trop grand: {M} points.")

        mu_post_grid, Sigma_post_grid = self.posterior_gp(XY_data, f_data_hat, mesh, eps_hat)
        f_hat = mu_post_grid        # Estimateur de la moyenne a posteriori

        mu_hat = mutilde_hat * self.sigma(f_hat)         # Calcul de l'intensité estimée
        mu_sample = ot.Sample([[mu_hat[i]] for i in range(len(mu_hat))])
        mu_field = ot.Field(mesh, mu_sample)

        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        # Subplot 1 : Données
        ax = axes[0]
        sc = ax.scatter(
            x,
            y,
            c=t,
            s=18,
            alpha=0.7,
            edgecolors="black",
        )
        ax.set_title("Données observées (couleur = temps)")
        ax.set_xlim(self.X_bounds)
        ax.set_ylim(self.Y_bounds)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        plt.colorbar(sc, ax=ax).set_label("t")

        # Subplot 2 : Intensité avec plot_field
        ax = axes[1]
        plot_field(mu_field, mode="subplot", ax=ax, title=r"Intensité postérieure $\hat{\mu}(s)$", 
                   add_colorbar=True)
        ax.scatter(x, y, s=10, alpha=0.5, color="white", edgecolors="black", linewidths=0.5)
        ax.set_xlim(self.X_bounds)
        ax.set_ylim(self.Y_bounds)
        ax.grid(alpha=0.3, color="white", linewidth=0.5)

        # Titre global
        fig.suptitle(r"Analyse postérieure : $\hat{\mu}(s) = \hat{\tilde{\mu}} \cdot \sigma(\hat{f}(s))$", 
                     fontsize=13, fontweight="bold",)
        plt.tight_layout()
        plt.show()

        return {
            "mu_hat": mu_hat,
            "eps_hat": eps_hat,
            "f_data_hat": f_data_hat,
            "mu_post_grid": mu_post_grid,
            "Sigma_post_grid": Sigma_post_grid,
            "mu_field": mu_field,
            "mesh": mesh,
        }





