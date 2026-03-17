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
import arviz as az
from visualizations.plot import plot_field
ot.RandomGenerator.SetSeed(42)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel


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
        lambda_nu,
        nu,
        a_mu,
        b_mu,
        delta,                 # hyperparamètre qui module confiance au prior
        polygons,              # attention ordre, identique à Areas
        jitter=1e-5,           # Paramètre sensible : trop grand peut biaiser, trop petit pas significatif
        rng_seed=None,
    ):
        self.X_bounds = tuple(X_bounds)
        self.Y_bounds = tuple(Y_bounds)
        self.T = T
        self.Areas = Areas
        self.lambda_nu = lambda_nu
        self.nu = ot.Point(nu)
        self.a_mu = a_mu
        self.b_mu = b_mu
        self.delta = ot.Point(delta)
        self.jitter = jitter
        self.areas = [a[0] for a in self.Areas]
        self.epsilons = [a[1] for a in self.Areas]
        self.J = len(self.areas)
        self.polygons = polygons
        if rng_seed is not None:
            ot.RandomGenerator.SetSeed(int(rng_seed))
            self.rng_state = ot.RandomGenerator.GetState()
        
        self.centroids_xy, self.Sigma_eps = self.compute_Sigma_eps()
        Sigma_eps_reg = ot.CovarianceMatrix((self.Sigma_eps + self.jitter * np.eye(self.J)).tolist())
        self.Sigma_eps_inv = Sigma_eps_reg.inverse()        # Calcul fait une fois, pas besoind d'être répété
        self.sd = 2.4**2 / 2.0         # Coefficient optimal pour MH adaptive (Haario et al. (2001) ; Gelma et al. (1996))
        self.eps_MH = 1e-6
        self.proposal_cov = None       # Sera initialisé dans le run

    # ==========================================================================
    # ------------------------------- Outillage --------------------------------
    # ==========================================================================

    @staticmethod
    def sigma(z):
        z_array = np.array(z)
        return ot.Point(1.0 / (1.0 + np.exp(-z_array)))
    
    @staticmethod
    def _acf(x, max_lag):
        """
        
        """
        x = np.asarray(x)
        x = x - x.mean()
        n = len(x)

        var = np.dot(x, x) / n
        if var == 0.0:
            return np.zeros(max_lag + 1)

        acf_vals = np.empty(max_lag + 1)
        for k in range(max_lag + 1):
            acf_vals[k] = np.dot(x[: n - k], x[k:]) / (n * var)

        return acf_vals
    
    #@staticmethod
    def compute_Sigma_eps(self):
        """

        """
        delta0, delta1 = map(float, self.delta)
        centroids_xy = np.array([[p.centroid.x, p.centroid.y] for p in self.polygons])
        dx = centroids_xy[:, 0].reshape(len(self.polygons),-1) - centroids_xy[:, 0]
        dy = centroids_xy[:, 1].reshape(len(self.polygons),-1) - centroids_xy[:, 1]
        dist2 = dx * dx + dy * dy

        Sigma_eps = delta0 * np.exp(-dist2 / (2.0 * delta1 ** 2))
        Sigma_eps = 0.5 * (Sigma_eps + Sigma_eps.T)       # Symétrisation

        return centroids_xy, Sigma_eps

    def compute_kernel(self, XY_data, XY_new=None):
        """
        
        """
        nu0, nu1 = map(float, self.nu)

        if not isinstance(XY_data, ot.Sample):
            XY_data = ot.Sample(np.asarray(XY_data).tolist())
        N_data = XY_data.getSize()

        kernel = ot.SquaredExponential([nu1, nu1], [nu0])

        if XY_new is None:
            K = kernel.discretize(XY_data)
            return ot.CovarianceMatrix(np.array(K).tolist())

        if not isinstance(XY_new, ot.Sample):
            XY_new = ot.Sample(np.asarray(XY_new).tolist())
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
            D_xy_sample = ot.Sample(np.asarray(D_xy).tolist())
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
        sample = distribution.getSample(int(N))

        return sample
    
    def log_posterior_nu(self, nu_vals, f_Df, m_f, D_f_sample):
        """

        """
        nu0, nu1 = map(float, nu_vals)
        log_prior = - self.lambda_nu * (nu0 + nu1)

        kernel = ot.SquaredExponential([nu1, nu1], [nu0])
        N = D_f_sample.getSize()
        K_mat = kernel.discretize(D_f_sample)
        
        for i in range(N):
            K_mat[i, i] += self.jitter       # Régularisation
        
        K_ff = ot.CovarianceMatrix(K_mat)
        dist_normal = ot.Normal(m_f, K_ff)
        log_likelihood = dist_normal.computeLogPDF(f_Df)
        
        return log_likelihood + log_prior
    

    # =================================================================================
    # -------------------------- Posteriors conditionnelles ---------------------------
    # =================================================================================

    def update_mu_tilde(self, Z, Pi_S):
        """
        
        """
        xmin, xmax = self.X_bounds
        ymin, ymax = self.Y_bounds

        N_0 = sum(1 for z in Z if z == 0.0)
        N_Pi = Pi_S.getSize()

        shape = self.a_mu + N_0 + N_Pi
        rate = self.b_mu + self.T * (xmax - xmin) * (ymax - ymin)

        mu_tilde = ot.Gamma(shape, rate, 0.0).getRealization()[0]

        return float(mu_tilde)

    def update_epsilons(self, f_Df, K_ff, U):
        """

        """
        n = K_ff.getDimension()

        K_cov_reg = ot.CovarianceMatrix(K_ff)
        for i in range(n):
            K_cov_reg[i, i] += self.jitter        # Régularisation
        K_inv = K_cov_reg.inverse()

        U_mat = ot.Matrix(U)
        U_T = U_mat.transpose()

        A_mat = U_T * K_inv * U_mat + self.Sigma_eps_inv        # A = U^T K^{-1} U + Sigam_eps
        A_np = np.array(A_mat)
        A_np = 0.5 * (A_np + A_np.T)            # Symétrisation, passage par numpy peut être pas nécessaire
        A_np += self.jitter * np.eye(self.J)
        A_cov = ot.CovarianceMatrix(A_np.tolist())
        Sigma_post_sym = A_cov.inverse()
        Sigma_post = ot.CovarianceMatrix(np.array(Sigma_post_sym).tolist())

        B = U_T * (K_inv * f_Df)
        mu = Sigma_post * B

        return ot.Normal(mu, Sigma_post).getRealization()

    def update_f(self, x, y, eps, Z, omega_D0, Pi_S):
        """
        
        """
        idx = [i for i in range(len(Z)) if Z[i] == 0.0]         # Indices des points D_0
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
            K_ff[i, i] += self.jitter
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
        A_array = 0.5 * (A_array + A_array.T)
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
        if hasattr(X_data, "getSize"):
            N = X_data.getSize()
        else:
            N = len(X_data)

        # N = X_data.getSize()
        
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
        
        # ------- SÉCURITÉ ANTI-EXPLOSION -------
        LIMIT_CANDIDATES = 1500
        if N_cand > LIMIT_CANDIDATES:
            N_cand = LIMIT_CANDIDATES
        # ---------------------------------------

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
            K_dd_reg[i, i] += self.jitter             # Régularisation
        K_inv = K_dd_reg.inverse()

        delta = f_data - m_data
        mu_star = m_cand + K_star_d * (K_inv * delta)

        Sigma_star_mat = K_star_star - K_star_d * (K_inv * K_star_d.transpose())
        Sigma_array = np.array(Sigma_star_mat)
        Sigma_array = 0.5 * (Sigma_array + Sigma_array.T)          # Symétrisation, passage par numpy peut être pas nécessaire
        Sigma_array += self.jitter * np.eye(N_cand)          # Régularisation
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

    # def update_nu_MH(self, f_Df, D_f, m_f, sigma_nu_RWMH=0.05):
    #     """

    #     """
    #     nu0, nu1 = map(float, self.nu)
    #     current_nu = [nu0, nu1]
        
    #     # log_post à l'état actuel
    #     log_post_current = self.log_posterior_nu(current_nu, f_Df, m_f, D_f)
        
    #     # log_post proposé (avec marche aléatoire log-normale)
    #     mu = ot.Point(2, 0.0)
    #     Sigma = ot.CovarianceMatrix(2, [sigma_nu_RWMH, 0.0, 0.0, sigma_nu_RWMH])
    #     perturbation = ot.Normal(mu, Sigma).getRealization()
    #     proposed_nu = [
    #         current_nu[0] * np.exp(perturbation[0]),
    #         current_nu[1] * np.exp(perturbation[1])
    #     ]

    #     log_post_proposal = self.log_posterior_nu(proposed_nu, f_Df, m_f, D_f)

    #     # Correction avec ratio d'Hastings 
    #     log_q_correction = np.sum(np.log(proposed_nu)) - np.sum(np.log(current_nu))
        
    #     log_accept_ratio = (log_post_proposal - log_post_current) + log_q_correction
    #     if np.log(ot.Uniform(0.0, 1.0).getRealization()) < log_accept_ratio:
    #         self.nu = ot.Point(proposed_nu)
    #         return self.nu, True
    #     else:
    #         return self.nu, False
        
    def update_nu_MH(self, f_Df, D_f, m_f, history_log_nu, it, step_nu_init=0.01):
        """

        """
        nu0, nu1 = map(float, self.nu)
        current_nu = [nu0, nu1]
        current_log_nu = np.log(current_nu)

        # Log-posterior actuelle
        log_post_current = self.log_posterior_nu(current_nu, f_Df, m_f, D_f)

        t0 = 50  
        if it > t0 and len(history_log_nu) > t0:
            cov_matrix = np.cov(np.array(history_log_nu).T)
            self.proposal_cov = self.sd * cov_matrix + self.sd * self.eps_MH * np.eye(2)
        elif self.proposal_cov is None:
             self.proposal_cov = step_nu_init * np.eye(2)

        # Génération du candidat (random walk multivariée sur le log)
        perturbation = np.random.multivariate_normal(np.zeros(2), self.proposal_cov)
        proposed_log_nu = current_log_nu + perturbation
        proposed_nu = np.exp(proposed_log_nu).tolist()
        log_post_proposal = self.log_posterior_nu(proposed_nu, f_Df, m_f, D_f)

        # Correction avec ratio d'Hastings 
        log_q_correction = np.sum(proposed_log_nu) - np.sum(current_log_nu)

        log_accept_ratio = (log_post_proposal - log_post_current) + log_q_correction

        # Acceptation / Rejet
        if np.log(np.random.rand()) < log_accept_ratio:
            self.nu = ot.Point(proposed_nu)
            return self.nu, True
        else:
            return self.nu, False



    # =====================================================================================
    # ----------------------------------- Run du Gibbs ------------------------------------
    # =====================================================================================

    def run(self, t, x, y, eps_init, mutilde_init, step_nu_init=0.01, n_iter=1000, verbose=True, verbose_every=100):
        N = len(t)

        # Seulement évènements de fond : ETAS = 0
        Z = ot.Point([0.0] * N)

        # Initialisations
        eps = ot.Point(eps_init)
        mu_tilde = mutilde_init
        XY_data = ot.Sample([[x[i], y[i]] for i in range(N)])
        U_data = self.compute_U_from_areas(XY_data)
        f_data = ot.Point(U_data * eps)

        # Stockage
        mu_chain = np.zeros(n_iter)
        eps_chain = np.zeros((n_iter, self.J))
        nPi_chain = np.zeros(n_iter)
        fdata_chain = np.zeros((n_iter, N))
        nu_chain = np.zeros((n_iter, 2))
        acc_nu_count = 0
        history_log_nu = []          # Liste qui stocker les log(nu) passés

        if verbose:
            print("\n" + "=" * 100)
            print(
                "-" * 29
                + f" Démarrage Gibbs : {n_iter} itérations, N={N} "
                + "-" * 29
            )
            print("=" * 100 + "\n")

        for it in range(n_iter):
            try:
                # 1) omega_D0 | ...
                omega_D0 = ot.Point(random_polyagamma(1.0, f_data))

                # 2) Pi_S | ...
                Pi_S = self.sample_Pi_S(mu_tilde, x, y, f_data, eps)

                # 3) f_Df | ...
                f_Df, D_f_xy, U_Df, K_ff, m_f = self.update_f(
                    x, y, eps, Z, omega_D0, Pi_S
                )
                f_data = ot.Point([f_Df[i] for i in range(N)])

                # 4) eps | ...
                eps = self.update_epsilons(f_Df, K_ff, U_Df)

                # 5) mu_tilde | ...
                mu_tilde = self.update_mu_tilde(Z, Pi_S)
                
                # # ============================================================
                # # 6) UPDATE NU 
                # # ============================================================
                # eps_mat = ot.Matrix([[float(eps[j])] for j in range(self.J)])
                # m_f_updated_mat = U_Df * eps_mat
                # m_f_updated = [m_f_updated_mat[i, 0] for i in range(m_f_updated_mat.getNbRows())]
                # #m_f_updated = U_Df * eps
                
                # new_nu, accepted = self.update_nu_MH(f_Df, D_f_xy, m_f_updated, sigma_nu_RWMH=step_nu_RWMH)
                
                # if accepted:
                #     acc_nu_count += 1
                # # ============================================================

                # ============================================================
                # 6) UPDATE NU (ADAPTIVE)
                # ============================================================
                # Recalcul moyenne m_f avec nouvel eps pour cohérence
                eps_mat = ot.Matrix([[float(eps[j])] for j in range(self.J)])
                m_f_updated_mat = U_Df * eps_mat
                m_f_updated = [m_f_updated_mat[i, 0] for i in range(m_f_updated_mat.getNbRows())]
                
                new_nu, accepted = self.update_nu_MH(f_Df, D_f_xy, m_f_updated, history_log_nu, it, step_nu_init)
                
                if accepted: 
                    acc_nu_count += 1
                
                # Update de l'historique (en log)
                history_log_nu.append(np.log(np.array(self.nu)))
                # ============================================================

                # Affichage
                if verbose and (it % verbose_every == 0 or it == n_iter - 1):
                    eps_arr = np.array(eps)
                    n_pi = Pi_S.getSize()
                    acc_rate = acc_nu_count / (it + 1) * 100
                    print(
                        f"[Gibbs iteration {it}] "
                        f"mu_tilde = {mu_tilde:.4f} | "
                        f"|Pi| = {n_pi} | "
                        #f"eps = {eps_arr}"
                        f"nu={np.array(self.nu)} (acc={acc_rate:.1f}%)"
                    )

                mu_chain[it] = mu_tilde
                eps_chain[it, :] = np.array(eps)
                nPi_chain[it] = Pi_S.getSize()
                fdata_chain[it, :] = np.array(f_data)
                nu_chain[it, :] = np.array(new_nu)

            except Exception as e:
                print(f"\nErreur iteration {it} : {e}")
                raise

        if verbose:
            print("\n" + "=" * 100)
            print("-" * 41 + " Gibbs terminé !! " + "-" * 41)
            print("=" * 100 + "\n")

        return {
            "mu_tilde": mu_chain,
            "eps": eps_chain,
            "nPi": nPi_chain,
            "f_data": fdata_chain,
            "last_state": {
                "mu_tilde": mu_tilde,
                "eps": np.array(eps),
                "nu": list(self.nu),
                "delta": self.delta,
            },
            "Sigma_eps": self.Sigma_eps,
            "centroids": self.centroids_xy,
            "nu": nu_chain,
            "acceptance_nu": acc_nu_count / n_iter,
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
        nu_chain = np.asarray(results["nu"])
        burn = int(len(mutilde_chain) * burn_in)

        return {
            "mutilde_hat": mutilde_chain[burn:].mean(),
            "eps_hat": eps_chain[burn:].mean(axis=0),
            "f_data_hat": f_chain[burn:].mean(axis=0),
            "nu_hat": nu_chain[burn:].mean(axis=0)
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
            K_dd_reg[i, i] += self.jitter    # Régularisation
        K_inv = K_dd_reg.inverse()

        # Moyenne postérieure : mu_post = m_grid + K_gd * K_dd^{-1} * (f_data - m_data)
        delta = f_data_hat - m_data
        mu_post = m_grid + K_gd * (K_inv * delta)

        # Covariance postérieure : Sigma_post = K_gg - K_gd * K_dd^{-1} * K_dg
        Sigma_post_mat = ot.Matrix(K_gg) - K_gd * (K_inv * K_gd.transpose())
        Sigma_post_np = np.array(Sigma_post_mat)
        Sigma_post_np = 0.5 * (Sigma_post_np + Sigma_post_np.T)   # Symétrisation, passage par numpy peut être pas nécessaire
        Sigma_post_np += self.jitter * np.eye(M)     # Régularisation
        Sigma_post = ot.CovarianceMatrix(Sigma_post_np.tolist())
        
        return mu_post, Sigma_post
    
    
    def plot_posterior_intensity(self, x, y, t, results, nx=70, ny=70, burn_in=0.3, save_path=None):
        """
        
        """
        post_sum = self.posterior_summary(results, burn_in)
        mutilde_hat = post_sum["mutilde_hat"]
        eps_hat = post_sum["eps_hat"]
        f_data_hat = post_sum["f_data_hat"]
        nu_hat = post_sum["nu_hat"]
        self.nu = ot.Point(nu_hat)

        # Récupération de la chaîne de mu_tilde pour vraie estimation de la moyenne a post
        mutilde_chain = np.asarray(results["mu_tilde"])
        burn_idx = int(len(mutilde_chain) * burn_in)
        mu_chain_burned = mutilde_chain[burn_idx:]     # Échantillons de la distribution a post de mu_tilde

        N = len(t)
        XY_data = ot.Sample([[x[i], y[i]] for i in range(N)])
        
        # Création du mesh
        xmin, xmax = self.X_bounds
        ymin, ymax = self.Y_bounds
        interval = ot.Interval([xmin, ymin], [xmax, ymax])
        mesher = ot.IntervalMesher([nx - 1, ny - 1])        # nb d'arêtes
        mesh = mesher.build(interval)

        M = mesh.getVertices().getSize()
        if M > 10000 :              # Critère pour éviter maillage trop grand (question de compléxité)
            raise ValueError(f"Mailage trop grand : {M} points")

        # Equations du Krigeage pour calcul posterior du GP
        mu_post_grid, Sigma_post_grid = self.posterior_gp(XY_data, f_data_hat, mesh, eps_hat)
        
        # ----------------------------------------------------------------------
        # MONTE CARLO pour estimation moyenne a posteriori
        # ----------------------------------------------------------------------
        Sigma_diag = np.diagonal(np.array(Sigma_post_grid))
        std_devs = np.sqrt(Sigma_diag) # Attention : racine carrée pour avoir l'écart-type !
        means = np.array(mu_post_grid).flatten()
        
        # Simulation MC : f_sim ~ N(mean, var) pour chaque point de la grille
        n_mc = 5000
        M = len(means)
        
        # 1. On génère le bruit pour f
        noise = np.random.randn(M, n_mc)
        f_sims = means[:, None] + std_devs[:, None] * noise
        
        # 2. On tire des n_mc échantillons de mu_tilde depuis la chaîne
        mu_samples = np.random.choice(mu_chain_burned, size=n_mc)
        
        # 3. Calcul de moyenne empirique
        sig_sims = 1.0 / (1.0 + np.exp(-f_sims))
        mu_hat_sims = sig_sims * mu_samples[None, :]
        squared_mu_hat_sims = (sig_sims * mu_samples[None, :])**2
        mu_hat = np.mean(mu_hat_sims, axis=1)
        squared_mu_hat = np.mean(squared_mu_hat_sims, axis=1)
        # ----------------------------------------------------------------------
        
        mu_hat_sample = ot.Sample([[val] for val in mu_hat])
        mu_hat_field = ot.Field(mesh, mu_hat_sample)

        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        # Subplot 1 : Données
        ax = axes[0]
        sc = ax.scatter(x, y, c=t, s=12, alpha=0.7, edgecolors="black")
        ax.set_title(f"Observed data ({N} events)")
        ax.set_xlim(self.X_bounds)
        ax.set_ylim(self.Y_bounds)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        #plt.colorbar(sc, ax=ax).set_label("t") 

        # Subplot 2 : Intensité avec plot_field
        ax = axes[1]
        plot_field(mu_hat_field, mode="subplot", ax=ax, title=r"Posterior intensity $\hat{\mu}(s)$", 
                   add_colorbar=True)
        ax.scatter(x, y, s=10, alpha=0.5, color="white", edgecolors="black", linewidths=0.5)
        ax.set_xlim(self.X_bounds)
        ax.set_ylim(self.Y_bounds)
        ax.grid(alpha=0.3, color="white", linewidth=0.5)

        # Titre global
        fig.suptitle(r"Analyse postérieure : $\hat{\mu}(s) = \mathbb{E} \left[ \hat{\tilde{\mu}} \cdot \sigma(\hat{f}(s)) \right]$", 
                     fontsize=13, fontweight="bold",)
        plt.tight_layout()

        # --- AJOUT SAUVEGARDE ---
        if save_path is not None:
            # bbox_inches='tight' coupe les marges blanches inutiles
            plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
            print(f"Graphique enregistré sous : {save_path}")
        # ------------------------

        plt.show()

        return {
            "mu_hat": mu_hat, 
            "squared_mu_hat": squared_mu_hat,
            "mu_field": mu_hat_field, 
            "mesh": mesh,
            "mu_post_gp": mu_post_grid, 
            "Sigma_post_gp": Sigma_post_grid,
            "eps_hat": eps_hat,
            "f_data_hat": f_data_hat
        }

        

    # def plot_posterior_intensity(self, x, y, t, results, nx=70, ny=70, burn_in=0.3):
    #     """
        
    #     """
    #     post_sum = self.posterior_summary(results, burn_in)
    #     mutilde_hat = post_sum["mutilde_hat"]
    #     eps_hat = post_sum["eps_hat"]
    #     f_data_hat = post_sum["f_data_hat"]
    #     N = len(t)
    #     XY_data = ot.Sample([[x[i], y[i]] for i in range(N)])
        
    #     # Création du mesh 
    #     xmin, xmax = self.X_bounds
    #     ymin, ymax = self.Y_bounds
    #     interval = ot.Interval([xmin, ymin], [xmax, ymax])
    #     mesher = ot.IntervalMesher([nx - 1, ny - 1])      # nb d'arêtes
    #     mesh = mesher.build(interval)

    #     M = mesh.getVertices().getSize()
    #     if M > 10000:            # Critère pour éviter maillage trop grand (question de compléxité)
    #         raise ValueError(f"Mailage trop grand: {M} points.")

    #     mu_post_grid, Sigma_post_grid = self.posterior_gp(XY_data, f_data_hat, mesh, eps_hat)
    #     f_hat = mu_post_grid        # Estimateur de la moyenne a posteriori

    #     mu_hat = mutilde_hat * self.sigma(f_hat)         # Calcul de l'intensité estimée
    #     mu_sample = ot.Sample([[mu_hat[i]] for i in range(len(mu_hat))])
    #     mu_field = ot.Field(mesh, mu_sample)

    #     fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    #     # Subplot 1 : Données
    #     ax = axes[0]
    #     sc = ax.scatter(x, y, c=t, s=12, alpha=0.7, edgecolors="black")
    #     ax.set_title("Données observées (couleur = temps)")
    #     ax.set_xlim(self.X_bounds)
    #     ax.set_ylim(self.Y_bounds)
    #     ax.set_aspect("equal")
    #     ax.grid(alpha=0.3)
    #     plt.colorbar(sc, ax=ax).set_label("t")

    #     # Subplot 2 : Intensité avec plot_field
    #     ax = axes[1]
    #     plot_field(mu_field, mode="subplot", ax=ax, title=r"Intensité postérieure $\hat{\mu}(s)$", 
    #                add_colorbar=True)
    #     ax.scatter(x, y, s=10, alpha=0.5, color="white", edgecolors="black", linewidths=0.5)
    #     ax.set_xlim(self.X_bounds)
    #     ax.set_ylim(self.Y_bounds)
    #     ax.grid(alpha=0.3, color="white", linewidth=0.5)

    #     # Titre global
    #     fig.suptitle(r"Analyse postérieure : $\hat{\mu}(s) = \hat{\tilde{\mu}} \cdot \sigma(\hat{f}(s))$", 
    #                  fontsize=13, fontweight="bold",)
    #     plt.tight_layout()
    #     plt.show()

    #     return {
    #         "mu_hat": mu_hat,
    #         "eps_hat": eps_hat,
    #         "f_data_hat": f_data_hat,
    #         "mu_post_grid": mu_post_grid,
    #         "Sigma_post_grid": Sigma_post_grid,
    #         "mu_field": mu_field,
    #         "mesh": mesh,
    #     }

    def plot_chains(self, results, figsize=(9, 5)):
        """

        """
        mutilde_chain = np.asarray(results["mu_tilde"])
        eps_chain = np.asarray(results["eps"])
        nu_chain = np.asarray(results["nu"])
        n_iter = len(mutilde_chain)
        iters = np.arange(n_iter)

        # =====================
        # 1) mu_tilde
        # =====================
        fig, ax = plt.subplots(1, 2, figsize=figsize)

        ax[0].plot(iters, mutilde_chain[:], linewidth=1)
        ax[0].set_title(r"Trace de $\tilde{\mu}$")
        ax[0].set_xlabel("Itération")
        ax[0].grid(alpha=0.3)

        ax[1].hist(mutilde_chain[:], bins=30, density=True, edgecolor="black", alpha=0.7)
        ax[1].set_title(r"Histogramme de $\tilde{\mu}$")
        ax[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

        # =====================
        # 2) epsilons
        # =====================
        J = eps_chain.shape[1]
        #if J <= 5 :
        fig, axes = plt.subplots(J, 2, figsize=(figsize[0], 3 * J), squeeze=False)

        for j in range(J):
            axes[j, 0].plot(iters, eps_chain[:, j], linewidth=1)
            axes[j, 0].set_title(rf"Trace de $\epsilon_{j}$")
            axes[j, 0].set_xlabel("Itération")
            axes[j, 0].grid(alpha=0.3)

            axes[j, 1].hist(
                eps_chain[:, j],
                bins=30,
                density=True,
                edgecolor="black",
                alpha=0.7,
            )
            axes[j, 1].set_title(rf"Histogramme de $\epsilon_{j}$")
            axes[j, 1].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

        # =====================
        # 3) nu
        # =====================
        fig, axes = plt.subplots(2, 2, figsize=(figsize[0], 3 * 2), squeeze=False)

        for j in range(2):
            axes[j, 0].plot(iters, nu_chain[:, j], linewidth=1)
            axes[j, 0].set_title(rf"Trace de $\nu_{j}$")
            axes[j, 0].set_xlabel("Itération")
            axes[j, 0].grid(alpha=0.3)

            axes[j, 1].hist(
                nu_chain[:, j],
                bins=30,
                density=True,
                edgecolor="black",
                alpha=0.7,
            )
            axes[j, 1].set_title(rf"Histogramme de $\nu_{j}$")
            axes[j, 1].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_acf(self, results, burn_in=0.3, max_lag=50, figsize=(8, 6)):
        """
        
        """
        mutilde_chain = np.asarray(results["mu_tilde"])
        eps_chain = np.asarray(results["eps"])
        n_iter = len(mutilde_chain)
        burn = int(burn_in * n_iter)
        lags = np.arange(max_lag + 1)

        plots = []
        plots.append((r"$\tilde{\mu}$", mutilde_chain[burn:]))
        for j in range(eps_chain.shape[1]):
            plots.append((rf"$\epsilon_{j}$", eps_chain[burn:, j]))

        n_plots = len(plots)
        fig, axes = plt.subplots(n_plots, 1, figsize=(figsize[0], 3.0 * n_plots))

        for ax, (param, chain) in zip(axes, plots):
            acf_vals = self._acf(chain, max_lag)

            ax.plot(lags, acf_vals)
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_xlim(0, max_lag)
            ax.set_ylim(-1.0, 1.0)
            ax.set_title(f"ACF de {param}")
            ax.set_xlabel("Lag")
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_ess_arviz(self, results, burn_in=0.3, kind="local", figsize=None):
        """
        
        """
        mutilde_chain = np.asarray(results["mu_tilde"])
        eps_chain = np.asarray(results["eps"])
        n_iter = len(mutilde_chain)
        burn = int(burn_in * n_iter)
        mutilde_post = mutilde_chain[burn:]
        eps_post = eps_chain[burn:, :]

        posterior = { "mu_tilde": mutilde_post[None, :] }
        for j in range(eps_post.shape[1]):
            posterior[f"eps_{j}"] = eps_post[:, j][None, :]

        idata = az.from_dict(posterior=posterior)
        ess = az.ess(idata)
        ess_dict = {
            var: ess[var].values for var in ess.data_vars
        }

        # Plot ESS
        az.plot_ess(idata, kind=kind, figsize=figsize)
        plt.suptitle( f"ESS | N = {mutilde_post.size}", fontsize=12)
        plt.tight_layout()
        plt.show()

        return ess_dict

    def plot_rhat_arviz(self, results_list, burn_in=0.3, figsize=(12, 4), rhat_bad=1.05):
        """

        """

        M = len(results_list)
        res = results_list[0]
        L = len(res["mu_tilde"])
        burn = int(burn_in * L)
        draws = L - burn
        mu_arr = np.zeros((M, draws))
        eps_arr = np.zeros((M, draws, self.J))
        for m, res in enumerate(results_list):
            mu = np.asarray(res["mu_tilde"])
            eps = np.asarray(res["eps"])
            mu_arr[m, :] = mu[burn:]
            eps_arr[m, :, :] = eps[burn:, :]

        idata = az.from_dict(
            posterior={"mu_tilde": mu_arr, "eps": eps_arr},
            coords={"eps_dim": np.arange(self.J)},
            dims={"eps": ["eps_dim"]}
        )

        r_hat = az.rhat(idata)
        rhat_mu = r_hat["mu_tilde"].values
        rhat_eps = np.asarray(r_hat["eps"].values) 

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        ax.scatter([0], [rhat_mu], s=50, label=r"$\tilde{\mu}$")
        ax.scatter(np.arange(1, self.J + 1), rhat_eps, s=50, label=r"$\epsilon_j$")
        ax.axhline(1.0, linestyle="--", color="green", linewidth=1.0)
        ax.axhline(rhat_bad, linestyle="--", color="red", linewidth=1.0)
        ax.set_xticks(np.arange(0, self.J + 1))
        ax.set_xticklabels([r"$\tilde{\mu}$"] + [rf"$\epsilon_{j}$" for j in range(self.J)])
        ax.set_ylabel(r"$\widehat{R}$")
        ax.set_title(rf"Gelman–Rubin $\widehat R$ sur {M} chains")
        ax.grid(alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.show()

        return {"mu_tilde": rhat_mu, "eps": rhat_eps}


class SGCP_GibbsSampler_noNu:
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
        delta,                 # hyperparamètre qui module confiance au prior
        polygons,              # attention ordre, identique à Areas
        jitter=1e-5,           # Paramètre sensible : trop grand peut biaiser, trop petit pas significatif
        rng_seed=None,
    ):
        self.X_bounds = tuple(X_bounds)
        self.Y_bounds = tuple(Y_bounds)
        self.T = T
        self.Areas = Areas
        self.nu = ot.Point(nu)
        self.a_mu = a_mu
        self.b_mu = b_mu
        self.delta = ot.Point(delta)
        self.jitter = jitter
        self.areas = [a[0] for a in self.Areas]
        self.epsilons = [a[1] for a in self.Areas]
        self.J = len(self.areas)
        self.polygons = polygons
        if rng_seed is not None:
            ot.RandomGenerator.SetSeed(int(rng_seed))
            self.rng_state = ot.RandomGenerator.GetState()
        
        self.centroids_xy, self.Sigma_eps = self.compute_Sigma_eps()
        Sigma_eps_reg = ot.CovarianceMatrix((self.Sigma_eps + self.jitter * np.eye(self.J)).tolist())
        self.Sigma_eps_inv = Sigma_eps_reg.inverse()        # Calcul fait une fois, pas besoind d'être répété
        #self.sd = 2.4**2 / 2.0         # Coefficient optimal pour MH adaptive (Haario et al. (2001) ; Gelma et al. (1996))
        #self.eps_MH = 1e-6
        #self.proposal_cov = None       # Sera initialisé dans le run

    # ==========================================================================
    # ------------------------------- Outillage --------------------------------
    # ==========================================================================

    @staticmethod
    def sigma(z):
        z_array = np.array(z)
        return ot.Point(1.0 / (1.0 + np.exp(-z_array)))
    
    @staticmethod
    def _acf(x, max_lag):
        """
        
        """
        x = np.asarray(x)
        x = x - x.mean()
        n = len(x)

        var = np.dot(x, x) / n
        if var == 0.0:
            return np.zeros(max_lag + 1)

        acf_vals = np.empty(max_lag + 1)
        for k in range(max_lag + 1):
            acf_vals[k] = np.dot(x[: n - k], x[k:]) / (n * var)

        return acf_vals
    
    #@staticmethod
    def compute_Sigma_eps(self):
        """

        """
        delta0, delta1 = map(float, self.delta)
        centroids_xy = np.array([[p.centroid.x, p.centroid.y] for p in self.polygons])
        dx = centroids_xy[:, 0].reshape(len(self.polygons),-1) - centroids_xy[:, 0]
        dy = centroids_xy[:, 1].reshape(len(self.polygons),-1) - centroids_xy[:, 1]
        dist2 = dx * dx + dy * dy

        Sigma_eps = delta0 * np.exp(-dist2 / (2.0 * delta1 ** 2))
        Sigma_eps = 0.5 * (Sigma_eps + Sigma_eps.T)       # Symétrisation

        return centroids_xy, Sigma_eps

    def compute_kernel(self, XY_data, XY_new=None):
        """
        
        """
        nu0, nu1 = map(float, self.nu)

        if not isinstance(XY_data, ot.Sample):
            XY_data = ot.Sample(np.asarray(XY_data).tolist())
        N_data = XY_data.getSize()

        kernel = ot.SquaredExponential([nu1, nu1], [nu0])

        if XY_new is None:
            K = kernel.discretize(XY_data)
            return ot.CovarianceMatrix(np.array(K).tolist())

        if not isinstance(XY_new, ot.Sample):
            XY_new = ot.Sample(np.asarray(XY_new).tolist())
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
            D_xy_sample = ot.Sample(np.asarray(D_xy).tolist())
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
        sample = distribution.getSample(int(N))

        return sample
    
    # def log_posterior_nu(self, nu_vals, f_Df, m_f, D_f_sample):
    #     """

    #     """
    #     nu0, nu1 = map(float, nu_vals)
    #     log_prior = - self.lambda_nu * (nu0 + nu1)

    #     kernel = ot.SquaredExponential([nu1, nu1], [nu0])
    #     N = D_f_sample.getSize()
    #     K_mat = kernel.discretize(D_f_sample)
        
    #     for i in range(N):
    #         K_mat[i, i] += self.jitter       # Régularisation
        
    #     K_ff = ot.CovarianceMatrix(K_mat)
    #     dist_normal = ot.Normal(m_f, K_ff)
    #     log_likelihood = dist_normal.computeLogPDF(f_Df)
        
    #     return log_likelihood + log_prior
    

    # =================================================================================
    # -------------------------- Posteriors conditionnelles ---------------------------
    # =================================================================================

    def update_mu_tilde(self, Z, Pi_S):
        """
        
        """
        xmin, xmax = self.X_bounds
        ymin, ymax = self.Y_bounds

        N_0 = sum(1 for z in Z if z == 0.0)
        N_Pi = Pi_S.getSize()

        shape = self.a_mu + N_0 + N_Pi
        rate = self.b_mu + self.T * (xmax - xmin) * (ymax - ymin)

        mu_tilde = ot.Gamma(shape, rate, 0.0).getRealization()[0]

        return float(mu_tilde)

    def update_epsilons(self, f_Df, K_ff, U):
        """

        """
        n = K_ff.getDimension()

        K_cov_reg = ot.CovarianceMatrix(K_ff)
        for i in range(n):
            K_cov_reg[i, i] += self.jitter        # Régularisation
        K_inv = K_cov_reg.inverse()

        U_mat = ot.Matrix(U)
        U_T = U_mat.transpose()

        A_mat = U_T * K_inv * U_mat + self.Sigma_eps_inv        # A = U^T K^{-1} U + Sigam_eps
        A_np = np.array(A_mat)
        A_np = 0.5 * (A_np + A_np.T)            # Symétrisation, passage par numpy peut être pas nécessaire
        A_np += self.jitter * np.eye(self.J)
        A_cov = ot.CovarianceMatrix(A_np.tolist())
        Sigma_post_sym = A_cov.inverse()
        Sigma_post = ot.CovarianceMatrix(np.array(Sigma_post_sym).tolist())

        B = U_T * (K_inv * f_Df)
        mu = Sigma_post * B

        return ot.Normal(mu, Sigma_post).getRealization()

    def update_f(self, x, y, eps, Z, omega_D0, Pi_S):
        """
        
        """
        idx = [i for i in range(len(Z)) if Z[i] == 0.0]         # Indices des points D_0
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
            K_ff[i, i] += self.jitter
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
        A_array = 0.5 * (A_array + A_array.T)
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
        if hasattr(X_data, "getSize"):
            N = X_data.getSize()
        else:
            N = len(X_data)

        # N = X_data.getSize()
        
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
        
        # ------- SÉCURITÉ ANTI-EXPLOSION -------
        LIMIT_CANDIDATES = 1500
        if N_cand > LIMIT_CANDIDATES:
            N_cand = LIMIT_CANDIDATES
        # ---------------------------------------

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
            K_dd_reg[i, i] += self.jitter             # Régularisation
        K_inv = K_dd_reg.inverse()

        delta = f_data - m_data
        mu_star = m_cand + K_star_d * (K_inv * delta)

        Sigma_star_mat = K_star_star - K_star_d * (K_inv * K_star_d.transpose())
        Sigma_array = np.array(Sigma_star_mat)
        Sigma_array = 0.5 * (Sigma_array + Sigma_array.T)          # Symétrisation, passage par numpy peut être pas nécessaire
        Sigma_array += self.jitter * np.eye(N_cand)          # Régularisation
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

    # def update_nu_MH(self, f_Df, D_f, m_f, sigma_nu_RWMH=0.05):
    #     """

    #     """
    #     nu0, nu1 = map(float, self.nu)
    #     current_nu = [nu0, nu1]
        
    #     # log_post à l'état actuel
    #     log_post_current = self.log_posterior_nu(current_nu, f_Df, m_f, D_f)
        
    #     # log_post proposé (avec marche aléatoire log-normale)
    #     mu = ot.Point(2, 0.0)
    #     Sigma = ot.CovarianceMatrix(2, [sigma_nu_RWMH, 0.0, 0.0, sigma_nu_RWMH])
    #     perturbation = ot.Normal(mu, Sigma).getRealization()
    #     proposed_nu = [
    #         current_nu[0] * np.exp(perturbation[0]),
    #         current_nu[1] * np.exp(perturbation[1])
    #     ]

    #     log_post_proposal = self.log_posterior_nu(proposed_nu, f_Df, m_f, D_f)

    #     # Correction avec ratio d'Hastings 
    #     log_q_correction = np.sum(np.log(proposed_nu)) - np.sum(np.log(current_nu))
        
    #     log_accept_ratio = (log_post_proposal - log_post_current) + log_q_correction
    #     if np.log(ot.Uniform(0.0, 1.0).getRealization()) < log_accept_ratio:
    #         self.nu = ot.Point(proposed_nu)
    #         return self.nu, True
    #     else:
    #         return self.nu, False
        
    # def update_nu_MH_adaptatif(self, f_Df, D_f, m_f, history_log_nu, it, step_nu_init=0.01):    ### 'adaptatif' à enlever
    #     """

    #     """
    #     nu0, nu1 = map(float, self.nu)
    #     current_nu = [nu0, nu1]
    #     current_log_nu = np.log(current_nu)

    #     # Log-posterior actuelle
    #     log_post_current = self.log_posterior_nu(current_nu, f_Df, m_f, D_f)

    #     t0 = 50  
    #     if it > t0 and len(history_log_nu) > t0:
    #         cov_matrix = np.cov(np.array(history_log_nu).T)
    #         self.proposal_cov = self.sd * cov_matrix + self.sd * self.eps_MH * np.eye(2)
    #     elif self.proposal_cov is None:
    #          self.proposal_cov = step_nu_init * np.eye(2)

    #     # Génération du candidat (random walk multivariée sur le log)
    #     perturbation = np.random.multivariate_normal(np.zeros(2), self.proposal_cov)
    #     proposed_log_nu = current_log_nu + perturbation
    #     proposed_nu = np.exp(proposed_log_nu).tolist()
    #     log_post_proposal = self.log_posterior_nu(proposed_nu, f_Df, m_f, D_f)

    #     # Correction avec ratio d'Hastings 
    #     log_q_correction = np.sum(proposed_log_nu) - np.sum(current_log_nu)

    #     log_accept_ratio = (log_post_proposal - log_post_current) + log_q_correction

    #     # Acceptation / Rejet
    #     if np.log(np.random.rand()) < log_accept_ratio:
    #         self.nu = ot.Point(proposed_nu)
    #         return self.nu, True
    #     else:
    #         return self.nu, False



    # =====================================================================================
    # ----------------------------------- Run du Gibbs ------------------------------------
    # =====================================================================================

    def run(self, t, x, y, eps_init, mutilde_init, step_nu_init=0.01, n_iter=1000, verbose=True, verbose_every=100):
        N = len(t)

        # Seulement évènements de fond : ETAS = 0
        Z = ot.Point([0.0] * N)

        # Initialisations
        eps = ot.Point(eps_init)
        mu_tilde = mutilde_init
        XY_data = ot.Sample([[x[i], y[i]] for i in range(N)])
        U_data = self.compute_U_from_areas(XY_data)
        f_data = ot.Point(U_data * eps)

        # Stockage
        mu_chain = np.zeros(n_iter)
        eps_chain = np.zeros((n_iter, self.J))
        nPi_chain = np.zeros(n_iter)
        fdata_chain = np.zeros((n_iter, N))
        #nu_chain = np.zeros((n_iter, 2))
        #acc_nu_count = 0
        #history_log_nu = []          # Liste qui stocker les log(nu) passés

        if verbose:
            print("\n" + "=" * 100)
            print(
                "-" * 29
                + f" Démarrage Gibbs : {n_iter} itérations, N={N} "
                + "-" * 29
            )
            print("=" * 100 + "\n")

        for it in range(n_iter):
            try:
                # 1) omega_D0 | ...
                omega_D0 = ot.Point(random_polyagamma(1.0, f_data))

                # 2) Pi_S | ...
                Pi_S = self.sample_Pi_S(mu_tilde, x, y, f_data, eps)

                # 3) f_Df | ...
                f_Df, D_f_xy, U_Df, K_ff, m_f = self.update_f(
                    x, y, eps, Z, omega_D0, Pi_S
                )
                f_data = ot.Point([f_Df[i] for i in range(N)])

                # 4) eps | ...
                eps = self.update_epsilons(f_Df, K_ff, U_Df)

                # 5) mu_tilde | ...
                mu_tilde = self.update_mu_tilde(Z, Pi_S)
                
                # # ============================================================
                # # 6) UPDATE NU 
                # # ============================================================
                # eps_mat = ot.Matrix([[float(eps[j])] for j in range(self.J)])
                # m_f_updated_mat = U_Df * eps_mat
                # m_f_updated = [m_f_updated_mat[i, 0] for i in range(m_f_updated_mat.getNbRows())]
                # #m_f_updated = U_Df * eps
                
                # new_nu, accepted = self.update_nu_MH(f_Df, D_f_xy, m_f_updated, sigma_nu_RWMH=step_nu_RWMH)
                
                # if accepted:
                #     acc_nu_count += 1
                # # ============================================================

                # ============================================================
                # 6) UPDATE NU (ADAPTIVE)
                # ============================================================
                # Recalcul moyenne m_f avec nouvel eps pour cohérence
                # eps_mat = ot.Matrix([[float(eps[j])] for j in range(self.J)])
                # m_f_updated_mat = U_Df * eps_mat
                # m_f_updated = [m_f_updated_mat[i, 0] for i in range(m_f_updated_mat.getNbRows())]
                
                # new_nu, accepted = self.update_nu_MH(f_Df, D_f_xy, m_f_updated, history_log_nu, it, step_nu_init)
                
                # if accepted: 
                #     acc_nu_count += 1
                
                # Update de l'historique (en log)
                # history_log_nu.append(np.log(np.array(self.nu)))
                # ============================================================

                # Affichage
                if verbose and (it % verbose_every == 0 or it == n_iter - 1):
                    eps_arr = np.array(eps)
                    n_pi = Pi_S.getSize()
                    #acc_rate = acc_nu_count / (it + 1) * 100
                    print(
                        f"[Gibbs iteration {it}] "
                        f"mu_tilde = {mu_tilde:.4f} | "
                        f"|Pi| = {n_pi} | "
                        f"eps = {eps_arr}"
                        #f"nu={np.array(self.nu)} (acc={acc_rate:.1f}%)"
                    )

                mu_chain[it] = mu_tilde
                eps_chain[it, :] = np.array(eps)
                nPi_chain[it] = Pi_S.getSize()
                fdata_chain[it, :] = np.array(f_data)
                #nu_chain[it, :] = np.array(new_nu)

            except Exception as e:
                print(f"\nErreur iteration {it} : {e}")
                raise

        if verbose:
            print("\n" + "=" * 100)
            print("-" * 41 + " Gibbs terminé !! " + "-" * 41)
            print("=" * 100 + "\n")

        return {
            "mu_tilde": mu_chain,
            "eps": eps_chain,
            "nPi": nPi_chain,
            "f_data": fdata_chain,
            "last_state": {
                "mu_tilde": mu_tilde,
                "eps": np.array(eps),
                "nu": list(self.nu),
                "delta": self.delta,
            },
            "Sigma_eps": self.Sigma_eps,
            "centroids": self.centroids_xy,
            #"nu": nu_chain,
            #"acceptance_nu": acc_nu_count / n_iter,
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
        #nu_chain = np.asarray(results["nu"])
        burn = int(len(mutilde_chain) * burn_in)

        return {
            "mutilde_hat": mutilde_chain[burn:].mean(),
            "eps_hat": eps_chain[burn:].mean(axis=0),
            "f_data_hat": f_chain[burn:].mean(axis=0),
            #"nu_hat": nu_chain[burn:].mean(axis=0)
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
            K_dd_reg[i, i] += self.jitter    # Régularisation
        K_inv = K_dd_reg.inverse()

        # Moyenne postérieure : mu_post = m_grid + K_gd * K_dd^{-1} * (f_data - m_data)
        delta = f_data_hat - m_data
        mu_post = m_grid + K_gd * (K_inv * delta)

        # Covariance postérieure : Sigma_post = K_gg - K_gd * K_dd^{-1} * K_dg
        Sigma_post_mat = ot.Matrix(K_gg) - K_gd * (K_inv * K_gd.transpose())
        Sigma_post_np = np.array(Sigma_post_mat)
        Sigma_post_np = 0.5 * (Sigma_post_np + Sigma_post_np.T)   # Symétrisation, passage par numpy peut être pas nécessaire
        Sigma_post_np += self.jitter * np.eye(M)     # Régularisation
        Sigma_post = ot.CovarianceMatrix(Sigma_post_np.tolist())
        
        return mu_post, Sigma_post
    

    def plot_posterior_intensity(self, x, y, t, results, nx=70, ny=70, burn_in=0.3, save_path=None):
        """
        
        """
        post_sum = self.posterior_summary(results, burn_in)
        mutilde_hat = post_sum["mutilde_hat"]
        eps_hat = post_sum["eps_hat"]
        f_data_hat = post_sum["f_data_hat"]
        #nu_hat = post_sum["nu_hat"]
        #self.nu = ot.Point(nu_hat)

        # Récupération de la chaîne de mu_tilde pour vraie estimation de la moyenne a post
        mutilde_chain = np.asarray(results["mu_tilde"])
        burn_idx = int(len(mutilde_chain) * burn_in)
        mu_chain_burned = mutilde_chain[burn_idx:]     # Échantillons de la distribution a post de mu_tilde

        N = len(t)
        XY_data = ot.Sample([[x[i], y[i]] for i in range(N)])
        
        # Création du mesh
        xmin, xmax = self.X_bounds
        ymin, ymax = self.Y_bounds
        interval = ot.Interval([xmin, ymin], [xmax, ymax])
        mesher = ot.IntervalMesher([nx - 1, ny - 1])        # nb d'arêtes
        mesh = mesher.build(interval)

        M = mesh.getVertices().getSize()
        if M > 10000 :              # Critère pour éviter maillage trop grand (question de compléxité)
            raise ValueError(f"Mailage trop grand : {M} points")

        # Equations du Krigeage pour calcul posterior du GP
        mu_post_grid, Sigma_post_grid = self.posterior_gp(XY_data, f_data_hat, mesh, eps_hat)
        
        # ----------------------------------------------------------------------
        # MONTE CARLO pour estimation moyenne a posteriori
        # ----------------------------------------------------------------------
        Sigma_diag = np.diagonal(np.array(Sigma_post_grid))
        std_devs = np.sqrt(Sigma_diag) # Attention : racine carrée pour avoir l'écart-type !
        means = np.array(mu_post_grid).flatten()
        
        # Simulation MC : f_sim ~ N(mean, var) pour chaque point de la grille
        n_mc = 5000
        M = len(means)
        
        # 1. On génère le bruit pour f
        noise = np.random.randn(M, n_mc)
        f_sims = means[:, None] + std_devs[:, None] * noise
        
        # 2. On tire des n_mc échantillons de mu_tilde depuis la chaîne
        mu_samples = np.random.choice(mu_chain_burned, size=n_mc)
        
        # 3. Calcul de moyenne empirique
        sig_sims = 1.0 / (1.0 + np.exp(-f_sims))
        mu_hat_sims = sig_sims * mu_samples[None, :]
        squared_mu_hat_sims = (sig_sims * mu_samples[None, :])**2
        mu_hat = np.mean(mu_hat_sims, axis=1)
        squared_mu_hat = np.mean(squared_mu_hat_sims, axis=1)
        # ----------------------------------------------------------------------
        
        mu_hat_sample = ot.Sample([[val] for val in mu_hat])
        mu_hat_field = ot.Field(mesh, mu_hat_sample)

        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        # Subplot 1 : Données
        ax = axes[0]
        sc = ax.scatter(x, y, c=t, s=12, alpha=0.7, edgecolors="black")
        ax.set_title(f"Observed data ({N} events)")
        ax.set_xlim(self.X_bounds)
        ax.set_ylim(self.Y_bounds)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        #plt.colorbar(sc, ax=ax).set_label("t") 

        # Subplot 2 : Intensité avec plot_field
        ax = axes[1]
        plot_field(mu_hat_field, mode="subplot", ax=ax, title=r"Posterior intensity $\hat{\mu}(s)$", 
                   add_colorbar=True)
        ax.scatter(x, y, s=10, alpha=0.5, color="white", edgecolors="black", linewidths=0.5)
        ax.set_xlim(self.X_bounds)
        ax.set_ylim(self.Y_bounds)
        ax.grid(alpha=0.3, color="white", linewidth=0.5)

        # Titre global
        fig.suptitle(r"Analyse postérieure : $\hat{\mu}(s) = \mathbb{E} \left[ \hat{\tilde{\mu}} \cdot \sigma(\hat{f}(s)) \right]$", 
                     fontsize=13, fontweight="bold",)
        plt.tight_layout()

        # --- AJOUT SAUVEGARDE ---
        if save_path is not None:
            # bbox_inches='tight' coupe les marges blanches inutiles
            plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
            print(f"Graphique enregistré sous : {save_path}")
        # ------------------------

        plt.show()

        return {
            "mu_hat": mu_hat, 
            "squared_mu_hat": squared_mu_hat,
            "mu_field": mu_hat_field, 
            "mesh": mesh,
            "mu_post_gp": mu_post_grid, 
            "Sigma_post_gp": Sigma_post_grid,
            "eps_hat": eps_hat,
            "f_data_hat": f_data_hat
        }

        

    # def plot_posterior_intensity(self, x, y, t, results, nx=70, ny=70, burn_in=0.3):
    #     """
        
    #     """
    #     post_sum = self.posterior_summary(results, burn_in)
    #     mutilde_hat = post_sum["mutilde_hat"]
    #     eps_hat = post_sum["eps_hat"]
    #     f_data_hat = post_sum["f_data_hat"]
    #     N = len(t)
    #     XY_data = ot.Sample([[x[i], y[i]] for i in range(N)])
        
    #     # Création du mesh 
    #     xmin, xmax = self.X_bounds
    #     ymin, ymax = self.Y_bounds
    #     interval = ot.Interval([xmin, ymin], [xmax, ymax])
    #     mesher = ot.IntervalMesher([nx - 1, ny - 1])      # nb d'arêtes
    #     mesh = mesher.build(interval)

    #     M = mesh.getVertices().getSize()
    #     if M > 10000:            # Critère pour éviter maillage trop grand (question de compléxité)
    #         raise ValueError(f"Mailage trop grand: {M} points.")

    #     mu_post_grid, Sigma_post_grid = self.posterior_gp(XY_data, f_data_hat, mesh, eps_hat)
    #     f_hat = mu_post_grid        # Estimateur de la moyenne a posteriori

    #     mu_hat = mutilde_hat * self.sigma(f_hat)         # Calcul de l'intensité estimée
    #     mu_sample = ot.Sample([[mu_hat[i]] for i in range(len(mu_hat))])
    #     mu_field = ot.Field(mesh, mu_sample)

    #     fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    #     # Subplot 1 : Données
    #     ax = axes[0]
    #     sc = ax.scatter(x, y, c=t, s=12, alpha=0.7, edgecolors="black")
    #     ax.set_title("Données observées (couleur = temps)")
    #     ax.set_xlim(self.X_bounds)
    #     ax.set_ylim(self.Y_bounds)
    #     ax.set_aspect("equal")
    #     ax.grid(alpha=0.3)
    #     plt.colorbar(sc, ax=ax).set_label("t")

    #     # Subplot 2 : Intensité avec plot_field
    #     ax = axes[1]
    #     plot_field(mu_field, mode="subplot", ax=ax, title=r"Intensité postérieure $\hat{\mu}(s)$", 
    #                add_colorbar=True)
    #     ax.scatter(x, y, s=10, alpha=0.5, color="white", edgecolors="black", linewidths=0.5)
    #     ax.set_xlim(self.X_bounds)
    #     ax.set_ylim(self.Y_bounds)
    #     ax.grid(alpha=0.3, color="white", linewidth=0.5)

    #     # Titre global
    #     fig.suptitle(r"Analyse postérieure : $\hat{\mu}(s) = \hat{\tilde{\mu}} \cdot \sigma(\hat{f}(s))$", 
    #                  fontsize=13, fontweight="bold",)
    #     plt.tight_layout()
    #     plt.show()

    #     return {
    #         "mu_hat": mu_hat,
    #         "eps_hat": eps_hat,
    #         "f_data_hat": f_data_hat,
    #         "mu_post_grid": mu_post_grid,
    #         "Sigma_post_grid": Sigma_post_grid,
    #         "mu_field": mu_field,
    #         "mesh": mesh,
    #     }

    def plot_chains(self, results, figsize=(9, 5)):
        """

        """
        mutilde_chain = np.asarray(results["mu_tilde"])
        eps_chain = np.asarray(results["eps"])
        #nu_chain = np.asarray(results["nu"])
        n_iter = len(mutilde_chain)
        iters = np.arange(n_iter)

        # =====================
        # 1) mu_tilde
        # =====================
        fig, ax = plt.subplots(1, 2, figsize=figsize)

        ax[0].plot(iters, mutilde_chain[:], linewidth=1)
        ax[0].set_title(r"Trace de $\tilde{\mu}$")
        ax[0].set_xlabel("Itération")
        ax[0].grid(alpha=0.3)

        ax[1].hist(mutilde_chain[:], bins=30, density=True, edgecolor="black", alpha=0.7)
        ax[1].set_title(r"Histogramme de $\tilde{\mu}$")
        ax[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

        # =====================
        # 2) epsilons
        # =====================
        J = eps_chain.shape[1]
        #if J <= 5 :
        fig, axes = plt.subplots(J, 2, figsize=(figsize[0], 3 * J), squeeze=False)

        for j in range(J):
            axes[j, 0].plot(iters, eps_chain[:, j], linewidth=1)
            axes[j, 0].set_title(rf"Trace de $\epsilon_{j}$")
            axes[j, 0].set_xlabel("Itération")
            axes[j, 0].grid(alpha=0.3)

            axes[j, 1].hist(
                eps_chain[:, j],
                bins=30,
                density=True,
                edgecolor="black",
                alpha=0.7,
            )
            axes[j, 1].set_title(rf"Histogramme de $\epsilon_{j}$")
            axes[j, 1].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

        # =====================
        # 3) nu
        # =====================
        # fig, axes = plt.subplots(2, 2, figsize=(figsize[0], 3 * 2), squeeze=False)

        # for j in range(2):
        #     axes[j, 0].plot(iters, nu_chain[:, j], linewidth=1)
        #     axes[j, 0].set_title(rf"Trace de $\nu_{j}$")
        #     axes[j, 0].set_xlabel("Itération")
        #     axes[j, 0].grid(alpha=0.3)

        #     axes[j, 1].hist(
        #         nu_chain[:, j],
        #         bins=30,
        #         density=True,
        #         edgecolor="black",
        #         alpha=0.7,
        #     )
        #     axes[j, 1].set_title(rf"Histogramme de $\nu_{j}$")
        #     axes[j, 1].grid(alpha=0.3)

        # plt.tight_layout()
        # plt.show()

    def plot_acf(self, results, burn_in=0.3, max_lag=50, figsize=(8, 6)):
        """
        
        """
        mutilde_chain = np.asarray(results["mu_tilde"])
        eps_chain = np.asarray(results["eps"])
        n_iter = len(mutilde_chain)
        burn = int(burn_in * n_iter)
        lags = np.arange(max_lag + 1)

        plots = []
        plots.append((r"$\tilde{\mu}$", mutilde_chain[burn:]))
        for j in range(eps_chain.shape[1]):
            plots.append((rf"$\epsilon_{j}$", eps_chain[burn:, j]))

        n_plots = len(plots)
        fig, axes = plt.subplots(n_plots, 1, figsize=(figsize[0], 3.0 * n_plots))

        for ax, (param, chain) in zip(axes, plots):
            acf_vals = self._acf(chain, max_lag)

            ax.plot(lags, acf_vals)
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_xlim(0, max_lag)
            ax.set_ylim(-1.0, 1.0)
            ax.set_title(f"ACF de {param}")
            ax.set_xlabel("Lag")
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_ess_arviz(self, results, burn_in=0.3, kind="local", figsize=None):
        """
        
        """
        mutilde_chain = np.asarray(results["mu_tilde"])
        eps_chain = np.asarray(results["eps"])
        n_iter = len(mutilde_chain)
        burn = int(burn_in * n_iter)
        mutilde_post = mutilde_chain[burn:]
        eps_post = eps_chain[burn:, :]

        posterior = { "mu_tilde": mutilde_post[None, :] }
        for j in range(eps_post.shape[1]):
            posterior[f"eps_{j}"] = eps_post[:, j][None, :]

        idata = az.from_dict(posterior=posterior)
        ess = az.ess(idata)
        ess_dict = {
            var: ess[var].values for var in ess.data_vars
        }

        # Plot ESS
        az.plot_ess(idata, kind=kind, figsize=figsize)
        plt.suptitle( f"ESS | N = {mutilde_post.size}", fontsize=12)
        plt.tight_layout()
        plt.show()

        return ess_dict

    def plot_rhat_arviz(self, results_list, burn_in=0.3, figsize=(12, 4), rhat_bad=1.05):
        """

        """

        M = len(results_list)
        res = results_list[0]
        L = len(res["mu_tilde"])
        burn = int(burn_in * L)
        draws = L - burn
        mu_arr = np.zeros((M, draws))
        eps_arr = np.zeros((M, draws, self.J))
        for m, res in enumerate(results_list):
            mu = np.asarray(res["mu_tilde"])
            eps = np.asarray(res["eps"])
            mu_arr[m, :] = mu[burn:]
            eps_arr[m, :, :] = eps[burn:, :]

        idata = az.from_dict(
            posterior={"mu_tilde": mu_arr, "eps": eps_arr},
            coords={"eps_dim": np.arange(self.J)},
            dims={"eps": ["eps_dim"]}
        )

        r_hat = az.rhat(idata)
        rhat_mu = r_hat["mu_tilde"].values
        rhat_eps = np.asarray(r_hat["eps"].values) 

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        ax.scatter([0], [rhat_mu], s=50, label=r"$\tilde{\mu}$")
        ax.scatter(np.arange(1, self.J + 1), rhat_eps, s=50, label=r"$\epsilon_j$")
        ax.axhline(1.0, linestyle="--", color="green", linewidth=1.0)
        ax.axhline(rhat_bad, linestyle="--", color="red", linewidth=1.0)
        ax.set_xticks(np.arange(0, self.J + 1))
        ax.set_xticklabels([r"$\tilde{\mu}$"] + [rf"$\epsilon_{j}$" for j in range(self.J)])
        ax.set_ylabel(r"$\widehat{R}$")
        ax.set_title(rf"Gelman–Rubin $\widehat R$ sur {M} chains")
        ax.grid(alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.show()

        return {"mu_tilde": rhat_mu, "eps": rhat_eps}


# Dernière version, les autres (au dessus sont amenés à disparaitre)
class iSGCP_GibbsSampler:
    """

    """

    def __init__(
        self,
        X_bounds,
        Y_bounds,
        T,
        Areas,
        lambda_nu,
        nu,
        delta,                  # [delta0, delta1] : variance and length-scale de Sigma_eps
        polygons,               # list of shapely polygons, même format que Areas
        jitter=1e-5,
        rng_seed=None,
    ):
        self.X_bounds = tuple(X_bounds)
        self.Y_bounds = tuple(Y_bounds)
        self.T = T
        self.Areas = Areas
        self.lambda_nu = lambda_nu
        self.nu = ot.Point(nu)
        self.delta = ot.Point(delta)
        self.jitter = jitter
        self.areas = [a[0] for a in self.Areas]
        self.epsilons = [a[1] for a in self.Areas]
        self.J = len(self.areas)
        self.polygons = polygons

        if rng_seed is not None:
            ot.RandomGenerator.SetSeed(int(rng_seed))
            self.rng_state = ot.RandomGenerator.GetState()

        self.centroids_xy, self.Sigma_eps = self.compute_Sigma_eps()
        Sigma_eps_reg = ot.CovarianceMatrix(
            (self.Sigma_eps + self.jitter * np.eye(self.J)).tolist()
        )
        self.Sigma_eps_inv = Sigma_eps_reg.inverse()


    # =============================================================================================
    # ----------------------------------------- Outillage -----------------------------------------
    # =============================================================================================

    @staticmethod
    def sigma(z):
        z_array = np.array(z)
        return ot.Point(1.0 / (1.0 + np.exp(-z_array)))

    @staticmethod
    def _acf(x, max_lag):
        x = np.asarray(x)
        x = x - x.mean()
        n = len(x)
        var = np.dot(x, x) / n
        if var == 0.0:
            return np.zeros(max_lag + 1)
        acf_vals = np.empty(max_lag + 1)
        for k in range(max_lag + 1):
            acf_vals[k] = np.dot(x[: n - k], x[k:]) / (n * var)
        return acf_vals

    def compute_Sigma_eps(self):
        """

        """
        delta0, delta1 = map(float, self.delta)
        centroids_xy = np.array(
            [[p.centroid.x, p.centroid.y] for p in self.polygons]
        )
        dx = centroids_xy[:, 0].reshape(-1, 1) - centroids_xy[:, 0]
        dy = centroids_xy[:, 1].reshape(-1, 1) - centroids_xy[:, 1]
        dist2 = dx * dx + dy * dy
        Sigma_eps = delta0 * np.exp(-dist2 / (2.0 * delta1 ** 2))
        Sigma_eps = 0.5 * (Sigma_eps + Sigma_eps.T)
        return centroids_xy, Sigma_eps

    def compute_kernel(self, XY_data, XY_new=None):
        """
        
        """
        nu0, nu1 = map(float, self.nu)

        if not isinstance(XY_data, ot.Sample):
            XY_data = ot.Sample(np.asarray(XY_data).tolist())
        N_data = XY_data.getSize()

        kernel = ot.SquaredExponential([nu1, nu1], [nu0])

        if XY_new is None:
            K = kernel.discretize(XY_data)
            return ot.CovarianceMatrix(np.array(K).tolist())

        if not isinstance(XY_new, ot.Sample):
            XY_new = ot.Sample(np.asarray(XY_new).tolist())
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

    def compute_mu_tilde(self, XY, eps=None):
        """

        """
        if not isinstance(XY, ot.Sample):
            XY = ot.Sample(np.asarray(XY).tolist())
        n = XY.getSize()
        mu_vals = np.zeros(n)
        eps_vals = eps if eps is not None else self.epsilons
        for k in range(n):
            pt = ShapelyPoint(float(XY[k, 0]), float(XY[k, 1]))
            for j, poly in enumerate(self.areas):
                if poly.covers(pt):
                    mu_vals[k] = np.exp(float(eps_vals[j]))
                    break
        return mu_vals

    def sample_candidats(self, N):
        """
        
        """
        xmin, xmax = self.X_bounds
        ymin, ymax = self.Y_bounds
        distribution = ot.ComposedDistribution(
            [ot.Uniform(xmin, xmax), ot.Uniform(ymin, ymax)]
        )
        return distribution.getSample(int(N))

    def _log_posterior_nu(self, nu_vals, f_Df, D_f_sample):
        """
        
        """
        nu0, nu1 = map(float, nu_vals)
        log_prior = -self.lambda_nu * (nu0 + nu1)
        kernel = ot.SquaredExponential([nu1, nu1], [nu0])
        N = D_f_sample.getSize()
        K_mat = kernel.discretize(D_f_sample)
        for i in range(N):
            K_mat[i, i] += self.jitter
        K_ff = ot.CovarianceMatrix(K_mat)
        m_f = ot.Point(N, 0.0)
        log_likelihood = ot.Normal(m_f, K_ff).computeLogPDF(f_Df)

        return log_likelihood + log_prior
    
    def _log_posterior_eps(self, eps_arr, N_j, M_j):
        """

        """
        Sigma_inv = np.array(self.Sigma_eps_inv)
        areas_j = np.array([self.polygons[j].area for j in range(self.J)])
        prior_term = -0.5 * eps_arr @ Sigma_inv @ eps_arr
        likelihood_term = np.sum(
            (N_j + M_j) * eps_arr - self.T * areas_j * np.exp(eps_arr)
        )
        return prior_term + likelihood_term

    def _grad_log_posterior_eps(self, eps_arr, N_j, M_j):
        """

        """
        Sigma_inv = np.array(self.Sigma_eps_inv)
        areas_j = np.array([self.polygons[j].area for j in range(self.J)])
        return (N_j + M_j) - self.T * areas_j * np.exp(eps_arr) - Sigma_inv @ eps_arr

    def _count_events_per_zone(self, x, y, Z, Pi_S):
        """

        """
        N_j = np.zeros(self.J)
        M_j = np.zeros(self.J)

        for i in range(len(x)):
            if float(Z[i]) == 0.0:
                pt = ShapelyPoint(float(x[i]), float(y[i]))
                for j, poly in enumerate(self.areas):
                    if poly.covers(pt):
                        N_j[j] += 1
                        break

        for m in range(Pi_S.getSize()):
            pt = ShapelyPoint(float(Pi_S[m, 0]), float(Pi_S[m, 1]))
            for j, poly in enumerate(self.areas):
                if poly.covers(pt):
                    M_j[j] += 1
                    break

        return N_j, M_j


    # ==============================================================================================
    # --------------------------------- Posteriors conditionnelles ---------------------------------
    # ==============================================================================================

    def update_f(self, x, y, Z, omega_D0, Pi_S):
        """

        """
        idx = [i for i in range(len(Z)) if Z[i] == 0.0]
        N_0 = len(idx)
        if N_0 == 0:
            raise ValueError("N_0 = 0 : pas de background events.")

        # 1) D_0 -> (x,y) + omega_D_0
        D_0 = ot.Sample(N_0, 2)
        omega_D_0 = ot.Point(N_0)
        for k, i in enumerate(idx):
            D_0[k, 0] = x[i]
            D_0[k, 1] = y[i]
            omega_D_0[k] = omega_D0[i]

        # 2) pi_S -> pi_S(x,y) + omega_Pi
        N_Pi = Pi_S.getSize()
        if N_Pi > 0:
            PiS_xy = ot.Sample(N_Pi, 2)
            omega_Pi = ot.Point(N_Pi)
            for i in range(N_Pi):
                PiS_xy[i, 0] = Pi_S[i, 0]
                PiS_xy[i, 1] = Pi_S[i, 1]
                omega_Pi[i]  = Pi_S[i, 2]
        else:
            PiS_xy   = ot.Sample(0, 2)
            omega_Pi = ot.Point(0)

       # 3) D_f = D_0 U Pi_S
        N_f = N_0 + N_Pi
        D_f = ot.Sample(N_f, 2)
        for i in range(N_0):
            D_f[i, 0] = D_0[i, 0]
            D_f[i, 1] = D_0[i, 1]
        for i in range(N_Pi):
            D_f[N_0 + i, 0] = PiS_xy[i, 0]
            D_f[N_0 + i, 1] = PiS_xy[i, 1]

        # 4) K_ff
        K_ff = self.compute_kernel(D_f)
        for i in range(N_f):
            K_ff[i, i] += self.jitter
        K_inv = K_ff.inverse()

        # 5) Omega = diag(omega_1, ..., omega_{N_f}) 
        Omega = ot.CovarianceMatrix(N_f)
        for i in range(N_0):
            Omega[i, i] = omega_D_0[i]
        for i in range(N_Pi):
            Omega[N_0 + i, N_0 + i] = omega_Pi[i]

        # 6) kappa : +1/2 pour D_0, -1/2 pour pi_S 
        kappa = ot.Point(N_f)
        for i in range(N_0):
            kappa[i] = 0.5
        for i in range(N_Pi):
            kappa[N_0 + i] = -0.5

        # 7) Posterior : Sigma_post = (K_ff^{-1} + Omega)^{-1}, mu_post = Sigma_post * kappa
        A_arr = np.array(K_inv) + np.array(Omega)
        A_arr = 0.5 * (A_arr + A_arr.T)
        A_arr += self.jitter * np.eye(N_f)
        Sigma_arr = np.linalg.inv(A_arr)
        Sigma_arr = 0.5 * (Sigma_arr + Sigma_arr.T)         # Symétrisation, passage par numpy peut être pas nécessaire
        Sigma_arr += self.jitter * np.eye(N_f)
        Sigma_post = ot.CovarianceMatrix(Sigma_arr.tolist())
        mu_post = Sigma_post * kappa

        f_new = ot.Normal(mu_post, Sigma_post).getRealization()
        return f_new, D_f, K_ff

    def sample_Pi_S(self, x, y, f_data, eps, LIM_CANDIDATES_ZONES = 1000, LIM_CANDIDATES = 2000):
        """
        
        """
        N = len(x)
        XY_data = ot.Sample([[x[i], y[i]] for i in range(N)])

        # 1) Draw candidates zone par zone 
        # Attention ! Reréfléchir au pk de passage par 'PreparedGeometry'
        # self.polygons : raw shapely Polygon  -> .bounds, .area
        # self.areas    : PreparedGeometry     -> .covers()
        XY_cand_list = []
        for j in range(self.J):
            raw_poly  = self.polygons[j]
            prep_poly = self.areas[j]
            bx, by, bx2, by2 = raw_poly.bounds
            area_j   = raw_poly.area
            mean_j   = self.T * area_j * np.exp(float(eps[j]))
            N_cand_j = int(ot.Poisson(mean_j).getRealization()[0])
            if N_cand_j == 0:
                continue
            # ------- SÉCURITÉ ANTI-EXPLOSION -------
            if N_cand_j > LIM_CANDIDATES_ZONES:
                N_cand_j = LIM_CANDIDATES_ZONES
            # ---------------------------------------
            accepted = []
            while len(accepted) < N_cand_j:
                pts = ot.ComposedDistribution(
                    [ot.Uniform(bx, bx2), ot.Uniform(by, by2)]
                ).getSample(N_cand_j * 3)
                for k in range(pts.getSize()):
                    if prep_poly.covers(ShapelyPoint(float(pts[k, 0]), float(pts[k, 1]))):
                        accepted.append([float(pts[k, 0]), float(pts[k, 1])])
                    if len(accepted) >= N_cand_j:
                        break
            XY_cand_list.extend(accepted)

        if len(XY_cand_list) == 0:
            return ot.Sample(0, 3)

        # -------------- SÉCURITÉ ANTI-EXPLOSION --------------
        if len(XY_cand_list) > LIM_CANDIDATES:
            XY_cand_list = XY_cand_list[:LIM_CANDIDATES]
        # -----------------------------------------------------

        N_cand  = len(XY_cand_list)
        XY_cand = ot.Sample(XY_cand_list)

        # 2) Conditional GP prediction at candidate locations
        K_dd, K_star_d, K_star_star = self.compute_kernel(XY_data, XY_cand)
        K_dd_reg = ot.CovarianceMatrix(K_dd)
        for i in range(N):
            K_dd_reg[i, i] += self.jitter       # Régularisation
        K_inv = K_dd_reg.inverse()

        f_data_pt = f_data if isinstance(f_data, ot.Point) else ot.Point(list(f_data))
        mu_star   = K_star_d * (K_inv * f_data_pt)

        Sigma_arr = np.array(K_star_star) - np.array(K_star_d) @ np.array(K_inv) @ np.array(K_star_d).T
        Sigma_arr = 0.5 * (Sigma_arr + Sigma_arr.T)         # Symétrisation, passage par numpy peut être pas nécessaire
        Sigma_arr += self.jitter * np.eye(N_cand)            # Régularisation
        Sigma_star = ot.CovarianceMatrix(Sigma_arr.tolist())

        f_star = ot.Normal(mu_star, Sigma_star).getRealization()

        # 3) Thinning : accept with proba sigma(-f) 
        accept_probs = self.sigma(ot.Point([-float(f_star[i]) for i in range(N_cand)]))
        Uu   = ot.Uniform(0.0, 1.0).getSample(N_cand)
        mask = [i for i in range(N_cand) if float(Uu[i, 0]) < float(accept_probs[i])]

        if len(mask) == 0:
            return ot.Sample(0, 3)

        XY_acc = ot.Sample(len(mask), 2)
        f_acc  = np.zeros(len(mask))
        for k, i in enumerate(mask):
            XY_acc[k, 0] = XY_cand[i, 0]
            XY_acc[k, 1] = XY_cand[i, 1]
            f_acc[k]     = float(f_star[i])

        # 4) Sample PG marks 
        omega_acc = random_polyagamma(1.0, f_acc)
        n_acc = len(omega_acc)
        Pi_S = ot.Sample(n_acc, 3)
        for i in range(n_acc):
            Pi_S[i, 0] = XY_acc[i, 0]
            Pi_S[i, 1] = XY_acc[i, 1]
            Pi_S[i, 2] = omega_acc[i]

        return Pi_S

    def update_eps(self, eps, N_j, M_j, step):
        """

        """
        eps_arr = np.array(eps)

        # Gradient current state
        grad_cur  = self._grad_log_posterior_eps(eps_arr, N_j, M_j)

        # MALA proposition
        eta = np.random.randn(self.J)
        eps_star = eps_arr + 0.5 * step ** 2 * grad_cur + step * eta

        # Gradient proposed state
        grad_star = self._grad_log_posterior_eps(eps_star, N_j, M_j)

        # Log posterior 
        log_p_cur  = self._log_posterior_eps(eps_arr, N_j, M_j)
        log_p_star = self._log_posterior_eps(eps_star, N_j, M_j)

        # Log ratio Hastings
        diff_fwd = eps_star - eps_arr - 0.5 * step ** 2 * grad_cur          # q(eps* | eps)
        diff_bwd = eps_arr - eps_star - 0.5 * step ** 2 * grad_star         # q(eps | eps*)
        log_q_ratio = (
            -0.5 / step ** 2 * np.dot(diff_bwd, diff_bwd)
            + 0.5 / step ** 2 * np.dot(diff_fwd, diff_fwd)
        )

        # Log alpha threshold
        log_alpha = min(0.0, (log_p_star - log_p_cur) + log_q_ratio)

        if np.log(np.random.uniform()) < log_alpha:
            return eps_star, True
        else:
            return eps_arr, False
            
    def update_nu_mh(self, f_Df, D_f_sample, history_log_nu, it, step_nu_init=0.1, t0=50, sd=2.38**2/2, eps_mh=1e-6):
        """

        """
        nu0, nu1     = map(float, self.nu)
        log_nu_cur   = np.log([nu0, nu1])
 
        # --- Adaptive proposal covariance ---
        if it > t0 and len(history_log_nu) > t0:
            cov_emp = np.cov(np.array(history_log_nu).T)
            proposal_cov = sd * cov_emp + eps_mh * np.eye(2)
        else:
            proposal_cov = step_nu_init * np.eye(2)
 
        # --- Proposal on log scale ---
        log_nu_star = log_nu_cur + np.random.multivariate_normal(np.zeros(2), proposal_cov)
 
        # Clip to avoid numerically zero or negative values in OT
        # v^2 in (1e-6, 10),  l in (1e-4, 20)
        LOG_NU_MIN = np.array([np.log(1e-6), np.log(1e-4)])
        LOG_NU_MAX = np.array([np.log(10.0),  np.log(20.0)])
        log_nu_star = np.clip(log_nu_star, LOG_NU_MIN, LOG_NU_MAX)
        nu_star = np.exp(log_nu_star).tolist()
 
        # --- Log acceptance ratio ---
        log_p_cur = self._log_posterior_nu(self.nu, f_Df, D_f_sample)
        try:
            log_p_star = self._log_posterior_nu(nu_star, f_Df, D_f_sample)
        except Exception:
            # Proposed nu invalid for OT -> reject
            return self.nu, False
 
        # Guard against -inf / nan
        if not np.isfinite(log_p_star):
            return self.nu, False
 
        # Jacobian correction for log-scale proposal: log|J| = sum(log nu_star - log nu_cur)
        log_jacobian = np.sum(log_nu_star - log_nu_cur)
 
        log_alpha = min(0.0, (log_p_star - log_p_cur) + log_jacobian)
 
        if np.log(np.random.uniform()) < log_alpha:
            self.nu = ot.Point(nu_star)
            return self.nu, True
        else:
            return self.nu, False
    
    
    # ==============================================================================================
    # ------------------------------ Calibration & Initialisation ----------------------------------
    # ==============================================================================================

    def estimate_eps_mle(self, x, y):
        """
        
        """
        counts = np.zeros(self.J)
        for i in range(len(x)):
            pt = ShapelyPoint(float(x[i]), float(y[i]))
            for j, poly in enumerate(self.areas):
                if poly.covers(pt):
                    counts[j] += 1
                    break
        eps_mle = np.zeros(self.J)
        for j in range(self.J):
            area_j = self.polygons[j].area
            rate_j = max(counts[j] / (self.T * area_j), 1e-6)
            eps_mle[j] = np.log(rate_j)
        return eps_mle
    
    def calibrate_nu(self, x, y, grid_size=50, verbose=True):
        """

        """
        xmin, xmax = self.X_bounds
        ymin, ymax = self.Y_bounds

        # Grid 
        gx = np.linspace(xmin, xmax, grid_size)
        gy = np.linspace(ymin, ymax, grid_size)
        GX, GY = np.meshgrid(gx, gy)
        grid_pts = np.column_stack([GX.ravel(), GY.ravel()])
        ot_grid = ot.Sample(grid_pts)

        # KDE -> p_hat 
        sample_ot = ot.Sample([[float(x[i]), float(y[i])] for i in range(len(x))])
        ks = ot.KernelSmoothing()
        kde = ks.build(sample_ot)
        p_hat = np.array(kde.computePDF(ot_grid)).flatten()

        # eps par MLE 
        eps_mle = self.estimate_eps_mle(x, y)

        if verbose:
            print(f"[calibrate_nu] eps_mle = {np.round(eps_mle, 4)}")

        # Target : z(x,y) = 2*N*|S_j|/N_j * p_hat - 2 
        N_obs = len(x)
        counts = np.zeros(self.J)
        for i in range(N_obs):
            pt = ShapelyPoint(float(x[i]), float(y[i]))
            for j, poly in enumerate(self.areas):
                if poly.covers(pt):
                    counts[j] += 1
                    break
        areas_j = np.array([self.polygons[j].area for j in range(self.J)])
        counts = np.maximum(counts, 1e-6)
        coefs = 2.0 * N_obs * areas_j / counts   # shape (J,)

        n_grid = len(grid_pts)
        z      = np.zeros(n_grid)
        for k in range(n_grid):
            pt = ShapelyPoint(float(grid_pts[k, 0]), float(grid_pts[k, 1]))
            for j, poly in enumerate(self.areas):
                if poly.covers(pt):
                    z[k] = coefs[j] * p_hat[k] - 2.0
                    break

        # GP regression : z ~ GP(0, v^2 * RBF(l)) + noise ---
        kernel = (
            C(0.1, (1e-3, 0.58 ** 2))
            * RBF(length_scale=0.3, length_scale_bounds=(1e-2, 5.0))
            + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-4, 1.0))
        )
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        gp.fit(grid_pts, z)

        # --- 6. Extract fitted hyperparameters ---
        k_params = gp.kernel_.get_params()
        v_sq = float(k_params["k1__k1__constant_value"])
        l = float(k_params["k1__k2__length_scale"])
        v = np.sqrt(v_sq)
        self.nu = ot.Point([v_sq, l])

        if verbose:
            print(f"[calibrate_nu] v = {np.round(v, 4)} (v^2 = {v_sq:.4f}) ; l = {l:.4f}")

        return v, l, eps_mle
    

    # =================================================================================================
    # ----------------------------------------- Run du Gibbs ------------------------------------------
    # =================================================================================================

    def run(self, t, x, y, mala_step=0.05, n_iter=1000, learn_nu=False, step_nu_init=0.1, verbose=True, verbose_every=100):
        """

        """
        N = len(t)
        Z = ot.Point([0.0] * N)
        N_j, _ = self._count_events_per_zone(x, y, Z, ot.Sample(0, 3))

        # Heuristic calibration des hyperparameters du GP
        if verbose:
            print("\n[Pre-run] Calibrating GP hyperparameters")
        _, _, eps_mle = self.calibrate_nu(x, y, grid_size=50, verbose=verbose)

        if learn_nu and verbose:
            print("[Pre-run] nu will be updated at each iteration (Adaptive MH).")
        elif verbose:
            print(f"[Pre-run] nu fixed at calibrated value : {np.round(np.array(self.nu), 4)}  [v^2, l]")

        if verbose:
            eps = eps_mle
            print(f"[Initialisation] Using eps_mle as eps_init : {np.round(eps_mle, 4)}")
            f_data = ot.Point(N, 0.0)
            print(f"[Initialisation] Initialise f to zero (zero-mean prior)")

        # ---------- Stockage ----------
        eps_chain = np.zeros((n_iter, self.J))
        nPi_chain = np.zeros(n_iter, dtype=int)
        fdata_chain = np.zeros((n_iter, N))
        nu_chain = np.zeros((n_iter, 2))
        acc_eps = 0
        acc_nu = 0
        history_log_nu = []      # used only when learn_nu=True

        if verbose:
            print("\n" + "=" * 100)
            print(
                "-" * 29
                + f" Démarrage Gibbs : {n_iter} itérations, N={N} "
                + "-" * 29
            )
            print("=" * 100)

        for it in range(n_iter):
            try:
                # Step 1 : omega_D0 | f ~ PG(1, f(x_i, y_i)) 
                f_data_np = np.array(f_data)
                omega_D0 = ot.Point(random_polyagamma(1.0, f_data_np))

                # Step 2 : pi_S | f, eps ~ PP(...)
                Pi_S = self.sample_Pi_S(x, y, f_data, eps)

                # Step 3 : f | omega_D0, pi_S ~ N(mu_post, Sigma_post) 
                f_Df, D_f_xy, K_ff = self.update_f(x, y, Z, omega_D0, Pi_S)

                # Extract f at observed locations D_0 (first N_0 entries of D_f)
                idx_D0 = [i for i in range(N) if Z[i] == 0.0]
                f_data = ot.Point([float(f_Df[k]) for k, i in enumerate(idx_D0)])

                # Step 4 : eps | f, pi_S (MALA) 
                # M_j changes at each iteration as pi_S is resampled
                _, M_j = self._count_events_per_zone(x, y, Z, Pi_S)
                eps_arr, accepted_eps = self.update_eps(eps, N_j, M_j, step=mala_step)
                eps = ot.Point(eps_arr.tolist())
                acc_eps += int(accepted_eps)

                # Step 5 (optional) : nu | f  (Adaptive MH)
                if learn_nu:
                    history_log_nu.append(np.log(np.array(self.nu)))
                    _, accepted_nu = self.update_nu_mh(
                        f_Df, D_f_xy, history_log_nu, it,
                        step_nu_init=step_nu_init
                    )
                    acc_nu += int(accepted_nu)

                
                # ---------- Affichage ----------
                if verbose and (it % verbose_every == 0 or it == n_iter - 1):
                    acc_rate_eps = acc_eps / (it + 1) * 100
                    msg = (
                        f"[Iter {it}] "
                        f"|pi_S| = {Pi_S.getSize()} | "
                        f"f_mean = {np.round(np.mean(np.array(f_data)), 3)} | "
                        f"eps = {np.round(eps_arr, 3)} | "
                        f"acc_eps = {np.round(acc_rate_eps, 1)}%"
                    )
                    if learn_nu:
                        acc_rate_nu = acc_nu / (it + 1) * 100
                        msg += (f" | nu = {np.round(np.array(self.nu), 4)}"
                                f" | acc_nu = {np.round(acc_rate_nu, 1)}%")
                    print(msg)

                # ---------- Stockage ----------
                eps_chain[it, :] = eps_arr
                nPi_chain[it] = Pi_S.getSize()
                fdata_chain[it,:] = np.array(f_data)
                nu_chain[it, :] = np.array(self.nu)

            except Exception as e:
                print(f"\nError at iteration {it} : {e}")
                raise            
        
        if verbose:
            print("=" * 100)
            print("-" * 41 + " Gibbs terminé !! " + "-" * 41)
            print("=" * 100 + "\n")
            print(f"eps acceptance rate : {np.round(acc_eps / n_iter * 100, 1)}%  "
                  f"(target ~57%  ->  {'increase' if acc_eps/n_iter > 0.57 else 'decrease'} mala_step)")
            if learn_nu:
                print(f"nu acceptance rate : {np.round(acc_nu  / n_iter * 100, 1)}%  "
                      f"(target ~23%  ->  {'increase' if acc_nu/n_iter > 0.23 else 'decrease'} step_nu_init)")

        return {
            "eps" : eps_chain,
            "nPi" : nPi_chain,
            "f_data" : fdata_chain,
            "nu" : nu_chain,
            "acceptance_eps" : acc_eps / n_iter,
            "acceptance_nu":  acc_nu  / n_iter if learn_nu else None,
            "last_state" : {
                "eps" : eps_arr,
                "nu" : list(self.nu),
                "delta" : list(self.delta),
            },
            "Sigma_eps" : self.Sigma_eps,
            "centroids" : self.centroids_xy,
        }
    

    # ================================================================================================
    # ---------------------------------- Analyse posterior -------------------------------------------
    # ================================================================================================

    def posterior_summary(self, results, burn_in=0.3):
        """
        """
        eps_chain = np.asarray(results["eps"])
        f_chain = np.asarray(results["f_data"])
        nu_chain = np.asarray(results["nu"])
        burn = int(eps_chain.shape[0] * burn_in)
        return {
            "eps_hat" : eps_chain[burn:].mean(axis=0),
            "f_data_hat" : f_chain[burn:].mean(axis=0),
            "nu_hat" : nu_chain[burn:].mean(axis=0),
        }

    def posterior_gp(self, XY_data, f_data_hat, mesh, eps_hat):
        """

        """
        XY_grid = mesh.getVertices()
        N = XY_data.getSize()
        M = XY_grid.getSize()

        K_dd, K_gd, K_gg = self.compute_kernel(XY_data, XY_grid)
        K_dd_reg = ot.CovarianceMatrix(K_dd)
        for i in range(N):
            K_dd_reg[i, i] += self.jitter
        K_inv = K_dd_reg.inverse()

        f_hat_pt = f_data_hat if isinstance(f_data_hat, ot.Point) else ot.Point(list(f_data_hat))
        mu_post = K_gd * (K_inv * f_hat_pt)

        Sigma_arr = np.array(K_gg) - np.array(K_gd) @ np.array(K_inv) @ np.array(K_gd).T
        Sigma_arr = 0.5 * (Sigma_arr + Sigma_arr.T)
        Sigma_arr += self.jitter * np.eye(M)
        Sigma_post = ot.CovarianceMatrix(Sigma_arr.tolist())

        return mu_post, Sigma_post

    def plot_posterior_intensity(self, x, y, t, results, nx=70, ny=70, burn_in=0.3, save_path=None):
        """
        
        """
        post_sum = self.posterior_summary(results, burn_in)
        eps_hat = post_sum["eps_hat"]
        f_data_hat = post_sum["f_data_hat"]
        nu_hat = post_sum["nu_hat"]
        self.nu = ot.Point(nu_hat)

        N = len(t)
        XY_data = ot.Sample([[x[i], y[i]] for i in range(N)])

        xmin, xmax = self.X_bounds
        ymin, ymax = self.Y_bounds
        interval = ot.Interval([xmin, ymin], [xmax, ymax])
        mesher = ot.IntervalMesher([nx - 1, ny - 1])
        mesh = mesher.build(interval)
        M = mesh.getVertices().getSize()
        if M > 10000:
            raise ValueError(f"Mesh too large : {M} points")

        mu_post_grid, Sigma_post_grid = self.posterior_gp(
            XY_data, ot.Point(list(f_data_hat)), mesh, eps_hat
        )

        # --- Monte Carlo for E[mu_tilde(s) * sigma(f(s))] ---
        means = np.array(mu_post_grid).flatten()
        std_devs = np.sqrt(np.diagonal(np.array(Sigma_post_grid)))
        n_mc = 5000

        noise = np.random.randn(M, n_mc)
        f_sims = means[:, None] + std_devs[:, None] * noise

        XY_grid = mesh.getVertices()
        mu_tilde_grid = self.compute_mu_tilde(XY_grid, eps=eps_hat)

        sig_sims = 1.0 / (1.0 + np.exp(-f_sims))
        mu_hat_sims = mu_tilde_grid[:, None] * sig_sims
        mu_hat = mu_hat_sims.mean(axis=1)
        squared_mu_hat = (mu_hat_sims ** 2).mean(axis=1)

        mu_hat_sample = ot.Sample([[val] for val in mu_hat])
        mu_hat_field = ot.Field(mesh, mu_hat_sample)

        fig, axes = plt.subplots(1, 2, figsize=(13, 6))

        ax = axes[0]
        ax.scatter(x, y, c=t, s=12, alpha=0.7, edgecolors="black")
        ax.set_title(f"Observed data ({N} events)")
        ax.set_xlim(self.X_bounds); ax.set_ylim(self.Y_bounds)
        ax.set_aspect("equal"); ax.grid(alpha=0.3)

        ax = axes[1]
        plot_field(mu_hat_field, mode="subplot", ax=ax,
                   title=r"Posterior intensity $\hat{\mu}(s)$", add_colorbar=True)
        ax.scatter(x, y, s=10, alpha=0.5, color="white", edgecolors="black", linewidths=0.5)
        ax.set_xlim(self.X_bounds); ax.set_ylim(self.Y_bounds)
        ax.grid(alpha=0.3, color="white", linewidth=0.5)

        fig.suptitle(
            r"Posterior analysis : $\hat{\mu}(s) = \mathbb{E}\left[\tilde{\mu}(s)\,\sigma(\hat{f}(s))\right]$",
            fontsize=13, fontweight="bold"
        )
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
            print(f"Saved to : {save_path}")

        plt.show()

        return {
            "mu_hat" : mu_hat,
            "squared_mu_hat" : squared_mu_hat,
            "mu_field" : mu_hat_field,
            "mesh" : mesh,
            "mu_post_gp" : mu_post_grid,
            "Sigma_post_gp" : Sigma_post_grid,
            "eps_hat" : eps_hat,
            "f_data_hat" : f_data_hat,
        }
    
    def plot_chains(self, results, figsize=(9, 5)):
        eps_chain = np.asarray(results["eps"])
        nu_chain = np.asarray(results["nu"])
        n_iter = eps_chain.shape[0]
        iters = np.arange(n_iter)

        J = eps_chain.shape[1]
        fig, axes = plt.subplots(J, 2, figsize=(figsize[0], 3 * J), squeeze=False)
        for j in range(J):
            axes[j, 0].plot(iters, eps_chain[:, j], linewidth=1)
            axes[j, 0].set_title(rf"Trace $\epsilon_{j}$")
            axes[j, 0].set_xlabel("Iteration")
            axes[j, 0].grid(alpha=0.3)
            axes[j, 1].hist(eps_chain[:, j], bins=30, density=True,
                            edgecolor="black", alpha=0.7)
            axes[j, 1].set_title(rf"Histogram $\epsilon_{j}$")
            axes[j, 1].grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Trace de nu si appris
        if results["acceptance_nu"] is not None:
            fig, axes = plt.subplots(2, 2, figsize=(figsize[0], 6), squeeze=False)
            labels = [r"$v^2$", r"$\ell$"]
            for k in range(2):
                axes[k, 0].plot(iters, nu_chain[:, k], linewidth=1)
                axes[k, 0].set_title(rf"Trace {labels[k]}")
                axes[k, 0].set_xlabel("Iteration")
                axes[k, 0].grid(alpha=0.3)
                axes[k, 1].hist(nu_chain[:, k], bins=30, density=True,
                                edgecolor="black", alpha=0.7)
                axes[k, 1].set_title(rf"Histogram {labels[k]}")
                axes[k, 1].grid(alpha=0.3)
            plt.tight_layout()
            plt.show()
    
    def plot_acf(self, results, burn_in=0.3, max_lag=50, figsize=(8, 6)):
        """
        
        """
        eps_chain = np.asarray(results["eps"])
        nu_chain = np.asarray(results["nu"])
        n_iter = eps_chain.shape[0]
        burn = int(burn_in * n_iter)
        lags = np.arange(max_lag + 1)

        plots = []
        for j in range(eps_chain.shape[1]):
            plots.append((rf"$\epsilon_{j}$", eps_chain[burn:, j]))

        # Add nu traces only if they were learned (i.e. they vary)
        if results["acceptance_nu"] is not None:
            plots.append((r"$v^2$", nu_chain[burn:, 0]))
            plots.append((r"$\ell$",  nu_chain[burn:, 1]))

        n_plots = len(plots)
        fig, axes = plt.subplots(n_plots, 1, figsize=(figsize[0], 3.0 * n_plots))
        if n_plots == 1:
            axes = [axes]

        for ax, (param, chain) in zip(axes, plots):
            acf_vals = self._acf(chain, max_lag)
            ax.plot(lags, acf_vals)
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_xlim(0, max_lag)
            ax.set_ylim(-1.0, 1.0)
            ax.set_title(f"ACF — {param}")
            ax.set_xlabel("Lag")
            ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_ess_arviz(self, results, burn_in=0.3, kind="local", figsize=None):
        """

        """
        eps_chain = np.asarray(results["eps"])
        nu_chain = np.asarray(results["nu"])
        n_iter = eps_chain.shape[0]
        burn = int(burn_in * n_iter)

        eps_post = eps_chain[burn:, :]
        nu_post = nu_chain[burn:, :]

        posterior = {}
        for j in range(eps_post.shape[1]):
            posterior[f"eps_{j}"] = eps_post[:, j][None, :]

        if results["acceptance_nu"] is not None:
            posterior["v_sq"] = nu_post[:, 0][None, :]
            posterior["l"]    = nu_post[:, 1][None, :]

        idata = az.from_dict(posterior=posterior)
        ess = az.ess(idata)
        ess_dict = {var: ess[var].values for var in ess.data_vars}

        az.plot_ess(idata, kind=kind, figsize=figsize)
        plt.suptitle(f"ESS | N_post = {eps_post.shape[0]}", fontsize=12)
        plt.tight_layout()
        plt.show()

        return ess_dict

    def plot_rhat_arviz(self, results_list, burn_in=0.3, figsize=(12, 4), rhat_bad=1.05):
        """

        """
        M = len(results_list)
        L = results_list[0]["eps"].shape[0]
        burn = int(burn_in * L)
        draws = L - burn

        eps_arr = np.zeros((M, draws, self.J))
        nu_arr = np.zeros((M, draws, 2))

        learn_nu = results_list[0]["acceptance_nu"] is not None

        for m, res in enumerate(results_list):
            eps_arr[m, :, :] = np.asarray(res["eps"])[burn:, :]
            nu_arr[m,  :, :] = np.asarray(res["nu"])[burn:,  :]

        posterior = {
            "eps": eps_arr,
        }
        coords = {"eps_dim": np.arange(self.J)}
        dims = {"eps": ["eps_dim"]}

        if learn_nu:
            posterior["v_sq"] = nu_arr[:, :, 0]
            posterior["l"] = nu_arr[:, :, 1]

        idata = az.from_dict(posterior=posterior, coords=coords, dims=dims)
        r_hat = az.rhat(idata)
        rhat_eps = np.asarray(r_hat["eps"].values)   # shape (J,)

        # --- Plot ---
        n_extra = 2 if learn_nu else 0
        x_eps = np.arange(self.J)
        x_start = self.J

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.scatter(x_eps, rhat_eps, s=50, label=r"$\epsilon_j$")

        if learn_nu:
            rhat_vsq = float(r_hat["v_sq"].values)
            rhat_l = float(r_hat["l"].values)
            ax.scatter([x_start], [rhat_vsq], s=50, marker="D", label=r"$v^2$")
            ax.scatter([x_start + 1], [rhat_l], s=50, marker="D", label=r"$\ell$")

        ax.axhline(1.0, linestyle="--", color="green", linewidth=1.0, label="R-hat = 1")
        ax.axhline(rhat_bad, linestyle="--", color="red",   linewidth=1.0,
                   label=f"R-hat = {rhat_bad}")

        xtick_pos = list(x_eps)
        xtick_labels = [rf"$\epsilon_{j}$" for j in range(self.J)]
        if learn_nu:
            xtick_pos += [x_start, x_start + 1]
            xtick_labels += [r"$v^2$", r"$\ell$"]

        ax.set_xticks(xtick_pos)
        ax.set_xticklabels(xtick_labels)
        ax.set_ylabel(r"$\widehat{R}$")
        ax.set_title(rf"Gelman–Rubin $\widehat{{R}}$ sur {M} chaînes")
        ax.grid(alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.show()

        out = {"eps": rhat_eps}
        if learn_nu:
            out["v_sq"] = rhat_vsq
            out["l"] = rhat_l
        return out



