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




