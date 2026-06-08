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
from scipy.special import expit
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
    def _acf(x, max_lag):
        x = np.asarray(x)
        x = x - x.mean()
        n = len(x)

        max_lag = min(max_lag, n - 1)       # max_lag ne peut pas dépasser n-1
        if max_lag < 0:
            return np.array([1.0])

        var = np.dot(x, x) / n
        if var == 0.0:
            return np.zeros(max_lag + 1)

        acf_vals = np.empty(max_lag + 1)
        for k in range(max_lag + 1):
            acf_vals[k] = np.dot(x[:n - k], x[k:]) / (n * var)
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
        sigma_amp = np.sqrt(nu0)      # OT attend sigma, pas sigma^2

        if not isinstance(XY_data, ot.Sample):
            XY_data = ot.Sample(np.asarray(XY_data).tolist())
        N_data = XY_data.getSize()

        kernel = ot.SquaredExponential([nu1, nu1], [sigma_amp])   

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

        # 1) D_0
        D_0 = ot.Sample(N_0, 2)
        omega_D_0 = ot.Point(N_0)
        for k, i in enumerate(idx):
            D_0[k, 0] = x[i]
            D_0[k, 1] = y[i]
            omega_D_0[k] = omega_D0[i]

        # 2) Pi_S
        N_Pi = Pi_S.getSize()
        if N_Pi > 0:
            PiS_xy = ot.Sample(N_Pi, 2)
            omega_Pi = ot.Point(N_Pi)
            for i in range(N_Pi):
                PiS_xy[i, 0] = Pi_S[i, 0]
                PiS_xy[i, 1] = Pi_S[i, 1]
                omega_Pi[i] = Pi_S[i, 2]
        else:
            PiS_xy = ot.Sample(0, 2)
            omega_Pi = ot.Point(0)

        # 3) D_f = D_0 ∪ Pi_S
        N_f = N_0 + N_Pi
        D_f = ot.Sample(N_f, 2)
        for i in range(N_0):
            D_f[i, 0] = D_0[i, 0]
            D_f[i, 1] = D_0[i, 1]
        for i in range(N_Pi):
            D_f[N_0 + i, 0] = PiS_xy[i, 0]
            D_f[N_0 + i, 1] = PiS_xy[i, 1]

        # 4) K_ff sur D_f
        K_ff = self.compute_kernel(D_f)
        for i in range(N_f):
            K_ff[i, i] += self.jitter
        K_inv = K_ff.inverse()

        # 5) Omega = diag(omega_{D_0}, omega_{Pi_S})
        Omega = ot.CovarianceMatrix(N_f)
        for i in range(N_0):
            Omega[i, i] = omega_D_0[i]
        for i in range(N_Pi):
            Omega[N_0 + i, N_0 + i] = omega_Pi[i]

        # 6) kappa : +1/2 pour D_0, -1/2 pour Pi_S
        kappa = ot.Point(N_f)
        for i in range(N_0):
            kappa[i] = 0.5
        for i in range(N_Pi):
            kappa[N_0 + i] = -0.5

        # 7) Posterior sur D_f
        A_arr = np.array(K_inv) + np.array(Omega)
        A_arr = 0.5 * (A_arr + A_arr.T) + self.jitter * np.eye(N_f)
        Sigma_arr = np.linalg.inv(A_arr)
        Sigma_arr = 0.5 * (Sigma_arr + Sigma_arr.T) + self.jitter * np.eye(N_f)
        Sigma_post = ot.CovarianceMatrix(Sigma_arr.tolist())
        mu_post = Sigma_post * kappa

        # 8) Tirage sur D_f entier
        f_Df = ot.Normal(mu_post, Sigma_post).getRealization()

        # Restriction explicite à D_0 (indices 0..N_0-1)
        # Les indices N_0..N_f-1 correspondent à Pi_S et doivent pas être utilisés comme vecteur de conditionnement à l'itération suivante
        f_D0 = ot.Point([float(f_Df[i]) for i in range(N_0)])

        return f_D0, f_Df, D_f, K_ff

    def sample_Pi_S(self, x, y, f_data, eps, LIM_CANDIDATES_ZONES=1000, LIM_CANDIDATES=2000):
        """

        """
        N = len(x)
        XY_data = ot.Sample([[x[i], y[i]] for i in range(N)])

        XY_cand_list = []
        for j in range(self.J):
            raw_poly = self.polygons[j]
            prep_poly = self.areas[j]
            bx, by, bx2, by2 = raw_poly.bounds
            area_j = raw_poly.area
            mean_j = self.T * area_j * np.exp(float(eps[j]))
            N_cand_j = int(ot.Poisson(mean_j).getRealization()[0])
            if N_cand_j == 0:
                continue
            if N_cand_j > LIM_CANDIDATES_ZONES:
                N_cand_j = LIM_CANDIDATES_ZONES
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

        if len(XY_cand_list) > LIM_CANDIDATES:
            XY_cand_list = XY_cand_list[:LIM_CANDIDATES]

        N_cand = len(XY_cand_list)
        XY_cand = ot.Sample(XY_cand_list)

        # Prédiction GP conditionnelle sur D_0
        K_dd, K_star_d, K_star_star = self.compute_kernel(XY_data, XY_cand)
        K_dd_reg = ot.CovarianceMatrix(K_dd)
        for i in range(N):
            K_dd_reg[i, i] += self.jitter
        K_inv = K_dd_reg.inverse()

        f_data_pt = f_data if isinstance(f_data, ot.Point) else ot.Point(list(f_data))
        mu_star = K_star_d * (K_inv * f_data_pt)

        Sigma_arr = (
            np.array(K_star_star)
            - np.array(K_star_d) @ np.array(K_inv) @ np.array(K_star_d).T
        )
        Sigma_arr = 0.5 * (Sigma_arr + Sigma_arr.T) + self.jitter * np.eye(N_cand)
        Sigma_star = ot.CovarianceMatrix(Sigma_arr.tolist())

        f_star = ot.Normal(mu_star, Sigma_star).getRealization()

        # Thinning 
        accept_probs = expit(ot.Point([-float(f_star[i]) for i in range(N_cand)]))
        Uu = ot.Uniform(0.0, 1.0).getSample(N_cand)
        mask = [i for i in range(N_cand) if float(Uu[i, 0]) < float(accept_probs[i])]

        if len(mask) == 0:
            return ot.Sample(0, 3)

        XY_acc = ot.Sample(len(mask), 2)
        f_acc = np.zeros(len(mask))
        for k, i in enumerate(mask):
            XY_acc[k, 0] = XY_cand[i, 0]
            XY_acc[k, 1] = XY_cand[i, 1]
            f_acc[k] = float(f_star[i])

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

        # MALA proposaal
        grad_cur  = self._grad_log_posterior_eps(eps_arr, N_j, M_j)
        eta = np.random.randn(self.J)
        eps_star = eps_arr + 0.5 * step**2 * grad_cur + step * eta

        grad_star = self._grad_log_posterior_eps(eps_star, N_j, M_j)
        log_p_cur  = self._log_posterior_eps(eps_arr, N_j, M_j)
        log_p_star = self._log_posterior_eps(eps_star, N_j, M_j)

        diff_fwd = eps_star - eps_arr - 0.5 * step**2 * grad_cur
        diff_bwd = eps_arr - eps_star - 0.5 * step**2 * grad_star 

        # Ratio d'Hasrtings
        log_q_ratio = (
            - 0.5 / step**2 * np.dot(diff_bwd, diff_bwd)
            + 0.5 / step**2 * np.dot(diff_fwd, diff_fwd)
        )

        log_alpha = min(0.0, (log_p_star - log_p_cur) + log_q_ratio)

        if np.log(np.random.uniform()) < log_alpha:
            return eps_star, True
        else:
            return eps_arr, False

    def update_nu(self, f_Df, D_f_sample, history_log_nu, it, step_nu_init=0.1, t0=50, sd=2.38**2/2, eps_mh=1e-6):
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
 
        # Jacobian correction for log-scale proposal : log|J| = sum(log nu_star - log nu_cur)
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
    
    def calibrate_nu(self, x, y, grid_size=None, verbose=True):
        """"""
        xmin, xmax = self.X_bounds
        ymin, ymax = self.Y_bounds

        if grid_size is None:
            ot_grid = ot.Sample( np.vstack((x, y)).reshape(-1,2) )
        else:
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
        coefs = 2.0 * N_obs * areas_j / counts 

        n_grid = len(grid_pts)
        z = np.zeros(n_grid)
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
    
        # ----- Extract fitted hyperparameters -----
        k_params = gp.kernel_.get_params()
        v_sq = float(k_params["k1__k1__constant_value"])
        l = float(k_params["k1__k2__length_scale"])
        v = np.sqrt(v_sq)
        
        # Attention différente écriture pour noyau SE entre Sklearn et OT
        l_ot = l * np.sqrt(2.0)
        self.nu = ot.Point([v_sq, l_ot])

        if verbose:
            print(f"[calibrate_nu] v = {np.round(v, 4)} ; l_ot = {l_ot:.4f}")

        return v, l_ot, eps_mle
    

    # =================================================================================================
    # ----------------------------------------- Run du Gibbs ------------------------------------------
    # =================================================================================================

    def run(self, t, x, y, mala_step=0.05, n_iter=1000, learn_nu=False, t0_nu=50,
        step_nu_init=0.1, verbose=True, verbose_every=100, use_calibration=True,
        mu_star_func=None,grid_nx=30, grid_ny=30, thin=1): 

        N = len(t)
        Z = ot.Point([0.0] * N)
        N_j, _ = self._count_events_per_zone(x, y, Z, ot.Sample(0, 3))

        if use_calibration:
            if verbose:
                print("[Pre-run] Calibrating GP hyperparameters")
            _, _, eps_mle = self.calibrate_nu(x, y, grid_size=50, verbose=verbose)
        else:
            if verbose:
                print(f"[Pre-run] Using provided nu_init = {list(self.nu)}")
            eps_mle = self.estimate_eps_mle(x, y)

        if learn_nu and verbose:
           print("[Pre-run] nu will be updated at each iteration (Adaptive MH).")
        elif verbose:
           print(f"[Pre-run] nu fixed at : {np.round(np.array(self.nu), 4)} [v^2, l]")

        eps = ot.Point(eps_mle.tolist())
        f_data = ot.Point([0.0] * N)

        if verbose:
            print(f"[Initialisation] Using eps_mle as eps_init : {np.round(eps_mle, 4)}")
            print(f"[Initialisation] Initialise f to zero (zero-mean prior)")

        # ---------- Grille fixe pour le calcul de Eps_mu ----------
        xmin, xmax = self.X_bounds
        ymin, ymax = self.Y_bounds
        gx = np.linspace(xmin, xmax, grid_nx)
        gy = np.linspace(ymin, ymax, grid_ny)
        GX, GY = np.meshgrid(gx, gy)
        grid_x = GX.ravel()
        grid_y = GY.ravel()
        M_grid = len(grid_x)
        domain_area = (xmax - xmin) * (ymax - ymin)

        # Précalcul de mu*(grille) si fourni
        if mu_star_func is not None:
            mu_star_grid = mu_star_func(grid_x, grid_y)
        else:
            mu_star_grid = None

        # ---------- Stockage ----------
        n_store = (n_iter + thin - 1) // thin
        eps_chain = np.zeros((n_store, self.J))
        nPi_chain = np.zeros(n_store, dtype=int)
        fdata_chain = np.zeros((n_store, N))
        nu_chain = np.zeros((n_store, 2))
        E_mu_chain = np.full(n_iter, np.nan)      # Je garde toutes les it pour E_mu
        store_idx = 0       # compteur de stockage
        acc_eps = 0
        acc_nu = 0
        history_log_nu = []         # used only when learn_nu=True

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
                f_D0, f_Df, D_f_xy, K_ff = self.update_f(x, y, Z, omega_D0, Pi_S)
                # f_data = f_D0    # restriction à D_0 déjà faite dans update_f

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
                    _, accepted_nu = self.update_nu(
                        f_Df, D_f_xy, history_log_nu, it, step_nu_init=step_nu_init, t0=t0_nu
                    )
                    acc_nu += int(accepted_nu)

                
                # ---------- Affichage ----------
                if verbose and (it % verbose_every == 0 or it == n_iter - 1):
                    acc_rate_eps = acc_eps / (it + 1) * 100
                    msg = (
                        f"[Iter {it}] "
                        f"|pi_S| = {Pi_S.getSize()} | "
                        f"eps = {np.round(eps_arr, 3)} | "
                        f"acc_eps = {np.round(acc_rate_eps, 1)}%"
                    )
                    if learn_nu:
                        acc_rate_nu = acc_nu / (it + 1) * 100
                        msg += (f" | nu = {np.round(np.array(self.nu), 4)}"
                                f" | acc_nu = {np.round(acc_rate_nu, 1)}%")
                    print(msg)

                # ---------- Calcul de Eps_mu^(t) ----------
                # Calcul de Eps_mu toutes les X itérations seulement
                if mu_star_func is not None and (it % 10 == 0):
                    XY_data_ot = ot.Sample([[x[i], y[i]] for i in range(N)])
                    XY_grid = ot.Sample(np.column_stack([grid_x, grid_y]).tolist())

                    K_dd, K_gd, K_gg = self.compute_kernel(XY_data_ot, XY_grid)
                    K_dd_reg = ot.CovarianceMatrix(K_dd)
                    for ii in range(N):
                        K_dd_reg[ii, ii] += self.jitter
                    K_inv = K_dd_reg.inverse()

                    f_data_pt = ot.Point(list(f_data))
                    mu_g = np.array(K_gd * (K_inv * f_data_pt)).flatten()
                    Sigma_g = (np.array(K_gg) - np.array(K_gd) @ np.array(K_inv) @ np.array(K_gd).T)
                    Sigma_g = 0.5 * (Sigma_g + Sigma_g.T) + self.jitter * np.eye(M_grid)

                    # Tirage via Cholesky
                    L_g = np.linalg.cholesky(Sigma_g)
                    f_draw_g = mu_g + L_g @ np.random.randn(M_grid)

                    #XY_full_ot   = ot.Sample(np.column_stack([grid_x, grid_y]).tolist())
                    mu_tilde_g = self.compute_mu_tilde(XY_grid, eps=eps_arr)
                    mu_draw_g = mu_tilde_g * (1.0 / (1.0 + np.exp(-f_draw_g)))

                    E_mu_chain[it] = (domain_area / M_grid) * np.sum((mu_draw_g - mu_star_grid) ** 2)

                # ---------- Stockage ----------
                if it % thin == 0:
                    eps_chain[store_idx, :] = eps_arr
                    nPi_chain[store_idx] = Pi_S.getSize()
                    fdata_chain[store_idx, :] = np.array(f_data)
                    nu_chain[store_idx, :] = np.array(self.nu)
                    store_idx += 1

            except Exception as e:
                print(f"\nError at iteration {it} : {e}")
                raise           
        
        if verbose:
            print("=" * 100)
            print("-" * 41 + " Gibbs terminé !! " + "-" * 41)
            print("=" * 100 + "\n")
            print(f"eps acceptance rate : {np.round(acc_eps / n_iter * 100, 1)}%"
                  f" (target ~57% -> {'increase' if acc_eps/n_iter > 0.57 else 'decrease'} mala_step)")
            if learn_nu:
                print(f"nu acceptance rate : {np.round(acc_nu  / n_iter * 100, 1)}%"
                      f" (target ~23% -> {'increase' if acc_nu/n_iter > 0.23 else 'decrease'} step_nu_init)")

        return {
            "eps"            : eps_chain[:store_idx],
            "nPi"            : nPi_chain[:store_idx],
            "f_data"         : fdata_chain[:store_idx],
            "nu"             : nu_chain[:store_idx],
            "E_mu"           : E_mu_chain,
            "acceptance_eps" : acc_eps / n_iter,
            "acceptance_nu"  : acc_nu / n_iter if learn_nu else None,
            "last_state"     : {"eps": eps_arr, "nu": list(self.nu), "delta": list(self.delta)},
            "Sigma_eps"      : self.Sigma_eps,
            "centroids"      : self.centroids_xy,
            "thin"           : thin,
            "n_iter"         : n_iter,
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
    





    ####################################################################################################
    ####################################################################################################
    ####################################################################################################
    ######################################### CHECKPOINT VERIF #########################################
    ####################################################################################################
    ####################################################################################################
    ####################################################################################################

    def plot_posterior_intensity(self, x, y, t, results, nx=70, ny=70, burn_in=0.3,
                             cmap="viridis", savefigure=False, title_savefig="posterior",
                             savefigure_Emu=False, title_savefig_Emu="Emu", color_Emu="steelblue",
                             mu_star_func=None):
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
        if M > 22500:
            raise ValueError(f"Mesh too large : {M} points")

        mu_post_grid, Sigma_post_grid = self.posterior_gp(
            XY_data, ot.Point(list(f_data_hat)), mesh, eps_hat
        )

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
        mu_hat_field  = ot.Field(mesh, mu_hat_sample)

        # ---------- Calcul de E_mu ----------
        domain_area = (xmax - xmin) * (ymax - ymin)
        E_mu_full = results["E_mu"]
        mask = ~np.isnan(E_mu_full)
        E_mu_post = E_mu_full[mask]
        iters_post = np.where(mask)[0]

        E_mu_bar = None
        if len(E_mu_post) > 0:
            E_mu_bar = E_mu_post.mean()

            fig_err, ax_err = plt.subplots(figsize=(9, 3))
            ax_err.plot(iters_post, E_mu_post, linewidth=0.8, color=color_Emu)
            ax_err.set_xlabel("Iteration")
            ax_err.set_ylabel(r"$\mathcal{E}_\mu^{(t)}$")
            ax_err.set_title(r"$L^2$ reconstruction error $\mathcal{E}_\mu^{(t)}$" + "\n")
            ax_err.grid(alpha=0.3)
            plt.tight_layout()

            if savefigure_Emu:
                try:
                    try:
                        ROOT = Path(__file__).resolve().parent.parent
                    except NameError:
                        ROOT = Path(".").resolve()
                    FIGURES_DIR = ROOT / "visualizations" / "figures"
                    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
                    save_path = FIGURES_DIR / Path(title_savefig_Emu).with_suffix(".pdf")
                    fig_err.savefig(save_path, format="pdf", dpi=200, bbox_inches="tight")
                    print(f"Figure E_mu sauvegardée : {save_path}")
                except Exception as e:
                    print(f"Erreur lors de la sauvegarde E_mu : {e}")

            plt.show()

        # ---------- Vraie intensité sur la grille (si fournie) ----------
        mu_star_grid = None
        if mu_star_func is not None:
            grid_xy = np.array(XY_grid)
            mu_star_grid = mu_star_func(grid_xy[:, 0], grid_xy[:, 1])

            mu_star_sample = ot.Sample([[val] for val in mu_star_grid])
            mu_star_field = ot.Field(mesh, mu_star_sample)

            #diff = np.abs(mu_hat - mu_star_grid)
            diff = np.abs(mu_hat - mu_star_grid) / (mu_star_grid + self.jitter)
            diff_sample = ot.Sample([[val] for val in diff])
            diff_field  = ot.Field(mesh, diff_sample)

        # ---------- Figure ----------
        if mu_star_func is not None:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

            ax = axes[0]
            plot_field(mu_star_field, mode="subplot", ax=ax, cmap=cmap, add_colorbar=True)
            ax.set_title(r"Vraie intensité $\mu^\star(s)$" + "\n")
            ax.set_xlim(self.X_bounds); ax.set_ylim(self.Y_bounds)
            ax.grid(alpha=0.3, color="white", linewidth=0.5)

            ax = axes[1]
            plot_field(mu_hat_field, mode="subplot", ax=ax, cmap=cmap, add_colorbar=True)
            ax.set_title(r"Intensité estimée $\hat{\mu}(s)$" + "\n")
            # ax.scatter(x, y, s=10, alpha=0.5, color="white", edgecolors="black", linewidths=0.5)
            ax.set_xlim(self.X_bounds); ax.set_ylim(self.Y_bounds)
            ax.grid(alpha=0.3, color="white", linewidth=0.5)

            ax = axes[2]
            plot_field(diff_field, mode="subplot", ax=ax, cmap=cmap, add_colorbar=True)
            #ax.set_title(r"$|\hat{\mu}(s) - \mu^\star(s)|$")
            ax.set_title(r"Erreur relative $\frac{|\hat{\mu}(s) - \mu^\star(s)|}{\mu^\star(s)}$" + "\n")
            ax.set_xlim(self.X_bounds); ax.set_ylim(self.Y_bounds)
            ax.grid(alpha=0.3, color="white", linewidth=0.5)

        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

            ax = axes[0]
            ax.scatter(x, y, c=t, s=12, alpha=0.7, edgecolors="black", cmap="plasma")
            ax.set_title(f"Données observées (N={N})" + "\n")
            ax.set_xlim(self.X_bounds); ax.set_ylim(self.Y_bounds)
            ax.set_aspect("equal", adjustable="box"); ax.grid(alpha=0.3)

            ax = axes[1]
            plot_field(mu_hat_field, mode="subplot", ax=ax, cmap=cmap, add_colorbar=True)
            # ax.set_title(r"Intensité a posteriori $\hat{\mu}(s)$" + "\n")
            # ax.scatter(x, y, s=10, alpha=0.5, color="white", edgecolors="black", linewidths=0.5)
            ax.set_title(r"Intensité estimée $\hat{\mu}(s)$" + "\n")
            ax.set_xlim(self.X_bounds); ax.set_ylim(self.Y_bounds)
            ax.grid(alpha=0.3, color="white", linewidth=0.5)

        plt.tight_layout()

        if savefigure:
            try:
                try:
                    ROOT = Path(__file__).resolve().parent.parent
                except NameError:
                    ROOT = Path(".").resolve()
                FIGURES_DIR = ROOT / "visualizations" / "figures"
                FIGURES_DIR.mkdir(parents=True, exist_ok=True)
                save_path = FIGURES_DIR / Path(title_savefig).with_suffix(".pdf")
                fig.savefig(save_path, format="pdf", dpi=200, bbox_inches="tight")
                print(f"Figure sauvegardée : {save_path}")
            except Exception as e:
                print(f"Erreur lors de la sauvegarde : {e}")

        plt.show()

        return {
            "mu_hat"         : mu_hat,
            "mu_star"        : mu_star_grid,
            #"diff"           : np.abs(mu_hat - mu_star_grid) if mu_star_grid is not None else None,
            "diff"           : diff if mu_star_grid is not None else None,
            "squared_mu_hat" : squared_mu_hat,
            "mu_field"       : mu_hat_field,
            "mesh"           : mesh,
            "mu_post_gp"     : mu_post_grid,
            "Sigma_post_gp"  : Sigma_post_grid,
            "eps_hat"        : eps_hat,
            "f_data_hat"     : f_data_hat,
            "E_mu_bar"       : E_mu_bar,
            "E_mu_chain"     : E_mu_post,
        }
    ####################################################################################################
    ####################################################################################################
    ####################################################################################################
    ########################################## FIN CHECKPOINT ##########################################
    ####################################################################################################
    ####################################################################################################
    ####################################################################################################









    
    def plot_chains(self, results, figsize=(9, 5), savefigure=False, title_savefig="traces_eps"):
        eps_chain = np.asarray(results["eps"])
        nu_chain = np.asarray(results["nu"])
        thin = results.get("thin", 1)
        n_iter = results.get("n_iter", eps_chain.shape[0])
        n_store = eps_chain.shape[0]

        # Axe x en vraies itérations
        iters = np.arange(n_store) * thin

        J = eps_chain.shape[1]
        fig, axes = plt.subplots(J, 2, figsize=(figsize[0], 3 * J), squeeze=False)
        for j in range(J):
            axes[j, 0].plot(iters, eps_chain[:, j], linewidth=1)
            axes[j, 0].set_title(rf"Trace $\epsilon_{j}$")
            axes[j, 0].set_xlabel(f"Iteration (thin={thin})")
            axes[j, 0].grid(alpha=0.3)
            axes[j, 1].hist(eps_chain[:, j], bins=30, density=True,
                            edgecolor="black", alpha=0.7)
            axes[j, 1].set_title(rf"Histogram $\epsilon_{j}$")
            axes[j, 1].grid(alpha=0.3)
        plt.tight_layout()
        if savefigure:
            try:
                try:
                    ROOT = Path(__file__).resolve().parent.parent
                except NameError:
                    ROOT = Path(".").resolve()
                FIGURES_DIR = ROOT / "visualizations" / "figures"
                FIGURES_DIR.mkdir(parents=True, exist_ok=True)
                save_path = FIGURES_DIR / Path(title_savefig).with_suffix(".pdf")
                fig.savefig(save_path, format="pdf", dpi=200, bbox_inches="tight")
                print(f"Figure sauvegardée : {save_path}")
            except Exception as e:
                print(f"Erreur lors de la sauvegarde : {e}")
        plt.show()

        if results["acceptance_nu"] is not None:
            fig, axes = plt.subplots(2, 2, figsize=(figsize[0], 6), squeeze=False)
            labels = [r"$v^2$", r"$\ell$"]
            for k in range(2):
                axes[k, 0].plot(iters, nu_chain[:, k], linewidth=1)
                axes[k, 0].set_title(rf"Trace {labels[k]}")
                axes[k, 0].set_xlabel(f"Iteration (thin={thin})")
                axes[k, 0].grid(alpha=0.3)
                axes[k, 1].hist(nu_chain[:, k], bins=30, density=True,
                                edgecolor="black", alpha=0.7)
                axes[k, 1].set_title(rf"Histogram {labels[k]}")
                axes[k, 1].grid(alpha=0.3)
            plt.tight_layout()
            if savefigure:
                try:
                    try:
                        ROOT = Path(__file__).resolve().parent.parent
                    except NameError:
                        ROOT = Path(".").resolve()
                    FIGURES_DIR = ROOT / "visualizations" / "figures"
                    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
                    save_path = FIGURES_DIR / Path("traces_nu").with_suffix(".pdf")
                    fig.savefig(save_path, format="pdf", dpi=200, bbox_inches="tight")
                    print(f"Figure sauvegardée : {save_path}")
                except Exception as e:
                    print(f"Erreur lors de la sauvegarde : {e}")
            plt.show()


    def plot_acf(self, results, burn_in=0.3, max_lag=50, figsize=(8, 6), savefigure=False, title_savefig="trace_acf"):
        eps_chain = np.asarray(results["eps"])
        nu_chain = np.asarray(results["nu"])
        thin = results.get("thin", 1)
        n_store = eps_chain.shape[0]
        burn = int(burn_in * n_store)

        # Garde : max_lag ne peut pas dépasser le nombre d'échantillons post burn-in - 1
        n_post = n_store - burn
        max_lag = min(max_lag, n_post - 1)
        if max_lag < 1:
            print(f"[plot_acf] Pas assez d'échantillons post burn-in ({n_post}) pour calculer l'ACF.")
            return

        lags = np.arange(max_lag + 1)

        plots = []
        for j in range(eps_chain.shape[1]):
            plots.append((rf"$\epsilon_{j}$", eps_chain[burn:, j]))

        if results["acceptance_nu"] is not None:
            plots.append((r"$v^2$", nu_chain[burn:, 0]))
            plots.append((r"$\ell$", nu_chain[burn:, 1]))

        n_plots = len(plots)
        fig, axes = plt.subplots(n_plots, 1, figsize=(figsize[0], 3.0 * n_plots))
        if n_plots == 1:
            axes = [axes]

        for ax, (param, chain) in zip(axes, plots):
            acf_vals = self._acf(chain, max_lag)
            ax.plot(lags[:len(acf_vals)], acf_vals)
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_xlim(0, max_lag)
            ax.set_ylim(-1.0, 1.0)
            ax.set_title(f"ACF — {param} (thin={thin})")
            ax.set_xlabel("Lag")
            ax.grid(alpha=0.3)

        plt.tight_layout()
        if savefigure:
            try:
                try:
                    ROOT = Path(__file__).resolve().parent.parent
                except NameError:
                    ROOT = Path(".").resolve()
                FIGURES_DIR = ROOT / "visualizations" / "figures"
                FIGURES_DIR.mkdir(parents=True, exist_ok=True)
                save_path = FIGURES_DIR / Path(title_savefig).with_suffix(".pdf")
                fig.savefig(save_path, format="pdf", dpi=200, bbox_inches="tight")
                print(f"Figure sauvegardée : {save_path}")
            except Exception as e:
                print(f"Erreur lors de la sauvegarde : {e}")
        plt.show()
    
    # def plot_ess_arviz(self, results, burn_in=0.3, kind="local", figsize=None, savefigure=False):
    #     """

    #     """
    #     eps_chain = np.asarray(results["eps"])
    #     n_iter = eps_chain.shape[0]
    #     burn = int(burn_in * n_iter)

    #     eps_post = eps_chain[burn:, :]
    #     posterior = {}
    #     for j in range(eps_post.shape[1]):
    #         posterior[f"eps_{j}"] = eps_post[:, j][None, :]

    #     if results["acceptance_nu"] is not None:
    #         nu_chain = np.asarray(results["nu"])
    #         nu_post = nu_chain[burn:, :]
    #         posterior["v"] = nu_post[:, 0][None, :]
    #         posterior["l"] = nu_post[:, 1][None, :]

    #     # Création de l'objet InferenceData
    #     idata = az.from_dict(posterior=posterior)
        
    #     ess_bulk = az.ess(idata, method="bulk")
    #     ess_tail = az.ess(idata, method="tail")
        
    #     ess_dict = {}
    #     for var in ess_bulk.data_vars:
    #         ess_dict[var] = {
    #             "Bulk_ESS" : float(ess_bulk[var].values),
    #             "Tail_ESS" : float(ess_tail[var].values)
    #         }

    #     # Affichage
    #     az.plot_ess(idata, kind=kind, figsize=figsize)
    #     plt.suptitle("ESS Diagnostic", fontsize=12)
    #     plt.tight_layout()
    #     if savefigure:
    #         try:
    #             try:
    #                 ROOT = Path(__file__).resolve().parent.parent
    #             except NameError:
    #                 ROOT = Path(".").resolve()
    #             FIGURES_DIR = ROOT / "visualizations" / "figures"
    #             FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    #             save_path = FIGURES_DIR / Path("ess").with_suffix(".pdf")
    #             fig.savefig(save_path, format="pdf", dpi=200, bbox_inches="tight")
    #             print(f"Figure sauvegardée : {save_path}")
    #         except Exception as e:
    #             print(f"Erreur lors de la sauvegarde : {e}")
    #     plt.show()

    #     return ess_dict

    # def plot_rhat_arviz(self, results_list, burn_in=0.3, figsize=(12, 4), rhat_bad=1.05, savefigure=False):
    #     """

    #     """
    #     M = len(results_list)
    #     L = results_list[0]["eps"].shape[0]
    #     burn = int(burn_in * L)
    #     draws = L - burn

    #     eps_arr = np.zeros((M, draws, self.J))
    #     for m, res in enumerate(results_list):
    #         eps_arr[m, :, :] = np.asarray(res["eps"])[burn:, :]

    #     posterior = {
    #         "eps" : eps_arr,
    #     }
    #     coords = {"eps_dim" : np.arange(self.J)}
    #     dims = {"eps" : ["eps_dim"]}

    #     idata = az.from_dict(posterior=posterior, coords=coords, dims=dims)
    #     r_hat = az.rhat(idata)
    #     rhat_eps = np.asarray(r_hat["eps"].values)

    #     # --- Plot ---
    #     x_eps = np.arange(self.J)

    #     fig, ax = plt.subplots(1, 1, figsize=figsize)
    #     ax.scatter(x_eps, rhat_eps, s=50, label=r"$\epsilon_j$")

    #     ax.axhline(1.0, linestyle="--", color="green", linewidth=1.0, label="R-hat = 1")
    #     ax.axhline(rhat_bad, linestyle="--", color="red",   linewidth=1.0,
    #                label=f"R-hat = {rhat_bad}")

    #     xtick_pos = list(x_eps)
    #     xtick_labels = [rf"$\epsilon_{j}$" for j in range(self.J)]

    #     ax.set_xticks(xtick_pos)
    #     ax.set_xticklabels(xtick_labels)
    #     ax.set_ylabel(r"$\widehat{R}$")
    #     ax.set_title(rf"Gelman–Rubin $\widehat{{R}}$ sur {M} chaînes")
    #     ax.grid(alpha=0.3)
    #     ax.legend()
    #     plt.tight_layout()
    #     if savefigure:
    #         try:
    #             try:
    #                 ROOT = Path(__file__).resolve().parent.parent
    #             except NameError:
    #                 ROOT = Path(".").resolve()
    #             FIGURES_DIR = ROOT / "visualizations" / "figures"
    #             FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    #             save_path = FIGURES_DIR / Path("r_hat").with_suffix(".pdf")
    #             fig.savefig(save_path, format="pdf", dpi=200, bbox_inches="tight")
    #             print(f"Figure sauvegardée : {save_path}")
    #         except Exception as e:
    #             print(f"Erreur lors de la sauvegarde : {e}")
    #     plt.show()

    #     out = {"eps": rhat_eps}

    #     return out
    
    def compute_diagnostics_multichain(self, results_list, burn_in=0.3):
        M = len(results_list)
        L = results_list[0]["eps"].shape[0]   # n_store (après thinning)
        burn = int(burn_in * L)
        draws = L - burn

        eps_arr = np.zeros((M, draws, self.J))
        for m, res in enumerate(results_list):
            eps_arr[m, :, :] = np.asarray(res["eps"])[burn:, :]

        posterior = {"eps": eps_arr}
        coords = {"eps_dim": np.arange(self.J)}
        dims = {"eps": ["eps_dim"]}

        idata = az.from_dict(posterior=posterior, coords=coords, dims=dims)
        r_hat = az.rhat(idata)["eps"].values
        ess_bulk = az.ess(idata, method="bulk")["eps"].values
        ess_tail = az.ess(idata, method="tail")["eps"].values

        return r_hat, ess_bulk, ess_tail




# %%
