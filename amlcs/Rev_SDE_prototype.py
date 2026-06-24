import numpy as np
import torch



class REVERSE_SDE:
    def __init__(self, pseudo_time_step, prior_ensemble, ensemble_size, obs, sigma, y_dim, scalefact, indxob, indx_indxob_linear, initial_std):
        self.p_time_step = pseudo_time_step
        self.x0 = np.copy(prior_ensemble)               ### Must use np.copy(), otherwise it will be changed globally when calling the class
        self.prior_ensemble = prior_ensemble
        self.eps_alpha = 0.05
        self.eps_beta = 0.025
        self.ensemble_size = ensemble_size              ### the number of ensembles.
        self.obs = np.copy(obs)                         ### observations
        self.obs_sigma = np.copy(sigma)                 ### the std of observations
        self.x_dim = prior_ensemble.shape[1]
        self.y_dim = y_dim                              ### y_dim = nobs
        self.scalefact = scalefact                      ### scalefact is a constant in pyhscis.
        self.indxob = indxob
        self.indx_indxob_linear = indx_indxob_linear
        self.initial_std = initial_std                  ### the initial sample std.


        ### if indx_obs_linear is None, it means the observation is fully nonlinear
        ### For example, the model state x = (x_0,x_1,x_2,x_3,x_4,x_5), totally 6 dimensions.
        ### Sparese observation y = (y_0,y_3,y_4,y_5), only the corresponding 4 elements are observable.
        ### Hence, indxob = [0,3,4,5], indx_unob = [1,2]
        ### Additionally, we assign linear operator to "y_0","y_4" and nonlinear operator to "y_3","y_5"
        ### indx_indxob_linear = [0,2], indxob[[0,2]]=[0,4]
        ### indx_indxob_nonlinear = [1,3], indxob[[1,3]] = [3,5]




    def cond_alpha(self,t):
        # \alpha_t
        return 1. - (1. - self.eps_alpha) * t

    def cond_sigma_sq(self,t):
        # \beta_t ** 2
        return self.eps_beta + (1. - self.eps_beta) * t

    def f(self,t):
        # f=d_(log_alpha)/dt
        alpha_t = self.cond_alpha(t)
        f_t = -(1. - self.eps_alpha) / alpha_t
        return f_t

    def g(self,t):
        # g = d(sigma_t^2)/dt -2f sigma_t^2
        d_sigma_sq_dt = 1.
        g2 = d_sigma_sq_dt * (1.- self.eps_beta) - 2. * self.f(t) * self.cond_sigma_sq(t)
        return np.sqrt(g2)

    def g_tau(self,t):
        return 1.-t

    def score_likelihood(self, xt, t, indx_indxob_linear, indx_indxob_nonlinear):
        # obs: (y_dim,)
        # xt: (ensemble, x_dim)
        # score_x: (ensemble, y_dim)
        score_x = np.zeros((self.ensemble_size,self.y_dim), np.float32)
        score_x[:, indx_indxob_linear] = -(xt[:, indx_indxob_linear] - self.obs[indx_indxob_linear]) / (self.obs_sigma[indx_indxob_linear] ** 2)
        score_x[:, indx_indxob_nonlinear] = (-(np.arctan(xt[:, indx_indxob_nonlinear]) - self.obs[indx_indxob_nonlinear]) / self.obs_sigma[indx_indxob_nonlinear] ** 2) * (1. / (1. + (xt[:, indx_indxob_nonlinear])**2))
        tau = self.g_tau(t)
        return tau * score_x

    def normalize(self, indx_indxob_linear, indx_indxob_nonlinear):
        ## Normalization for x0
        mean_X0 = np.mean(self.x0[:,self.indxob], axis=0)                            ### (y_dim,)
        std_X0 = np.std(self.x0[:,self.indxob], axis=0)                              ### (y_dim,)
        self.x0[:,self.indxob] = (self.x0[:,self.indxob] - mean_X0) / std_X0         ### (ensemble, y_dim)
        ## Normalization for obs and obs_sigma
        self.obs[indx_indxob_linear] = (self.obs[indx_indxob_linear] - self.scalefact * mean_X0[indx_indxob_linear]) / std_X0[indx_indxob_linear]
        self.obs[indx_indxob_nonlinear] = np.arctan(((np.tan(self.obs[indx_indxob_nonlinear]) - mean_X0[indx_indxob_nonlinear]) / std_X0[indx_indxob_nonlinear]))
        self.obs_sigma[indx_indxob_linear] = (self.obs_sigma[[self.indx_indxob_linear]] / std_X0[indx_indxob_linear])
        self.obs_sigma[indx_indxob_nonlinear] = np.where(abs(self.obs[indx_indxob_nonlinear]) < 1.55,(self.obs_sigma[indx_indxob_nonlinear] * 1.),self.obs_sigma[indx_indxob_nonlinear] * 1e6)
        return mean_X0, std_X0

    def reverse_SDE(self):
        indxunob = np.sort(np.setdiff1d(np.arange(self.x_dim), self.indxob))
        indx_indxob_linear = self.indx_indxob_linear
        indx_indxob_nonlinear = np.sort(np.setdiff1d(np.arange(self.y_dim), indx_indxob_linear))


        ### Normalization step
        mean_X0, std_X0 = self.normalize(indx_indxob_linear, indx_indxob_nonlinear)         ### (ensemble, y_dim)
        dt = 1.0 / self.p_time_step
        #xt = torch.randn(self.ensemble_size, self.x_dim, device='cuda')
        torch.manual_seed(42)
        xt = torch.randn(self.ensemble_size, self.y_dim)
        x_ens_full = np.zeros((self.ensemble_size, self.x_dim))
        x_ens_full[:, indxunob] = self.prior_ensemble[:, indxunob]
        xt_array = np.zeros((int(0.1*self.p_time_step), self.y_dim))
        t = 1.0
        tolerance = 0.03
        for i in range(self.p_time_step):
            # prior score evaluation
            alpha_t = self.cond_alpha(t)
            sigma2_t = self.cond_sigma_sq(t)
            diffuse = self.g(t)
            drift_fun = self.f
            # Update
            xt_temp = xt - dt * (drift_fun(t) * xt + diffuse ** 2 * ((xt - alpha_t * self.x0[:,self.indxob]) / sigma2_t) -  self.score_likelihood(xt, t, indx_indxob_linear,  indx_indxob_nonlinear)) + np.sqrt(dt) * diffuse * torch.randn_like(xt)
            if (abs(xt_temp.mean(dim=0) - xt_array.mean(axis=0))).all() < tolerance and i > 0.2 * self.p_time_step:
                xt = xt_temp
                print("Pseudo time stops at ", i)
                break
            else:
                xt_array[i % xt_array.shape[0],:] = xt_temp.mean(dim=0)
            if i > 0.5 * self.p_time_step:
                print("Pseudo time stops at ", i)
                break
            xt = xt_temp
            t = t - dt

        ### Denormalization fot xt
        #x_ens_analysis_new = np.array(xt)
        x_ens = mean_X0 + np.array(xt) * std_X0
        x_ens_full[:, self.indxob] = x_ens
        x_ens_analysis_new = x_ens_full

        """
        ## Inflation: we want to restore "std(x_ens)" to the initial std "self.initial_std" in order to discentralize the ensembles.
        ## Here we change the std(x_ens) without changing the mean(x_ens).
        mean_infla = np.mean(x_ens_full, axis=0)
        std_infla = np.std(x_ens_full, axis=0)
        Xens_infla = (x_ens_full - mean_infla) / std_infla
        x_ens_analysis_new = Xens_infla * self.initial_std + mean_infla
        """



        return np.array(x_ens_analysis_new)
