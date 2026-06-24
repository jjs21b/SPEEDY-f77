import numpy as np
import torch



class REVERSE_SDE2:
    def __init__(self, pseudo_time_step, prior_ensemble, ensemble_size, obs, sigma, y_dim, indxob, indx_indxob_linear):
        self.p_time_step = pseudo_time_step
        self.x0 = torch.clone(torch.tensor(prior_ensemble))
        self.prior_ensemble = torch.clone(torch.tensor(prior_ensemble))
        self.eps_alpha = 0.5
        self.eps_beta = 0.025
        self.ensemble_size = ensemble_size              ### the number of ensembles.
        self.obs = torch.clone(torch.tensor(obs))                         ### observations
        self.obs_sigma = torch.clone(torch.tensor(sigma))                 ### the std of observations
        self.x_dim = prior_ensemble.shape[1]
        self.y_dim = y_dim                              ### y_dim = nobs
        self.indxob = indxob
        self.indx_indxob_linear = indx_indxob_linear



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
        score_x = torch.zeros((self.ensemble_size, self.y_dim),dtype=self.prior_ensemble.dtype)
        score_x[:, indx_indxob_linear] = -(xt[:, indx_indxob_linear] - self.obs[indx_indxob_linear]) / (self.obs_sigma[indx_indxob_linear] ** 2)
        score_x[:, indx_indxob_nonlinear] = (-(torch.atan(xt[:, indx_indxob_nonlinear]) - self.obs[indx_indxob_nonlinear]) / self.obs_sigma[indx_indxob_nonlinear] ** 2) * (1. / (1. + xt[:, indx_indxob_nonlinear]**2))
        tau = self.g_tau(t)
        return tau * score_x

    def reverse_SDE(self):
        indxunob = np.sort(np.setdiff1d(np.arange(self.x_dim), self.indxob))
        indx_indxob_linear = self.indx_indxob_linear
        indx_indxob_nonlinear = np.sort(np.setdiff1d(np.arange(self.y_dim), indx_indxob_linear))


        dt = 1.0 / self.p_time_step
        torch.manual_seed(42)
        #xt = torch.randn(self.ensemble_size, self.x_dim, device='cuda')
        xt = torch.randn(self.ensemble_size, self.y_dim)
        xt = (xt - xt.mean(dim=0)) / xt.std(dim=0)
        x_ens_full = torch.zeros((self.ensemble_size, self.x_dim),dtype=self.prior_ensemble.dtype)
        x_ens_full[:, indxunob] = self.prior_ensemble[:, indxunob]
        t = 1.0
        for i in range(self.p_time_step):
            # prior score evaluation
            alpha_t = self.cond_alpha(t)
            sigma2_t = self.cond_sigma_sq(t)
            diffuse = self.g(t)
            drift_fun = self.f(t)
            prior_score = - (xt - alpha_t * self.x0[:,self.indxob]) / sigma2_t
            damping_likelihood = self.score_likelihood(xt, t, indx_indxob_linear, indx_indxob_nonlinear)
            posterior_score = prior_score + damping_likelihood
            posterior_score = torch.clip(posterior_score, min=-1000., max=1000.)
            # Update
            xt_temp = xt - dt * (drift_fun * xt - diffuse ** 2 * posterior_score) + np.sqrt(dt) * diffuse * torch.randn_like(xt)
            #xt_temp = xt - dt * (drift_fun * xt - diffuse ** 2 * prior_score - damping_likelihood) + np.sqrt(dt) * diffuse * torch.randn_like(xt)
            xt = xt_temp
            t = t - dt


        x_ens_full[:, self.indxob] = xt
        x_ens_analysis_new = x_ens_full
        return np.array(x_ens_analysis_new)
