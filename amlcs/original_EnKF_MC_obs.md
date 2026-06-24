```python
##########################################################################################
##########################################################################################
##########################################################################################
# Observation space version of:
# Nino-Ruiz, E. D., Sandu, A., & Deng, X. (2018). An ensemble Kalman filter implementation based on modified Cholesky decomposition for inverse covariance matrix estimation. SIAM Journal on Scientific Computing, 40(2), A867-A886.
# To be published
##########################################################################################
##########################################################################################
##########################################################################################
class EnKF_MC_obs(ensemble_DA):
            
    
    def prepare_background(self):
        mask_cor = self.nm.mask_cor;
        var_names = self.nm.var_names;
        Nens = self.Nens;
        self.XB_map = [];
        gr = self.nm.gs;
        for msk_cor, pre_info in zip(mask_cor, gr.lpr):
            XB_block = self.get_ensemble_block(msk_cor);
            xb_block = XB_block.mean(axis=1).reshape(-1,1);
            DX_block = XB_block-xb_block;
            Binv_sqrt_block = compute_modified_Cholesky_decomposition(DX_block, pre_info, thr=0.15);

            self.XB_map.append({'XB_b':XB_block, 'xb_b':xb_block, 'Binv_s_b':Binv_sqrt_block});
     
    def prepare_analysis(self, ob, k, args = None):
        self.Ys = [];
        Nens = self.Nens;
        mask_cor = self.nm.mask_cor;
        N_blocks = len(mask_cor);
        for block in range(0, N_blocks):
            Ys_block = ob.get_perturbed_observations(block, k, Nens);  
            self.Ys.append(Ys_block); 
    
    def perform_assimilation(self, ob):
        self.XA_map = [];
        for XB_info, Ys_block, R_info, H_block in zip(self.XB_map, self.Ys, ob.obs_R_sparse, ob.obs_H_sparse):
            XB_block = XB_info['XB_b'];
            R_block  = R_info['R'];
            Binv_sqrt_block = XB_info['Binv_s_b'];
            XA_block = self.perform_assimilation_block(XB_block, Binv_sqrt_block, H_block, R_block, Ys_block);
            XA_block = self.covariance_inflation(XA_block);
            self.XA_map.append(XA_block);
        self.map_vector_states(); #Update ensemble folders
    
    def perform_assimilation_block(self, XB, Binv_sqrt, H, R, Ys):
      
          Ds = Ys - H @ XB;
           
          H_spar = H.toarray();
          
          P = spa.linalg.spsolve_triangular(Binv_sqrt, H_spar.T, lower=False, check_finite=False);
          
          Inno = R + P.T @ P;

          Q_temp = P @ spa.linalg.spsolve(Inno, Ds);
          
          #Q_temp = spa.linalg.spsolve_triangular(Binv_sqrt,Q_temp,lower=False);
          
          DXa = spa.linalg.spsolve_triangular(Binv_sqrt.T, Q_temp, lower=True, check_finite=False);

          XA = XB + DXa;  
          
          return XA;
      
```