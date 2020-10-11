# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 10:41:39 2020

@author: Lara Vomfell

Description: Define class to simplify the simulation of contamination
"""

import numpy as np
import pandas as pd
import os
import pystan
from NPL import NPL
import platform

class simulations():
    """this class first sets up a generic simulation environment. 
    After filling the data_setup we are then ready to simulate based on some
    simulation parameters
    
    Args:
      nsim: Number of dataset simulations
      test_size: share of data to split into test set
      lik: Likelihood from likelihood_functions to use
      B: number of 'boostrap' samples in NPL
      save_path: where to save the simulation results
      stan_model: name and path to stan model
      var_par: which parameter is being varied
    """
    
    def __init__(self, nsim, lik, B, test_size, save_path, stan_model, var_par):
        assert var_par == 'share' or var_par == 'contam_par', 'var_par needs to match data_simulator arguments'
        self.nsim = nsim
        self.test_size = test_size
        self.lik = lik
        self.B = B
        self.save_path = save_path
        self.stan_model = stan_model
        self.var_par = var_par
        
    def data_setup(self, contam_type, n, p, params, continuous_x, 
                   share, contam_par, extra = 0):
        
        """take data setup information and assign to self
        
        Args:
            all the arguments needed for contamination class in data_simulators:
        n, p, params, continuous_x for PoissonSim() and
        share, contam_par for Contam class
        
        Returns:
            X,Y: X generated from PoissonSim() and contaminated Y
        """

        self.contam_type = contam_type
        self.n = n
        self.p = p
        self.params = params
        self.continuous_x = continuous_x
        self.share = share        
        self.contam_par = contam_par
        self.extra = extra
        
       
    def parse_setup(self):
        if self.extra == 0:
            X,Y = self.contam_type(share = self.share, 
                   contam_par = self.contam_par,
                   n = self.n, p = self.p, params = self.params,
                   continuous_x = self.continuous_x).contaminate()
        else:
            X,Y = self.contam_type(share = self.share, 
                               contam_par = self.contam_par,
                               n = self.n, p = self.p, params = self.params,
                               continuous_x = self.continuous_x,
                               n_trials = self.extra).contaminate()
        return(X, Y)
    
        
    def simulate(self):
        # set up npl optimizer
        L = self.lik()
        npl = NPL(L, optimizer = "BFGS")
        # set up stan mode
        model = pystan.StanModel(self.stan_model)
    
        # right now we're tracking param deviations and absolute prediction error
        # and predictive log likelihoods
        npl_dev = []
        kld_dev = []
        stan_dev = []
        npl_ae = []
        kld_ae = []
        stan_ae = []
        npl_loglik = []
        kld_loglik = []
        stan_loglik = []        
        
        # set train dataset size
        n_test = int(np.floor(self.n * self.test_size))
        n_train = self.n - n_test
        
        # check platform and run multicore if not windows
        if platform.system() == 'Windows':
            n_jobs = 1
        else: 
            n_jobs = -1
        
        for i in range(0, self.nsim):
            # set seed
            np.random.seed(i)
            # generate data
            X, Y = self.parse_setup()
            # split into train/test
            train_idx = np.random.choice(self.n, size = n_train, replace = False)
            test_idx = np.setdiff1d(np.linspace(0, self.n-1, self.n, dtype = int), 
                                    train_idx)
            X_train = X[train_idx,:]
            Y_train = Y[train_idx]
            X_test = X[test_idx,:]
            Y_test = Y[test_idx]
            
            # get npl and log score samples
            npl.draw_samples(Y = Y_train, 
                             X = X_train, 
                             B = self.B,
                             display_opt = False)
            
            
            # run stan model
            data = {'N': n_train,
                    'P': self.p,
                    'Y': Y_train,
                    'X': X_train,
                    'N_test': n_test,
                    'X_test': X_test,
                    'Y_test': Y_test
                    }
            
            fit = model.sampling(data = data,
                                  n_jobs = n_jobs, 
                                  iter = 2000, 
                                  warmup = 1000,
                                  chains = 4).extract(permuted=True)
            
            # track param deviation            
            npl_dev.append(np.abs(self.params - npl.sample))
            kld_dev.append(np.abs(self.params - npl.mle))
            stan_dev.append(np.abs(self.params - fit['beta']))
            # track out-of-sample prediction
            x1, x2, x3 = npl.predict(Y_test, X_test)
            npl_loglik.append(x1)
            npl_ae.append(x3)
            
            x1, x2, x3 = npl.predict_log_loss(Y_test, X_test)
            kld_loglik.append(x1)
            kld_ae.append(x3)
            
            stan_loglik.append(fit['loglik'])
            stan_ae.append(np.abs(Y_test - fit['Y_pred']))
        
            if i == 50:
                print("finished nsim = 50")
        # save the results to path
        if self.var_par == 'share':
            if (self.share * 100) < 10:
                num = '0' + str(int(self.share * 100))
            else:
                num = str(int(self.share * 100))
        else:
            if self.contam_par < 10:
                num = "0" + str(self.contam_par)
            else:
                num = str(self.contam_par)
        
        if self.save_path.endswith("/"):
            save_path = self.save_path
        else:
            save_path = self.save_path + "/"
        # create a folder for the simulations
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # save param deviations
        npl_dev = pd.DataFrame(np.concatenate(npl_dev, axis = 0))
        npl_dev.to_csv(save_path + num + "_param_tvd.csv",
                       index = False, header = False)
        kld_dev = pd.DataFrame(np.concatenate(kld_dev, axis = 0))
        kld_dev.to_csv(save_path + num + "_param_kld.csv",
                       index = False, header = False)
        stan_dev = pd.DataFrame(np.concatenate(stan_dev, axis = 0))
        stan_dev.to_csv(save_path + num + "_param_stan.csv",
                        index = False, header = False)
        
        # save predictive stuff
                    
        # predictive log likelihood
        npl_loglik = pd.DataFrame(np.concatenate(npl_loglik).ravel())
        npl_loglik.to_csv(save_path + num + "_loglik_tvd.csv",
                       index = False, header = False)
        
        kld_loglik = pd.DataFrame(np.concatenate(kld_loglik).ravel())
        kld_loglik.to_csv(save_path + num + "_loglik_kld.csv",
                          index = False, header = False)
        
        stan_loglik = pd.DataFrame(np.concatenate(stan_loglik).ravel())
        stan_loglik.to_csv(save_path + num + "_loglik_stan.csv",
                           index = False, header= False)
        
        # # absolute error
        npl_ae = pd.DataFrame(np.concatenate(npl_ae).ravel())
        npl_ae.to_csv(save_path + num + "_ae_tvd.csv",
                      index = False, header= False)
        kld_ae = pd.DataFrame(np.concatenate(kld_ae).ravel())
        kld_ae.to_csv(save_path + num + "_ae_kld.csv",
                      index = False, header= False)
        stan_ae = pd.DataFrame(np.concatenate(stan_ae).ravel())
        stan_ae.to_csv(save_path + num + "_ae_stan.csv",
                       index = False, header = False)        
        
    # def calc_quantiles(self, q = [0.1, 0.5, 0.9]):
        
    #     def f(input):
    #         return np.quantile(input, q = q, axis = 0).transpose()
        
    #     quantile_dict = {k: f(v) for k, v in self.result.items()}
        
    #     return(quantile_dict)