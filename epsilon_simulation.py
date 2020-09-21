# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 10:41:39 2020

@author: Lara Vomfell

Description: Define class to simplify the simulation of epsilon contamination
"""

import numpy as np
import os
from NPL import NPL

class simulations():
    """this class first sets up a generic simulation environment. 
    After filling the data_setup we are then ready to simulate based on some
    simulation parameters"""
    
    def __init__(self, nsim, lik, B, test_size, save_path):
        self.nsim = nsim
        self.test_size = test_size
        self.lik = lik
        self.B = B
        self.save_path = save_path
        
    def data_setup(self, contam_type, n, p, params, continuous_x, 
                   share, contam_par):
        
        """take data setup information and assign to self"""
        self.contam_type = contam_type
        self.n = n
        self.p = p
        self.params = params
        self.continuous_x = continuous_x
        self.share = share        
        self.contam_par = contam_par
        
    def parse_setup(self):
        X,Y = self.contam_type(share = self.share, 
                               contam_par = self.contam_par,
                               n = self.n, p = self.p, params = self.params,
                               continuous_x = self.continuous_x).contaminate()
        return(X, Y)
        
        
    def simulate(self):
        # set up npl optimizer
        L = self.lik()
        npl = NPL(L, optimizer = "BFGS")
    
        # right now we're tracking param deviations and absoluteprediction error
        # and predictive log likelihoods
        npl_dev = []
        kld_dev = []
        npl_ae = []
        kld_ae = []
        npl_loglik = []
        kld_loglik = []
        
        
        # set train dataset size
        n_train = int(np.floor(self.n * self.test_size))
        
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
            
            # track param deviation            
            npl_dev.append(np.abs(self.params - npl.sample))
            kld_dev.append(np.abs(self.params - npl.mle))
            # track out-of-sample prediction
            x1, x2, x3 = npl.predict(Y_test, X_test)
            npl_loglik.append(x1)
            npl_ae.append(x3)
            
            x1, x2, x3 = npl.predict_log_loss(Y_test, X_test)
            kld_loglik.append(x1)
            kld_ae.append(x3)
        
        # save the results to path
        if self.contam_par < 10:
            num = "0" + str(self.contam_par)
        else :
            num = str(self.contam_par)
        
        if self.save_path.endswith("/"):
            save_path = self.save_path
        else:
            save_path = self.save_path + "/"
        # create a folder for the simulations
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # save npl and kld param deviations
        np.savetxt(save_path + num + "_param_tvd.txt", 
                   np.concatenate(npl_dev, axis = 0))
        np.savetxt(save_path + num + "param_kld.txt",
                   np.concatenate(kld_dev, axis = 0))
        
        # save predictive stuff
                    
        # predictive log likelihood
        np.savetxt(save_path + num + "_loglik_tvd.txt",
                   np.concatenate(npl_loglik).ravel())
        np.savetxt(save_path + num + "_loglik_kld.txt",
                   np.concatenate(kld_loglik).ravel())
        
        # absolute error
        np.savetxt(save_path + num + "_ae_tvd.txt",
                   np.concatenate(npl_ae).ravel())
        np.savetxt(save_path + num + "_ae_kld.txt",
                   np.concatenate(kld_ae).ravel())        
        
        
    # def calc_quantiles(self, q = [0.1, 0.5, 0.9]):
        
    #     def f(input):
    #         return np.quantile(input, q = q, axis = 0).transpose()
        
    #     quantile_dict = {k: f(v) for k, v in self.result.items()}
        
    #     return(quantile_dict)