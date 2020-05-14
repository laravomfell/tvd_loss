#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 10:15:17 2020

@author: jeremiasknoblauch

Description:
    Class hierarchy
"""

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd.scipy.stats import poisson
from autograd import grad


# 

""" 

Class instantiator

"""

class TVDMaster():
    """
    Role: This class contains all objects relevant to inference, including
          model description, data and the inference algorithm
          
    Attributes:
        
    
    """
    
    def __init__(self, X, Y, lik):
        self.X = X
        self.Y = Y
        self.lik = lik        
        self.n = X.shape[0]
        self.p = X.shape[1]
        self.params = None
        
        
    def get_histogram(self):       
        
        # get empirical pmf for X
        self.hist_X = np.unique(self.X, return_counts = True, 
                                return_inverse = True, axis=0)
        self.epmf_X = self.hist_X[2][self.hist_X[1]] / self.n
        
        # get empirical pmf for Y
        self.hist_Y = np.unique(self.Y, return_counts = True, 
                                return_inverse = True)
        self.epmf_Y = self.hist_Y[2][self.hist_Y[1]] / self.n
        
        # get empirical pmf for the joint distribution of (Y,X)
        self.hist_YX = np.unique(np.hstack((np.atleast_2d(self.Y).T, self.X)),
                            return_counts = True, axis=0, return_inverse = True)
        self.epmf_YX = self.hist_YX[2][self.hist_YX[1]] / self.n
        
        # get empirical pmf for the conditional distro Y|X
        self.epmf_Y_cond_X = self.epmf_YX / self.epmf_X
    
    
    def run(self):
        
        # Step 1: construct the empirical pmfs
        self.get_histogram()
        
        # Step 2: Initialize the parameter vector
        self.params = np.zeros(self.p)  # self.params_initializer()
        
        # Step 3: Define/obtain gradient w.r.t. the parameters
        def objective(params):
            # define objective w.r.t. params
            
            # evaluate p_{\theta}(y_j|x_i) for all i,j
            lambdas = np.exp(np.matmul(self.X, params))
            data_lik = poisson.pmf(np.repeat(self.Y,self.n),
                        np.tile(lambdas,self.n))
            
            # evaluate p_{\theta}(y_j \notin {y_1, ... y_n}|x_i) for all i,j
            remainder_lik = 1.0 - np.sum( 
                    data_lik.reshape(self.n,self.n), axis=0)
            
            # compute the estimated TVD
            
            
            
            return objective_value
                    
        gradient = grad(objective)
            
        # Step 4: Perform iterative optimization
        for i in range(0,10000):
            
            step_size = 0.0001
            
            # Step 4.1: Compute gradient
            gradient_value = gradient(self.params)
            
            # Step 4.2: Gradient step
            self.params = self.params + step_size * gradient_value
            
class PoissonSim():
    
    def __init__(self, n, p, params):
        self.params = params
        self.n = n
        self.p = p
        
    def run(self):
        # generate covariates
        self.X = np.random.poisson(2, self.n * self.p).reshape(self.n, self.p)
        
        # generate y
        self.Y = np.random.poisson(np.exp(np.matmul(self.X, self.params)),
                                   self.n)
        
        return (self.X, self.Y)
        
mySims = PoissonSim(100, 2, np.array([0.1, 0.5]))            
X, Y = mySims.run()

inference = TVDMaster(X, Y, None)
inference.run()

print(inference.params)



"""
def objective(params):
            # define objective w.r.t. params
            
            objective_value = np.sum(
                        poisson.logpmf(self.Y, 
                                       np.exp(np.matmul(self.X, params)))
                    )
"""