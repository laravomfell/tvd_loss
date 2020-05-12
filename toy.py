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
        self.hist_X = np.unique(self.X, return_counts = True, axis=0)
        self.hist_Y = np.unique(self.Y, return_counts = True)
        self.hist_YX = np.unique(np.hstack((np.atleast_2d(self.Y).T, self.X)),
                            return_counts = True, axis=0)
    
    def run(self):
        
        # Step 1: construct the empirical pmfs
        self.get_histogram()
        
        # Step 2: Initialize the parameter vector
        self.params = np.zeros(self.p)  # self.params_initializer()
        
        # Step 3: Define/obtain gradient w.r.t. the parameters
        def objective(params):
            # define objective w.r.t. params
            
            objective_value = np.sum(
                        poisson.logpmf(self.Y, 
                                       np.exp(np.matmul(self.X, params)))
                    )
                        
            observed_counts = self.
            
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





