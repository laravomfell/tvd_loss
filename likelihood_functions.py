#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 09:35:37 2020

@author: jeremiasknoblauch

Description: Likelihood function wrappers for use within NPL class
"""

import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize
import statsmodels.api as sm


class Likelihood():
    """An empty/abstract class that dictates what functions a sub-class of the 
    likelihood type needs to have"""
    
    def __init__(self, d):
        self.d = d
        
    def initialize(self,Y,X):
        """Returns an initialization for the likelihood parameters, typically
        based on the maximum likelihood estimate."""
        return 0

    def evaluate(self, params, Y, X):
        """Letting Y be of shape (n,) and X of shape (n,d), compute the 
        likelihoods of each pair (Y[i], X[i,:]) at parameter value param"""
        return 0


class PoissonLikelihoodSqrt(Likelihood):
    """Use the link function lambda(x) = |abs(x)|^1/2 to make the gradients
    nicer. We still use the (transformed) MLE for initialization"""
    
    def __init__(self, d):
        self.d = d
        self.X_mean = None
        
    def set_X_mean(self, X_mean):
        self.X_mean = X_mean
    
    def initialize(self, Y, X, weights = None):
        
        # check if the mean has been computed before
        if self.X_mean is None:
            self.set_X_mean(np.mean(X,0))
                
        MLE = sm.GLM(Y, X, family = sm.families.Poisson(), 
                     freq_weights = weights).fit().params
        
        # Taking a = parameter for the link function lambda(x) = exp(a*x), 
        # we use the standard MLE procedure to get the best a. Then,
        # we want to solve for param b in link function lambda(x) = |bx|^1/2.
        # PROBLEM: We won't get a one-to-one mapping because x_i varies with i
        # SOLUTION: Solve b for exp(a * E[x]) = |b * E[x]|^{1/2}
        # RATIONALE: E[x] should be representative for x_i
        params = np.power(np.exp(MLE * self.X_mean), 2.0) /np.abs(self.X_mean)
        return params

    
    def evaluate(self, params, Y_unique, X_unique):
        # First, compute lambda(x) = |bx|^1/2
        lambdas = np.power(np.abs(np.matmul(X_unique, 
                                            np.transpose(params))),1.0/2.0)
        # Second, use the standard poisson pmf to evaluate the likelihood
        n_X_unique = X_unique.shape[0]
        n_Y_unique = Y_unique.shape[0]
        
        Y_given_X_model = poisson.pmf(
                np.repeat(Y_unique,n_X_unique).reshape(n_Y_unique, n_X_unique),
                np.tile(lambdas, n_Y_unique).reshape(n_Y_unique, n_X_unique) 
                )
                
        return Y_given_X_model