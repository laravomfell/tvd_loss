# -*- coding: utf-8 -*-
"""
Created on Wed May 20 10:38:23 2020

@author: Lara Vomfell

This file provides simple classes that allow one to specify both the 'pure'
(uncontaminated) DGP as well as the type of contamination. 

Defining things this way is useful because it plays nice with the simulation 
template we use.
"""

import numpy as np
from scipy.special import logit, expit

class PoissonSim():
    """Poisson outcomes without covariates

    Args:
      n: Number of observations to simulate
      params: coefficients to determine Y

    Returns:
      X, Y: covariates X and Poisson outcome Y 

    """
    
    def __init__(self, n, params, p, continuous_x):
        self.n = n
        self.params = params
            
    def run(self):
        # generate covariates
        self.X = np.ones(self.n)
        # generate y
        self.Y = np.random.poisson(self.X * self.params)
        
        return (self.X[:, np.newaxis], self.Y)
     
        
class EpsilonContam(PoissonSim):
    """adds epsilon contamination to poisson outcome
    
    Args:
        share: share of contaminated data
        contam_par: constant epsilon to add to data
        
    Returns:
        X, Y: covariates X and contaminated Poisson outcome Y
    """
    
    def __init__(self, share, contam_par, **kw):
        assert 0 <= share <= 1, "share of data to contaminate needs to be [0,1]"
        self.share = share
        self.contam_par = contam_par
        super(EpsilonContam, self).__init__(**kw)
        
    def contaminate(self):
        # get uncontaminated data
        self.X, self.Y = self.run()
        
        # number of observations to contaminate
        n_contam = int(np.floor(self.n * self.share))
        # add constant epsilon to Y
        self.Y[0:n_contam] += self.contam_par
        return(self.X, self.Y)

class BinomSim():
    def __init__(self, n, p, params, n_trials, continuous_x = True):
        assert p == params.shape[0], "params need to be of shape p"
        assert type(continuous_x) == bool, "need boolean flag for type of X"
        self.n = n
        self.p = p
        self.params = params
        self.n_trials = n_trials
        self.continuous_x = continuous_x
    
        if continuous_x and p == 1: 
            print("""Returning an intercept-only model despite continuous_x = True""")
        
    def generate_X(self):
        if self.p == 1:
            self.X = np.ones(self.n)
        
        if self.p > 1 and self.continuous_x:
            self.X = np.array([
                np.ones(self.n),
                np.random.normal(loc = 0.0, 
                                 scale = 1.0, 
                                 size = self.n * (self.p-1))]).transpose()
        
        if self.p > 1 and self.continuous_x is False:
            self.X = np.random.choice(4, self.n * self.p).reshape(self.n, self.p)
        
    def run(self):
        # generate covariates
        self.generate_X()
        # generate y
        self.Y = np.random.binomial(self.n_trials,
                                    expit(np.matmul(self.X, self.params)),
                                    self.n)
        
        return (self.X, self.Y)
    
class ZeroInflBinom(BinomSim):
    def __init__(self, share, contam_par, **kw):
        assert 0 <= share <= 1, "share of data to contaminate needs to be [0,1]"
        self.share = share
        self.contam_par = contam_par
        super(ZeroInflBinom, self).__init__(**kw)
    
    def contaminate(self):
        self.X, self.Y = self.run()
        
        # number of observations to contaminate
        n_contam = int(np.floor(self.n * self.share))
        # draw zero-part
        zero_part = np.concatenate((np.random.binomial(1, 1 - self.contam_par, n_contam),
                                    np.ones(self.n - n_contam, dtype = int)))
        return(self.X, (self.Y * zero_part))
