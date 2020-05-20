# -*- coding: utf-8 -*-
"""
Created on Wed May 20 10:38:23 2020

@author: Lara Vomfell
Description:
    Data simulation templates
"""

import numpy as np

class PoissonSim():
    """simulates covariates and Poisson outcomes

    Args:
      n: Number of observations to simulate
      p: number of dimensions of X to simulate
      params: coefficients to determine Y

    Returns:
      X, Y: covariates X and Poisson outcome Y 

    """
    
    def __init__(self, n, p, params):
        assert params.shape == p, "params shape needs to be p"
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
        


# dispersion such that small inv_disp = large dispersion
class NBPoissonSim():
    """simulates covariates and Poisson outcomes with negative binomial contamination

    Args:
      n: Number of observations to simulate
      p: number of dimensions of X to simulate
      params: coefficients to determine Y
      contam: share of contaminated data
      inv_disp: 1/overdispersion in the contamined data

    Returns:
      X, Y: covariates X and Poisson outcome Y 

    """
    
    
    def __init__(self, n, p, params, contam, inv_disp):
        assert 0 <= contam <= 1, "contam needs to be [0,1]"
        assert params.shape == p, "params shape needs to be p"
        assert inv_disp > 0, "inv_disp can't be negative"
        self.params = params
        self.n = n
        self.p = p
        self.contam = contam
        self.inv_disp = inv_disp
        
    def run(self):
        # generate poisson outcome + covariates
        poisson = PoissonSim(self.n, self.p, self.params)
        self.X, self.Y = poisson.run()
        # calculate mu
        self.mu = np.exp(np.matmul(self.X, self.params))
        
        # get contamination indices
        n_contam = int(np.floor(self.n * self.contam))
        # create neg bin params
        # to go from mu, disp to n,p 
        # we do n = disp, p = disp/(disp + mu)
        prob = self.inv_disp / (self.inv_disp + self.mu[0:n_contam])
        Y_contam = np.random.negative_binomial(self.inv_disp, prob, n_contam)
        
        return(self.X, np.concatenate((Y_contam, self.Y[(n_contam + 1):self.n]),
                                      axis = 0))
        



class ZeroInflPoissonSim():
    """simulates covariates and Poisson outcomes with zeroinflated contamination

    Args:
      n: Number of observations to simulate
      p: number of dimensions of X to simulate
      params: coefficients to determine Y
      contam: share of contaminated data
      prob0: probability of 0's in contaminated part

    Returns:
      X, Y: covariates X and Poisson outcome Y 

    """
    
    
    def __init__(self, n, p, params, contam, prob0):
        assert 0 <= contam <= 1, "contam needs to be [0,1]"
        assert 0 <= prob0 <= 1, "prob0 needs to be [0,1]"
        assert params.shape == p, "params shape needs to be p"
        self.params = params
        self.n = n
        self.p = p
        self.contam = contam
        self.prob0 = prob0
        
    def run(self):
        # generate poisson outcomes + covariates
        # generate poisson outcome + covariates
        poisson = PoissonSim(self.n, self.p, self.params)
        self.X, self.Y = poisson.run()
        
        # get contamination indices
        n_contam = int(np.floor(self.n * self.contam))
        # from 1:n_contam multiply Y with 0,1 from 
        zero_part = np.concatenate((np.random.random(n_contam) > self.prob0,
                                    np.ones(self.n - n_contam)))
        return(self.X, self.Y * zero_part)
        
# start simulating
poisson = PoissonSim(100, 2, np.array([0.1, 0.5]))            
poisson.run() 
        
nb = NBPoissonSim(100, 2, np.array([0.1, 0.5]), 0.1, disp = 1/10)
nb.run()


zeroinfl = ZeroInflPoissonSim(100, 2, np.array([0.1, 0.5]), 0.1, 0.75)
zeroinfl.run()