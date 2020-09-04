# -*- coding: utf-8 -*-
"""
Created on Wed May 20 10:38:23 2020

@author: Lara Vomfell
Description:
    Data simulation templates
"""

import numpy as np

class PoissonSim():
    """Simulates covariates and Poisson outcomes

    Args:
      n: Number of observations to simulate
      p: number of dimensions of X to simulate
      params: coefficients to determine Y
      continuous_x: switch whether to simulate continuous X or simulate
                    covariates as uniform from categories

    Returns:
      X, Y: covariates X and Poisson outcome Y 

    """
    
    def __init__(self, n, p, params, continuous_x = True):
        assert params.shape[0] == p, "params shape needs to be p"
        assert type(continuous_x) == bool, "continuous_x must be boolean"
        self.n = n
        self.p = p
        self.continuous_x = continuous_x
        self.params = params
        
        if continuous_x and p == 1: 
            print("""continuous_x is True and p = 1 and 
                  will be treated as continuous_x is False""")
        
    def generate_X(self):
        if self.p == 1:
            self.X = np.ones(self.n)
        
        if self.p > 1 and self.continuous_x:
            self.X = np.array([
                np.ones(self.n),
                np.random.normal(loc = 2.0, 
                                 scale = 1.0, 
                                 size = self.n * (self.p-1))]).transpose()
        
        if self.p > 1 and self.continuous_x is False:
            self.X = np.random.choice(4, self.n * self.p).reshape(self.n, self.p)
        
    def run(self):
        # generate covariates
        self.generate_X()
        # generate y
        self.Y = np.random.poisson(np.exp(np.matmul(self.X, self.params)),
                                   self.n)
        
        return (self.X, self.Y)


class NegBinContam(PoissonSim):
    """Inherits Poisson outcomes and contaminates them 
     with negative binomial over-dispersion.
     Parametrization goes from mu, (inverse) overdispersion
     to Python negative_binomial param in term of (n, p) where 
     n = number of successes and p = probability of success.
     Transformation is n = contam_par and p = contam_par/(contam_par + mu).
     In other words, small inverse disp -> large dispersion.
     
     Args:
         share: share of contaminated data
         contam_par: 1/overdispersion in the contamined data

    Returns:
      X, Y: covariates X and contaminated outcome Y """
    
    def __init__(self, share, contam_par, **kw):
        assert 0 <= share <= 1, "share of data to contaminate needs to be [0,1]"
        assert contam_par > 0, "inv_disp can't be negative"
        self.share = share
        self.contam_par = contam_par
        super(NegBinContam, self).__init__(**kw)
        
    def contaminate(self):
        # get uncontaminated data
        self.X, self.Y = self.run()
        
        # estimate mu
        self.mu = np.mean(self.Y)
        
        # number of obversations to contaminate
        n_contam = int(np.floor(self.n * self.share))
        
        # create negative binomial params
        prob = self.contam_par / (self.contam_par + self.mu)
        
        # generate n_contam overdispersed counts
        Y_contam = np.random.negative_binomial(self.contam_par, prob, n_contam)
        
        # return result
        return(self.X,
               np.concatenate((Y_contam, self.Y[n_contam:self.n]), axis = 0))


class ZeroInflContam(PoissonSim):
    """adds zeroinflated contamination to Poisson outcome

    Args:
      share: share of contaminated data
      contam_par: probability of 0's in contaminated part

    Returns:
      X, Y: covariates X and contaminated Poisson outcome Y 
    """
    
    
    def __init__(self, share, contam_par, **kw):
        assert 0 <= share <= 1, "share of data to contaminate needs to be [0,1]"
        assert 0 <= contam_par <= 1, "prob of 0 needs to be [0,1]"
        self.share = share
        self.contam_par = contam_par
        super(ZeroInflContam, self).__init__(**kw)
        
    def contaminate(self):
        # get uncontaminated data
        self.X, self.Y = self.run()
        
        # number of observations to contaminate
        n_contam = int(np.floor(self.n * self.share))
        # multiply Y at 0:n_contam with vector of (0,1) to inflate
        # draw 0's with probability prob0
        zero_part = np.concatenate((np.random.random(n_contam) > self.contam_par,
                                    np.ones(self.n - n_contam)))
        return(self.X, self.Y * zero_part)
        
        
        
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


