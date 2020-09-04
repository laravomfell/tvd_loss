# -*- coding: utf-8 -*-
"""
Created on Wed May 20 10:38:23 2020

@author: Lara Vomfell
Description:
    Data simulation templates
"""

import numpy as np
from sklearn.model_selection import train_test_split
from NPL import NPL
from likelihood_functions import PoissonLikelihood
import pdb

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


class simulations():
    """this class first sets up a generic simulation environment. 
    After filling the data_setup we are then ready to simulate based on some
    simulation parameters"""
    
    def __init__(self, nsim, lik, B = 100, test_size = 0.2):
        self.nsim = nsim
        self.test_size = test_size
        self.lik = lik
        self.B = B
        
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
    
        # right now we're only tracking param deviations and prediction error
        npl_dev = []
        log_dev = []
        npl_pred = []
        log_pred = []
        
        for i in range(0, self.nsim):
            X, Y = self.parse_setup()
            X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, test_size=self.test_size, random_state=42)
            
            # get npl and log score samples
            npl.draw_samples(Y = Y_train, 
                             X = X_train, 
                             B = self.B,
                             seed = 0, 
                             display_opt = False)
            
            # track param deviation            
            npl_dev.append(np.absolute(self.params - npl.sample))
            log_dev.append(np.absolute(self.params - npl.mle))
            # track out-of-sample prediction errors
            x1, x2, x3 = npl.predict(Y_test, X_test)
            npl_pred.append(x3)
            
            x1, x2, x3 = npl.predict_log_loss(Y_test, X_test)
            
            log_pred.append(x3)
            
            
        # create dictionary of results
         # I need to reshape things here
        #pdb.set_trace()
        keys = ['npl_dev', 'log_dev', 'npl_pred', 'log_pred']
        result = dict(zip(keys,
                          [np.concatenate(npl_dev, axis = 0), 
                           np.concatenate(log_dev, axis = 0),
                           np.concatenate(npl_pred).ravel(), 
                           np.concatenate(log_pred).ravel()]))
        self.result = result
        
    def calc_quantiles(self, q = [0.1, 0.5, 0.9]):
        
        def f(input):
            return np.quantile(input, q = q, axis = 0).transpose()
        
        quantile_dict = {k: f(v) for k, v in self.result.items()}
        
        return(quantile_dict)