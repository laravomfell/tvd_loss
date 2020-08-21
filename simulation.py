#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 10:15:17 2020

@author: laravomfell

Description:
    Early simulations
    
    The structure I was thinking of:
        1. Generate noisy data
        2. infer params with tvd and glm
        (Q here: better to model noisy dgp or robust pois?)
        3. prediction (possibly later)
"""

import autograd.numpy as np
import autograd.numpy.random as npr
#from autograd.scipy.misc import logsumexp
from autograd.scipy.stats import poisson
from autograd import grad

import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.discrete.count_model as d_sm

from data_simulators import NBPoissonSim, ZeroInflPoissonSim, EpsilonPoissonSim
from toy import TVDMaster

def model_wrap(X, Y, type):

    # next, estimate the models
    #1. standard poisson
    std_pois = sm.GLM(Y, X, family = sm.families.Poisson()).fit()
    # 2. actual model    
    if type == "NB":
        std_act = sm.GLM(Y, X, 
                         family = sm.families.NegativeBinomial()).fit()
        act_params = std_act.params
    if type == "ZI":
        std_act = d_sm.ZeroInflatedPoisson(Y, X, None, inflation='logit').fit()
        act_params = np.delete(std_act.params, 0)
        
    # 3. TVD
    tvd = TVDMaster(X, Y, None)
    tvd.run()
    
    # return params
    out = np.column_stack((std_pois.params,
                           act_params,
                           tvd.params.x))
    
    # get fitted values
    val = np.column_stack((std_pois.fittedvalues,
                           std_act.fittedvalues,
                           tvd.fittedvalues
                           ))
    
    res = Y[:, None] - val
    
    return out, res



# let's begin with a negbin example
truth = np.array([0.5, -1.2, 1])

#negbin = NBPoissonSim(2000, 3, truth, 3, 1, 0.5)
#X, Y = negbin.run()
#
##print(np.mean(X, axis = 0))
##print(np.mean(Y))
#plt.hist(Y, bins=range(0, max(Y) + 5, 5))
#
#params, res = model_wrap(X, Y, "NB")
#
#print(params)
## evaluation 


# epsilon example

eps = EpsilonPoissonSim(200, 3, truth, 3, 0.05, 50)
X,Y = eps.run()

#print(np.mean(X, axis = 0))
#print(np.mean(Y))

#plt.hist(Y, bins=range(0, max(Y) + 5, 5))

params, res = model_wrap(X, Y, "NB")

plt.plot(Y, res[:,2], "b.")

print(params)





# so with this example, tvd is doing pretty badly

## ZERO INFLATION
#zeroinfl = ZeroInflPoissonSim(2000, 3, truth, 3, 0.3, 0.8)
#X, Y = zeroinfl.run()
#
## standard glm where zeroinfl does not depend on X
#std_zero = d_sm.ZeroInflatedPoisson(Y, X, None, inflation='logit').fit()
#
## robust tvd loss
#zero_tvd = TVDMaster(X, Y, None)
#zero_tvd.run()
#
## evaluation
#out = np.column_stack((truth, np.delete(std_zero.params, 0), zero_tvd.params.x))
#print(out)