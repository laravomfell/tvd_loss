#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 1 2020

@author: laravomfell

Description:
    Early data tests on some built in R data    

"""

import autograd.numpy as np
import autograd.numpy.random as npr
#from autograd.scipy.misc import logsumexp
from autograd.scipy.stats import poisson
from autograd import grad

import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.discrete.count_model as d_sm

from toy import TVDMaster

import pandas as pd

def model_wrap(X, Y, type, model_matrix = True):
    if model_matrix:
        X = pd.get_dummies(X, drop_first = True)
        X = sm.add_constant(X)
       
    Y = Y.to_numpy()    
    X = X.to_numpy()
    
    # next, estimate the models
    #1. standard poisson
    std_pois = sm.GLM(Y, X, family = sm.families.Poisson()).fit()
    # 2. actual model    
    if type == "NB":
        std_act = sm.GLM(Y, X, family = sm.families.NegativeBinomial()).fit()
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
    
    return out, val, res

# NEGATIVE BINOMIAL

quine = pd.read_csv('data/quine.csv',delimiter=',')
# separate Y and X, create model matrix
Y = quine['Days']
X = quine.drop('Days', axis = 1)

params, fitted, res = model_wrap(X, Y, type = "NB")

print("Quine:")
print(params)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
fig.suptitle('Quine fit')
ax1.hist(Y)
ax2.plot(Y, res[:,0], 'bo')
ax3.plot(Y, res[:,1], 'bo')
ax4.plot(Y, res[:,2], 'bo')
#plt.close()

# CRABS NEXT

crabs = pd.read_csv('data/crabs.csv', delimiter = ',')
Y = crabs['satellites']
X = crabs.drop(['satellites', 'id', 'weight', 'carapace_width'], axis = 1)

params, fitted, res = model_wrap(X, Y, type = "NB")

print("Crabs:")
print(params)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
fig.suptitle('Crabs fit')
ax1.hist(Y)
ax2.plot(Y, res[:,0], 'bo')
ax3.plot(Y, res[:,1], 'bo')
ax4.plot(Y, res[:,2], 'bo')

# next up, zero inflation

biochem = pd.read_csv('data/bioChemists.csv', delimiter = ',')
Y = biochem['art']
X = biochem.drop('art', axis = 1)

params, fitted, res = model_wrap(X, Y, type = "ZI")
print("bioChem")
print(params)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
fig.suptitle('bioChem fit')
ax1.hist(Y)
ax2.plot(Y, res[:,0], 'bo')
ax3.plot(Y, res[:,1], 'bo')
ax4.plot(Y, res[:,2], 'bo')

# fish last
fish = pd.read_csv('data/fish.csv', delimiter = ',')
Y = fish['count']
X = fish.drop(['count', 'nofish', 'xb', 'zg'], axis = 1)
params, fitted, res = model_wrap(X, Y, type = "NB")

print("Fish:")
print(params)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
fig.suptitle('Fish fit')
ax1.hist(Y)
ax2.plot(Y, res[:,0], 'bo')
ax3.plot(Y, res[:,1], 'bo')
ax4.plot(Y, res[:,2], 'bo')