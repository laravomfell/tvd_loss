#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 09:34:38 2020

@author: jeremiasknoblauch

Description: Test the TVD-NPL algorithm on some artificual data
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.discrete.count_model as d_sm
from sklearn.model_selection import train_test_split

from scipy.stats import poisson

from NPL import NPL
from likelihood_functions import PoissonLikelihoodSqrt, PoissonLikelihood
from likelihood_functions import SoftMaxNN
from data_simulators import NBPoissonSim, ZeroInflPoissonSim, EpsilonPoissonSim


n = 2000
truth = np.array([0.5, -1.2, 1])
X = np.array([np.ones(n),
              np.random.rand(n),
              np.random.normal(loc=2.0, scale=1.0, size = n)]).reshape(n, 3)
Y = poisson.rvs(np.exp(np.matmul(X, truth)))

Y[0:100] += 10
# X, Y = EpsilonPoissonSim(1000, 3, truth, 3, 0.1, 10).run()
#X, Y = ZeroInflPoissonSim(2000, 3, truth, 3, 0.2, 1).run()
#X, Y = NBPoissonSim(2000, 3, truth, 3, 0.1, 5).run()


# test if the NN works (if both covariates are positive, Y=1. Y=0 otherwise)
X = np.array([[1.0, 2.0], 
              [2.0, 3.1], 
              [5.0, 6.3],
              [5.3, 2.9],
              [6.7, 9.0],
              [-2.0, -1.0],
              [-2.0, -3.1], 
              [-5.0, -6.3],
              [-5.3, -2.9],
              [-6.7, -9.0]])

Y = np.array([0,
              0,
              0,
              0,
              0,
              1,
              1,
              1,
              1,
              1], dtype=int)    

nn_lklh = SoftMaxNN(2, 2, 2, reset_initializer=True, 
            batch_size = 10,
            epochs_TVD = 1000, epochs_vanilla = 10) #network with 10 hidden layers

n=10 
d=2
npl_sampler = NPL(nn_lklh, optimizer = "BFGS")
B=100
npl_sampler.draw_samples(Y,X,B)


npl_sampler.predict(Y,X)

# nn_lklh.initialize(Y,X)
# weights = np.ones(10) * 1.0/ 10
# nn_lklh.minimize_TVD(Y,X,weights)



#1. standard poisson
#std_pois = sm.GLM(Y, X, family = sm.families.Poisson()).fit()

if False:
    # TVD 2 -- new inference method
    n, d = X.shape
    L = PoissonLikelihood(d)
    
    npl_sampler = NPL(L, optimizer = "BFGS")
    B=100
    npl_sampler.draw_samples(Y,X,B)