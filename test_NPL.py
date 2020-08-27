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
import scipy.stats as stats

import statsmodels.api as sm
import statsmodels.discrete.count_model as d_sm
from sklearn.model_selection import train_test_split

from scipy.stats import poisson, norm

from NPL import NPL
from likelihood_functions import ProbitLikelihood, PoissonLikelihood
from likelihood_functions import SoftMaxNN
from data_simulators import NBPoissonSim, ZeroInflPoissonSim, EpsilonPoissonSim

# L1 = PoissonLikelihood(X.shape[1])
# npl_1 = NPL(L1, optimizer = "BFGS")
# B = 100
# npl_1.draw_samples(Y, X, B)

# # get MLE
# mle = sm.GLM(Y, X, family = sm.families.Poisson()).fit().params()
# # npl
# npl = npl_1.sample.mean(axis = 0)

# plt.hist(Y)

if False:
    n = 2000
    truth = np.array([0.5, -1.2, 1])
    X = np.array([np.ones(n),
                  np.random.rand(n),
                  np.random.normal(loc=2.0, scale=0.25, size = n)]).reshape(n, 3)
    Y = np.floor(np.exp(np.matmul(X, truth)))
    Y[0:100] += 10
    # # X, Y = EpsilonPoissonSim(1000, 3, truth, 3, 0.1, 10).run()
    # #X, Y = ZeroInflPoissonSim(2000, 3, truth, 3, 0.2, 1).run()
    # #X, Y = NBPoissonSim(2000, 3, truth, 3, 0.1, 5).run()
    
    
    #1. standard poisson
    #std_pois = sm.GLM(Y, X, family = sm.families.Poisson()).fit()
    
    # TVD 2 -- new inference method
    n, d = X.shape
    L = PoissonLikelihood("name")
    
    npl_sampler = NPL(L, optimizer = "BFGS")
    B=100
    npl_sampler.draw_samples(Y,X,B)
    predictive_likelihoods, SE, AE = npl_sampler.predict(Y[100:],X[100:,:])
    
    print("avg pred lklh", np.mean(predictive_likelihoods))
    print("MSE", np.mean(SE))
    print("MAE", np.mean(AE))
    
    predictive_likelihoods, SE, AE = npl_sampler.predict_log_loss(Y[100:],X[100:,:])
    
    print("avg pred lklh MLE", np.mean(predictive_likelihoods))
    print("MSE MLE", np.mean(SE))
    print("MAE MLE", np.mean(AE))


n=300
truth = np.array([0.01, -0.25, 0.25])
X = np.array([np.ones(n),
              np.random.normal(loc = 1.0, scale = 0.7,size = n),
              np.random.normal(loc= -1.0, scale=0.25, size = n)]).reshape(n, 3)
X = X / np.var(X)
eps = np.random.normal(loc=0.0, scale = 1.0, size = n)
probs = norm.cdf(np.matmul(X, truth) + eps)

Y = np.zeros(n, dtype = int)
Y[np.where(probs > 0.5)] = 1


L = ProbitLikelihood()
npl_sampler = NPL(L, optimizer = "BFGS")
B=100
npl_sampler.draw_samples(Y,X,B)


if False:
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
    
    X_test = np.array([[10.0, 9.0],
                       [0.5, 1.3], 
                       [-5.4, -20.8], 
                       [-0.2, -2]])
    Y_test = np.array([0,0,1,1])
    
    X = X + np.random.normal(0, 100, (10,2))   
    
    nn_lklh = SoftMaxNN(2, 100, 2, reset_initializer=True, 
                batch_size = 64,
                epochs_TVD = 1000, epochs_vanilla = 10000) #network with 10 hidden layers
    
    n=10 
    d=2
    npl_sampler = NPL(nn_lklh, optimizer = "BFGS")
    B=100
    npl_sampler.draw_samples(Y,X,B)
    
    
    predictions, accuracy, cross_entropy = npl_sampler.predict(Y_test,X_test)
    predictions_init, accuracy_init, cross_entropy_init = npl_sampler.lklh.predict_initializer(Y_test,X_test)

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
