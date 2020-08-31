#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 09:34:38 2020

@author: Lara Vomfell

Description: Simulations for the experiments.
Rough structure:
    - Figure 1: show univariate Poisson process with epsilon-contamination,
                overplot MLE Poisson density and TVD
    - Actual experiments:
        Epsilon-contamination: Param inference and out-of-sample prediction
        Zero-inflation: Param inference and out-of-sample prediction
"""

import numpy as np

import matplotlib.pyplot as plt
import scipy.stats as stats

import statsmodels.api as sm
import statsmodels.discrete.count_model as d_sm

from scipy.stats import poisson
from sklearn.model_selection import train_test_split

from NPL import NPL
from likelihood_functions import PoissonLikelihood
import data_simulators

# Figure 1
# n = 250
# Y = np.random.poisson(2, n)
# Y[0:24] += 15
# X = np.ones([n, 1])
# L1 = PoissonLikelihood(X.shape[1])
# npl_1 = NPL(L1, optimizer = "BFGS")
# B = 100
# npl_1.draw_samples(Y, X, B)

# # get MLE
# mle = sm.GLM(Y, X, family = sm.families.Poisson()).fit().params
# # npl
# npl = npl_1.sample.mean(axis = 0)

# plt.hist(Y, range(Y.max() + 1), density = 1)
# x = range(Y.max() + 1)
# ppmf = plt.plot(x, stats.poisson.pmf(x, np.exp(mle)), color = "red")
# tpmf = plt.plot(x, stats.poisson.pmf(x, np.exp(npl)), color = "green")
# plt.show()


# SIMULATIONS
stest = data_simulators.simulations(nsim = 500, B = 1000, 
                                    lik = PoissonLikelihood)
# right now we're only tracking param deviations and prediction error
q_results = []

for i in range(0, 20, 5):
    print("on epsilon:", i)
    stest.data_setup(data_simulators.EpsilonContam, 
                     n = 500, p = 2, params = np.array([0.1, 2]), 
                     continuous_x = True, share = 0.2, contam_par = i)
    stest.simulate()
    q_results = q_results + [stest.calc_quantiles()]
    
