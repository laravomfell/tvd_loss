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

from scipy.stats import poisson

from NPL import NPL
from likelihood_functions import PoissonLikelihoodSqrt, PoissonLikelihood
from data_simulators import NBPoissonSim, ZeroInflPoissonSim, EpsilonPoissonSim

L1 = PoissonLikelihood(X.shape[1])
npl_1 = NPL(L1, optimizer = "BFGS")
B = 100
npl_1.draw_samples(Y, X, B)

# get MLE
mle = sm.GLM(Y, X, family = sm.families.Poisson()).fit().params()
# npl
npl = npl_1.sample.mean(axis = 0)

plt.hist(Y)



# truth = np.array([0.5, -1.2, 1])
# X = np.array([np.ones(2000),
#               np.random.rand(2000),
#               np.random.normal(loc=2.0, scale=1.0, size = 2000)]).reshape(2000, 3)
# Y = np.floor(np.exp(np.matmul(X, truth)))
# Y[0:100] += 10
# # X, Y = EpsilonPoissonSim(1000, 3, truth, 3, 0.1, 10).run()
# #X, Y = ZeroInflPoissonSim(2000, 3, truth, 3, 0.2, 1).run()
# #X, Y = NBPoissonSim(2000, 3, truth, 3, 0.1, 5).run()


# #1. standard poisson
# #std_pois = sm.GLM(Y, X, family = sm.families.Poisson()).fit()

# # TVD 2 -- new inference method
# n, d = X.shape
# L = PoissonLikelihood(d)

# npl_sampler = NPL(L, optimizer = "BFGS")
# B=100
# npl_sampler.draw_samples(Y,X,B)