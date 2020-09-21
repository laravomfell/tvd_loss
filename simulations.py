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
import time

from likelihood_functions import PoissonLikelihood, BinomialLikelihood
from epsilon_simulation import simulations
from data_simulators import EpsilonContam, ZeroInflContam, ZeroInflBinom

from NPL import NPL

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

#SIMULATIONS
# t0 = time.time()
# stest = simulations(nsim = 100, B = 1000, 
#                     lik = PoissonLikelihood, 
#                     save_path = "D:/research/tvd_loss/sim_eps/",
#                     stan_model = "stan_poisson.stan",
#                     test_size = 0.2)

# for i in range(0, 20):
#     print("on epsilon:", i)
#     stest.data_setup(EpsilonContam, 
#                   n = 500, p = 2, params = np.array([0.1, 0.6]), 
#                   continuous_x = True, share = 0.2, contam_par = i)
#     stest.simulate()
    
# t1 = time.time()
# print((t1 - t0)/60)

X, Y = ZeroInflBinom(n = 250, p = 2, params = np.array([-0.5, 0.6]), u_bound = 5,
                      continuous_x=True, share = 0.3, contam_par = 1).contaminate()

npl_b = NPL(BinomialLikelihood())
npl_b.draw_samples(Y,X, 500)



# t0 = time.time()
# stest = simulations(nsim = 100, B = 1000, 
#                     lik = PoissonLikelihood, 
#                     save_path = "D:/research/tvd_loss/sim_zeroinfl/",
#                     stan_model = "stan_poisson.stan",
#                     test_size = 0.2,
#                     var_par = 'share')
# for i in range(0, 60, 10):
#     print("on p:", i)
#     stest.data_setup(ZeroInflContam, 
#                      n = 250, p = 2, params = np.array([0.5, 0.6]),
#                      continuous_x = True, share = i/100, contam_par = 1)
#     stest.simulate()
    
# t1 = time.time()
# print((t1 - t0)/60)