#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 09:34:38 2020

@author: Lara Vomfell

This file runs the full simulations reported in our paper.
1. The epsilon-contaminated Poisson model with \lambda = 3 and k = 0, ..., 20
2. The zero-inflated binomial model with \theta = [0.8, 0.25] and \varepsilon = 0, ..., 0.3

For each setting, we create 100 dataset replicates and then infer 
a) a model minimizing the KLD (with B = 1,000)
b) a model minimizing the TVD (with B = 1,000)
c) a fully Bayesian model using pystan (drawing 4,000 posterior samples)
"""

import numpy as np

from likelihood_functions import SimplePoissonLikelihood, BinomialLikelihood
from epsilon_simulation import simulations
from data_simulators import EpsilonContam, ZeroInflBinom



# EPSILON CONTAMINATION

# set up simulation params 
eps = simulations(nsim = 100, B = 1000, 
                  lik = SimplePoissonLikelihood, 
                  save_path = "D:/research/tvd_loss/sim_eps/",
                  stan_model = "stan_poisson.stan",
                  test_size = 0.2, 
                  var_par = 'contam_par')

# for each k, generate 100 datasets of size 500 with lambda = 3 
# and 15% contamination
for k in range(0, 24, 4):
    print("on k:", k)
    eps.data_setup(EpsilonContam, 
                   n = 500, 
                   p = 1, params = 3.0, 
                   continuous_x = False, 
                   share = 0.15, 
                   contam_par = k)
    eps.simulate()
    

#  ZERO-INFLATION

zero = simulations(nsim = 100, B = 1000, 
                   lik = BinomialLikelihood,
                   save_path = "D:/research/tvd_loss/sim_zeroinfl/",
                   stan_model = "stan_binomial.stan",
                   test_size = 0.2,
                   var_par = 'share')

# for each varepsilon, generate 100 datasets of size 1000
# with alpha = 0.8 and beta = 0.25 
for i in range(0, 40, 10):
    print("on varepsilon:", i)
    zero.data_setup(ZeroInflBinom,
                    n = 1000, 
                    p = 2, 
                    params = np.array([0.8, 0.25]), 
                    # number of trials
                    extra = 8,
                    continuous_x = False, 
                    share = i/100, 
                    contam_par = 1)
    zero.simulate()
    

