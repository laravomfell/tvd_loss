#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 09:34:38 2020

@author: Lara Vomfell

This code generates Figure 1 in our code by generating some Poisson outcomes,
contaminating 10% of the data with outliers and then running a KLD-minimizing
and a TVD-minimizing model and plotting the resulting pmfs.
"""

import numpy as np

from likelihood_functions import SimplePoissonLikelihood

import matplotlib.pyplot as plt
import scipy.stats as stats

from NPL import NPL

# Figure 1

# set seed
np.random.seed(16)

# generate n Poisson outcomes with lambda = 3
n = 500
Y = np.random.poisson(3, n)
# contaminate 10% of the data by adding k = 15
Y[0:50] += 15

# tell NPL to use 'SimplePoissonLikelihood', a Poisson lik without covariates
npl_fig1 = NPL(SimplePoissonLikelihood(), optimizer = "BFGS")
# generate intercept for NPL
X = np.ones([n, 1])
# (quietly) run models
npl_fig1.draw_samples(Y, X, B = 500, display_opt = False)

# get MLE and TVD
mle = npl_fig1.mle.mean()
tvd = npl_fig1.sample[npl_fig1.sample >= 0].mean()


# set up figure
plt.figure(figsize = (5,3))
x = range(Y.max() + 1)
# plot pmf of data
plt.hist(Y, x, density = 1, color = '#bababa', ec='#9f9f9f',align = 'left')

# plot implied pmfs of both models as lines + dots
plt.plot(x, stats.poisson.pmf(x, tvd), color = '#009E73', label = 'TVD')
plt.plot(x, stats.poisson.pmf(x, mle), color = '#56B4E9', label = 'KLD')

plt.plot(x, stats.poisson.pmf(x, mle), color = '#56B4E9', marker = 'o')
plt.plot(x, stats.poisson.pmf(x, tvd), color = '#009E73', marker = 'o')

plt.xticks(np.array(range(0, 21, 5)))
plt.ylabel('Probability mass')
plt.legend(frameon = False)
plt.tight_layout()
plt.savefig("fig1.png")