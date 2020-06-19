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

from data_simulators import NBPoissonSim
from data_simulators import ZeroInflPoissonSim
from toy import TVDMaster


# One thing I didn't think about too hard is that we basically do not care
# about the inference of the other params, just about the coefficients
# so maybe none of this is as big a deal as I thought.

# let's begin with a negbin example
truth = np.array([0.5, 1.2, 0])

negbin = NBPoissonSim(2000, 3, truth, 3, 0.4, 1/10)
X, Y = negbin.run()

std_nb = sm.GLM(Y, X, family = sm.families.NegativeBinomial()).fit()

# robust tvd
negbin_tvd = TVDMaster(X, Y, None)
negbin_tvd.run()

# evaluation 
out = np.column_stack((truth, std_nb.params, negbin_tvd.params.x))
print(out)

# so with this example, tvd is doing pretty badly

# ZERO INFLATION
#
#zeroinfl = ZeroInflPoissonSim(2000, 3, truth, 3, 0.2, 0.3)
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