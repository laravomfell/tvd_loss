#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 10:46:44 2020

@author: jeremiasknoblauch
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 1 2020

@author: laravomfell

Description:
    Early data tests on some built in R data    

"""

import jax.numpy as jnp
import numpy as np
import jax.random as npr
from jax.scipy.special import logsumexp
from jax.scipy.stats import poisson
from jax import grad

import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.discrete.count_model as d_sm

from toy import TVDMaster
from BBGVI import BBGVI_TVD
from Divergence import MFN_MFN_KLD 
from Loss import TvdLossPoissonGLM 

import pandas as pd

from sklearn.model_selection import train_test_split


# This is needed so that I can use scipy optimizers on jax-np arrays 
# (see https://github.com/google/jax/issues/936)
from jax.config import config
config.update("jax_enable_x64", True)




epil = pd.read_csv('/Users/jeremiasknoblauch/Documents/OxWaSP/tvd_loss/tvd_loss/data/epilepsy.csv',delimiter=',')
# separate Y and X, create model matrix
Y = epil['Y4']
X = epil.drop(['Y1', 'Y2', 'Y3', 'Y4', 'ID'], axis = 1)
    
    
X = pd.get_dummies(X, drop_first = True)
X = sm.add_constant(X)

Y = Y.to_numpy()    
X = X.to_numpy()
Y = Y.astype('float64') 
X = X.astype('float64') 


# split data into training and test data.#
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, 
                                                    train_size=0.8,
                                                    test_size=0.2,
                                                    random_state=123)

# next, estimate the models
#1. standard poisson
std_pois = sm.GLM(train_Y, train_X, family = sm.families.Poisson()).fit()
# 2. actual model    
if type == "NB":
    std_act = sm.GLM(train_Y, train_X, 
                     family = sm.families.NegativeBinomial()).fit()
    act_params = std_act.params
    act_test = std_act.predict(test_X)
if type == "ZI":
    std_act = d_sm.ZeroInflatedPoisson(train_Y, train_X, None, inflation='logit').fit()
    act_params = np.delete(std_act.params, 0)
    act_test = std_act.predict(test_X, exog_infl = np.ones((len(test_X), 1)))
    
# 3. TVD
#tvd = TVDMaster(train_X, train_Y, None)
#tvd.run()

# TVD 2 -- new inference method
n, d = X.shape
loss_weight = 1.0
print(X.shape)
D = MFN_MFN_KLD(np.zeros(d), np.ones(d)*5)
L = TvdLossPoissonGLM(d, loss_weight)
jax_key_seed = 1
learning_rate = 0.01
tvd2 = BBGVI_TVD(D, L, jax_key_seed)
K=100
epochs = 500
# optimizer methods: BFGS, Nelder-Mead, ADAM
tvd2.fit_q(Y,X,K, epochs, learning_rate, optimizer = "ADAM")
tvd2.report_parameters()
    

