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

from NPL import NPL
from likelihood_functions import PoissonLikelihoodSqrt


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

# normalize your covariates
#X = X - np.mean(X, 0) 

# add some jitter to avoid that the first column (which consists in 1s) is 
# exactly zero
#


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
print(X.shape)
L = PoissonLikelihoodSqrt(d) # by default, we take the 2-nd root. 
                                       # Can change this to p-th root by passing argument

npl_sampler = NPL(L, optimizer = "BFGS")
B=100
npl_sampler.draw_samples(Y,X,B)

    