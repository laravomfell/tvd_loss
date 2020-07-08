#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 1 2020

@author: laravomfell

Description:
    Early data tests on some built in R data    

"""

import autograd.numpy as np
import autograd.numpy.random as npr
#from autograd.scipy.misc import logsumexp
from autograd.scipy.stats import poisson
from autograd import grad

import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.discrete.count_model as d_sm

from toy import TVDMaster

import pandas as pd

from sklearn.model_selection import train_test_split

def model_wrap(X, Y, type):
    
    
    X = pd.get_dummies(X, drop_first = True)
    X = sm.add_constant(X)
    
    Y = Y.to_numpy()    
    X = X.to_numpy()
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
    tvd = TVDMaster(train_X, train_Y, None)
    tvd.run()
    
    # return params
    out = np.column_stack((std_pois.params,
                           act_params,
                           tvd.params.x))
    
    # get fitted values
    val = np.column_stack((std_pois.fittedvalues,
                           std_act.fittedvalues,
                           tvd.fittedvalues
                           ))
    
    res = train_Y[:, None] - val
    
    # now test on hold out data
    test_val = np.column_stack((std_pois.predict(test_X),
                                act_test,
                                np.exp(np.matmul(test_X, tvd.params.x))))
    
    test_fit = test_Y[:, None] - test_val
    
    return train_Y, test_Y, out, res, test_fit


# Contamination
    
#carrots = pd.read_csv('data/carrots.csv',delimiter=',')
## separate Y and X, create model matrix
#Y = carrots['success']
#X = carrots.drop('success', axis = 1)
#
#train_Y, test_Y, params, res, test_fit = model_wrap(X, Y, type = "NB")
#
#print("Carrots:")
#print(params)
#fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5)
#fig.suptitle('Carrot fit')
#ax1.hist(Y)
#ax2.plot(train_Y, res[:,0], 'b.')
#ax3.plot(train_Y, res[:,2], 'b.')
#ax4.plot(test_Y, test_fit[:,0], 'r.')
#ax5.plot(test_Y, test_fit[:, 2], 'r.')


epil = pd.read_csv('data/epilepsy.csv',delimiter=',')
# separate Y and X, create model matrix
Y = epil['Y4']
X = epil.drop(['Y1', 'Y2', 'Y3', 'Y4', 'ID'], axis = 1)

train_Y, test_Y, params, res, test_fit = model_wrap(X, Y, type = "NB")

print("Epilepsy:")
print(params)
#fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5)
#fig.suptitle('Epilepsy fit')
#ax1.hist(Y)
#ax2.plot(train_Y, res[:,0], 'b.')
#ax3.plot(train_Y, res[:,2], 'b.')
#ax4.plot(test_Y, test_fit[:,0], 'r.')
#ax5.plot(test_Y, test_fit[:, 2], 'r.')


# NEGATIVE BINOMIAL

#quine = pd.read_csv('data/quine.csv',delimiter=',')
## separate Y and X, create model matrix
#Y = quine['Days']
#X = quine.drop('Days', axis = 1)
#
#train_Y, test_Y, params, fitted, res, test_fit = model_wrap(X, Y, type = "NB")
#
#print("Quine:")
#print(params)
#fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7)
#fig.suptitle('Quine fit')
#ax1.hist(Y)
#ax2.plot(train_Y, res[:,0], 'b.')
#ax3.plot(train_Y, res[:,1], 'b.', sharex = ax2)
#ax4.plot(train_Y, res[:,2], 'b.')
#ax5.plot(test_Y, test_fit[:,0], 'r.')
#ax6.plot(test_Y, test_fit[:,1], 'r.')
#ax7.plot(test_Y, test_fit[:, 2], 'r.')
#plt.close()

# CRABS NEXT
#
#crabs = pd.read_csv('data/crabs.csv', delimiter = ',')
#Y = crabs['satellites']
#X = crabs.drop(['satellites', 'id', 'weight', 'carapace_width'], axis = 1)
#
#train_Y, test_Y, params, fitted, res, test_fit = model_wrap(X, Y, type = "NB")
#
#print("Crabs:")
#print(params)
#fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7)
#fig.suptitle('Crabs fit')
#ax1.hist(Y)
#ax2.plot(train_Y, res[:,0], 'b.')
#ax3.plot(train_Y, res[:,1], 'b.')
#ax4.plot(train_Y, res[:,2], 'b.')
#ax5.plot(test_Y, test_fit[:,0], 'r.')
#ax6.plot(test_Y, test_fit[:,1], 'r.')
#ax7.plot(test_Y, test_fit[:, 2], 'r.')
## next up, zero inflation
#
#biochem = pd.read_csv('data/bioChemists.csv', delimiter = ',')
#Y = biochem['art']
#X = biochem.drop('art', axis = 1)
#
#train_Y, test_Y, params, fitted, res, test_fit = model_wrap(X, Y, type = "ZI")
#print("bioChem")
#print(params)
#fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7)
#fig.suptitle('bioChem fit')
#ax1.hist(Y)
#ax2.plot(train_Y, res[:,0], 'b.')
#ax3.plot(train_Y, res[:,1], 'b.')
#ax4.plot(train_Y, res[:,2], 'b.')
#ax5.plot(test_Y, test_fit[:,0], 'r.')
#ax6.plot(test_Y, test_fit[:,1], 'r.')
#ax7.plot(test_Y, test_fit[:, 2], 'r.')
## fish last
#fish = pd.read_csv('data/fish.csv', delimiter = ',')
#Y = fish['count']
#X = fish.drop(['count', 'nofish', 'xb', 'zg'], axis = 1)
#train_Y, test_Y, params, fitted, res, test_fit = model_wrap(X, Y, type = "NB")
#
#print("Fish:")
#print(params)
#fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7)
#fig.suptitle('Fish fit')
#ax1.hist(Y)
#ax2.plot(train_Y, res[:,0], 'b.')
#ax3.plot(train_Y, res[:,1], 'b.')
#ax4.plot(train_Y, res[:,2], 'b.')
#ax5.plot(test_Y, test_fit[:,0], 'r.')
#ax6.plot(test_Y, test_fit[:,1], 'r.')
#ax7.plot(test_Y, test_fit[:, 2], 'r.')