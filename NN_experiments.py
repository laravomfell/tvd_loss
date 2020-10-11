#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 10:24:04 2020

@author: jeremiasknoblauch

Description: File to execute some experiments for NNs
"""

from experiment_auxiliaries import get_probit_data, get_test_performance 
from likelihood_functions import SoftMaxNN
import numpy as np

# set the data path (where is the data stored?)
data_path = "/Users/jeremiasknoblauch/Documents/OxWaSP/tvd_loss/tvd_loss/data/NN/"

# set the save path (where are the results stored?)
save_path = "/Users/jeremiasknoblauch/Documents/OxWaSP/tvd_loss/experiments/NN"

# decide which data you want to run the experiments for
# choices: ionos, madelon, pima, banknote_authentication, diabetic, ilpd, rice
data_name = "rice"

# get the data
X, Y = get_probit_data(data_path, data_name)

X, indices = np.unique(X, return_inverse=True, axis=0) # ionos has a duplicate row;need to remove it
Y= Y[indices]

# set the parameters for the test performance
num_splits = 50
train_proportion = 0.9
d = X.shape[1]
layer_size = 50
num_classes = 2
L = SoftMaxNN(d, layer_size, num_classes, epochs_vanilla = 10,
                             epochs_TVD = 10)
L.deactivate_printing()
B = 100

# run the experiments
get_test_performance(X, Y, num_splits, train_proportion, L, B, 
                             save_path, data_name)

