#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 10:24:04 2020

@author: jeremiasknoblauch

Description: File to execute the experiments for NNs
"""

from auxiliaries.experiment_auxiliaries import get_data, get_test_performance 
from npl.likelihood_functions import SoftMaxNN
import numpy as np

# set the data path (where is the data stored?)
data_path = "data/NN/"

# set the save path (where are the results stored?)
save_path = "data/NN"

# decide which data you want to run the experiments for
nn_data = ["pima", "diabetic", 
            "banknote_authentication", "ilpd", "rice"]

for d in nn_data:
    X, Y = get_data(data_path, d)
    
    # ionos has a duplicate row;need to remove it
    X, indices = np.unique(X, return_inverse=True, axis=0) 
    Y= Y[indices]
    
    # set the parameters for the test performance
    num_splits = 50
    train_proportion = 0.9
    k = X.shape[1]
    layer_size = 50
    num_classes = 2
    L = SoftMaxNN(k, layer_size, num_classes, epochs_vanilla = 10,
                                 epochs_TVD = 10)
    L.deactivate_printing()
    B = 100
    # run the experiments
    get_test_performance(X, Y, num_splits, train_proportion, L, B, 
                         save_path, d)

