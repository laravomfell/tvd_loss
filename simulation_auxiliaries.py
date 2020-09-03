#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 17:13:47 2020

@author: jeremiasknoblauch

Description: Some helper files for running splits with NPL
"""

import numpy as np
from NPL import NPL
import os
import pandas as pd


def get_probit_data(data_path):
    data = pd.read_csv(data_path, sep=" ", header=None)
    X = np.array(data)[:,:-1]
    Y = np.array(data)[:,-1] 
    Y = Y.astype(int)
    return X, Y


def get_test_performance(X, Y, num_splits, train_proportion, L, B, 
                             save_path, data_name, print_process = True):
    """Take in a data set (X,Y), split it (randomly) and train on a 
    portion of the data (test on the remainder)"""
    
    # Create the NPL object
    npl_sampler = NPL(L)
    n, d = X.shape
    n_train = np.floor(n * train_proportion)
    n_test = n - n_train
    
    # Loop over the splits
    for i in range(0, num_splits):
        
        # notify user of split
        print("Starting to process split " + str(i) + " / " + str(num_splits))
    
        # Create the split for seed
        np.random.seed(i)
        train_indices = np.random.choice(n, size = n_train, replace=False)
        test_indices = np.setdiff1d(np.linspace(0,n,n-1), train_indices)           
        X_train = X[:,train_indices]
        Y_train = Y[train_indices]
        X_test = X[:,test_indices]
        Y_test = Y[test_indices]
        
        # Sample from NPL object, based on the train proportion of (X,Y)
        npl_sampler.draw_samples(Y_train, X_train,B, display_opt=False)
        
        # Test on the remainder of (X,Y)
        log_probs, accuracy, cross_entropy = npl_sampler.predict(Y_test, X_test)
        log_probs_init, accuracy_init, cross_entropy_init = npl_sampler.predict_log_loss(Y_test, X_test)
        accuracy_prob = 1 - np.mean(np.abs(np.exp(log_probs) - 
                                           Y_test[:,np.newaxis]))
        accuracy_prob_init = 1 - np.mean(np.abs(np.exp(log_probs_init) - 
                                                Y_test[:,np.newaxis]))

        # save the results to path
        if i < 10:
            num = "0" + str(i)
        else:
            num = str(i)
            
        # create a folder with the data name in which to save all results
        if not os.path.exists(data_name):
            os.makedirs(data_name)
        
        file_path = save_path + "/" + data_name + "/"
            
        # log probs
        np.savetxt(file_path + num + "_log_probs_TVD.txt", log_probs)
        np.savetxt(file_path + num + "_log_probs_KLD.txt", log_probs_init)
        # accuracy
        np.savetxt(file_path + num + "_accuracy_TVD.txt", accuracy)
        np.savetxt(file_path + num + "_accuracy_KLD.txt", accuracy_init)
        # probabilistic accuracy
        np.savetxt(file_path + num + "_probabilistic_accuracy_TVD.txt", 
                   accuracy_prob)
        np.savetxt(file_path + num + "_probabilistic_accuracy_KLD.txt", 
                   accuracy_prob_init)
        # cross-entropy
        np.savetxt(file_path + num + "_cross_entropy_TVD.txt", cross_entropy)
        np.savetxt(file_path + num + "_cross_entropy_KLD.txt", cross_entropy_init)

