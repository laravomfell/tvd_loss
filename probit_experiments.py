#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 17:23:25 2020

@author: jeremiasknoblauch

Description: File to execute some experiments
"""

from auxiliaries.experiment_auxiliaries import get_data, get_test_performance 
from npl.likelihood_functions import ProbitLikelihood
import numpy as np

# set the data path (where is the data stored?)
data_path = "data/probit/"

# set the save path (where are the results stored?)
save_path = "data/probit"

# decide which data you want to run the experiments for
probit_data = ["mammographic_mass", "fourclass", "heart", "haberman", 
        "breast-cancer-wisconsin"]

for d in probit_data:
    X, Y = get_data(data_path, d)
    
    
    # set the parameters for the test performance
    num_splits = 50
    train_proportion = 0.9
    L = ProbitLikelihood()
    B = 100
    
    # run the experiments
    get_test_performance(X, Y, num_splits, train_proportion, L, B, 
                         save_path, d)

