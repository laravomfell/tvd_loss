#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 17:23:25 2020

@author: jeremiasknoblauch

Description: File to execute some experiments
"""

from experiment_auxiliaries import get_probit_data, get_test_performance 
from likelihood_functions import ProbitLikelihood

# set the data path (where is the data stored?)
data_path = "/Users/jeremiasknoblauch/Documents/OxWaSP/tvd_loss/tvd_loss/data/binary_classification/"

# set the save path (where are the results stored?)
save_path = "/Users/jeremiasknoblauch/Documents/OxWaSP/tvd_loss/experiments"

# decide which data you want to run the experiments for
# choices: haberman, fourclass, heart, mammographic_mass, statlog-shuttle, 
#          breast-cancer-wisconsin
data_name = "statlog-shuttle"

# ran: haberman, fourclass, heart, mammographic_mass,statlog-shuttle

# get the data
X, Y = get_probit_data(data_path, data_name)

# set the parameters for the test performance
num_splits = 50
train_proportion = 0.9
L = ProbitLikelihood()
B = 1000

# run the experiments
get_test_performance(X, Y, num_splits, train_proportion, L, B, 
                             save_path, data_name)