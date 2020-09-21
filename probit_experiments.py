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
save_path = "/Users/jeremiasknoblauch/Documents/OxWaSP/tvd_loss/experiments_new"
save_path_contam = "/Users/jeremiasknoblauch/Documents/OxWaSP/tvd_loss/experiments_contamination"

# decide which data you want to run the experiments for
# choices: haberman, fourclass, heart, mammographic_mass, statlog-shuttle, 
#          breast-cancer-wisconsin
data_name = "breast-cancer-wisconsin"


 # get the data
X, Y = get_probit_data(data_path, data_name)

# set the parameters for the test performance
num_splits = 50
train_proportion = 0.9
L = ProbitLikelihood()
B = 100


# whether or not you want to inject contamination
contamination = True

# size of contamination (in std-deviation units)
contamination_factor = 5.0 

# proportion of training observations we contaminate
contamination_proportion = 0.05

if contamination:
    
    # add contamination size to save path
    save_path_contam_ = save_path_contam + "_factor=" + str(int(contamination_factor)) + "_prop=" + str(contamination_proportion)
    
    # run the experiments
    get_test_performance(X, Y, num_splits, train_proportion, L, B, 
                                 save_path_contam_, data_name,
                                 contamination_factor, contamination_proportion, True)


if not contamination:
    
    # run the experiments
    get_test_performance(X, Y, num_splits, train_proportion, L, B, 
                                 save_path, data_name)

