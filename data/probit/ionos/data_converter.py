#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 15:15:04 2019

@author: jeremiasknoblauch

Description: Convert raw Ionosphere data into numerical
"""


import numpy as np

import pandas as pd


# We set the random seed

np.random.seed(1) #was 1

# We load the data

#data = np.loadtxt('/Users/jeremiasknoblauch/Documents/OxWaSP/GVI/code/BayesianProbit/data/ionos/data.txt',
#                  delimiter = ",")
data = pd.read_csv('/Users/jeremiasknoblauch/Documents/OxWaSP/GVI/code/BayesianProbit/data/ionos/data_orig.txt',sep=',', 
                   header = None)
data = np.array(data)
dat = np.zeros((351, 35), dtype=float)
for i in range(0, 351):
    dat[i,:-1] = data[i][:-1]
for j in range(0, 351):
    ent = 0.0
    if data[j][-1] == 'g':
        ent = 1.0
    dat[j,-1] = ent
#data[:,-1] = np.array([0 if i =='b' else 1 for i in data[:,-1]])
#data = np.array(data, dtype=float)
np.savetxt("data.txt",dat,fmt='%10.10f')

#data = np.append(data, labs, axis=1) #append labs as last column.

n = data.shape[ 0 ]

# We generate the training test splits

n_splits = 50
for i in range(n_splits):

    permutation = np.random.choice(range(n), n, replace = False)

    end_train = int(round(n * 9.0 / 10))
    end_test = n

    index_train = permutation[ 0 : end_train ]
    index_test = permutation[ end_train : n ]

    np.savetxt("index_train_{}.txt".format(i+1), index_train, fmt = '%d')
    np.savetxt("index_test_{}.txt".format(i+1), index_test, fmt = '%d')

    print i

np.savetxt("n_splits.txt", np.array([ n_splits ]), fmt = '%d')

# We store the index to the features and to the target

index_features = np.array(range(data.shape[ 1 ] - 1), dtype = int)
index_target = np.array([ data.shape[ 1 ] - 1 ])

np.savetxt("index_features.txt", index_features, fmt = '%d')
np.savetxt("index_target.txt", index_target, fmt = '%d')


