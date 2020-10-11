# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 09:21:55 2020

@author: phd17lv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pdb

# # read in each file, then calculate medians, then plot
# models = ['tvd', 'kld', 'stan']
# result_path = "d:/research/tvd_loss/sim_eps"
# results = ['param', 'loglik', 'ae']

# # list all files in result dir
# files = os.listdir(result_path)
# # drop all the ones not starting with a number
# eps = []
# for i in files:
#     if i[0].isdigit():
#         eps.append(i[0:2])
# eps = np.unique(eps)

def get_type_results(type, eps, models, result_path, write = None):
    # if write is unspecified, check if results already exist
    if write is None:
        write = not os.path.isfile(result_path + '/' + type + '_q10.csv')
    
    quantile_10 = []
    quantile_50 = []
    quantile_90 = []
    
    for m in models:
        print("on model:", m)
        m10 = []
        m50 = []
        m90 = []
        
        for e in eps:
            print("reading epsilon:", e)
            result = pd.read_csv(result_path +'/' + e + '_' + type + '_' + m + '.csv',
                                 delimiter=',')
            # drop the intercept results if multiple dimensions
            if result.shape[1] > 1:
                result = result.iloc[:, 1]
                
            # calculate quantiles
            q = np.nanquantile(result, q = [0.1, 0.5, 0.9])
            m10.append(q[0])
            m50.append(q[1])
            m90.append(q[2])
        
        quantile_10.append(m10)
        quantile_50.append(m50)
        quantile_90.append(m90)
    
    # collapse results
    quantile_10 = np.array(quantile_10).transpose()
    quantile_50 = np.array(quantile_50).transpose()
    quantile_90 = np.array(quantile_90).transpose()
    
    
    if write:
        np.savetxt(result_path + '/' + type + '_q10.csv', quantile_10)
        np.savetxt(result_path + '/' + type + '_q50.csv', quantile_50)
        np.savetxt(result_path + '/' + type + '_q90.csv', quantile_90)
        
    return quantile_10, quantile_50, quantile_90


def get_mean_results(type, eps, models, result_path, write = None):
    # if write is unspecified, check if results already exist
    if write is None:
        write = not os.path.isfile(result_path + '/' + type + '_q10_mean.csv')
    
    quantile_10 = []
    quantile_50 = []
    quantile_90 = []
    
    for m in models:
        print("on model:", m)
        m10 = []
        m50 = []
        m90 = []
        
        for e in eps:
            print("reading epsilon:", e)
            result = pd.read_csv(result_path +'/' + e + '_' + type + '_' + m + '.csv',
                                 delimiter=',')
            # drop the intercept results if multiple dimensions
            if result.shape[1] > 1:
                result = result.iloc[:, 1]
                
            # calculate quantiles for each chunk
            if m == "stan":
                size = 4000
            else:
                size = 1000
                
            
            q = np.quantile(np.array(
                [np.nanmean(
                    result.iloc[i:i+size]) for i in range(0, len(result), size)
                    ]),
                q = [0.1, 0.5, 0.9])
            
            # q = np.mean(np.array(
            #     [np.nanquantile(
            #         result.iloc[i:i+size], q = [0.1, 0.5, 0.9]) for i in range(0, len(result), size)
            #         ]),
            #     axis = 0)
            
            m10.append(q[0])
            m50.append(q[1])
            m90.append(q[2])

        
        quantile_10.append(m10)
        quantile_50.append(m50)
        quantile_90.append(m90)
        
    # collapse results
    quantile_10 = np.array(quantile_10).transpose()
    quantile_50 = np.array(quantile_50).transpose()
    quantile_90 = np.array(quantile_90).transpose()
    
    
    if write:
        np.savetxt(result_path + '/' + type + '_q10_mean.csv', quantile_10)
        np.savetxt(result_path + '/' + type + '_q50_mean.csv', quantile_50)
        np.savetxt(result_path + '/' + type + '_q90_mean.csv', quantile_90)
        
    return quantile_10, quantile_50, quantile_90

#par10, par50, par90 = get_type_results('param', eps, models, result_path)

# x = eps.astype('int')
# labs = ['TVD', 'KLD', 'Bayes']
# cols = ['#009E73', '#56B4E9', '#E69F00']

# # fig, ax = plt.subplots(1,1)

# # for i in range(len(models)):
# #     ax.plot(x, par50[:, i], label = labs[i], c = cols[i])
# #     ax.fill_between(x, par10[:,i], par90[:,i], alpha = .4, facecolors = cols[i])
    
# # ax.legend()


# lik10, lik50, lik90 = get_type_results('loglik', eps, models, 
#                                         result_path, write = True)

# # turn the loglikelihoods into likelihoods
# lik10 = np.exp(lik10)
# lik50 = np.exp(lik50)
# lik90 = np.exp(lik90)
# fig, ax = plt.subplots(1,1)

# for i in range(len(models)):
#     ax.plot(x, lik50[:, i], label = labs[i], c = cols[i])
#     #ax.fill_between(x, lik10[:,i], lik90[:,i], alpha = .4, facecolors = cols[i])
# ax.legend()

# ae10, ae50, ae90 = get_type_results('ae', eps, models, result_path)

# for i in range(len(models)):
#     plt.plot(x, ae50[:, i], label = labs[i], c = cols[i])
#     #plt.fill_between(x, ae10[:, i], ae90[:, i], alpha = .4, facecolors = cols[i])
# plt.legend()
# plt.show()

# read in each file, then calculate medians, then plot
models = ['tvd', 'kld', 'stan']
result_path = "d:/research/tvd_loss/sim_zeroinfl"
results = ['param', 'loglik', 'ae']

# list all files in result dir
files = os.listdir(result_path)
# drop all the ones not starting with a number
zero = []
for i in files:
    if i[0].isdigit():
        zero.append(i[0:2])
zero = np.unique(zero)


x = zero.astype('int')/100
labs = ['TVD', 'KLD', 'Bayes']
cols = ['#009E73', '#56B4E9', '#E69F00']

par10, par50, par90 = get_mean_results('param', zero, models, result_path, write = True)

for i in range(len(models)):
    plt.plot(x, par50[:, i], label = labs[i], c = cols[i])
    #plt.fill_between(x, par10[:,i], par90[:,i], alpha = .4, facecolors = cols[i])
plt.legend()
plt.show()

ae10, ae50, ae90 = get_mean_results('ae', zero, models, result_path, write = True)

for i in range(len(models)):
    plt.plot(x, ae50[:, i], label = labs[i], c = cols[i])
    #plt.fill_between(x, ae10[:, i], ae90[:, i], alpha = .4, facecolors = cols[i])
plt.legend()
plt.show()



lik10, lik50, lik90 = get_mean_results('loglik', zero, models, result_path, write = True)

lik10 = np.exp(lik10)
lik50 = np.exp(lik50)
lik90 = np.exp(lik90)



fig, ax = plt.subplots(1,1)
for i in range(len(models)):
    ax.plot(x, lik50[:, i], label = labs[i], c = cols[i])
    #ax.fill_between(x, lik10[:,i], lik90[:,i], alpha = .4, facecolors = cols[i])
    
ax.legend()