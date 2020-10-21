#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 09:30:54 2020

@author: jeremiasknoblauch

Description: Read in the results and produce plots. Before creating the plots
create .txt files holding the results to be plotted.
"""

import numpy as np
import matplotlib.pyplot as plt

# global variables
color_TVD = '#009E73'
color_KLD = '#56B4E9'
median_color = 'k'
linewidth_boxplot = 2


def aggregate_splits(path, data_name, split_num=50):
    # construct the path
    file_path = path + "/" + data_name + "/"
    
    # for each of the result types, set the result and inference types
    result_type = ["_log_probs_", "_accuracy_", "_probabilistic_accuracy_",
                   "_cross_entropy_"]
    inference_type = ["KLD", "TVD"]
    
    # for each (result, inference) combination, extract all results and 
    # aggregate into single file named "aggregate_" + result + inference
    for result in result_type:
        for inference in inference_type:
            
            # template name that needs a number still
            template_file_name = result + inference + ".txt"
            
            # create the aggregate output file and write all the results
            # to it
            aggregate_file = file_path + "aggregate" + template_file_name
            with open(aggregate_file, 'w') as aggregate:
                for i in range(0, split_num):
                    
                    # get the right string to append to front of template
                    if i < 10:
                        num = "0" + str(i)
                    else:
                        num = str(i)
                    
                    # new filename
                    file_name = file_path + num + template_file_name
                    
                    # append template to aggregate
                    with open(file_name) as current_split_results:
                        for line in current_split_results:
                            aggregate.write(line)



def single_boxplot_grouped_comparison(base_path, fig, ax, data_name, criterion):
    """Create a single boxplot comparison between TVD and KLD on data set 
    data_name with the given criterion"""
    
    # get the path at which relevant aggregates are stored
    path_name_TVD = (base_path + "/" + data_name + "/" + 
                     "aggregate" + criterion + "TVD" + ".txt")
    path_name_KLD = (base_path + "/" + data_name + "/" + 
                     "aggregate" + criterion + "KLD" + ".txt") 
    
    # read in the aggregate data
    data_TVD = np.loadtxt(path_name_TVD)
    data_KLD = np.loadtxt(path_name_KLD)
    
    
    # if the data was stored as matrix
    if len(data_TVD.shape) > 1:
        n_total, B = data_KLD.shape
        n = int(n_total/50)
        
        # get the number of rows corresponding to a single iteration/split
        means_TVD = np.zeros(50)
        means_KLD = np.zeros(50)
        for i in range(0, 50):
            start = i*n
            stop = (i+1)*n
            means_TVD[i] = np.mean(data_TVD[start:stop,:])
            means_KLD[i] = np.mean(data_KLD[start:stop,:])
    
    # if the data was stored as a single vector, this is an NN result
    else:
        n_total = len(data_TVD)
        n_split = int(n_total/50)
        
        # get the number of rows corresponding to a single iteration/split
        means_TVD = np.zeros(50)
        means_KLD = np.zeros(50)
        for i in range(0, 50):
            start = i*n_split
            stop = (i+1)*n_split
            means_TVD[i] = np.mean(data_TVD[start:stop])
            means_KLD[i] = np.mean(data_KLD[start:stop])
        
    
    
    dats = [means_TVD, means_KLD]    
    # if entropy, do log scale
    if criterion == "_cross_entropy_":
        max1, max2 = np.max(means_TVD), np.max(means_KLD)
        max_ = abs(max(max1, max2) + 100)
        dats = [np.log(-means_TVD+max_), np.log(-means_KLD+max_)]
        
    # group and plot the data
    medianprops = dict(linestyle='-', color='black')
    bp = ax.boxplot(dats, notch = False, showfliers = False, patch_artist=True,
                    medianprops = medianprops, widths=0.6) 
    
    # set colors for boxplot outlines
    cols = [color_TVD, color_KLD]
    for box, whisker, cap, median, flier, i in zip(bp['boxes'], bp['whiskers'], 
             bp['caps'], bp['medians'], bp['fliers'], range(0,2)):
        box.set( color=cols[i], linewidth=linewidth_boxplot)
        whisker.set( color=cols[i], linewidth=linewidth_boxplot)
        cap.set( color=cols[i], linewidth=linewidth_boxplot)
        median.set(color = "black")
        flier.set(False)
    
    for patch, color in zip(bp['boxes'], cols):
        patch.set_facecolor(color)
        
    # set x-axis label
    ax.set_xticklabels(['TVD', 'KLD'], fontsize = 13)
    ax.yaxis.set_tick_params(labelsize=12)
    
    # remove top axes and right axes ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
    return fig, ax


def single_boxplot_comparison(base_path, fig, ax, data_name, criterion):
    """Create a single boxplot comparison between TVD and KLD on data set 
    data_name with the given criterion"""
    
    # get the path at which relevant aggregates are stored
    path_name_TVD = (base_path + "/" + data_name + "/" + 
                     "aggregate" + criterion + "TVD" + ".txt")
    path_name_KLD = (base_path + "/" + data_name + "/" + 
                     "aggregate" + criterion + "KLD" + ".txt") 
    
    # read in the aggregate data
    data_TVD = np.loadtxt(path_name_TVD).flatten()
    data_KLD = np.loadtxt(path_name_KLD).flatten()
    
    # group and plot the data
    dats = [data_TVD, data_KLD]
    bp = ax.boxplot(dats, notch = False, showfliers = False, widths = 0.6)
    
    # set colors for boxplot outlines
    cols = [color_TVD, color_KLD]
    for box, whisker, cap, median, flier, i in zip(bp['boxes'], bp['whiskers'], 
             bp['caps'], bp['medians'], bp['fliers'], range(0,2)):
        box.set( color=cols[i], linewidth=linewidth_boxplot)
        whisker.set( color=cols[i], linewidth=linewidth_boxplot)
        cap.set( color=cols[i], linewidth=linewidth_boxplot)
        median.set(color = median_color,linewidth=linewidth_boxplot)
        flier.set(False)
        
        
    # set x-axis label
    ax.set_xticklabels(['TVD', 'KLD'])
    
    # remove top axes and right axes ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
    return fig, ax


def boxplot_comparison(base_path, list_of_data_names, list_of_criteria, fig_size):
    """Create a plot s.t. each row gives a criterion, each col a data set"""
    
    # create panels
    num_rows = len(list_of_criteria)
    num_cols = len(list_of_data_names)
    fig, ax_array = plt.subplots(num_rows, num_cols, figsize = fig_size)
    
    for ax, i in zip(ax_array.flatten(), range(0, num_rows * num_cols)):
        ax = single_boxplot_comparison(base_path, fig,ax, 
                            list_of_data_names[i], list_of_criteria[i])
    
    return fig, ax_array


def boxplot_grouped_comparison(base_path, list_of_data_names, list_of_plot_names, 
                               list_of_criteria, fig_size, ylim=[0.48, 0.699]):
    """Create a plot s.t. each row gives a criterion, each col a data set"""
    
    # create panels
    num_rows = len(list_of_criteria)
    num_cols = len(list_of_data_names)
    fig, ax_array = plt.subplots(num_rows, num_cols, figsize = fig_size)
    
    ylabel_names = ["predictive likelihood", "accuracy"]
    
    for row in range(0, num_rows):
        for col in range(0, num_cols):
            
            fig, ax_array[row, col] = single_boxplot_grouped_comparison(base_path, 
                  fig,ax_array[row,col], list_of_data_names[col], list_of_criteria[row])
            ax_array[row, col].set_ylim(ylim)
            
            if row == 0:
                ax_array[row, col].set_title(list_of_plot_names[col], fontsize=15)
            
            if col == 0:
                ax_array[row, col].set_ylabel(ylabel_names[row], fontsize=15)

    
    return fig, ax_array
    


'''Aggregate the full experimental files into a single dataset'''
# note: this might take some time

# set the save paths (where are the results stored?)
probit_path = "data/probit"
nn_path = "data/NN"

# datasets to evaluate
probit_data = ["mammographic_mass", "fourclass", "heart", "haberman", 
        "breast-cancer-wisconsin"]
# NN datasets evaluated 
nn_data = ["pima", "diabetic", 
            "banknote_authentication", "ilpd", "rice"]

for d in probit_data:
    aggregate_splits(probit_path, d)
    
for d in nn_data:
    aggregate_splits(nn_path, d)



"""Plots"""
fig_path = "figures"
    
    
# evaluation criteria
crits = ["_accuracy_", "_log_probs_", ]

# Probit results

# probit plot headers with padded white space
probit_headers = ["mammograph", "fourclass", "heart", "haberman  ", "breast cancer   "]
    
# create boxplots with one panel for each dataset, 
# top row = predictive likelihood, bottom row = accuracy
fig, ax = boxplot_grouped_comparison(probit_path, probit_data, probit_headers, crits, (7.5,8))

fig.suptitle("Probit results", fontsize=18  )
fig.tight_layout()
        
fig.savefig(fig_path + "/" + "probit_results.pdf")
    
    
# NN results

# plot headers
nn_headers = ["pima", "diabetic", 
            "banknote", "ilpd", "rice"]
    
# create boxplots with one panel for each dataset, 
# top row = predictive likelihood, bottom row = accuracy
fig, ax = boxplot_grouped_comparison(nn_path, nn_data, nn_headers, crits, (7.5,8))
fig.tight_layout()
fig.suptitle("Neural Network results", fontsize=18  )
fig.tight_layout()
fig.savefig(fig_path + "/" + "NN_results.pdf")

    

