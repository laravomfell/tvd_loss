#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 09:30:54 2020

@author: jeremiasknoblauch

Description: Read in the results and produce plots
"""

import numpy as np
import matplotlib.pyplot as plt

# global variables
lightblue = '#56B4E9'
black = '#000000'
darkblue = '#0072B2'
green = '#009E73'
orange = "#D55E00"

color_TVD = darkblue
color_KLD = green
median_color = orange
linewidth_boxplot = 4


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


def single_histogram_comparison(base_path, fig, ax, data_name, criterion):
    """Create a single histogram comparison between TVD and KLD on data set 
    data_name with the given criterion"""
    
    # get the path at which relevant aggregates are stored
    path_name_TVD = (base_path + "/" + data_name + "/" + 
                     "aggregate" + criterion + "TVD" + ".txt")
    path_name_KLD = (base_path + "/" + data_name + "/" + 
                     "aggregate" + criterion + "KLD" + ".txt") 
    
    # read in the aggregate data
    data_TVD = np.loadtxt(path_name_TVD).flatten()
    data_KLD = np.loadtxt(path_name_KLD).flatten()
    
    # convert into logs if we process cross-entropy
    if criterion == "_cross_entropy_":
        data_TVD = np.log(-data_TVD)
        data_KLD = np.log(-data_KLD)
    
    # exclude outliers
    outlier_indices_TVD = data_TVD < 50
    outlier_indices_KLD = data_KLD < 50
    
    # get the data and its maxima/minima
    dats = [data_TVD[outlier_indices_TVD], data_KLD[outlier_indices_KLD]]  
    min1, min2 = np.min(dats[0]), np.min(dats[1])
    max1, max2 = np.max(dats[0]), np.max(dats[1])
    min_, max_ = min(min1, min2), max(max1, max2)
    hist_range = (min_, max_)
    print(hist_range)
    
    # pre-compute the histograms for easier computation
    number_of_bins = 50
    binned_data_sets = [
        np.histogram(d, range=hist_range, bins=number_of_bins)[0]
        for d in dats
        ]
    binned_maximums = np.max(binned_data_sets, axis=1)
    x_locations = np.arange(0, sum(binned_maximums), np.max(binned_maximums))
    print(binned_data_sets[0])
    
    # The bin_edges are the same for all of the histograms
    bin_edges = np.linspace(hist_range[0], hist_range[1], number_of_bins + 1)
    centers = 0.5 * (bin_edges + np.roll(bin_edges, 1))[:-1]
    heights =  np.diff(bin_edges)
    
    # Cycle through and plot each histogram
    for x_loc, binned_data in zip(x_locations, binned_data_sets):
        lefts = x_loc - 0.5 * binned_data
        ax.barh(centers, binned_data, height=heights, left=lefts)
        
    # set x-axis label
    ax.set_xticks(x_locations)
    ax.set_xticklabels(['TVD', 'KLD'])
    
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
    
    # convert into logs if we process cross-entropy
    if criterion == "_cross_entropy_":
        data_TVD = np.log(-data_TVD)
        data_KLD = np.log(-data_KLD)
    
    # group and plot the data
    dats = [data_TVD, data_KLD]
    bp = ax.boxplot(dats, notch = True, showfliers = False)
    
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


def single_whiskerplot_comparison(base_path, fig, ax, data_name, criterion):
    """Create a single whisker plot comparison between TVD and KLD on data set 
    data_name with the given criterion"""
    
    
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
    

if True:
    # set the save path (where are the results stored?)
    base_path = "/Users/jeremiasknoblauch/Documents/OxWaSP/tvd_loss/experiments"
    
    # decide which data you want to run the experiments for
    # choices: haberman, fourclass, heart, mammographic_mass, statlog-shuttle, 
    #          breast-cancer-wisconsin
    data_name = "haberman"
    
    # criterion types
    criteria = ["_log_probs_", "_accuracy_", "_probabilistic_accuracy_",
                   "_cross_entropy_"]
    
    fig, ax = plt.subplots(1, 1, figsize = (10,10))
    
    fig, ax = single_boxplot_comparison(base_path, fig, ax, data_name, criteria[1])


if False:
    # set the save path (where are the results stored?)
    save_path = "/Users/jeremiasknoblauch/Documents/OxWaSP/tvd_loss/experiments"
    
    # decide which data you want to run the experiments for
    # choices: haberman, fourclass, heart, mammographic_mass, statlog-shuttle, 
    #          breast-cancer-wisconsin
    data_name = "mammographic_mass"
    
    # done for: haberman, fourclass, heart, mammographic_mass
    aggregate_splits(save_path, data_name)
