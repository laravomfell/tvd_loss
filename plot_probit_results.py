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

TVD_col =  '#009E73'
KLD_col =  '#56B4E9'

color_TVD = TVD_col
color_KLD = KLD_col
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


def single_violinplot_comparison(base_path, fig, ax, data_name, criterion, 
                                 bw_method = None):
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
        max1 = np.max(data_TVD)
        max2 = np.max(data_KLD)
        max_ = max(max1, max2) + 2
        data_TVD = np.log(-(data_TVD - max_))
        data_KLD = np.log(-(data_KLD - max_))
    
    # exclude outliers
    # outlier_indices_TVD = data_TVD < 50
    # outlier_indices_KLD = data_KLD < 50
    
    # get the data and its maxima/minima
    # dats = [data_TVD[outlier_indices_TVD], data_KLD[outlier_indices_KLD]]  
    dats = [data_TVD, data_KLD]  
    
    # axes labels
    labels = ["TVD", "KLD"]
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')
    
    # get violinplot
    parts = ax.violinplot(
        dats, showmeans=False, showmedians=False,
        showextrema=False, widths = [1.25, 0.5], bw_method=bw_method)
    
    # set colors for the density 
    colors = [color_TVD, color_KLD]
    for pc, i in zip(parts['bodies'], range(0,2)):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(1)
    
    # compute mean and median + plot
    means = np.array([np.mean(dat) for dat in dats])
    medians = np.array([np.median(dat) for dat in dats])
    xpos = np.array([1,2])
    ax.scatter(xpos, medians, s=250, c="black", marker = "D")
    ax.scatter(xpos,means, s=250, c="black")
    
        
    # set colors for means, medians, ...
    # parts['cmeans'].set_edgecolor("black")
    # parts['cmeans'].set_linewidth(4)

    
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
    # if criterion == "_cross_entropy_":
    #     data_TVD = np.log(-data_TVD)
    #     data_KLD = np.log(-data_KLD)
    
    #      btvd = ax.boxplot(tvd, positions = np.array(range(len(tvd))) * 3 - 0.8,
    #                            sym = '', widths = 0.6, patch_artist = True,
    #                            boxprops = dict(facecolor = '#009E73'),
    #                            medianprops = dict(color = 'black'))

    # Color scheme:
    # TVD '#009E73'
    # KLD '#56B4E9'
    # Stan '#E69F00'
    
    #      btvd = ax.boxplot(tvd, positions = np.array(range(len(tvd))) * 3 - 0.8,
    #                            sym = '', widths = 0.6, patch_artist = True,
    #                            boxprops = dict(facecolor = '#009E73'),
    #                            medianprops = dict(color = 'black'))

    
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
    
    
     # Color scheme:
    # TVD '#009E73'
    # KLD '#56B4E9'
    # Stan '#E69F00'
    
    #      btvd = ax.boxplot(tvd, positions = np.array(range(len(tvd))) * 3 - 0.8,
    #                            sym = '', widths = 0.6, patch_artist = True,
    #                            boxprops = dict(facecolor = '#009E73'),
    #                            medianprops = dict(color = 'black'))

    
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
    # (0.8, 0.8)
    
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



def single_whiskerplot_comparison(base_path, fig, ax, data_name, criterion):
    """Create a single whisker plot comparison between TVD and KLD on data set 
    data_name with the given criterion"""
    
    # get the path at which relevant aggregates are stored
    path_name_TVD = (base_path + "/" + data_name + "/" + 
                     "aggregate" + criterion + "TVD" + ".txt")
    path_name_KLD = (base_path + "/" + data_name + "/" + 
                     "aggregate" + criterion + "KLD" + ".txt") 
    
    # read in the aggregate data
    data_TVD = np.loadtxt(path_name_TVD).flatten()
    data_KLD = np.loadtxt(path_name_KLD).flatten()

    
    # get the means and stddevs (for 'whiskers')
    dats = [data_TVD, data_KLD]
    means = np.array([np.mean(dat) for dat in dats])
    std_devs = np.array([np.std(dat) for dat in dats])
    medians = np.array([np.median(dat) for dat in dats])
    
    
    """Plot all settings and put a vertical line through the baseline"""
    colors = [color_TVD, color_KLD]
    xpos = np.array([0.5,1.0])
    ax.errorbar(y=means, x=xpos, yerr = std_devs, fmt = 'none', ecolor = colors)
    ax.scatter(xpos,medians, s=80, c=colors, marker = "D")
    ax.set_xlim(0, 1.5)
    #for m,y, c in zip(means, ypos, colors):
    ax.scatter(xpos,means, s=80, c=colors) #marker

    ax.set_xticklabels(["TVD", "KLD"], fontsize = 11)
    ax.set_xticks(xpos)
    
    #ax.axvline(x = means[baseline_index],linestyle = '--', color='grey')   
    
    """plot name of dataset on top"""
    ax.set_title(data_name)
    
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
    
    # for ax, i in zip(ax_array.flatten(), range(0, num_rows * num_cols)):
    #     ax = single_boxplot_grouped_comparison(base_path, fig,ax, 
    #                         list_of_data_names[i], list_of_criteria[i])
    
    return fig, ax_array
    


"""PLOT THE PROBIT RESULTS"""
if False:
    # set the save path (where are the results stored?)
    base_path = "/Users/jeremiasknoblauch/Documents/OxWaSP/tvd_loss/experiments_new"
    fig_path = "/Users/jeremiasknoblauch/Documents/OxWaSP/tvd_loss/tvd_loss/figures"
    
    # decide which data you want to run the experiments for
    # choices: haberman, fourclass, heart, mammographic_mass, statlog-shuttle, 
    #          breast-cancer-wisconsin
    data_name = "fourclass"
    
    # criterion types
    criteria = ["_log_probs_", "_accuracy_", "_probabilistic_accuracy_",
                   "_cross_entropy_"]
    
    crits = [criteria[2], criteria[1]]
    sets = ["mammographic_mass", "fourclass", "heart", "haberman", 
            "breast-cancer-wisconsin"]
    plot_headers = ["mammograph", "fourclass", "heart", "haberman  ", "breast cancer   "]
    
    #if criterion != "_cross_entropy_":  
    fig, ax = boxplot_grouped_comparison(base_path, sets,plot_headers, crits, (7.5,8))
    # (10,5)
    fig.suptitle("Probit results", fontsize=18  )
    fig.tight_layout()
        
    fig.savefig(fig_path + "/" + "probit_results.pdf")
            #data_name, criterion)
        # , bw_method = 0.2
    #else:
    #fig, ax = single_boxplot_comparison(base_path, fig, ax, 
    #        data_name, criterion)
    
    
    #  btvd = ax.boxplot(tvd, positions = np.array(range(len(tvd))) * 3 - 0.8,
    #                            sym = '', widths = 0.6, patch_artist = True,
    #                            boxprops = dict(facecolor = '#009E73'),
    #                            medianprops = dict(color = 'black'))

    # Color scheme:
    # TVD '#009E73'
    # KLD '#56B4E9'
    # Stan '#E69F00'

"""PLOT THE PROBIT RESULTS WITH CONTAMINATION"""
if False:
    # set the save path (where are the results stored?)
    base_path = "/Users/jeremiasknoblauch/Documents/OxWaSP/tvd_loss/experiments_new"
    save_path_contam = "/Users/jeremiasknoblauch/Documents/OxWaSP/tvd_loss/experiments_contamination"
    
    # decide which data you want to run the experiments for
    # choices: haberman, fourclass, heart, mammographic_mass, statlog-shuttle, 
    #          breast-cancer-wisconsin
    data_name = "fourclass"
    
    # criterion types
    criteria = ["_log_probs_", "_accuracy_", "_probabilistic_accuracy_",
                   "_cross_entropy_"]
    
    fig, ax = plt.subplots(2, 2, figsize = (5,5))
    
    crits = [criteria[2], criteria[1]]
    sets = ["mammographic_mass", "fourclass", "heart", "haberman", 
            "breast-cancer-wisconsin"]
    plot_headers = ["mammographic mass", "fourclass", "heart", "haberman", "breast cancer"]
    
    # contamination
    contam = "_factor=5_prop=0.05"
    save_path_contam = save_path_contam + contam
    
    #if criterion != "_cross_entropy_":  
    fig, ax = boxplot_grouped_comparison(save_path_contam, sets, plot_headers, crits, (10,5))
    fig.tight_layout()
    
    
"""PLOT THE NN RESULTS"""
if True:
    # set the save path (where are the results stored?)
    base_path = "/Users/jeremiasknoblauch/Documents/OxWaSP/tvd_loss/experiments/NN"
    fig_path = "/Users/jeremiasknoblauch/Documents/OxWaSP/tvd_loss/tvd_loss/figures"
    
    # decide which data you want to run the experiments for
    # choices: pima, madelon, ionos, diabetic, banknote_authentication, ilpd
    
    # criterion types
    criteria = ["_log_probs_", "_accuracy_", "_probabilistic_accuracy_",
                   "_cross_entropy_"]
    
    crits = [criteria[2], criteria[1]]
    sets = ["pima", "diabetic", 
            "banknote_authentication", "ilpd", "rice"]
    plot_headers = ["pima", "diabetic", 
            "banknote", "ilpd", "rice"]
    
    #if criterion != "_cross_entropy_":  
    fig, ax = boxplot_grouped_comparison(base_path, sets, plot_headers, crits, (7.5,8))
    fig.tight_layout()
    fig.suptitle("Neural Network results", fontsize=18  )
    fig.tight_layout()
    fig.savefig(fig_path + "/" + "NN_results.pdf")
            #data_name, criterion)
        # , bw_method = 0.2
    #else:
    #fig, ax = single_boxplot_comparison(base_path, fig, ax, 
    #        data_name, criterion)


"""AGGREGATE INDIVIDUAL SPLITS INTO A SINGLE RESULTS DATA SET"""
if False:
    # set the save path (where are the results stored?)
    save_path = "/Users/jeremiasknoblauch/Documents/OxWaSP/tvd_loss/experiments_new"
    save_path_NN = "/Users/jeremiasknoblauch/Documents/OxWaSP/tvd_loss/experiments/NN"
    save_path_contam = "/Users/jeremiasknoblauch/Documents/OxWaSP/tvd_loss/experiments_contamination"
    
    
    # contamination params
    contamination_factor = str(int(5.0)) 
    contamination_proportion = str(0.05) 
    
    # decide which data you want to run the experiments for
    # probit: haberman, fourclass, heart, mammographic_mass, statlog-shuttle, 
    #          breast-cancer-wisconsin
    data_name = "rice"
    
    # NN: pima, madelon, ionos, diabetic, banknote_authentication, ilpd
    data_name_NN = "ilpd"
    
    contam = "_factor=5_prop=0.05"
    save_path_contam = save_path_contam + contam
    
    # done for: haberman, fourclass, heart, mammographic_mass
    aggregate_splits(save_path_NN, data_name)
  
    
"""RE-FORMAT THE DATA"""    
if False:
    # set the data path (where are the results stored?)
    data_path = "/Users/jeremiasknoblauch/Documents/OxWaSP/tvd_loss/tvd_loss/data"
    
    # data name: ilpd, diabetic, banknote_authentication, Rice_Osmancik_Cammeo_Dataset.xlsx
    data_name = "rice"
    
    import pandas as pd
    data = pd.read_excel(data_path + "/" + "Rice_Osmancik_Cammeo_Dataset.xlsx")
    data['CLASS'] = data['CLASS'].astype('category').cat.codes
    
    # dat = pd.read_csv(data_path + "/" + data_name + ".txt", delimiter = " ")
    dat = np.array(data)
    # means = np.nanmean(dat, axis=0)
    # dat[np.isnan(dat[:,-1]),-1] = means[-1]
    
    
    # read in, transform first column from +1/-1 to 0/1 and put in last col
    dat = np.savetxt(data_path + "/" + data_name + ".txt", dat, delimiter = " ")
    
    dat_ = np.zeros(dat.shape)
    dat_[:, :-1] = dat[:, 1:]
    dat_[:,-1] = ((dat[:,0] + 1) / 2)
    
    np.savetxt(data_path + "/" + data_name + ".txt", dat_, delimiter = " ")
