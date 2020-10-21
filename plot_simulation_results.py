# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 14:38:06 2020

@author: Lara Vomfell

This file collects all simulation results (epsilon-contamination and 
zero-inflation) and produces the three-panel figures in our paper. 
Left: absolute difference
Middle: absolute error
Right: predictive likelihods

In the appendix, we also display the results for a standard (non-bootstrapped) 
Bayesian procedure using pystan. These results can be additionally generated
by specifying 'bayes = True below'

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def result_panels(result_path, xlab, figname, setting, bayes= True):
    
    def result_boxplot(ax, result_type, result_path, param, x, xlab, ylab, ymax, bayes = True):     
        def get_bootstrap_means(model, result_type, result_path, param):
            """ 
            This function loads the results from .csv files and calculates
            the means over j=1, ..., B. 
            We exclude results > 100 because the optimization clearly did not 
            converge.
            """
            l = []
            for p in param:
                result = pd.read_csv(result_path + '/' + p + '_' + result_type + '_' + model + '.csv',
                                     delimiter = ',', header = None).to_numpy()
                
                if result.shape[1] > 1:
                    result = result[:,1]
                    
                if model == 'stan':
                    size = 4000
                else:
                    size = 1000
                    
                result[np.where(result > 100)[0]] = np.nan
                vec = [np.nanmean(result[i:i+size]) for i in range(0, len(result), size)]
                l.append(vec)
                
            return l
    
        # standard main paper plots
        if not bayes:
                
            # get the tvd and kld results
            tvd = get_bootstrap_means('tvd', result_type, result_path, param)
            kld = get_bootstrap_means('kld', result_type, result_path, param)
            
            # the predictive likelihood results are on the log-scale,
            # exponentiate for plots
            if result_type == 'loglik':
                tvd = [np.exp(tvd[i]) for i in range(len(tvd))]
                kld = [np.exp(kld[i]) for i in range(len(kld))]
        
            
            # show dodged boxplots with corresponding colors
            btvd = ax.boxplot(tvd, positions = np.array(range(len(tvd))) * 3 - 0.4,
                               sym = '', widths = 0.6, patch_artist = True,
                               boxprops = dict(facecolor = '#009E73'),
                               medianprops = dict(color = 'black'))
            bkld = ax.boxplot(kld, positions = np.array(range(len(kld))) * 3 + 0.4,
                               sym = '', widths = 0.6, patch_artist = True,
                               boxprops = dict(facecolor = '#56B4E9'),
                               medianprops = dict(color = 'black'))
        
            # add legend
            ax.legend([btvd["boxes"][0], bkld["boxes"][0]], ['TVD', 'KLD'], 
                      loc=2,frameon = False)
        
            # add a little extra padding on y-axis for legend box if
            # we're showing likelihoods
            if result_type == 'loglik':
                ax.set_ylim(top = ymax)
            # add axis 
            ax.set_xticks(range(0, len(x) * 3, 3))
            ax.set_xticklabels(x)
            ax.set_xlim(-1.5, len(x)*3 - 1.5)
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
            ax.set_title(ylab)
            
            return ax
        
        # Bayesian plot otherwise
        else:
            # all results                   
            tvd = get_bootstrap_means('tvd', result_type, result_path, param)
            kld = get_bootstrap_means('kld', result_type, result_path, param)
            stan = get_bootstrap_means('stan', result_type, result_path, param)
            
            # the predictive likelihood results are on the log-scale,
            # exponentiate for plots
            if result_type == 'loglik':
                tvd = [np.exp(tvd[i]) for i in range(len(tvd))]
                kld = [np.exp(kld[i]) for i in range(len(kld))]
                stan = [np.exp(stan[i]) for i in range(len(stan))]
            
            # show dodged boxplots with corresponding colors
            btvd = ax.boxplot(tvd, positions = np.array(range(len(tvd))) * 3 - 0.8,
                               sym = '', widths = 0.6, patch_artist = True,
                               boxprops = dict(facecolor = '#009E73'),
                               medianprops = dict(color = 'black'))
            bkld = ax.boxplot(kld, positions = np.array(range(len(kld))) * 3,
                               sym = '', widths = 0.6, patch_artist = True,
                               boxprops = dict(facecolor = '#56B4E9'),
                               medianprops = dict(color = 'black'))
            bstan = ax.boxplot(stan, positions = np.array(range(len(stan))) * 3 + 0.8,
                               sym = '', widths = 0.6, patch_artist = True,
                               boxprops = dict(facecolor = '#E69F00'),
                               medianprops = dict(color = 'black'))
            
    
            # add legend
            ax.legend([btvd["boxes"][0], bkld["boxes"][0], bstan['boxes'][0]], 
                      ['TVD', 'KLD', 'Bayes'], 
                      loc=2,frameon = False)
            
            # add a little extra padding on y-axis for legend box if
            # we're showing likelihoods
            if result_type == 'loglik':
                ax.set_ylim(top = ymax + 0.1)
            
            # add axis labels
            ax.set_xticks(range(0, len(x) * 3, 3))
            ax.set_xticklabels(x)
            ax.set_xlim(-1.5, len(x)*3 - 1.5)
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
            ax.set_title(ylab)
            
            return ax
    
        
    # list all files in result dir
    files = os.listdir(result_path)
    # drop all the ones not starting with a number 
    # (in case there are extra files floating around)
    pars = []
    for i in files:
        if i[0].isdigit():
            pars.append(i[0:2])
    pars = np.unique(pars)
    
    # extract x-axis labels from pars
    x = pars.astype('int')
    
    # x should be a share if we're looking at zero-inflations
    if setting == 'Zero-inflation':
        x = x/100
        ymax = 0.8
    else:
        ymax = 0.3
    
    # define three plot panels
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10, 4))
    
    # fill the subplot axes
    result_boxplot(ax1, 'param', result_path, pars, x, xlab, 
                   'Absolute difference', ymax, bayes = False)
    result_boxplot(ax2, 'ae', result_path, pars, x, xlab, 
                   'Absolute error', ymax, bayes = False)
    result_boxplot(ax3, 'loglik', result_path, pars, x, xlab, 
                   'Predictive likelihoods', ymax, bayes = False)
        
    # save result
    fig.tight_layout()
    fig.savefig(figname)

    # if we also want the appendix plots, redo the whole thing including Stan results
    if bayes:
        # define three plot panels
        fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10, 4))
    
        # fill the subplot axes
        result_boxplot(ax1, 'param', result_path, pars, x, xlab, 
                       'Absolute difference', ymax, bayes = True)
        result_boxplot(ax2, 'ae', result_path, pars, x, xlab, 
                       'Absolute error', ymax, bayes = True)
        result_boxplot(ax3, 'loglik', result_path, pars, x, xlab, 
                       'Predictive likelihoods', ymax, bayes = True)
    
        fig.tight_layout()
        fig.savefig(figname + '_stan')
        
        
# create three-panel figures for each simulation type:
# specify result_path
zeroinfl_path = 'data/sim_zeroinfl'
result_panels(result_path = zeroinfl_path, 
              xlab = 'Proportion of zero-inflation', 
              figname = 'figures/zero_inflation', 
              setting = 'Zero-inflation', 
              bayes = True)

eps_path = 'data/sim_eps'
result_panels(result_path = eps_path, 
              xlab = 'k', 
              figname = 'figures/epsilon', 
              setting = 'Epsilon contamination', 
              bayes = True)

