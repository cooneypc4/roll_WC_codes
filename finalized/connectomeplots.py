#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 15:15:17 2023

Plots of connectome data

including
heatmaps of connectivity mats - 'plot_p2m_weights'
histograms of LR laterality preference - 'plot_LRwavg_dists'
swarmplot of influence strengths of PMNs = 'swarm_influs'
swarmplot of contralateral influences = 'swarm_contrain'

@author: PatriciaCooney
"""
#%% imports
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

#%%heatmap connectivity matrices
def plot_p2m_weights(mat_pm,outnum,outname,names,plotti,saveti):
    f,ax = plt.subplots()
    sb.heatmap(mat_pm,cmap = 'Blues')
    #option: sb.clustermap?
    ax.set(ylabel="MNs", xticks=np.arange(len(names)), xticklabels=names, yticks = outnum, yticklabels = outname, title = plotti)
    ax.set_xlabel('Left PMNs', loc = 'left')
    ax.set_xlabel('Right PMNs', loc = 'right')
    #plt.xticks(fontsize=10, rotation=0)
    for tick in ax.xaxis.get_majorticklabels():  # example for xaxis
        tick.set_fontsize(4)
    for tick in ax.yaxis.get_majorticklabels():  # example for xaxis
        tick.set_fontsize(6)
    #add line to show LR division of MNs and PMNs
    ax.vlines(len(names)/2, *ax.get_ylim(), color='k')
    ax.hlines(len(outname)/2, *ax.get_xlim(), color='k')


    plt.tight_layout()
    plt.show()
    f.savefig(saveti+'_LR_PtoM_weights.svg', format = 'svg', dpi = 1200)
       
#%% #plot hists for optional number of data groups, smushed into 3D array - iterate plot over 3rd dimension
#also iterate over # threshold lines to plot, depdneing on needs
def plot_LRwavg_dists(data,datlabs,thresh,plotti,saveti):
    f,ax = plt.subplots()
    colarr = ['c', 'm', 'b', 'r']
    for i,d in enumerate(data):
        cd = colarr[i]
        plt.hist(data[i], bins = 20, color = cd,  alpha = 0.6, label = datlabs[i])
    ax = plt.gca()
    for t in thresh:
        ax.vlines(t, *ax.get_ylim(), color='k', alpha = 0.8)
    ax.set(xlabel = 'Left vs. Right Synaptic Weight Average', ylabel = 'Frequency of PMNs')
    plt.legend()
    plt.title(plotti)
    
    f.savefig(saveti+'histogram_wavgLR.svg', format = 'svg', dpi = 1200)


#%%swarmplot of influs
def swarm_influs(influs,plotti,saveti):
    f,ax = plt.subplots()
    sb.swarmplot(data = influs,palette = "rocket", dodge=True)
    #add medianlines - show the medians bc run MWU test
    sb.boxplot(showmeans=False,
            meanline=False,
            meanprops={'color': 'k', 'ls': '-', 'lw': 2},
            whiskerprops={'visible': True},
            boxprops = {'facecolor':'none'},
            zorder=10,
            data=influs,
            showfliers=False,
            showbox=False,
            showcaps=True)
    
    ax.set(xlabel="Secondary to MN Influence per PMN",xticklabels=['E-->E','E-->I','I-->E','I-->I'],ylabel="Normalized Motor Influence",title = plotti)
    plt.savefig(saveti + ".svg")


#%%swarmplot of LR wavgs
def swarm_contrain(data,xticklabs,ylabtype,plotti,saveti):
    f,ax = plt.subplots()
    sb.swarmplot(data = data,palette = "rocket", dodge=True)
    #add medianlines - show the medians bc run MWU test
    sb.boxplot(showmeans=False,
            meanline=False,
            meanprops={'color': 'k', 'ls': '-', 'lw': 2},
            medianprops={'visible': True},
            whiskerprops={'visible': True},
            boxprops = {'facecolor':'none'},
            zorder=10,
            data=data,
            showfliers=False,
            showbox=False,
            showcaps=True)
    
    if ylabtype == 0:
        ylabel = "Number of Contralateral Inputs"
    elif ylabtype == 1:
        ylabel = "Proportion of Contralateral Inputs"
    elif ylabtype == 2:
        ylabel = "Proportion of Contra to Ipsi Influence"
    else:
        ylabel = "Mean Influence of Contralateral Inputs"
    
    ax.set(xlabel="PMN-MN Type",xticklabels=xticklabs, ylabel=ylabel,title = plotti)
    ax.set_ylim(0.2,0.8)
    plt.savefig(saveti + ".svg")
