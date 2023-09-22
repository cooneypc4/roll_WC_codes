#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 15:55:37 2023
plots for ring simulations

including
activity traces each ring - 'plotbothmodovertime'
phase diffs each ring - 'plotbothphiovertime'
heatmaps of many iterations phase diffs in param space - ''

@author: PatriciaCooney
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import cmasher as cmr #custom colormap = cyclic
#%%plot both rings, node activity traces over time
def plotbothmodovertime(nodes,timesteps_local,x_freqs,y_freqs,omegaeach,coupling_within,coupling_between,plotti,ti):
    fig,ax = plt.subplots()
    for node in nodes:
        #subplots for each node
        plt.subplot(len(nodes), 1, node+1)
        plt.plot(timesteps_local,x_freqs[node,:]%(2*np.pi), color = "b")
        plt.plot(timesteps_local,y_freqs[node,:]%(2*np.pi), color = "m")
        plt.ylim([0,2*np.pi])
        for i in np.arange(x_freqs.shape[1]):
            if i > 0:
                #check x peaks
                p = (x_freqs[node,i]%(2*np.pi))
                o = (x_freqs[node,i-1]%(2*np.pi))
                if i < len(timesteps_local)-1:
                    q = (x_freqs[node,i+1]%(2*np.pi))
                    if p > o and p > q:
                        plt.axvline(timesteps_local[i],0,2*np.pi,color = 'g')
                #check y peaks
                p = (y_freqs[node,i]%(2*np.pi))
                o = (y_freqs[node,i-1]%(2*np.pi))
                if i < len(timesteps_local)-1:
                    q = (y_freqs[node,i+1]%(2*np.pi))
                    if p > o and p > q:
                        plt.axvline(timesteps_local[i],0,2*np.pi,color = 'y')

        if node == 0:
            plt.title(plotti)
        if node == len(nodes)/2:
            plt.ylabel('Nodes', fontsize=10)
        if node == nodes[-1]:
            plt.xlabel('Time', fontsize=10)
    
    nodenames = ['n'+str(nodes[i]) for i in nodes]
    allstrs = [('within coupling = ' + str(coupling_within)), ('between coupling = ' + str(coupling_between))]
    for i,n in enumerate(nodenames):
        allstrs.append(n + '=' + str(omegaeach[i]))
    #for loop looping list - textstr, plot, add inc
    inc=0
    for l in allstrs:
        plt.gcf().text(0.92, 0.92+inc, l, fontsize = 6)
        inc = inc-0.025

    #plt.tight_layout()
    plt.show()
    fig.savefig((ti+'_BOTH_activityovertime_redo.svg'), format = 'svg', dpi = 1200)
    fig.savefig((ti+'_BOTH_activityovertime_redo.png'), format = 'png', dpi = 1200)
    
#%% plot both ring phase diffs over time
def plotbothphiovertime(nodes,phi_xy,timesteps,omegaeach,coupling_within,coupling_between,plotti,ti):
    #one time plot but with diff colors per node pair
    fig,ax = plt.subplots()
    nodenames = ['n'+str(nodes[i]) for i in nodes]
    if phi_xy.shape[0]<=6:
        colorsn = ['m','b','c','g','y','r']
        for node in np.arange(phi_xy.shape[0]):
            #diff lines
            plt.plot(np.arange(len(phi_xy[0,:])), phi_xy[node,:]%2*np.pi, c = colorsn[node], alpha = 0.5)
    
    for node in np.arange(phi_xy.shape[0]):
        #diff lines
        plt.plot(np.arange(len(phi_xy[0,:])), phi_xy[node,:]%2*np.pi, color = 'k', alpha = 0.4)
    plt.plot(np.arange(len(phi_xy[0,:])), np.mean(phi_xy,0)%2*np.pi, color = 'k', alpha = 0.8, label = 'Average')
    
    plt.axhline(np.pi, color='r', ls='--')
    plt.axhline(2*np.pi, color='y', ls='--')
    plt.axhline(0, color='y', ls='--')
    plt.title(plotti)
    #axis labels etc
    plt.ylim([-0.2,2*np.pi+0.2])
    plt.ylabel(r"$\phi$")
    plt.xlabel('Time')
    
    nodenames = ['n'+str(nodes[i]) for i in nodes]
    allstrs = [('within coupling = ' + str(coupling_within)), ('between coupling = ' + str(coupling_between))]
    for ring in range(2):
        if ring == 0:
            inc=0
            omegar = omegaeach[0]
        elif ring == 1:
            inc = inc-0.05
            omegar = omegaeach[1]
            allstrs = []
        for i,n in enumerate(nodenames):
            allstrs.append(n + '=' + str(omegar[i]))
        #for loop looping list - textstr, plot, add inc
        for l in allstrs:
            plt.gcf().text(0.92, 0.92+inc, l, fontsize = 6)
            inc = inc-0.025

    #plt.tight_layout()
    plt.show()
    fig.savefig((ti+'_phasediffsBETWEEN_discret0.6_redo.svg'), format = 'svg', dpi = 1200)
    fig.savefig((ti+'_phasediffsBETWEEN_discret0.6_redo.png'), format = 'png', dpi = 1200) 
    
#%%heatmap plot of intra and interring phase diffs in one, like the larva plots
#also with cyclic colormap
def heatmaps_allparams_allphis(meanphix,varphix,meanphiy,varphiy,meanphixy,varphixy,within_range,between_range,stype,itype,plottitle):
    f,((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2)
    winlabels = [str(round(w,2)) for w in within_range]
    btlabels = [str(round(b,2)) for b in between_range]
    
    #avg interseg phis for L and R, pull out nans to make diff color
    meanphi_win = np.mean([meanphix,meanphiy],axis=0)
    varphi_win = np.mean([varphix,varphiy],axis=0)
    cmap = cmr.copper_s

    #heatmaps - interseg and contra -- interseg is row 1, contra is row 2
    sb.heatmap(meanphi_win,ax=ax1,vmin=0,vmax=2*np.pi,cmap=cmap,cbar_kws={"ticks":[0,round(np.pi,2),round(2*np.pi,2)]})
    sb.heatmap(varphi_win,ax=ax2,vmin=0,vmax=2*np.pi,cmap=cmap,cbar_kws={"ticks":[0,round(np.pi,2),round(2*np.pi,2)]})
    sb.heatmap(meanphixy,ax=ax3,vmin=0,vmax=2*np.pi,cmap=cmap,cbar_kws={"ticks":[0,round(np.pi,2),round(2*np.pi,2)]})
    sb.heatmap(varphixy,ax=ax4,vmin=0,vmax=2*np.pi,cmap=cmap,cbar_kws={"ticks":[0,round(np.pi,2),round(2*np.pi,2)]})
    
    if len(between_range)>10:
        ax1.set_xticks(np.arange(0,len(between_range),2))
        ax2.set_xticks(np.arange(0,len(between_range),2))
        ax3.set_xticks(np.arange(0,len(between_range),2))
        ax4.set_xticks(np.arange(0,len(between_range),2))
        
        # ax1.set_xticklabels(btlabels[::2],fontsize=7)
        # ax2.set_xticklabels(btlabels[::2],fontsize=7)
        ax3.set_xticklabels(btlabels[::2],fontsize=7)
        ax4.set_xticklabels(btlabels[::2],fontsize=7)
    else:
        ax1.set_xticklabels(btlabels)
        ax2.set_xticklabels(btlabels)
    if len(within_range)>7:
        ax1.set_yticklabels(winlabels[::2],fontsize=7)
        ax2.set_yticklabels(winlabels[::2],fontsize=7)
        ax3.set_yticklabels(winlabels[::2],fontsize=7)
        ax4.set_yticklabels(winlabels[::2],fontsize=7)
    else:
        ax1.set_yticklabels(winlabels,fontsize=7)
        ax2.set_yticklabels(winlabels,fontsize=7)
        ax3.set_yticklabels(winlabels,fontsize=7)
        ax4.set_yticklabels(winlabels,fontsize=7)
    
    if stype == 0:
        tist = "Wave - "
    elif stype == 1:
        tist = "Global Oscill - "
    if itype == 0:
        iti = "In-Phase"
    elif itype == 1:
        iti = "Quarter-Phase"
    elif itype == 2:
        iti = "Antiphase"
    ax1.set(xlabel="", ylabel="Coupling Within", title = tist + r"$<\phi>_{within}$")
    ax2.set(xlabel="", ylabel="",  title = r"$\sigma^2_{within}$" + iti)
    ax3.set(xlabel="Coupling Between", ylabel="Coupling Within", title = tist + r"$<\phi>_{between}$")
    ax4.set(xlabel="Coupling Between", ylabel="", title = r"$\sigma^2_{between}$" + iti)
 
    plt.show()
    f.savefig(plottitle+'_redo.svg', format = 'svg', dpi = 1200)
    f.savefig(plottitle+'_redo.png', format = 'png', dpi = 1200)