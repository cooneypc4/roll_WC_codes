#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 14:54:30 2023

All plotting fxns of use for the larval model

including
activity plots = 'plotbothsidesheatmap'
external input plots = 'plot_extinput'
phase different heatmaps = 'heatmaps_allparams_allphis'

@author: PatriciaCooney
"""

#%%imports
# Imports
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import cmasher as cmr #custom colormap = cyclic

#%%plot heatmaps of model activity through time
def plotbothsidesheatmap(pars,n_t,rEms,pulse_vals,contra_weights,offsetcontra,contra_dur,perturb_input,EIprop,plotti,ti):
    f,(ax1,ax2) = plt.subplots(nrows=2)
    sb.heatmap(rEms[n_t,:,0].T,ax=ax1,cbar=True)
    ax1.set(ylabel="Segments", title = plotti)
    ax1.set_xticks([])
    ax1.tick_params(axis = 'y', labelrotation = 0)
    ax1.set_yticklabels(['A'+str(n+1) for n in np.arange(rEms.shape[1])])
    sb.heatmap(rEms[n_t,:,1].T,ax=ax2,cbar=True)
    ax2.set(xlabel="Timesteps", ylabel="Segments")
    ax2.tick_params(axis = 'y', labelrotation = 0)
    ax2.tick_params(axis = 'x', labelrotation = 60)
    ax2.set_yticklabels(['A'+str(n+1) for n in np.arange(rEms.shape[1])])

    I_mag = pulse_vals[0,0]
    I_dur = pulse_vals[0,1]
    inc = -0.1
    contra_names = ['LR-EE','LR-EI','LR-IE','LR-II'] 

    allstrs = [('L mag: ' + str(I_mag)), ('L dur: ' + str(I_dur*pars['dt']) + 's'), ('R mag: ' + str(round((I_mag*offsetcontra),2))),
               ('R dur: ' + str(I_dur*contra_dur*pars['dt'])), ('EIprop = '+str(EIprop))]
    #for loop looping list - textstr, plot, add inc
    for l in allstrs:
        plt.gcf().text(0.92, 0.92+inc, l, fontsize = 6)
        inc = inc-0.1
    for iw,w in enumerate(contra_weights):
        inc = inc-0.05
        textstr = contra_names[iw] + '=' + str(w)
        plt.gcf().text(0.92, 0.92+inc, textstr, fontsize = 6)

    #plt.tight_layout()
    plt.show()
    f.savefig((ti + str(n_t[0]) + str(n_t[-1]) + '_2side_multiseg_heatmaps.svg'), format = 'svg', dpi = 1200)
    f.savefig((ti + str(n_t[0]) + str(n_t[-1]) + '_2side_multiseg_heatmaps.png'), format = 'png', dpi = 1200)

#%%plot input to nodes -- test if level of noise seems appropriate
def plot_extinput(ext_input_E,ext_input_I,n_t,pulse_vals,noisevar,dt,ti,savetitle):
    fig = plt.figure()
    for s in np.arange(ext_input_E.shape[2]):
        for n in np.arange(ext_input_E.shape[1]):
            plt.plot(ext_input_E[n_t,n,s],label='E nodes')
            #plt.plot(ext_input_I[:,n,s],label='I nodes')

    plt.xlabel('Timesteps')
    plt.ylabel('External Input')
    plt.title('External Input + Noise to E nodes -' + ti)
    
    I_mag = pulse_vals[0,0]
    I_dur = pulse_vals[0,1]
    noise_sigma = noisevar*np.sqrt(dt)
    inc = -0.1

    allstrs = [('input mag: ' + str(I_mag)), 'input dur: ' + str(I_dur), 'noise_var: ' + str(noise_sigma)]
    #for loop looping list - textstr, plot, add inc
    for l in allstrs:
        plt.gcf().text(0.92, 0.92+inc, l, fontsize = 6)
        inc = inc-0.1
        
    #plt.tight_layout()
    plt.show()   
    fig.savefig(savetitle+'.svg', format = 'svg', dpi = 1200)
    fig.savefig(savetitle+'.png', format = 'png', dpi = 1200)


#%%heatmaps for phase diffs -- all in one plot -- interseg + contra, circular colormap
def heatmaps_allparams_allphis(meanphi_interseg_L,varphi_interseg_L,meanphi_interseg_R,varphi_interseg_R,meanphi_LR,varphi_LR,intersegprops,contraprops,stype,svector,savetitle):
    f,((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2)
    #reset the labels - flipped
    ilabels = [str(round(-i)) for i in intersegprops]
    clabels = [str(round(c)) for c in contraprops]
    
    #avg interseg phis for L and R, pull out nans to make diff color
    meanphi_interseg = np.mean([meanphi_interseg_L,meanphi_interseg_R],axis=0)
    varphi_interseg = np.mean([varphi_interseg_L,varphi_interseg_R],axis=0)
    
    #set linear colormap to match the ringsims
    cmap = cmr.copper_s
    cmap.set_bad('darkgray')
    
    #mask out the nan vals
    maskmeaninter = np.isnan(meanphi_interseg)
    maskvarinter = np.isnan(varphi_interseg)
    maskmeancontra = np.isnan(meanphi_LR)
    maskvarcontra = np.isnan(varphi_LR)

    #heatmaps - interseg and contra -- interseg is row 1, contra is row 2
    sb.heatmap(meanphi_interseg,ax=ax1,cmap=cmap,mask=maskmeaninter,vmin=0,vmax=1)
    sb.heatmap(varphi_interseg,ax=ax2,cmap=cmap,mask=maskvarinter,vmin=0,vmax=1)
    sb.heatmap(meanphi_LR,ax=ax3,cmap=cmap,mask=maskmeancontra,vmin=0,vmax=1)
    sb.heatmap(varphi_LR,ax=ax4,cmap=cmap,mask=maskvarcontra,vmin=0,vmax=1)
    
    #set axis labels
    ax1.set_xticks([])
    ax2.set_xticks([])
    ax1.set_xticklabels("")
    ax2.set_xticklabels("")
    ax3.set_xticklabels(clabels[::2],fontsize=7)
    ax4.set_xticklabels(clabels[::2],fontsize=7)
    ax1.set_yticklabels(ilabels[::3],fontsize=7)
    ax2.set_yticklabels(ilabels[::3],fontsize=7)
    ax3.set_yticklabels(ilabels[::3],fontsize=7)
    ax4.set_yticklabels(ilabels[::3],fontsize=7)
    
    if stype == 0:
        tist = "Crawl - "
    elif stype == 1:
        tist = "Roll - "
        
    #set axes according to the sweep type performed
    #svector [wEIadj, wEEcontra, wEIcontra, wIEcontra, wIIcontra]
    #reset since the varying changes svector vals
    svector[abs(svector[:])>5]=1
    if svector[0] == 1:
        yvar = "wEsEs"
        if svector[1] == 1:
            xvar = "wIsEg"
        if svector[3] == 1:
            xvar = "wIsIs"
    elif svector[0] > 1:
        yvar = "wIsEg"
        if svector[2] == 1:
            xvar = "wEgEs"
        elif svector[3] == 1:
            xvar = "wIsIs"

    ax1.set(xlabel="", title = tist + r"$<\phi>_{interseg}$")
    ax2.set(xlabel="", ylabel="",  title = r"$\sigma^2_{interseg}$")
    ax3.set(xlabel=xvar, ylabel=yvar, title = tist + r"$<\phi>_{contra}$")
    ax4.set(xlabel=xvar, ylabel=str(svector), title = r"$\sigma^2_{contra}$")
    
    f.tight_layout(pad=0.2)
    plt.show()
    f.savefig(savetitle+'.svg', format = 'svg', dpi = 1200)
    f.savefig(savetitle+'.png', format = 'png', dpi = 1200)
