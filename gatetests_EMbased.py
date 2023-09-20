#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 15:05:59 2023

@author: PatriciaCooney
"""

# Imports
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import cmasher as cmr #custom colormap = cyclic

#%% plotting fxns
#plot heatmaps through time
def plotbothsidesheatmap(n_t,rEms,pulse_vals,contra_weights,offsetcontra,contra_dur,perturb_input,EIprop,plotti,ti):
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


#fxn for plotting the LR phase diff with diff params
def heatmap_params_LRphis(meanphi_LR,varphi_LR,intersegprops,contraprops,stype,svector,savetitle):
    f,[ax1,ax2] = plt.subplots(ncols=2)
    ilabels = intersegprops
    clabels = contraprops
    # ilabels = [str(round(i,2)) for i in intersegprops]
    # clabels = [str(round(c,2)) for c in contraprops]
    #pull out nans to make diff color
    cmap = plt.cm.get_cmap("rocket").copy()
    cmap.set_bad('darkgray')
    maskmean = np.isnan(meanphi_LR)
    maskvar = np.isnan(varphi_LR)
    sb.heatmap(meanphi_LR,ax=ax1,cmap=cmap,mask=maskmean,vmin=0,vmax=0.5)
    sb.heatmap(varphi_LR,ax=ax2,cmap=cmap,mask=maskvar,vmin=0,vmax=0.1)
    if len(contraprops)>10:
        ax1.set_xticks(np.arange(0,len(contraprops),2))
        ax2.set_xticks(np.arange(0,len(contraprops),2))
        ax1.set_yticklabels(ilabels,fontsize=8)
        ax1.set_xticklabels(clabels[::2],fontsize=8)
        ax2.set_yticklabels(ilabels,fontsize=8)
        ax2.set_xticklabels(clabels[::2],fontsize=8)
    else:
        ax1.set_yticklabels(ilabels)
        ax1.set_xticklabels(clabels)
        ax2.set_yticklabels(ilabels)
        ax2.set_xticklabels(clabels)
    if stype == 0:
        tist = "Crawl - "
    elif stype == 1:
        tist = "Roll - "
    
    #set axes according to the sweep type performed
    #svector [wEIadj, wEEcontra, wEIcontra, wIEcontra, wIIcontra]
    svector[abs(svector[:])>5]=1
    if svector[0] == 1:
        yvar = "wEEadj - wEIadj"
        if svector[1] == 5 and svector[3]==1:
            xvar = "wEEcontra - wIEcontra"
        elif svector[1] == 5 and svector[2]==1:
            xvar = "wEEcontra - wEIcontra"
        elif svector[3] == 5:
            xvar = "wIEcontra + wEEcontra"
        elif svector[4] == -5:
            xvar = "wIIcontra - wEIcontra"
    if svector[0] == 0: 
        if svector[1] == 1:
            yvar = "wEEcontra"
            if svector[2] == 1:
                xvar = "wEIcontra"
            elif svector[3] == 1:
                xvar = "wIEcontra"
        if svector[2] == 1 and svector[3] == 1:
            yvar = "wEIcontra"
            xvar = "wIEcontra"
        if svector[3] == 1 and svector[4] == 1:
            yvar = "wIEcontra"
            xvar = "wIIcontra"
    
    ax1.set(xlabel=xvar, ylabel=yvar, title = tist + r"$<\phi>_{contra}$")
    ax2.set(xlabel=xvar, ylabel=str(svector),  title = r"$\sigma^2_\phi - contra$")
 
    plt.show()
    f.savefig(savetitle+'.svg', format = 'svg', dpi = 1200)
    f.savefig(savetitle+'.png', format = 'png', dpi = 1200)


#fxn for plotting the interseg phase diff with diff params
def heatmap_params_intersegphis(meanphi_interseg_L,varphi_interseg_L,meanphi_interseg_R,varphi_interseg_R,intersegprops,contraprops,stype,svector,savetitle):
    f,((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2)
    ilabels = intersegprops
    clabels = contraprops
    # ilabels = [str(round(i,2)) for i in intersegprops]
    # clabels = [str(round(c,2)) for c in contraprops]
    #pull out nans to make diff color
    cmap = plt.cm.get_cmap("rocket").copy()
    cmap.set_bad('darkgray')
    maskmean1 = np.isnan(meanphi_interseg_L)
    maskvar1 = np.isnan(varphi_interseg_L)
    maskmean2 = np.isnan(meanphi_interseg_R)
    maskvar2 = np.isnan(varphi_interseg_R)
    
    sb.heatmap(meanphi_interseg_L,ax=ax1,cmap=cmap,mask=maskmean1,vmin=0,vmax=0.5)
    sb.heatmap(varphi_interseg_L,ax=ax2,cmap=cmap,mask=maskvar1,vmin=0,vmax=0.1)
    sb.heatmap(meanphi_interseg_R,ax=ax3,cmap=cmap,mask=maskmean2,vmin=0,vmax=0.5)
    sb.heatmap(varphi_interseg_R,ax=ax4,cmap=cmap,mask=maskvar2,vmin=0,vmax=0.1)
    if len(contraprops)>=10:
        # ax1.set_xticks(np.arange(0,len(contraprops),2))
        # ax2.set_xticks(np.arange(0,len(contraprops),2))
        ax3.set_xticks(np.arange(0,len(contraprops),2))
        ax4.set_xticks(np.arange(0,len(contraprops),2))
        
        # ax1.set_xticklabels(clabels[::2],fontsize=7)
        # ax2.set_xticklabels(clabels[::2],fontsize=7)
        ax3.set_xticklabels(clabels[::2],fontsize=7)
        ax4.set_xticklabels(clabels[::2],fontsize=7)
    else:
        ax1.set_xticklabels(clabels)
        ax2.set_xticklabels(clabels)
    if len(intersegprops)>7:
        ax1.set_yticklabels(ilabels[::3],fontsize=7)
        ax2.set_yticklabels(ilabels[::3],fontsize=7)
        ax3.set_yticklabels(ilabels[::3],fontsize=7)
        ax4.set_yticklabels(ilabels[::3],fontsize=7)
    else:
        ax1.set_yticklabels(ilabels,fontsize=7)
        ax2.set_yticklabels(ilabels,fontsize=7)
        ax3.set_yticklabels(ilabels,fontsize=7)
        ax4.set_yticklabels(ilabels,fontsize=7)
    
    if stype == 0:
        tist = "Crawl - "
    elif stype == 1:
        tist = "Roll - "
        
    #set axes according to the sweep type performed
    #svector [wEIadj, wEEcontra, wEIcontra, wIEcontra, wIIcontra]
    svector[abs(svector[:])>5]=1
    if svector[0] == 1:
        yvar = "wEEadj - wEIadj"
        if svector[1] == 5 and svector[3]==1:
            xvar = "wEEcontra - wIEcontra"
        elif svector[1] == 5 and svector[2]==1:
            xvar = "wEEcontra - wEIcontra"
        elif svector[3] == 5:
            xvar = "wIEcontra + wEEcontra"
        elif svector[4] == -5:
            xvar = "wIIcontra - wEIcontra"
    if svector[0] == 0: 
        if svector[1] == 1:
            yvar = "wEEcontra"
            if svector[2] == 1:
                xvar = "wEIcontra"
            elif svector[3] == 1:
                xvar = "wIEcontra"
        if svector[2] == 1 and svector[3] == 1:
            yvar = "wEIcontra"
            xvar = "wIEcontra"
        if svector[3] == 1 and svector[4] == 1:
            yvar = "wIEcontra"
            xvar = "wIIcontra"
            
    ax1.set(xlabel="", title = tist + r"$<\phi>_{interseg}$")
    ax2.set(xlabel="", ylabel="",  title = r"$\sigma^2_\phi - interseg$")
    ax3.set(xlabel=xvar, ylabel=yvar)
    ax4.set(xlabel=xvar, ylabel=str(svector))
 
    plt.show()
    f.savefig(savetitle+'.svg', format = 'svg', dpi = 1200)
    f.savefig(savetitle+'.png', format = 'png', dpi = 1200)


#new heatmaps for phase diffs -- all in one plot -- interseg + contra
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
    #rescale phase diffs from 0 to pi instead of 0 to 0.5
    # meanphi_interseg = rescalepi(meanphi_interseg)
    # varphi_interseg = rescalepi(varphi_interseg)
    # meanphi_LR = rescalepi(meanphi_LR)
    # varphi_LR = rescalepi(varphi_LR)
    
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




#plot input to nodes -- test if level of noise seems appropriate
def plot_extinput(ext_input_E,ext_input_I,n_t,pulse_vals,noisevar,dt,ti,savetitle):
    fig = plt.figure()
    for s in np.arange(I_ext_E.shape[2]):
        for n in np.arange(I_ext_E.shape[1]):
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


#%% functions for setting up the model - param setting, nonlinearity (FI curve) - sigmoid, derivative of activation function
#setup the default parameters to match Gjorgjieva et al., 2013
def default_pars(**kwargs):
  pars = {}

  # Excitatory parameters
  pars['tau_E'] = 0.5     # Timescale of the E population [ms]
  pars['b_E'] = 1.3      # Gain of the E population
  pars['theta_E'] = 4  # Threshold of the E population
  pars['kmax_E'] = 0.9945   #max f-i value; max possible proportion of population to be active at once

  # Inhibitory parameters
  pars['tau_I'] = 0.5    # Timescale of the I population [ms]
  pars['b_I'] = 2      # Gain of the I population
  pars['theta_I'] = 3.7  # Threshold of the I population
  pars['kmax_I'] = 0.9994  #max f-i value; max possible proportion of population to be active at once

  # Connection strength - self and AP interseg
  pars['wEEself'] = 16.   # E to E
  pars['wEEadj'] = 20.   # E to E
  pars['wEIself'] = -12.   # I to E
  pars['wEIadj'] = -20.   # I to E
  pars['wIEself'] = 15.  # E to I
  pars['wIIself'] = -3.  # I to I

  # External input
  pars['I_ext_E'] = 1.7
  #pars['I_ext_I'] = 0.

  # simulation parameters
  pars['T'] = 100.        # Total duration of simulation [ms]
  pars['dt'] = .1        # Simulation time step [ms]
  pars['rE_init'] = 0.017  # Initial value of E
  pars['rI_init'] = 0.011 # Initial value of I
  pars['n_segs'] = 8
  pars['rest_dur'] = 1

  # External parameters if any
  for k in kwargs:
      pars[k] = kwargs[k]

  # Vector of discretized time points [ms]
  pars['range_t'] = np.arange(0, pars['T'], pars['dt'])

  return pars

#sigmoid
def G(x, b, theta):
  """
  Population activation function, F-I curve -- sigmoidal

  Args:
    x     : the population input
    b     : the gain of the function
    theta : the threshold of the function
    
  Returns:
    g     : the population activation response f(x) for input x
  """

  # add the expression of f = F(x)  
  g = (1 + np.exp(-b * (x - theta)))**-1 - (1 + np.exp(b * theta))**-1

  return g

#%% Peak-finding and phase calculation functions
#removed previous code from crawl_and_roll_singleseg-multiseg-2sides.py that calc'd, stored the ISI, contractiondur
#just use phase diff calc here to get interseg and LR phase diffs
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def motor_output_check(n_t,E,pulse_vals,c_thresh,titype):
    #find contraction onset and dur by checking drE vals
    segx = np.arange(E.shape[1])
    side = np.arange(E.shape[2])

    #track contraction start and end per seg and per side - catch partial waves
    if np.where(E[:,:,:] > c_thresh)[0].size>0:
        left_starts = -np.ones([E.shape[1],100])*np.inf
        left_ends = -np.ones([E.shape[1],100])*np.inf
        totalwaves = 0 #update in each step below to capture max num waves, all segs, both sides
        suprathresh_left = np.where(E[:,:,0] > c_thresh, 1, 0)
        for seg in segx:
            left_ws = np.where(np.diff(suprathresh_left[:,seg],axis=0)==1)[0]
            left_we = np.where(np.diff(suprathresh_left[:,seg],axis=0)==-1)[0]
            if left_we.size < left_ws.size:
                left_we = np.hstack([left_we,int(E.shape[0])])
            num_waves = left_ws.shape[0]
            nwends = left_we.shape[0]
            if num_waves > totalwaves:
                totalwaves = num_waves
            if num_waves > nwends:
                wdiff = totalwaves - num_waves
                if wdiff > 0:
                    left_ws = np.concatenate((np.reshape(left_ws,[left_ws.shape[0],-1]),np.zeros([left_starts.shape[0],wdiff])),1)
                elif wdiff < 0:
                    left_starts = np.concatenate(((left_starts,np.zeros([left_starts.shape[0],wdiff]))),1)
            elif nwends > num_waves:
                totalwaves = nwends
                wdiff = totalwaves - nwends
                if wdiff > 0:
                    left_we = np.concatenate((np.reshape(left_we,[left_we.shape[0],-1]),np.zeros([left_starts.shape[0],wdiff])),1)
                elif wdiff < 0:
                    left_ends = np.concatenate(((left_ends,np.zeros([left_ends.shape[0],wdiff]))),1)   
            left_starts[seg,:num_waves] = left_ws+1
            left_ends[seg,:nwends] = left_we
        if side.size>1:
            right_starts = -np.ones([E.shape[1],100])*np.inf
            right_ends = -np.ones([E.shape[1],100])*np.inf
            suprathresh_right = np.where(E[:,:,1] > c_thresh, 1, 0)
            for seg in segx:
                right_ws = np.where(np.diff(suprathresh_right[:,seg],axis=0)==1)[0]
                right_we = np.where(np.diff(suprathresh_right[:,seg],axis=0)==-1)[0]
                if right_we.size < right_ws.size:
                    right_we = np.hstack([right_we,int(E.shape[0])])
                num_waves = right_ws.shape[0]
                nwends = right_we.shape[0]
                
                if num_waves > totalwaves:
                    totalwaves = num_waves
                if num_waves > nwends:
                    wdiff = totalwaves - num_waves
                    if wdiff > 0:
                        right_ws = np.concatenate((np.reshape(right_ws,[right_ws.shape[0],-1]),np.zeros([right_starts.shape[0],wdiff])),1)
                    elif wdiff < 0:
                        right_starts = np.concatenate(((right_starts,np.zeros([right_starts.shape[0],wdiff]))),1)
                elif nwends > num_waves:
                    totalwaves = nwends
                    wdiff = totalwaves - nwends
                    if wdiff > 0:
                        right_we = np.concatenate((np.reshape(right_we,[right_we.shape[0],-1]),np.zeros([right_starts.shape[0],wdiff])),1)
                    elif wdiff < 0:
                        right_ends = np.concatenate(((right_ends,np.zeros([right_ends.shape[0],wdiff]))),1)
                right_starts[seg,:num_waves] = right_ws+1
                right_ends[seg,:nwends] = right_we
            cstart = np.dstack((left_starts[:,0:totalwaves],right_starts[:,0:totalwaves]))*pars['dt']
            cend = np.dstack((left_ends[:,0:totalwaves],right_ends[:,0:totalwaves]))*pars['dt']
        else:
            cstart = left_starts[:,0:totalwaves]*pars['dt']
            cend = left_ends[:,0:totalwaves]*pars['dt']

        #do phase diff for 2-sided system
        if cstart.shape[1]>1:
            if side.size>1:
                phasediff_LR = -np.ones([cstart.shape[1]-2,cstart.shape[0]-1])*np.inf
                mean_phasediff_LR = -np.ones(cstart.shape[1]-2)*np.inf
                phasediff_interseg = -np.ones([cstart.shape[1]-2,cstart.shape[0]-1,2])*np.inf
                mean_phasediff_interseg = -np.ones([cstart.shape[1]-2,2])*np.inf
            #wave
            for wa in np.arange(cstart.shape[1]-2):
                #seg
                for seg in segx:
                    if seg < cstart.shape[0]-1: 
                        #interseg comparison each side
                        for si in side:
                            #phase diff
                            wavecurr = cstart[seg,wa,si]
                            #print('currentwave '+str(wavecurr))
                            compwavenext = cstart[seg,wa+1,si]
                            #print('next wave in same seg '+str(compwavenext))
                            comparray_seg = cstart[seg+1,:,si] #compare to all possible waves in neighboring segs
                            adjval = find_nearest(comparray_seg, wavecurr)
                            #print('closest wave in adj seg '+str(adjval))
                            phasediff_interseg[wa,seg,si] = abs(wavecurr - adjval)/abs(wavecurr - compwavenext)
                            #print(phasediff_interseg[wa,seg,si])
                            
                            #LR comparison
                            if si == 0:
                                comparray_side = cstart[seg,:,1] #compare to all possible waves in this seg on the contralateral side to find nearest
                                adjval = find_nearest(comparray_side, wavecurr)
                                #print('closest wave in contra seg '+str(adjval))
                                phasediff_LR[wa,seg] = abs(wavecurr - adjval)/abs(wavecurr - compwavenext)
                                #print(phasediff_LR[wa,seg])
                #mean across segs - interseg phi and contra phi    
                mean_phasediff_interseg[wa,0] = np.nanmean(phasediff_interseg[wa,:,0]) #mean, remove the weird entries
                mean_phasediff_interseg[wa,1] = np.nanmean(phasediff_interseg[wa,:,1]) #mean, remove the weird entries
                mean_phasediff_LR[wa] = np.nanmean(phasediff_LR[wa,:]) #mean, remove the weird entries
                # print(mean_phasediff_interseg[wa,0])
                # print(mean_phasediff_interseg[wa,1])
                # print(mean_phasediff_LR[wa])
                
            # else:
            #     #no phase diff to calculate in 1-sided system
            #     mean_phasediff_LR, phasediff_LR, mean_phasediff_interseg, phasediff_interseg = np.nan, np.nan, np.nan, np.nan
        else:
            #no phase diff because system --- traveling front - stays elevated/saturated with activity
            mean_phasediff_LR, phasediff_LR, mean_phasediff_interseg, phasediff_interseg = np.nan, np.nan, np.ones([2])*np.nan, np.ones([2])*np.nan
        
    # else:
    #     cstart,cend,totalwaves = np.nan, np.nan, np.nan
    #     mean_phasediff_LR, phasediff_LR, mean_phasediff_interseg, phasediff_interseg = np.nan, np.nan, np.nan, np.nan
    
    # print(mean_phasediff_interseg)
    # print(mean_phasediff_LR)
    # print(phasediff_interseg)
    # print(phasediff_LR)
    
    return cstart, cend, totalwaves, mean_phasediff_LR, phasediff_LR, mean_phasediff_interseg, phasediff_interseg

#%% rescale all your phase diff phi's from 0 to pi for the colormapping
def rescalepi(inputarr):
    arrmin = np.min(inputarr)
    arrmax = np.max(inputarr)
    pinormarr = np.pi * ((inputarr - arrmin)/(inputarr - arrmax))
    
    return pinormarr

#%% Define noise term according to Euler Maruyama equation
def addnoise(noisemean, noisescale, dt):
    """Add Euler Maruyama noise term so noise scale is not impacted by length of simulation"""
    return noisescale*np.random.normal(noisemean,np.sqrt(dt))

#%% Simulate 2 sided WC EI eq's for multiple interconnected segments
def simulate_wc_multiseg(tau_E, b_E, theta_E, tau_I, b_I, theta_I,
                    wEEself, wEIself, wIEself, wIIself, wEEadj, wEIadj,
                    rE_init, rI_init, dt, range_t, kmax_E, kmax_I, n_segs, rest_dur, 
                    n_sides, sim_input, pulse_vals, EEadjtest, EIadjtest, contra_weights, offsetcontra, contra_dur, 
                    offsetcontra_sub, contra_dur_sub, perturb_init, 
                    perturb_input, noisemean, noisescale, **otherpars):
    """
    Simulate the Wilson-Cowan equations

    Args:
      Parameters of the Wilson-Cowan model
    
    Returns:
      rE1-8, rI1-8 (arrays) : Activity of excitatory and inhibitory populations
    """  
    # Initialize activity arrays
    Lt = range_t.size
    rEms_sides, rEms_gates, rIms, drEms_sides, drIms, drEms_gates, I_ext_E, I_ext_I = np.zeros([Lt+1,n_segs,n_sides]), np.zeros([Lt+1,n_segs,n_sides]), np.zeros([Lt+1,n_segs,n_sides]), np.zeros([Lt+1,n_segs,n_sides]), np.zeros([Lt+1,n_segs,n_sides]), np.zeros([Lt+1,n_segs,n_sides]), np.zeros([Lt,n_segs,n_sides]), np.zeros([Lt,n_segs,n_sides])
    
    #initialize E and I activity
    rIms[0,:,:] = rI_init #both
    rEms_sides[0,:,:] = rE_init #both
    rEms_gates[0,:,:] = rE_init

    if perturb_init[0] == 1:
        if perturb_init[1] == 1: #excitatory ipsi or contra
            rEms_sides[perturb_init[5]:perturb_init[5]+perturb_init[6],perturb_init[2]] = perturb_init[4]
        else:
            rIms[perturb_init[5]:perturb_init[5]+perturb_init[6],perturb_init[3],perturb_init[2]] = perturb_init[4]
    
    #setup external input mat
    if sim_input == 0:
        #just posterior seg input - crawl
        #print('crawling')
        I_ext_E[:,n_segs-1,0] = np.concatenate((np.zeros(rest_dur), pulse_vals[0,0] * np.ones(int(pulse_vals[0,1])), np.zeros(Lt-int(pulse_vals[0,1])-rest_dur))) #ipsi
    if sim_input == 0 and n_sides>1:
        I_ext_E[:,n_segs-1,1] = np.concatenate((np.zeros(rest_dur), round(pulse_vals[0,0] * offsetcontra,2) * np.ones(int(pulse_vals[0,1]*contra_dur)), 
                                                              np.zeros(Lt-int(pulse_vals[0,1]*contra_dur)-rest_dur))) #contra
    elif sim_input == 1: #simultaneous drive
        #print('rolling')
        if pulse_vals[0,2] == 0: #tonic input, single pulse
            I_ext_E[:,:,0] = np.repeat(np.reshape(np.concatenate((np.zeros(rest_dur), pulse_vals[0,0] * np.ones(int(pulse_vals[0,1])), np.zeros(Lt-int(pulse_vals[0,1])-rest_dur))),[Lt,1]),n_segs,axis=1) #ipsi
            if n_sides>1:
                #print('side2')
                I_ext_E[:,:,1] = np.repeat(np.reshape(np.concatenate((np.zeros(rest_dur), round(pulse_vals[0,0] * offsetcontra_sub,2) * np.ones(int(pulse_vals[0,1]*contra_dur_sub)),
                                                                      round(pulse_vals[0,0] * offsetcontra,2) * np.ones(int((pulse_vals[0,1]*contra_dur))), 
                                                                      np.zeros(Lt-int((pulse_vals[0,1]*(contra_dur+contra_dur_sub)))-rest_dur))),[Lt,1]),n_segs,axis=1) #contra
        else: #alternating sine waves
            sine_ipsi = np.sin(np.linspace(0,60*np.pi*pulse_vals[0,1],Lt))
            sine_contra = -np.sin(np.linspace(0,60*np.pi*pulse_vals[0,1],Lt))
            I_ext_E[:,:,0] = np.repeat(np.reshape(np.where(sine_ipsi>0, sine_ipsi*pulse_vals[0,0], 0),[Lt,1]),n_segs,axis=1)
            if n_sides>1:
                I_ext_E[:,:,1] = np.repeat(np.reshape(np.where(sine_contra>0, sine_contra*round(pulse_vals[0,0]*offsetcontra), 0),[Lt,1]),n_segs,axis=1)
    
    #perturb_input = [1, sign, 0, sloop, inputl, rest_dur+1, pulse-rest_dur] #yesno, I E or both, ipsi contra or both, seg, mag, time of onset, duration of input
    if perturb_input[0] == 1:
        start = int(perturb_input[5])
        end = start + int(perturb_input[6])
        if perturb_input[1] == 1: #excitatory ipsi or contra
            if perturb_input[2] != 2:
                if 'all' in perturb_input:
                    I_ext_E[start:end,:,perturb_input[2]] = np.repeat(perturb_input[4] * (np.ones([int(end-start),1])),n_segs,axis=1)
                else:
                    I_ext_E[start:end,perturb_input[3],perturb_input[2]] = perturb_input[4] * (np.ones(int(end-start)))
            else:
                I_ext_E[:,perturb_input[3],0] = np.concatenate((np.zeros(perturb_input[5]), perturb_input[4] * np.ones(perturb_input[6]), 
                                                                                                                                np.zeros(Lt-int(perturb_input[5]+perturb_input[6]))))
                I_ext_E[:,perturb_input[3],1] = np.concatenate((np.zeros(perturb_input[5]), perturb_input[4] * np.ones(perturb_input[6]), 
                                                                                                                                np.zeros(Lt-int(perturb_input[5]+perturb_input[6]))))
        else: #inhib
            I_ext_I[:,perturb_input[3],perturb_input[2]] = np.concatenate((np.zeros(perturb_input[5]), perturb_input[4] * np.ones(perturb_input[6]), 
                                                                                                                            np.zeros(Lt-int(perturb_input[5]+perturb_input[6]))))
        
        #perturb_input = [1, 1, 1, 7, pars['I_ext_E']*2, rest_dur+110, 100] #yesno, E or I or both, ipsi contra or both, seg, mag, time of onset, duration of input
    
   #pull out the individual contra weights -- these are contra weights to the gate neurons now - NOT directly to the contralateral neurons
    wEsEs, wIsEg, wEgEs, wIsIs = contra_weights

    # Simulate the Wilson-Cowan equations
    for k in np.arange(Lt):
        for s in np.arange(n_sides):
            if s == 0 and n_sides==1:
                cs = 0
            elif s == 0:
                cs = 1
            else:
                cs = 0
            for seg in np.arange(n_segs):
                noise_s = addnoise(noisemean, noisescale, dt)
                # Calculate the derivative of the I population - same update for all seg
                drIms[k,seg,s] = dt / tau_I * (-rIms[k,seg,s] + (kmax_I - rIms[k,seg,s]) 
                                               * G((wIIself * rIms[k,seg,s] + wIEself * rEms_sides[k,seg,s]
                                                   + wIsEg * rEms_gates[k,seg,cs] + wIsIs * rIms[k,seg,cs] + I_ext_I[k,seg,s]), b_I, theta_I)) + noise_s
                
                #store noisy input
                I_ext_I[k,seg,s] = I_ext_I[k,seg,s] + noise_s
                
                if seg == n_segs-1:
                    noise_s = addnoise(noisemean, noisescale, dt)
                    #eq without +1 terms
                    # Calculate the derivative of the E population
                    drEms_sides[k,seg,s] = dt / tau_E * (-rEms_sides[k,seg,s] + (kmax_E - rEms_sides[k,seg,s]) * G(((EEadjtest*2 * rEms_sides[k,seg-1,s]) + (wEEself * rEms_sides[k,seg,s]) 
                                                                   + (EIadjtest*2 * rIms[k,seg-1,s]) + (wEIself * rIms[k,seg,s])
                                                                   + (wEsEs * rEms_sides[k,seg,cs]) + I_ext_E[k,seg,s]), b_E, theta_E)) + noise_s
                    #store noisy input
                    I_ext_E[k,seg,s] = I_ext_E[k,seg,s] + noise_s
                    
                elif seg == 0:
                    noise_s = addnoise(noisemean, noisescale, dt)
                    #eq without the i-1 terms
                    # Calculate the derivative of the E population
                    drEms_sides[k,seg,s] = dt / tau_E * (-rEms_sides[k,seg,s] + (kmax_E - rEms_sides[k,seg,s]) * G(((wEEself * rEms_sides[k,seg,s]) + (EEadjtest*2 * rEms_sides[k,seg+1,s]) 
                                                                       + (wEIself * rIms[k,seg,s]) + (EIadjtest*2 * rIms[k,seg+1,s]) 
                                                                       + (wEsEs * rEms_sides[k,seg,cs]) + I_ext_E[k,seg,s]), b_E, theta_E)) + noise_s
                    #store noisy input
                    I_ext_E[k,seg,s] = I_ext_E[k,seg,s] + noise_s
                    
                else: #seg a2-7
                    noise_s = addnoise(noisemean, noisescale, dt)
                    #the eq's for all mid segs
                    drEms_sides[k,seg,s] = dt / tau_E * (-rEms_sides[k,seg,s] + (kmax_E - rEms_sides[k,seg,s]) * G(((EEadjtest * rEms_sides[k,seg-1,s]) + (wEEself * rEms_sides[k,seg,s]) + (EEadjtest * rEms_sides[k,seg+1,s]) 
                                                                   + (EIadjtest * rIms[k,seg-1,s]) + (wEIself * rIms[k,seg,s]) + (EIadjtest * rIms[k,seg+1,s]) 
                                                                   + (wEsEs * rEms_sides[k,seg,cs]) + I_ext_E[k,seg,s]), b_E, theta_E)) + noise_s
                    #store noisy input
                    I_ext_E[k,seg,s] = I_ext_E[k,seg,s] + noise_s
                
                #gate E pop -- SAME for allsegs, both sides #NOTE - FOR RIGHT NOW, Recurrence at gate, consistent with assumptions of the WC model
                drEms_gates[k,seg,s] = dt / tau_E * (-rEms_gates[k,seg,s] + (kmax_E - rEms_gates[k,seg,s]) * G(((wEEself * rEms_gates[k,seg,s])+(wEgEs * rEms_sides[k,seg,s])), b_E, theta_E))
                
                # Update using Euler's method
                if rEms_sides[k+1,seg,s] == 0:
                    rEms_sides[k+1,seg,s] = float(rEms_sides[k,seg,s] + drEms_sides[k,seg,s])
                else:
                    rEms_sides[k+1,seg,s] = rEms_sides[k+1,seg,s]
                if rIms[k+1,seg,s] == 0:
                    rIms[k+1,seg,s] = float(rIms[k,seg,s] + drIms[k,seg,s])
                else:
                    rIms[k+1,seg,s] = rIms[k+1,seg,s]
                if rEms_gates[k+1,seg,s] == 0:
                    rEms_gates[k+1,seg,s] = float(rEms_gates[k,seg,s] + drEms_gates[k,seg,s])
                else:
                    rEms_gates[k+1,seg,s] = rEms_gates[k+1,seg,s]
        
    return rEms_sides, rEms_gates, rIms, I_ext_E, I_ext_I

#%% run simulations - just single value variables - good for single runs + Goro input type tests
n_t = np.arange(0,200)
simname = ['crawl', 'roll']
pars = default_pars()
n_sides = 2
pulse = 900 #this is for your regular input pulse NOT FOR YOUR PERTURBATIONs
alt = 0#0 #1 = alternating goro inputs in sine wave pattern
pulse_vals = np.array([[pars['I_ext_E'], pulse, alt]])
c_thresh = 0.3
noisemean = 0
noisescale = 0.005*pars['I_ext_E']
contra_weights = [0,0,0,0]

#setup sim type
sim = 1
sim_input = sim
if sim == 0: #crawl input
    offsetcontra = 1.1
    offsetcontra_sub = 0
    contra_dur_sub = 0
    contra_dur = 1
else: #roll input - commented out Goro tests
    offsetcontra = 1.1#0#0.5
    offsetcontra_sub = 0.05#0
    contra_dur_sub = 0.01#0
    contra_dur = 1-contra_dur_sub#0

#temp for roll/crawl plots
wEEadjtest = 20
wEIadjtest = -20
wEsEs = 5
wIsEg = 12
wEgEs = 2
wIsIs = -2


contra_weights = [wEsEs, wIsEg, wEgEs, wIsIs]

#no perturbations       
perturb_init = [0]
perturb_input = [0]

#run that sim
rEms_sides,rEms_gates,rIms,I_ext_E,I_ext_I = simulate_wc_multiseg(**default_pars(n_sides=n_sides, sim_input=sim_input,
                                                    pulse_vals=pulse_vals, EEadjtest=wEEadjtest, EIadjtest=wEIadjtest, contra_weights=contra_weights, 
                                                    offsetcontra = offsetcontra, contra_dur = contra_dur,
                                                    offsetcontra_sub = offsetcontra_sub, contra_dur_sub = contra_dur_sub,
                                                    perturb_init = perturb_init, perturb_input = perturb_input, noisemean = noisemean, noisescale = noisescale))

#plot the activity heatmap
plotbothsidesheatmap(n_t,rEms_sides,pulse_vals,contra_weights,offsetcontra,contra_dur,perturb_input,[],plotti = simname[sim],ti = simname[sim] + '_gateex'+ str(contra_weights))
#plot the external input used
plot_extinput(I_ext_E,I_ext_I,n_t,pulse_vals,noisescale,pars['dt'],ti = simname[sim],savetitle = simname[sim] + '_gateex'+ str(contra_weights))


#%% run over parameter space with noise many times - iterating different combos of interseg and contra weights
n_t = np.arange(0,200)
simname = ['crawl', 'roll']
pars = default_pars()
n_sides = 2
pulse = 500 #this is for your regular input pulse NOT FOR YOUR PERTURBATIONs
alt = 0 #1 = alternating goro inputs in sine wave pattern
pulse_vals = np.array([[pars['I_ext_E'], pulse, alt]])
c_thresh = 0.3

#set the noise params = mean, var for Gauss draw to test (white noise, nm=0,nv=0.1, made 0.1 b/c 1 = too high -- dominates frequencies - this is congruent with random draw init)
noisemean = 0
noisescale = 0.01*pars['I_ext_E']

#setup many runs and ranges



#PUT ALL VECS in
it_range = np.arange(0,100) #run 100 times then average
varyinter = np.arange(0,40,2)
varycontra = np.arange(0,20,1)
testlen = np.arange(len(varyinter))
yvals = -np.ones(len(testlen))
xvals = -np.ones(len(testlen))

#paramtestsweep - vectors for choosing which weights to fix vs. which to vary - each row is sweep type; each col is variable of interest
sweep = np.array([[1,1,0,0], [1,1,0,-5], [5,1,0,1], [5,1,1,0], [5,1,1,-5]])

#setup sim and go loop
simtype = [0,1]
for sim in simtype:
    sim_input = sim
    if sim == 0: #crawl input
        offsetcontra_sub = 0
        contra_dur_sub = 0
        offsetcontra = 1.1
        contra_dur = 1
    else: #roll input
        offsetcontra_sub = 0.05
        contra_dur_sub = 0.01
        offsetcontra = 1.1
        contra_dur = 1-contra_dur_sub
        
    wEEadj = pars['wEEadj']
    wEIadj = pars['wEIadj']
    
    #preset heatmap mats for storing the various interseg and contra phi values - for proportions plots - EE contra and EE interseg both fixed; vary EI vals and calculate proportion
    phi_propL = -np.ones([len(testlen), len(testlen)])
    phi_propR =  -np.ones([len(testlen), len(testlen)])
    phi_propcontra =  -np.ones([len(testlen), len(testlen)])
    varphi_propL = -np.ones([len(testlen), len(testlen)])
    varphi_propR =  -np.ones([len(testlen), len(testlen)])
    varphi_propcontra =  -np.ones([len(testlen), len(testlen)])
    
    #pick params
    for s in sweep:
        #print(s)
        svec = s
        varyiind = np.where(svec==1)[0][0]
        varyjind = np.where(svec==1)[0][1]
        for i in testlen:
            for j in testlen:
                #pull out the individual contra weights -- these are contra weights to the gate neurons now - NOT directly to the contralateral neurons
                #wEsEs, wIsEg, wEgEs, wIsIs = contra_weights
                contra_weights = svec
                #do the effective couple calc before reassigning weights to the variable values so that sum is correct since 2 variables in contra
                if any(contra_weights) > 1:
                    yvals[i] = str(np.sum(np.where(contra_weights)!=1) + varycontra[i])
                    xvals[j] = str(np.sum(np.where(contra_weights)!=1) + varycontra[j])
                else:
                    yvals[i] = str(i)
                    xvals[j] = str(j)
                #always wEsEs or wIsEg, so positive vary
                contra_weights[varyiind] = varycontra[i]
                #sometimes varyjind is for wIsIs, so need to make negative
                if varyjind == 3:
                    contra_weights[varyjind] = -varycontra[j]
                else:
                    contra_weights[varyjind] = varycontra[j] 
                        
                #print(contra_weights)
                
                #no perturbations       
                perturb_init = [0]
                perturb_input = [0]
                
                #run that sim
                rEms_sides, rEms_gates, rIms, I_ext_E, I_ext_I = simulate_wc_multiseg(**default_pars(n_sides=n_sides, sim_input=sim_input,
                                                                    pulse_vals=pulse_vals, EEadjtest=wEEadj, EIadjtest=wEIadj, contra_weights=contra_weights, 
                                                                    offsetcontra = offsetcontra, contra_dur = contra_dur,
                                                                    offsetcontra_sub = offsetcontra_sub, contra_dur_sub = contra_dur_sub,
                                                                    perturb_init = perturb_init, perturb_input = perturb_input, noisemean = noisemean, noisescale = noisescale))
                
                #plot the activity heatm
                #plotbothsidesheatmap(n_t,rEms,pulse_vals,contra_weights,offsetcontra,contra_dur,perturb_input,-i,plotti = simname[sim],ti = simname[sim] + '_TRUETEST_IE=5_Ipropinter=' + str(-i) + '_Ipropcontra=' + str(-j))
                
                #calc + store the phi's - interseg + LR
                cstart, cend, totalwaves, mean_phasediff_LR, phasediff_LR, mean_phasediff_interseg, phasediff_interseg = motor_output_check(n_t, rEms_sides, pulse_vals, c_thresh, titype = simname[sim] + '_')
                
                #take mean of the interseg and LR phase diffs across all "waves"
                phi_propcontra[i,j] = np.nanmean(mean_phasediff_LR)
                varphi_propcontra[i,j] = np.nanvar(mean_phasediff_LR)
                
                phi_propL[i,j] = np.nanmean(mean_phasediff_interseg)
                varphi_propL[i,j] = np.nanvar(mean_phasediff_interseg)
                phi_propR[i,j] = np.nanmean(mean_phasediff_interseg)
                varphi_propR[i,j] = np.nanvar(mean_phasediff_interseg)
                    
        # #plot the phi's over the whole param space
        filename = simname[sim]+'_sweep_'+str(s)+'_'+'x_'+'_noise['+str(noisemean)+','+str(noisescale)+']'+'_heatmap_allparams_gatedmodel_fixed5s_'
        
        heatmaps_allparams_allphis(phi_propL,varphi_propL,phi_propR,varphi_propR,phi_propcontra,varphi_propcontra,yvals,xvals,sim,s,filename)

        # save your output arrays
        np.save(filename + '_phi_contra', phi_propcontra)
        np.save(filename + '_varphi_contra', varphi_propcontra)
        np.save(filename + '_phi_L', phi_propL)
        np.save(filename + '_phi_R', phi_propR)
        np.save(filename + '_varphi_L', varphi_propL)
        np.save(filename + '_varphi_R', varphi_propR)
        
        