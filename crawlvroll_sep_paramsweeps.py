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
    #fig.savefig((ti + str(n_t) + '_2side_multiseg_traces.svg'), format = 'svg', dpi = 1200)
    f.savefig((ti + str(n_t[0]) + str(n_t[-1]) + '_2side_multiseg_heatmaps.png'), format = 'png', dpi = 1200)


#fxn for plotting the LR phase diff with diff params
def heatmap_params_LRphis(meanphi_LR,varphi_LR,intersegprops,contraprops,stype,plottitle):
    f,[ax1,ax2] = plt.subplots(ncols=2)
    ilabels = intersegprops
    clabels = contraprops
    ilabels = [str(round(i,2)) for i in intersegprops]
    clabels = [str(round(c,2)) for c in contraprops]
    sb.heatmap(meanphi_LR,ax=ax1)
    sb.heatmap(varphi_LR,ax=ax2)
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
    ax1.set(xlabel="EE Contra Coupling", ylabel="EI Contra Coupling", title = tist + r"$<\phi>_{contra}$")
    ax2.set(xlabel="EE Contra Coupling", ylabel="",  title = r"$\sigma^2_\phi - contra$")
 
    plt.show()
    f.savefig(plottitle+'.svg', format = 'svg', dpi = 1200)
    f.savefig(plottitle+'.png', format = 'png', dpi = 1200)


#fxn for plotting the interseg phase diff with diff params
def heatmap_params_intersegphis(meanphi_interseg_L,varphi_interseg_L,meanphi_interseg_R,varphi_interseg_R,intersegprops,contraprops,stype,plottitle):
    f,((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2)
    ilabels = intersegprops
    clabels = contraprops
    ilabels = [str(round(i,2)) for i in intersegprops]
    clabels = [str(round(c,2)) for c in contraprops]
    sb.heatmap(meanphi_interseg_L,ax=ax1)
    sb.heatmap(varphi_interseg_L,ax=ax2)
    sb.heatmap(meanphi_interseg_R,ax=ax3)
    sb.heatmap(varphi_interseg_R,ax=ax4)
    if len(contraprops)>10:
        ax1.set_xticks(np.arange(0,len(contraprops),2))
        ax2.set_xticks(np.arange(0,len(contraprops),2))
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
        ax1.set_yticklabels(ilabels[::2],fontsize=7)
        ax2.set_yticklabels(ilabels[::2],fontsize=7)
        ax3.set_yticklabels(ilabels[::2],fontsize=7)
        ax4.set_yticklabels(ilabels[::2],fontsize=7)
    else:
        ax1.set_yticklabels(ilabels,fontsize=7)
        ax2.set_yticklabels(ilabels,fontsize=7)
        ax3.set_yticklabels(ilabels,fontsize=7)
        ax4.set_yticklabels(ilabels,fontsize=7)
    
    if stype == 0:
        tist = "Crawl - "
    elif stype == 1:
        tist = "Roll - "
    ax1.set(xlabel="", title = tist + r"$<\phi>_{interseg}$")
    ax2.set(xlabel="", ylabel="",  title = r"$\sigma^2_\phi - interseg$")
    ax3.set(xlabel="EE Contra Coupling", ylabel="EI Contra Coupling")
    ax4.set(xlabel="EE Contra Coupling", ylabel="")
 
    plt.show()
    f.savefig(plottitle+'.svg', format = 'svg', dpi = 1200)
    f.savefig(plottitle+'.png', format = 'png', dpi = 1200)


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


#%% Simulate 2 sided WC EI eq's for multiple interconnected segments
def simulate_wc_multiseg(tau_E, b_E, theta_E, tau_I, b_I, theta_I,
                    wEEself, wEIself, wIEself, wIIself, wEEadj, wEIadj,
                    rE_init, rI_init, dt, range_t, kmax_E, kmax_I, n_segs, rest_dur, 
                    n_sides, sim_input, pulse_vals, contra_weights, EEadjtest, EIadjtest, offsetcontra, contra_dur, 
                    offsetcontra_sub, contra_dur_sub, perturb_init, 
                    perturb_input, **otherpars):
    """
    Simulate the Wilson-Cowan equations

    Args:
      Parameters of the Wilson-Cowan model
    
    Returns:
      rE1-8, rI1-8 (arrays) : Activity of excitatory and inhibitory populations
    """  
    # Initialize activity arrays
    Lt = range_t.size
    rEms, rIms, drEms, drIms, I_ext_E, I_ext_I = np.zeros([Lt+1,n_segs,n_sides]), np.zeros([Lt+1,n_segs,n_sides]), np.zeros([Lt+1,n_segs,n_sides]), np.zeros([Lt+1,n_segs,n_sides]), np.zeros([Lt,n_segs,n_sides]), np.zeros([Lt,n_segs,n_sides])
    
    #initialize E and I activity
    # if inhib_init[0] == 1 and inhib_init[1] == 0:
    #     rIms[0,:,0] = inhib_init[2] #ipsi
    # elif inhib_init[0] == 1 and inhib_init[1] == 1:
    #     rIms[0,:,1] = inhib_init[2] #contra
    # elif inhib_init[0] == 1 and inhib_init[1] == 2:
    rIms[0,:,:] = rI_init #both
    # if excit_init[0] == 1 and excit_init[1] == 0:
    #     rEms[0,:,0] = excit_init[2] #ipsi
    # elif excit_init[0] == 1 and excit_init[1] == 1:
    #     rEms[0,:,1] = excit_init[2] #contra 
    # elif excit_init[0] == 1 and excit_init[1] == 2:
    rEms[0,:,:] = rE_init #both

    if perturb_init[0] == 1:
        if perturb_init[1] == 1: #excitatory ipsi or contra
            rEms[perturb_init[5]:perturb_init[5]+perturb_init[6],perturb_init[2]] = perturb_init[4]
        else:
            rIms[perturb_init[5]:perturb_init[5]+perturb_init[6],perturb_init[3],perturb_init[2]] = perturb_init[4]
    
    #setup external input mat
    if sim_input == 0:
        #just posterior seg input - crawl
        print('crawling')
        I_ext_E[:,n_segs-1,0] = np.concatenate((np.zeros(rest_dur), pulse_vals[0,0] * np.ones(int(pulse_vals[0,1])), np.zeros(Lt-int(pulse_vals[0,1])-rest_dur))) #ipsi
    if sim_input == 0 and n_sides>1:
        I_ext_E[:,n_segs-1,1] = np.concatenate((np.zeros(rest_dur), round(pulse_vals[0,0] * offsetcontra,2) * np.ones(int(pulse_vals[0,1]*contra_dur)), 
                                                              np.zeros(Lt-int(pulse_vals[0,1]*contra_dur)-rest_dur))) #contra
    elif sim_input == 1: #simultaneous drive
        print('rolling')
        if pulse_vals[0,2] == 0: #tonic input, single pulse
            I_ext_E[:,:,0] = np.repeat(np.reshape(np.concatenate((np.zeros(rest_dur), pulse_vals[0,0] * np.ones(int(pulse_vals[0,1])), np.zeros(Lt-int(pulse_vals[0,1])-rest_dur))),[Lt,1]),n_segs,axis=1) #ipsi
            if n_sides>1:
                I_ext_E[:,:,1] = np.repeat(np.reshape(np.concatenate((np.zeros(rest_dur), round(pulse_vals[0,0] * offsetcontra_sub,2) * np.ones(int(pulse_vals[0,1]*contra_dur_sub)),
                                                                      round(pulse_vals[0,0] * offsetcontra,2) * np.ones(int((pulse_vals[0,1]*contra_dur))), 
                                                                      np.zeros(Lt-int((pulse_vals[0,1]*(contra_dur+contra_dur_sub)))-rest_dur))),[Lt,1]),n_segs,axis=1) #contra
        else: #alternating sine waves
            sine_ipsi = np.sin(np.linspace(0,np.pi*2*pulse_vals[0,1],Lt))
            sine_contra = -np.sin(np.linspace(0,np.pi*2*pulse_vals[0,1],Lt))
            I_ext_E[:,:,0] = np.repeat(np.reshape(np.where(sine_ipsi>0, sine_ipsi*pulse_vals[0,0], 0),[Lt,1]),n_segs,axis=1)
            if n_sides>1:
                I_ext_E[:,:,1] = np.repeat(np.reshape(np.where(sine_contra>0, sine_contra*pulse_vals[0,0], 0),[Lt,1],n_segs,axis=1))
    
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
    #pull out the individual contra weights
    wEEcontra, wEIcontra, wIEcontra, wIIcontra = contra_weights
    
    print('EEadj = '+str(EEadjtest))
    print('EIadj = '+str(EIadjtest))
    print('EEcontra = '+str(wEEcontra))
    print('EIcontra = '+str(wEIcontra))
    
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
                # Calculate the derivative of the I population - same update for all seg
                drIms[k,seg,s] = dt / tau_I * (-rIms[k,seg,s] + (kmax_I - rIms[k,seg,s]) 
                                               * G((wIIself * rIms[k,seg,s] + wIEself * rEms[k,seg,s]
                                                   + wIEcontra * rEms[k,seg,cs] + wIIcontra * rIms[k,seg,cs] + I_ext_I[k,seg,s]), b_I, theta_I))
                if seg == n_segs-1:
                    #eq without +1 terms
                    # Calculate the derivative of the E population
                    drEms[k,seg,s] = dt / tau_E * (-rEms[k,seg,s] + (kmax_E - rEms[k,seg,s]) * G(((EEadjtest*2 * rEms[k,seg-1,s]) + (wEEself * rEms[k,seg,s]) 
                                                                   + (EIadjtest*2 * rIms[k,seg-1,s]) + (wEIself * rIms[k,seg,s])
                                                                   + (wEEcontra * rEms[k,seg,cs]) + (wEIcontra * rIms[k,seg,cs]) + I_ext_E[k,seg,s]), b_E, theta_E))
                elif seg == 0:
                    #eq without the i-1 terms
                    # Calculate the derivative of the E population
                    drEms[k,seg,s] = dt / tau_E * (-rEms[k,seg,s] + (kmax_E - rEms[k,seg,s]) * G(((wEEself * rEms[k,seg,s]) + (EEadjtest*2 * rEms[k,seg+1,s]) 
                                                                       + (wEIself * rIms[k,seg,s]) + (EIadjtest*2 * rIms[k,seg+1,s]) 
                                                                       + (wEEcontra * rEms[k,seg,cs]) + (wEIcontra * rIms[k,seg,cs]) + I_ext_E[k,seg,s]), b_E, theta_E))
                else: #seg a2-7
                    #the eq's for all mid segs
                    drEms[k,seg,s] = dt / tau_E * (-rEms[k,seg,s] + (kmax_E - rEms[k,seg,s]) * G(((EEadjtest * rEms[k,seg-1,s]) + (wEEself * rEms[k,seg,s]) + (EEadjtest * rEms[k,seg+1,s]) 
                                                                   + (EIadjtest * rIms[k,seg-1,s]) + (wEIself * rIms[k,seg,s]) + (EIadjtest * rIms[k,seg+1,s]) 
                                                                   + (wEEcontra * rEms[k,seg,cs]) + (wEIcontra * rIms[k,seg,cs]) + I_ext_E[k,seg,s]), b_E, theta_E))
                # Update using Euler's method
                if rEms[k+1,seg,s] == 0:
                    rEms[k+1,seg,s] = float(rEms[k,seg,s] + drEms[k,seg,s])
                else:
                    rEms[k+1,seg,s] = rEms[k+1,seg,s]
                if rIms[k+1,seg,s] == 0:
                    rIms[k+1,seg,s] = float(rIms[k,seg,s] + drIms[k,seg,s])
                else:
                    rIms[k+1,seg,s] = rIms[k+1,seg,s]
        
    return rEms, rIms, I_ext_E, I_ext_I



#%% run simulations - iterate over the parameter space - proportion EI adj weights; proportion EI contra weights 
n_t = np.arange(0,200)
simname = ['crawl', 'roll']
pars = default_pars()
n_sides = 2
pulse = 900 #this is for your regular input pulse NOT FOR YOUR PERTURBATIONs
alt = 0 #1 = alternating goro inputs in sine wave pattern
pulse_vals = np.array([[pars['I_ext_E'], pulse, alt]])
c_thresh = 0.3

#setup sim type
sim = 1
sim_input = sim
if sim == 0: #crawl input
    offsetcontra = 1.1
    contra_dur = 1
    contra_dur_sub = 0
    offsetcontra_sub = 0
else: #roll input
    offsetcontra_sub = 0.05
    contra_dur_sub = 0.01
    offsetcontra = 1.1
    contra_dur = 1-contra_dur_sub

#setup interseg weights to test
intersegwprop = np.arange(0,2.75,0.25)
wEEadjtest = 20 # keep interseg E the same, but vary I according to proportions
#setup contra weights to test
#contrawprop = np.arange(0.25,5.25,0.25) #iterate over diff ranges for contra weights
contraEEfix = 20 #starting out trying this as same mag of the interseg EE, but could also try this at lower mags since in main model tests before, contras were~5mag (so 0.25 of interseg mag)

#calculate the proportion of E/I for contra weight type to use as x,y axes heatmap
intersegxvals_prop = [str(i) for i in intersegwprop]
contrayvals_prop = [str(i) for i in intersegwprop]
    
#preset heatmap mats for storing the various interseg and contra phi values - for proportions plots - EE contra and EE interseg both fixed; vary EI vals and calculate proportion
phi_propL = -np.ones([len(intersegwprop), len(intersegwprop)])
phi_propR =  -np.ones([len(intersegwprop), len(intersegwprop)])
phi_propcontra =  -np.ones([len(intersegwprop), len(intersegwprop)])
varphi_propL = -np.ones([len(intersegwprop), len(intersegwprop)])
varphi_propR =  -np.ones([len(intersegwprop), len(intersegwprop)])
varphi_propcontra =  -np.ones([len(intersegwprop), len(intersegwprop)])

#iterate over different interseg and contralateral weights
for indi,i in enumerate(intersegwprop):
    wEIadjtest = -i * wEEadjtest
    for indj,j in enumerate(intersegwprop):
        EI = -j * contraEEfix
        print(wEEadjtest)
        print(wEIadjtest)
        contra_weights = [contraEEfix,EI,0,0]
        
        #no perturbations       
        perturb_init = [0]
        perturb_input = [0]
        
        #run that sim
        rEms,rIms,I_ext_E,I_ext_I = simulate_wc_multiseg(**default_pars(n_sides=n_sides, sim_input=sim_input,
                                                            pulse_vals=pulse_vals, EEadjtest=wEEadjtest, EIadjtest=wEIadjtest, contra_weights=contra_weights, 
                                                            offsetcontra = offsetcontra, contra_dur = contra_dur,
                                                            offsetcontra_sub = offsetcontra_sub, contra_dur_sub = contra_dur_sub,
                                                            perturb_init = perturb_init, perturb_input = perturb_input))
        
        #plot the activity heatmap
        plotbothsidesheatmap(n_t,rEms,pulse_vals,contra_weights,offsetcontra,contra_dur,perturb_input,-i,plotti = simname[sim],ti = simname[sim] + '_Ipropinter=' + str(-i) + '_Ipropcontra=' + str(-j))
        
        #calc + store the phi's - interseg + LR
        cstart, cend, totalwaves, mean_phasediff_LR, phasediff_LR, mean_phasediff_interseg, phasediff_interseg = motor_output_check(n_t, rEms, pulse_vals, c_thresh, titype = simname[sim] + '_')
        
        #take mean of the interseg and LR phase diffs across all "waves"
        phi_propcontra[indi,indj] = np.nanmean(mean_phasediff_LR)
        varphi_propcontra[indi,indj] = np.nanvar(mean_phasediff_LR)
        
        phi_propL[indi,indj] = np.nanmean(mean_phasediff_interseg)
        varphi_propL[indi,indj] = np.nanvar(mean_phasediff_interseg)
        phi_propR[indi,indj] = np.nanmean(mean_phasediff_interseg)
        varphi_propR[indi,indj] = np.nanvar(mean_phasediff_interseg)
        
#plot the phi's over the whole param space
heatmap_params_LRphis(phi_propcontra,varphi_propcontra,intersegxvals_prop,contrayvals_prop,sim,simname[sim] + '_ei_props_intersegvcontra')
heatmap_params_intersegphis(phi_propL,varphi_propL,phi_propR,varphi_propR,intersegxvals_prop,contrayvals_prop,sim,simname[sim] + '_ei_props_eachside')

#stopped here 
#figure out what's going on with weights actually being used appropriately
#test taht input iteration is correct and that fxn is using that instead of defaultpars vals
#fix where divide by zero in phi calcs so that just nan in output and plots

#get plot and cross-check with Gjogjieva paper

#do again for next set of plots

#then check weights compared to ring

#then do midseg test

#fin by Mon; move onto EM

#Tues - keep going EM, be prepping Allen
#if time, set up stability sims too and run overnight some time

#when Ashok back, update on model and EM


#%% run simulations - iterate over the parameter space - contra magnitudes, fixed interseg weights
n_t = np.arange(0,200)
simname = ['crawl', 'roll']
pars = default_pars()
n_sides = 2
pulse = 900 #this is for your regular input pulse NOT FOR YOUR PERTURBATIONs
alt = 0 #1 = alternating goro inputs in sine wave pattern
pulse_vals = np.array([[pars['I_ext_E'], pulse, alt]])
c_thresh = 0.3

#setup sim type
sim = 0
sim_input = sim
if sim == 0: #crawl input
    offsetcontra = 1.1
    contra_dur = 1
    contra_dur_sub = 0
    offsetcontra_sub = 0
else: #roll input
    offsetcontra_sub = 0.05
    contra_dur_sub = 0.01
    offsetcontra = 1.1
    contra_dur = 1-contra_dur_sub
    
#interseg params (fixed to orig Gjorgjieva interseg weights) via default_pars()

#contra params
contraEErange = np.arange(0,22,2)
contraEIrange = -np.arange(0,22,2)

#preset heatmap mats for storing the various interseg and contra phi values - for proportions plots - EE contra and EE interseg both fixed; vary EI vals and calculate proportion
phi_magL = -np.ones([len(contraEErange),len(contraEIrange)])
phi_magR =  -np.ones([len(contraEErange),len(contraEIrange)])
phi_magcontra =  -np.ones([len(contraEErange),len(contraEIrange)])
varphi_magL = -np.ones([len(contraEErange),len(contraEIrange)])
varphi_magR =  -np.ones([len(contraEErange),len(contraEIrange)])
varphi_magcontra =  -np.ones([len(contraEErange),len(contraEIrange)])

for ce,EE in enumerate(contraEErange):
    for ci,EI in enumerate(contraEIrange): 
        contra_weights = [EE,EI,0,0]
        #wEEcontra, wEIcontra, wIEcontra, wIIcontra = contra_weights
        
        #no perturbations       
        perturb_init = [0]
        perturb_input = [0]
        
        #run that sim
        rEms,rIms,I_ext_E,I_ext_I = simulate_wc_multiseg(**default_pars(n_sides=n_sides, sim_input=sim_input,
                                                            pulse_vals=pulse_vals, EEadjtest=pars['wEEadj'], EIadjtest=pars['wEIadj'], contra_weights=contra_weights, 
                                                            offsetcontra = offsetcontra, contra_dur = contra_dur,
                                                            offsetcontra_sub = offsetcontra_sub, contra_dur_sub = contra_dur_sub,
                                                            perturb_init = perturb_init, perturb_input = perturb_input))
        
        #plot the activity heatmap      
        # if ci%4 == 0: #plot just a subset of the activity graphs to check that simulations matchup w/ phase diagram plots
        #     plotbothsidesheatmap(n_t,rEms,pulse_vals,contra_weights,offsetcontra,contra_dur,perturb_input,1,plotti = simname[sim],ti = simname[sim] + '_' + 'EE_'+str(ce) + 'EI_'+str(ci))
        
        #take mean of the interseg and LR phase diffs across all "waves"
        phi_magcontra[ce,ci] = np.nanmean(mean_phasediff_LR)
        varphi_magcontra[ce,ci] = np.nanvar(mean_phasediff_LR)
        
        phi_magL[ce,ci] = np.nanmean(mean_phasediff_interseg)
        varphi_magL[ce,ci] = np.nanvar(mean_phasediff_interseg)
        phi_magR[ce,ci] = np.nanmean(mean_phasediff_interseg)
        varphi_magR[ce,ci] = np.nanvar(mean_phasediff_interseg)
        
        
#plot the phi's over the whole param space
heatmap_params_LRphis(phi_propcontra,varphi_propcontra,contraEErange,contraEIrange,sim,simname[sim] + '_ei_mags_contracomp')
heatmap_params_intersegphis(phi_propL,varphi_propL,phi_propR,varphi_propR,contraEErange,contraEIrange,sim,simname[sim] + '_ei_mags_eachside')


#%%
##problem to think about - though the itnerseg wegiths are equal, 
#the I pop threshold is lower and gain is higher... so faster inhibition probably part of the key togettign wave? need to think more about consolidating w/ Harris too

#then try roll input to iterate over number of midsegs -- how many necessary to generate the rolling?