#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 11:15:50 2022

@author: PatriciaCooney
"""
# Imports
import matplotlib.pyplot as plt
import numpy as np

#%% functions for plotting
def plot_multiseg_sameaxes(n_t,rEms,pulse_vals,contra_weights,offsetcontra,contra_dur,perturb_init,perturb_input,titype):
    #look at LR waves in diff vertical subplots, diff colors
    cthresh = 0.3
    #perturb_input = [1, 1, 0, 7, 1.7, rest_dur+1, 50] #yesno, E or I, ipsi contra, seg, mag, time of onset, duration of input
    fig = plt.figure(figsize=(7,5))
    
    if rEms.shape[1] > 20:
        seginds = np.linspace(0,rEms.shape[1],8,dtype=int)
    else:
        seginds = np.arange(0,rEms.shape[1])
        
    for seg in seginds:
        for s in np.arange(rEms.shape[2]):
            if s == 0:
                co = 'b'
                la = 'Left rE'
            else:
                co = 'm'
                la = 'Right rE'
            #do subplots for each segment and side
            plt.subplot(len(seginds),1,seg+1)
            plt.plot(rEms[0:n_t,seg,s], co, label=la)
            ax = plt.gca()
            ax.set_xticks([])
            ax.set_yticks([])
            if any(rEms[0:n_t,seginds[seg],s]>cthresh):
                xset = np.where(rEms[0:n_t,seginds[seg],s]>cthresh)[0]
                xends = xset[np.where(np.diff(xset)>1)]
                xends = np.append(xends,xset[-1])
                a = np.where(np.diff(xset)>1)[0] + 1
                xstarts = np.append(xset[0],xset[a])
                ystarts = np.zeros(len(xstarts))
                heights = np.repeat(1,len(xstarts))
                for p in np.arange(0,len(xstarts)):
                    plt.axvline(xstarts[p],ystarts[p],heights[p],color = 'g')
                    plt.axvline(xends[p],ystarts[p],heights[p],color = 'k')
            if perturb_input[0] == 1 and seg == perturb_input[3] and s == perturb_input[2]: 
                plt.axvline(x=perturb_input[5],ymin=0,ymax=1,color='r')
                if perturb_input[6] > n_t:
                    plt.axvline(x=n_t,ymin=0,ymax=1,color='r')
                else:
                    plt.axvline(x=perturb_input[5]+perturb_input[6],ymin=0,ymax=1,color='r')
            if seg+1 == len(seginds)/2 and s == 0:
                plt.ylabel('L & R rE', fontsize=12)
            if seg == seginds[-1] and s == 0:
                plt.xlabel('seconds', fontsize=10)
                xints = np.arange(0,n_t+20,40)
                plt.xticks(xints)
                ax = plt.gca()
                ax.set_xticklabels(str(int(xi*pars['dt'])) for xi in xints) 
    plt.legend(loc='lower right')
            
    I_mag = pulse_vals[0,0]
    I_dur = pulse_vals[0,1]
    inc = -0.05
    contra_names = ['EsideE','EsideI','EgateE','IgateE','IgateI'] 
    allstrs = [('L mag: ' + str(I_mag)), ('L dur: ' + str(I_dur*pars['dt'])), ('R mag: ' + str(round((I_mag*offsetcontra),2))),
               ('R dur: ' + str(I_dur*contra_dur*pars['dt'])), 'perturb_init:' + str(perturb_init), 'perturb_input:' + str(perturb_input)]
    #for loop looping list - textstr, plot, add inc
    for l in allstrs:
        plt.gcf().text(0.92, 0.92+inc, l, fontsize = 6)
        inc = inc-0.025
    for iw,w in enumerate(contra_weights):
        inc = inc-0.025
        textstr = contra_names[iw] + '=' + str(w)
        plt.gcf().text(0.92, 0.92+inc, textstr, fontsize = 6)
    
    #plt.tight_layout()
    plt.show()
    fig.savefig((titype + str(n_t) + '_multiseg_traces_singleaxesperseg.svg'), format = 'svg', dpi = 1200)
    fig.savefig((titype + str(n_t) + '_multiseg_traces_singleaxesperseg.png'), format = 'png', dpi = 1200)


#plot contract_dur, isi, and lr diff, per seg, per wave --
def plot_motor_out(segx_in,cdur_in,isi_in,side_diff,numwaves_in,I_pulse_in,ti):
    if numwaves_in >= 8:
        selwa = np.linspace(0,numwaves_in-1,8,dtype=int)
    else:
        selwa = np.arange(0,numwaves_in-1)
    
    if cdur_in.ndim==2:
        subp = 2
        num_sides = 1
        s = 0
    else:
        subp = 3
        num_sides = 2
        k = np.arange(3,len(selwa)*subp+1,subp) #row-wise idxs for separate col plots vars
        
    i = np.arange(1,len(selwa)*subp+1,subp) #row-wise idxs for separate col plots vars
    j = np.arange(2,len(selwa)*subp+1,subp) #row-wise idxs for separate col plots vars
    
    fig = plt.figure(figsize = (len(selwa),subp*3))
    for wi,w in enumerate(selwa):
        #contract dur
        plt.subplot(len(selwa),subp,i[wi])
        if num_sides == 2:
            for s in np.arange(num_sides):
                if s == 0:
                    clr = 'b'
                else:
                    clr = 'r'
                plt.scatter(segx_in,cdur_in[:,w,s], c=clr)
                plt.plot(segx_in,cdur_in[:,w,s], c=clr)
        else:
            clr = 'b'
            plt.scatter(segx_in,cdur_in[:,w], c=clr)
            plt.plot(segx_in,cdur_in[:,w], c=clr)
        if np.max(cdur_in)>0.2:
            offsetdur = np.nanmax(cdur_in)
        else:
            offsetdur = 0.3
        plt.ylim([np.nanmin(cdur_in)-offsetdur,np.nanmax(cdur_in)+offsetdur])
        #labels - contract dur
        if s==0 and wi+1 == round(len(selwa)/2)+1:
            plt.ylabel('normalized contraction duration (s)')
            ax = plt.gca()
            ax.set_xticks([])
        elif wi+1 == len(selwa):
            plt.xlabel('segments')
            plt.xticks(np.arange(0,8))
            ax = plt.gca()
            ax.set_xticklabels(['a1','a2','a3','a4','a5','a6','a7','a8'])
        else:
            ax = plt.gca()
            ax.set_xticks([])
    
        #isi
        plt.subplot(len(selwa),subp,j[wi])
        if num_sides == 2:
            for s in np.arange(num_sides):
                if s == 0:
                    clr = 'b'
                else:
                    clr = 'r'
                plt.scatter(segx_in[:-1],isi_in[:,w,s], c=clr)
                plt.plot(segx_in[:-1],isi_in[:,w,s], c=clr)
        else:
            plt.scatter(segx_in[:-1],isi_in[:,w], c=clr)
            plt.plot(segx_in[:-1],isi_in[:,w], c=clr)                
        
        plt.axhline(0, color='k', ls='--')
        if np.max(isi_in)>0.2:
            offsetisi = np.max(isi_in)
        else:
            offsetisi = 0.3
        plt.ylim([np.nanmin(isi_in)-(offsetisi),np.nanmax(isi_in)+offsetisi])   
        #labels - isi
        if s==0 and wi+1 == round(len(selwa)/2)+1:
            plt.ylabel('normalized intersegmental phase lag (s)')
            ax = plt.gca()
            ax.set_xticks([])
        elif wi+1 == len(selwa):                  
            plt.xlabel('segments')
            plt.xticks(np.arange(0,7))
            ax = plt.gca()
            ax.set_xticklabels(['a2-a1','a3-a2','a4-a3','a5-a4','a6-a5','a7-a6','a8-a7'])
            ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
        else:
            ax = plt.gca()
            ax.set_xticks([])
            
        #side diff
        if num_sides == 2:
            plt.subplot(len(selwa),subp,k[wi])
            plt.scatter(segx_in,side_diff[:,w], c='b')
            plt.plot(segx_in,side_diff[:,w], c='b')
            plt.axhline(0, color='k', ls='--')
            ax = plt.gca()
            
            if s==0 and np.max(side_diff)>0.2:
                offsetsidediff = np.nanmax(side_diff)
            else:
                offsetsidediff = 0.2
            plt.ylim([np.nanmin(side_diff)-offsetsidediff,np.nanmax(side_diff)+offsetsidediff])
            #labels - side diff
            if wi+1 == round(len(selwa)/2)+1:
                plt.ylabel('LR onset difference (s)')
                ax = plt.gca()
                ax.set_xticks([])
            elif wi+1 == len(selwa):
                plt.xlabel('segments')
                plt.xticks(np.arange(0,8))
                ax = plt.gca()
                ax.set_xticklabels(['a1','a2','a3','a4','a5','a6','a7','a8'])
            else:
                ax = plt.gca()
                ax.set_xticks([])
            
    I_mag = pulse_vals[0,0]
    I_dur = pulse_vals[0,1]

    inc = -0.025
    contra_names = ['EsideE','EsideI','EgateE','IgateE','IgateI'] 

    allstrs = [('L mag: ' + str(I_mag)), ('L dur: ' + str(I_dur*pars['dt'])), ('R mag: ' + str(round((I_mag*offsetcontra),2))),
               ('R dur: ' + str(I_dur*contra_dur*pars['dt'])), 'perturb_init:' + str(perturb_init), 'perturb_input:' + str(perturb_input)]
    #for loop looping list - textstr, plot, add inc
    for l in allstrs:
        plt.gcf().text(0.985, 0.98+inc, l, fontsize = 6)
        inc = inc-0.025
    for iw,w in enumerate(contra_weights):
        inc = inc-0.025
        textstr = contra_names[iw] + '=' + str(w)
        plt.gcf().text(0.985, 0.98+inc, textstr, fontsize = 6)
    
    fig.tight_layout()
    plt.show()
    fig.savefig((ti+'_cntrct_isi.svg'), format = 'svg', dpi = 1200)
    fig.savefig((ti+'_cntrct_isi.png'), format = 'png', dpi = 1200)
    

#phase difference plot
def plot_phase_diff(numwaves_in, avg, ant, mid, post, I_pulse_in, ti):
    fig = plt.figure()
    print(numwaves_in)
    if numwaves_in > 20:
        selwa = np.linspace(0,numwaves_in-2,20,dtype=int)
    else:
        selwa = np.arange(0,numwaves_in-2)
        
    plt.scatter(selwa,ant[selwa],c ="m")
    plt.plot(selwa,ant[selwa],c ="m",label='A1')
    plt.scatter(selwa,mid[selwa],c ="b")
    plt.plot(selwa,mid[selwa],c ="b",label='A3')
    plt.scatter(selwa,post[selwa],c ="g")
    plt.plot(selwa,post[selwa],c ="g",label='A8/9')
    plt.scatter(selwa,avg[selwa],c ="k")
    plt.plot(selwa,avg[selwa],c ="k",label='Average')
    plt.axhline(0.5, color='r', ls='--')
    plt.axhline(1, color='y', ls='--')
    plt.axhline(0, color='y', ls='--')
    
    plt.ylim([-0.1,1.5])
    plt.ylabel(r"$\phi$")
    plt.xlabel('Activity Waves')
    ax = plt.gca()
    wavect = [str(si+1) for si in selwa]
    ax.set_xticks(selwa)
    ax.set_xticklabels(wavect)
    plt.legend(loc='lower left')
       
    I_mag = pulse_vals[0,0]
    I_dur = pulse_vals[0,1]
    inc = -0.025
    contra_names = ['EsideE','EsideI','EgateE','IgateE','IgateI']  
    
    allstrs = [('L mag: ' + str(I_mag)), ('L dur: ' + str(I_dur*pars['dt'])), ('R mag: ' + str(round((I_mag*offsetcontra),2))),
               ('R dur: ' + str(I_dur*contra_dur*pars['dt'])), 'perturb_init:' + str(perturb_init), 'perturb_input:' + str(perturb_input)]
    #for loop looping list - textstr, plot, add inc
    for l in allstrs:
        plt.gcf().text(0.985, 0.98+inc, l, fontsize = 6)
        inc = inc-0.025
    for iw,w in enumerate(contra_weights):
        inc = inc-0.025
        textstr = contra_names[iw] + '=' + str(w)
        plt.gcf().text(0.985, 0.98+inc, textstr, fontsize = 6)
    
    plt.tight_layout()
    plt.show()
    fig.savefig((ti+'_phase_diff.svg'), format = 'svg', dpi = 1200)
    fig.savefig((ti+'_phase_diff.png'), format = 'png', dpi = 1200)


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

  # simulation parameters
  pars['T'] = 80.        # Total duration of simulation timesteps
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


#calc population input flexibly w/ changing variables for manual derivation of gate weights
def popinput_exc(wEEself, wEEadj, wEIself, wEIadj, I_ext_E, startvalE, startvalI, contraweight, startEcontra, startIcontra):
    """
    calc population input based on AP and contra neighbors, plus recurrence in node

    Args:
        weights for recurrence - wEEself, wEEadj, wEIself, wEIadj
        I_ext_E - ext input to this node
        startvalE - if timestep 0, rE_init; if timestep>0, see t-1 val from simulation, ipsi side for same
        startvalI - same as above but for I
        startEcontra - if timestep 0, rE_init; if timestep>0, see t-1 val from simulation, CONTRA SIDE
        startIcontra - same as above but for I
        contraweight - if + --> EE, crawl just A8; if - --> EI roll midseg
      
    Returns:
      x_pop     : the population input x, to be used in sigmoid FI calc
      
    """
    if contraweight > 0:
        x_pop = (wEEself * startvalE + wEEadj * startvalE) + (wEIself * startvalI + wEIadj * startvalI) + (contraweight * startEcontra) + I_ext_E
    elif contraweight < 0:
        x_pop = (wEEself * startvalE + wEEadj * startvalE) + (wEIself * startvalI + wEIadj * startvalI) + (contraweight * startIcontra) + I_ext_E
    else:
        x_pop = (wEEself * startvalE + wEEadj * startvalE) + (wEIself * startvalI + wEIadj * startvalI) + (contraweight * startIcontra) + I_ext_E
        
    return x_pop

#%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% Simulate 2 sided WC EI eq's for multiple interconnected segments -- includes extra gating interneurons per segment
def simulate_wc_multiseg(tau_E, b_E, theta_E, tau_I, b_I, theta_I,
                    wEEself, wEIself, wIEself, wIIself, wEEadj, wEIadj,
                    rE_init, rI_init, dt, range_t, kmax_E, kmax_I, n_segs, rest_dur, 
                    n_sides, sim_input, pulse_vals, contra_weights, offsetcontra, contra_dur, 
                    offsetcontra_sub, contra_dur_sub, perturb_init, perturb_input, **otherpars):
    """
    Simulate the Wilson-Cowan equations

    Args:
      Parameters of the Wilson-Cowan model
    
    Returns:
      rE1-8, rI1-8 (arrays) : Activity of excitatory and inhibitory populations
    """  
    # Initialize activity arrays
    Lt = range_t.size
    rEms_sides, rIms_sides, rEms_gates, rIms_gates, drEms_sides, drIms_sides, drEms_gates, drIms_gates, I_ext_E, I_ext_I = np.zeros([Lt+1,n_segs,n_sides]), np.zeros([Lt+1,n_segs,n_sides]), np.zeros([Lt+1,n_segs,n_sides]), np.zeros([Lt+1,n_segs,n_sides]), np.zeros([Lt+1,n_segs,n_sides]), np.zeros([Lt+1,n_segs,n_sides]), np.zeros([Lt+1,n_segs,n_sides]), np.zeros([Lt+1,n_segs,n_sides]), np.zeros([Lt,n_segs,n_sides]), np.zeros([Lt,n_segs,n_sides])
    
    #initialize E and I activity -- just initialize the LR side neurons
    rIms_sides[0,:,:] = rI_init #both
    rEms_sides[0,:,:] = rE_init #both

    if perturb_init[0] == 1:
        if perturb_init[1] == 1: #excitatory ipsi or contra
            rEms_sides[rest_dur,perturb_init[3],perturb_init[2]] = perturb_init[4]
        else:
            rIms_sides[rest_dur,perturb_init[3],perturb_init[2]] = perturb_init[4]
    
    #setup external input mat
    if sim_input == 0:
        #just posterior seg input - crawl
        print('crawling')
        I_ext_E[:,n_segs-1,0] = np.concatenate((np.zeros(rest_dur), pulse_vals[0,0] * np.ones(int(pulse_vals[0,1])), np.zeros(Lt-int(pulse_vals[0,1])-rest_dur))) #ipsi
        if n_sides>1:
            I_ext_E[:,n_segs-1,1] = np.concatenate((np.zeros(rest_dur), round(pulse_vals[0,0] * offsetcontra,2) * np.ones(int(pulse_vals[0,1]*contra_dur)), 
                                                                  np.zeros(Lt-int(pulse_vals[0,1]*contra_dur)-rest_dur))) #contra
    if sim_input == 1: #simultaneous drive
        print('rolling')
        if pulse_vals[0,2] == 0: #tonic input, single pulse
            I_ext_E[:,:,0] = np.repeat(np.reshape(np.concatenate((np.zeros(rest_dur), pulse_vals[0,0] * np.ones(int(pulse_vals[0,1])), np.zeros(Lt-int(pulse_vals[0,1])-rest_dur))),[Lt,1]),8,axis=1) #ipsi
        if n_sides>1:
            I_ext_E[:,:,1] = np.repeat(np.reshape(np.concatenate((np.zeros(rest_dur), round(pulse_vals[0,0] * offsetcontra_sub,2) * np.ones(int(pulse_vals[0,1]*contra_dur_sub)),
                                                              round(pulse_vals[0,0] * offsetcontra,2) * np.ones(int((pulse_vals[0,1]*contra_dur))), 
                                                              np.zeros(Lt-int((pulse_vals[0,1]*(contra_dur+contra_dur_sub)))))),[Lt,1]),8,axis=1) #contra
        else: #alternating sine waves
            sine_ipsi = np.sin(np.linspace(0,np.pi*2*pulse_vals[0,1],Lt))
            sine_contra = -np.sin(np.linspace(0,np.pi*2*pulse_vals[0,1],Lt))
            I_ext_E[:,:,0] = np.repeat(np.reshape(np.where(sine_ipsi>0, sine_ipsi*pulse_vals[0,0], 0),[Lt,1]),8,axis=1)
            if n_sides>1:
                I_ext_E[:,:,1] = np.repeat(np.reshape(np.where(sine_contra>0, sine_contra*pulse_vals[0,0], 0),[Lt,1],8,axis=1))
                                          
    #perturb_input = [1, sign, 0, sloop, inputl, rest_dur+1, pulse-rest_dur] #yesno, I E or both, ipsi contra or both, seg, mag, time of onset, duration of input
    if perturb_input[0] == 1:
        start = int(perturb_input[5])
        end = int(perturb_input[6])
        if perturb_input[1] == 1: #excitatory ipsi or contra
            if perturb_input[2] != 2:
                if 'all' in perturb_input:
                    I_ext_E[start:end,:,perturb_input[2]] = np.repeat(perturb_input[4] * (np.ones([int(end-start),1])),8,axis=1)
                else:
                    I_ext_E[start:end,perturb_input[3],perturb_input[2]] = perturb_input[4] * (np.ones(int(end-start)))
            else:
                I_ext_E[:,perturb_input[3],0] = np.concatenate((np.zeros(perturb_input[5]), perturb_input[4] * np.ones(perturb_input[6]), 
                                                                                                                                np.zeros(Lt-int(perturb_input[5]+perturb_input[6]))))
                I_ext_E[:,perturb_input[3],1] = np.concatenate((np.zeros(perturb_input[5]), perturb_input[4] * np.ones(perturb_input[6]), 
                                                                                                                                np.zeros(Lt-int(perturb_input[5]+perturb_input[6]))))
        else:
            I_ext_I[:,perturb_input[3],perturb_input[2]] = np.concatenate((np.zeros(perturb_input[5]), perturb_input[4] * np.ones(perturb_input[6]), 
                                                                                                                            np.zeros(Lt-int(perturb_input[5]+perturb_input[6]))))
    #print(I_ext_E)
        #perturb_input = [1, 1, 1, 7, pars['I_ext_E']*2, rest_dur+110, 100] #yesno, E or I or both, ipsi contra or both, seg, mag, time of onset, duration of input
    #pull out the individual contra weights -- these are contra weights to the gate neurons now - NOT directly to the contralateral neurons
    wEsideE, wEsideI, wEgateE, wIgateE, wIgateI = contra_weights
    
    #rEms_sides, rIms_sides, rEms_gates, rIms_gates, drEms_sides, drIms_sides, drEms_gates, drIms_gates
    
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
                #true side I pop -- all segments, all sides - keep exact same equation as before except no direct contralateral input
                drIms_sides[k,seg,s] = dt / tau_I * (-rIms_sides[k,seg,s] + (kmax_I - rIms_sides[k,seg,s]) 
                                               * G((wIIself * rIms_sides[k,seg,s] + wIEself * rEms_sides[k,seg,s] + I_ext_I[k,seg,s]), b_I, theta_I))
                
                #gate E pop -- allsegs, both sides #NOTE - FOR RIGHT NOW, THE GATE NEURONS JUST GET THEIR INPUT FROM SIDES; OPTION NEXT - TRY WITH RECURRENCE AT GATE
                #if add gate recurrence,add (wEEself * rEms_gates[k,seg,s]) in G  and (wIIself * rIms_gates[k,seg,s]) in G for I gate
                drEms_gates[k,seg,s] = dt / tau_E * (-rEms_gates[k,seg,s] + (kmax_E - rEms_gates[k,seg,s]) * G(((wEEself * rEms_gates[k,seg,s])+(wEgateE * rEms_sides[k,seg,cs])), b_E, theta_E))
                
                #gate I pop -- allsegs, both sides #NOTE - FOR RIGHT NOW, THE GATE NEURONS JUST GET THEIR INPUT FROM SIDES; OPTION NEXT - TRY WITH RECURRENCE AT GATE
                drIms_gates[k,seg,s] = dt / tau_I * (-rIms_gates[k,seg,s] + (kmax_I - rIms_gates[k,seg,s]) * G(((wIIself * rIms_gates[k,seg,s])+(wIgateE * rEms_sides[k,seg,s]) 
                                                                                                                + (wIgateI * rIms_gates[k,seg,cs])), b_I, theta_I))
                
                #double check -- each E side should get E gate ipsi, which gets Eside contra; also get I gate contra, which gets Eside contra - correct.
                if seg == n_segs-1:
                    #eq without +1 terms
                    #true side E pop -- A8/9, both sides - update to include inputs from gate I and gate E neurons
                    drEms_sides[k,seg,s] = dt / tau_E * (-rEms_sides[k,seg,s] + (kmax_E - rEms_sides[k,seg,s]) * G(((wEEadj*2 * rEms_sides[k,seg-1,s]) + (wEEself * rEms_sides[k,seg,s]) 
                                                                   + (wEIadj*2 * rIms_sides[k,seg-1,s]) + (wEIself * rIms_sides[k,seg,s])
                                                                   + (wEsideE * rEms_gates[k,seg,s]) + (wEsideI * rIms_gates[k,seg,cs]) + I_ext_E[k,seg,s]), b_E, theta_E))
                    
                elif seg == 0:
                    #eq without the i-1 terms
                    #true side E pop -- A1, both sides - update to include inputs from gate I and gate E neurons
                    drEms_sides[k,seg,s] = dt / tau_E * (-rEms_sides[k,seg,s] + (kmax_E - rEms_sides[k,seg,s]) * G(((wEEself * rEms_sides[k,seg,s]) + (wEEadj*2 * rEms_sides[k,seg+1,s]) 
                                                                       + (wEIself * rIms_sides[k,seg,s]) + (wEIadj*2 * rIms_sides[k,seg+1,s]) 
                                                                       + (wEsideE * rEms_gates[k,seg,s]) + (wEsideI * rIms_gates[k,seg,cs]) + I_ext_E[k,seg,s]), b_E, theta_E))
                    
                else: #seg a2-7
                    #true side E pop -- all segments, both sides - update to include inputs from gate I and gate E neurons
                    drEms_sides[k,seg,s] = dt / tau_E * (-rEms_sides[k,seg,s] + (kmax_E - rEms_sides[k,seg,s]) * G(((wEEadj * rEms_sides[k,seg-1,s]) + (wEEself * rEms_sides[k,seg,s]) + (wEEadj * rEms_sides[k,seg+1,s]) 
                                                                   + (wEIadj * rIms_sides[k,seg-1,s]) + (wEIself * rIms_sides[k,seg,s]) + (wEIadj * rIms_sides[k,seg+1,s]) 
                                                                   + (wEsideE * rEms_gates[k,seg,s]) + (wEsideI * rIms_gates[k,seg,cs]) + I_ext_E[k,seg,s]), b_E, theta_E))
                    
                # Update all population nodes using Euler's method
                rEms_sides[k+1,seg,s] = float(rEms_sides[k,seg,s] + drEms_sides[k,seg,s])
                rIms_sides[k+1,seg,s] = float(rIms_sides[k,seg,s] + drIms_sides[k,seg,s])
                rEms_gates[k+1,seg,s] = float(rEms_gates[k,seg,s] + drEms_gates[k,seg,s])
                rIms_gates[k+1,seg,s] = float(rIms_gates[k,seg,s] + drIms_gates[k,seg,s])
        
    return rEms_sides, rIms_sides, rEms_gates, rIms_gates, I_ext_E, I_ext_I

#%% find nearest fxn to make the side and seg comparisons easier
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

#%% calculate side diffs and motor output for 2-sided models
def motor_output_check(E,pulse_vals,c_thresh,titype):
    #find contraction onset and dur by checking drE vals
    segx = np.arange(E.shape[1])
    side = np.arange(E.shape[2])

    #track contraction start and end per seg and per side - catch partial waves
    if np.where(E[:,:,:] > c_thresh)[0].size>0:
        left_starts = np.ones([8,100])*99999
        left_ends = np.ones([8,100])*99999
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
                    left_ws = np.concatenate((left_ws,np.zeros([left_starts.shape[0],wdiff])),1)
                elif wdiff < 0:
                    left_starts = np.concatenate(((left_starts,np.zeros([left_starts.shape[0],wdiff]))),1)
            elif nwends > num_waves:
                totalwaves = nwends
                wdiff = totalwaves - nwends
                if wdiff > 0:
                    left_we = np.concatenate((left_we,np.zeros([left_starts.shape[0],wdiff])),1)
                elif wdiff < 0:
                    left_ends = np.concatenate(((left_ends,np.zeros([left_ends.shape[0],wdiff]))),1)   
            left_starts[seg,:num_waves] = left_ws+1
            left_ends[seg,:nwends] = left_we
        if side.size>1:
            right_starts = np.ones([8,1000])*99999
            right_ends = np.ones([8,1000])*99999
            suprathresh_right = np.where(E[:,:,1] > c_thresh, 1, 0)
            for seg in segx:
                right_ws = np.where(np.diff(suprathresh_right[:,seg],axis=0)==1)[0]
                right_we = np.where(np.diff(suprathresh_right[:,seg],axis=0)==-1)[0]
                num_waves = right_ws.shape[0]
                nwends = right_we.shape[0]
                
                print(right_ws)
                print(right_we)
                print(num_waves)
                print(nwends)
                print(totalwaves)
                
                if num_waves > totalwaves:
                    totalwaves = num_waves
                if num_waves > nwends:
                    wdiff = totalwaves - num_waves
                    if wdiff > 0:
                        right_ws = np.concatenate((right_ws,np.zeros([right_starts.shape[0],wdiff])),1)
                    elif wdiff < 0:
                        right_starts = np.concatenate(((right_starts,np.zeros([right_starts.shape[0],wdiff]))),1)
                elif nwends > num_waves:
                    totalwaves = nwends
                    wdiff = totalwaves - nwends
                    if wdiff > 0:
                        right_we = np.concatenate((right_we,np.zeros([right_starts.shape[0],wdiff])),1)
                    elif wdiff < 0:
                        right_ends = np.concatenate(((right_ends,np.zeros([right_ends.shape[0],wdiff]))),1)
                right_starts[seg,:num_waves] = right_ws+1
                right_ends[seg,:nwends] = right_we
            cstart = np.dstack((left_starts[:,0:totalwaves],right_starts[:,0:totalwaves]))*pars['dt']
            cend = np.dstack((left_ends[:,0:totalwaves],right_ends[:,0:totalwaves]))*pars['dt']
        else:
            cstart = left_starts[:,0:totalwaves]*pars['dt']
            cend = left_ends[:,0:totalwaves]*pars['dt']
            
        #take latency
        lat = cstart[7,0]
        
        #use nearest approach to calculate isi
        #to get norm isi, take nearest A8 start val to nearest A1 end val
        if side.size == 1:    
            isi = np.ones([cstart.shape[0]-1,cstart.shape[1]])*99999
            cdur = np.ones(cstart.shape)*99999
            cnorm = np.ones(cstart.shape[1])*99999
        else:
            isi = np.ones([cstart.shape[0]-1,cstart.shape[1],2])*99999
            cdur = np.ones([cstart.shape[0],cstart.shape[1],2])*99999
            cnorm = np.ones([cstart.shape[1],2])*99999
        for si in side: #figure out how to do this w/ 2D array and 3D array in same loop
            for seg in segx:
                for wa in np.arange(cstart.shape[1]):
                    if seg != 0:
                        comparray_ant = cstart[seg-1,:,si]
                        adjval = find_nearest(comparray_ant, cstart[seg,wa,si])
                        isi[seg-1,wa,si] = cstart[seg,wa,si] - adjval
                        if isi[seg-1,wa,si] > 5:
                            isi[seg-1,wa,si] = np.nan
                        cdur[seg-1,wa,si] = abs(cend[seg,wa,si] - cstart[seg,wa,si])
                    elif seg == 0:
                        comparray_end = cstart[-1,:,si]
                        adjval = find_nearest(comparray_end, cstart[0,wa,si])
                        cnorm[wa,si] = adjval - cstart[0,wa,si]
                        cdur[seg-1,wa,si] = abs(cend[seg,wa,si] - cstart[seg,wa,si])
        cdurnorm = cdur/cnorm
        isinorm = isi/cnorm

        #do phase diff for 2-sided system
        side_diff = np.ones([cstart.shape[0],cstart.shape[1]])*99999
        phasediff = np.ones([cstart.shape[1],cstart.shape[0]])*99999
        ant_phasediff = np.ones(cstart.shape[1]-1)*99999
        mid_phasediff = np.ones(cstart.shape[1]-1)*99999
        post_phasediff = np.ones(cstart.shape[1]-1)*99999
        mean_phasediff = np.ones(cstart.shape[1]-1)*99999
        if side.size>1:
            for seg in segx:
                for wa in np.arange(cstart.shape[1])-2:
                    #side diff
                    compwave = cstart[seg,wa,0]
                    comparray_side = cstart[seg,:,1] #compare to all possible waves in this seg on the contralateral side to find nearest
                    adjval = find_nearest(comparray_side, compwave)
                    side_diff[seg,wa] = abs(compwave - adjval)
                    
                    #phase diff
                    compwavenext = cstart[seg,wa+1,0]
                    comparray_side = cstart[seg,:,1] #compare to all possible waves in this seg on the contralateral side to find nearest
                    adjval = find_nearest(comparray_side, cstart[seg,wa,0])
                    adjvalnext = find_nearest(comparray_side, compwavenext)
                    phasediff[wa,seg] = abs(compwavenext - adjval)/abs(adjvalnext - adjval)
                    if seg == 0:
                        ant_phasediff[wa] = phasediff[wa,seg]
                    elif seg == 3:
                        mid_phasediff[wa] = phasediff[wa,seg]
                    elif seg == 7:
                        post_phasediff[wa] = phasediff[wa,seg]
                        mean_phasediff[wa] = np.mean(phasediff[wa,:],0) #mean, remove the weird entries
                    if side_diff[seg,wa] > 5:
                        side_diff[seg,wa] = np.nan
                        phasediff[wa] = np.nan
                        ant_phasediff[wa] = np.nan
                        mid_phasediff[wa] = np.nan
                        post_phasediff[wa] = np.nan
                        mean_phasediff[wa] = np.nan
            
            # #plot phase diffs for 2-sided system
            plot_phase_diff(num_waves, mean_phasediff, ant_phasediff, mid_phasediff, post_phasediff, pulse_vals, ti = titype + 'offsetcontra_' + str(offsetcontra) + '_' + str(contra_weights) + '_')
                
        else:
            #no phase diff to calculate in 1-sided system
            mean_phasediff, ant_phasediff, mid_phasediff, post_phasediff, side_diff = np.nan, np.nan, np.nan, np.nan, np.nan

        #plot contraction duration of all segs, interseg phase lag, peak E amp as subplots 1 fig, all segs (fig 3 and then some)
        plot_motor_out(segx,cdur,isi,side_diff,totalwaves,pulse_vals,ti = titype + 'offsetcontra_' + str(offsetcontra) + '_' + str(contra_weights) + '_')
        
    else:
        cstart,cend,cdur,cdurnorm,lat,isi,isinorm,totalwaves = np.nan, np.nan, np.nan*np.ones([8]), np.nan*np.ones([8]), np.nan, np.nan*np.ones([7]), np.nan*np.ones([7]), np.nan
        mean_phasediff, ant_phasediff, mid_phasediff, post_phasediff = np.nan, np.nan, np.nan, np.nan
        side_diff = np.nan
        
    return cstart, cend, cdur, cdurnorm, lat, isi, isinorm, side_diff, totalwaves, mean_phasediff, ant_phasediff, mid_phasediff, post_phasediff

#%% build the gated model and do initial test of how works/if can generate the two patterns
#%% setup general gate model and run
simname = ['crawl', 'roll']
sim = 0 #MAKE 0 OR 1 FOR CRAWL OR ROLL
n_t = 900
pars = default_pars()
n_sides = 2
pulse = 900
alt = 0 #1 = alternating goro inputs in sine wave pattern
pulse_vals = np.array([[pars['I_ext_E'], pulse, alt]])

#null perturbations
perturb_init = [0,0] #none
#perturb_input = [0,0]
perturb_input = [1, 0, 0, 7, 1.2, pars['rest_dur']+1, pulse-pars['rest_dur']+1]

#NEW CONTRA GATE WEIGHTS--post,pre
#EE gate to side, EI gate to side, EE side to gate, EI side to gate, IE side to gate, II gate to gate
# wEsideE, wEsideI, wEgateE, wIgateE, wIgateI = contra_weights
#contra_weights = [0,0,0,0,0]
#contra_weights = [2.5,-1.5,2,2,-1.5]
contra_weights = [2,-6,6,12,-4]

#setup crawl or roll input
sim_input = sim
if sim == 0:
    offsetcontra = 1.1
    contra_dur = 1
    contra_dur_sub = 0
    offsetcontra_sub = 0
elif sim ==1: 
    offsetcontra_sub = 0.05
    contra_dur_sub = 0.01
    offsetcontra = 1.1
    contra_dur = 1-contra_dur_sub

#run that sim
rEms_sides, rIms_sides, rEms_gates, rIms_gates, I_ext_E, I_ext_I = simulate_wc_multiseg(**default_pars(n_sides=n_sides, sim_input=sim_input,
                                                    pulse_vals=pulse_vals, contra_weights=contra_weights, 
                                                    offsetcontra = offsetcontra, contra_dur = contra_dur,
                                                    offsetcontra_sub = offsetcontra_sub, contra_dur_sub = contra_dur_sub,
                                                    perturb_init = perturb_init, perturb_input = perturb_input))

plot_multiseg_sameaxes(n_t,rEms_sides,pulse_vals,contra_weights,offsetcontra,contra_dur,perturb_init,perturb_input,
                  titype = simname[sim] +'offsetcontra_' + str(offsetcontra) + '_' + str(contra_weights)+'_twosided_fpinit_rest1_gatewithrecurr_fulllength_900')

cstart, cend, cdur, cdurnorm, lat, isi, isinorm, side_diff, right, mean_phasediff, ant_phasediff, mid_phasediff, post_phasediff = motor_output_check(rEms_sides,
      pulse_vals,c_thresh = 0.3,titype = simname[sim] +'_twosided_fpinit_rest1_gatewithrecurr_fulllength_900')

#%% test stability of gated model w/ crawl v roll inputs to diff perturbations

initloop = np.arange(0,2,0.6)
#inputloop = np.arange(0,3.6,0.4)
inputloop = np.arange(2.4,3.6,0.4)
segloop = [7]
#segloop = [0,2,4,7]
#simtype = [0,1]
sim = 0
n_t = 300

#make naming options
simname = ['crawl', 'roll']
segname = ['A1','A3','A5','A8']

#loop for running stability analysis on all possible simulations of interest
pars = default_pars()
n_sides = 2
pulse = 550
alt = 0 #1 = alternating goro inputs in sine wave pattern
pulse_vals = np.array([[pars['I_ext_E'], pulse, alt]])

#NEW CONTRA GATE WEIGHTS--post,pre
#EE gate to side, EI gate to side, EE side to gate, EI side to gate, IE side to gate, II gate to gate
# wEsideE, wEsideI, wEgateE, wIgateE, wIgateI = contra_weights
contra_weights = [2,-6,6,12,-4]

#setup crawl or roll input
sim_input = sim
if sim == 0:
    offsetcontra = 1.1
    contra_dur = 1
    contra_dur_sub = 0
    offsetcontra_sub = 0
else: 
    offsetcontra_sub = 0.05
    contra_dur_sub = 0.01
    offsetcontra = 1.1
    contra_dur = 1-contra_dur_sub

#set this loop to match the perturb_init and perturb_input above
sign = 1

for inputl in inputloop:
    initl = 0
    for seg,sloop in enumerate(segloop):
        #perturbations
        perturb_init = [1, sign, 0, sloop, initl] #no yes, E or I, ipsi contra, seg, init_val
        perturb_input = [1, sign, 0, sloop, inputl, pars['rest_dur']+1, pulse-pars['rest_dur']+1] #yesno, I E or both, ipsi contra or both, seg, mag, time of onset, duration of input
        
        #run that sim
        rEms_sides, rIms_sides, rEms_gates, rIms_gates, I_ext_E, I_ext_I = simulate_wc_multiseg(**default_pars(n_sides=n_sides, sim_input=sim_input,
                                                            pulse_vals=pulse_vals, contra_weights=contra_weights, 
                                                            offsetcontra = offsetcontra, contra_dur = contra_dur,
                                                            offsetcontra_sub = offsetcontra_sub, contra_dur_sub = contra_dur_sub,
                                                            perturb_init = perturb_init, perturb_input = perturb_input))
        
        plot_multiseg_sameaxes(n_t,rEms_sides,pulse_vals,contra_weights,offsetcontra,contra_dur,perturb_init,perturb_input,
                          titype = simname[sim] +'_twosided_gatewrecurr_fpinit_rest1_' + segname[seg] +'_initpert' + str(initl) +'_excinputpert' + str(inputl))
        
        cstart, cend, cdur, cdurnorm, lat, isi, isinorm, side_diff, right, mean_phasediff, ant_phasediff, mid_phasediff, post_phasediff = motor_output_check(rEms_sides,
              pulse_vals,c_thresh = 0.3,titype = simname[sim] +'_twosided_gatewrecurr_fpinit_rest1_' + segname[seg] +'_initpert' + str(initl) +'_excinputpert' + str(inputl))

for initl in initloop:
    inputl = 0
    for seg,sloop in enumerate(segloop):
        #perturbations
        perturb_init = [1, sign, 0, sloop, initl] #no yes, E or I, ipsi contra, seg, init_val
        perturb_input = [1, sign, 0, sloop, inputl, pars['rest_dur']+1, pulse-pars['rest_dur']+1] #yesno, I E or both, ipsi contra or both, seg, mag, time of onset, duration of input 
        
        #run that sim
        rEms_sides, rIms_sides, rEms_gates, rIms_gates, I_ext_E, I_ext_I = simulate_wc_multiseg(**default_pars(n_sides=n_sides, sim_input=sim_input,
                                                            pulse_vals=pulse_vals, contra_weights=contra_weights, 
                                                            offsetcontra = offsetcontra, contra_dur = contra_dur,
                                                            offsetcontra_sub = offsetcontra_sub, contra_dur_sub = contra_dur_sub, 
                                                            perturb_init = perturb_init, perturb_input = perturb_input))
        
        plot_multiseg_sameaxes(n_t,rEms_sides,pulse_vals,contra_weights,offsetcontra,contra_dur,perturb_init,perturb_input,
                          titype = simname[sim] +'_twosided_gatewrecurr_fpinit_rest1_' + segname[seg] +'_excinitpert' + str(initl) +'_inputpert' + str(inputl))
        
        cstart, cend, cdur, cdurnorm, lat, isi, isinorm, side_diff, right, mean_phasediff, ant_phasediff, mid_phasediff, post_phasediff = motor_output_check(rEms_sides,
              pulse_vals,c_thresh = 0.3,titype = simname[sim] +'_twosided_gatewrecurr_fpinit_rest1_' + segname[seg] +'_excinitpert' + str(initl) +'_inputpert' + str(inputl))


# ## note - for input perturb... need a temporal component of interest relative to the roll freq and crawl freq? -- for now, doing tonic perturb and see what happens; can shorten later
# # NOTE FOR LATER - to clean this up, make default params for crawl and default for roll -- general shared then specific
