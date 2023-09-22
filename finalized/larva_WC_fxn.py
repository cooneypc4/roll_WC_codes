#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 14:02:14 2023

Larval model simulation function

Takes parameters for Wilson-Cowan equations to simulate activity for each segment in the larval model
Each hemisegment's E and I populations are represented by the equations 
Euler integration is used to update each population's activity at a future timestep

For rolling/multipattern generation paper, the following parameters are fixed:
tau_E, b_E, theta_E, tau_I, b_I, theta_I, wEEself, wEIself, wIEself, wIIself, wEEadj, 
wEIadj, rE_init, rI_init, dt, range_t, kmax_E, kmax_I, n_segs, rest_dur, n_sides

The following parameters are tested/varied throughout:
sim_input, pulse_vals, EEadjtest, EIadjtest, contra_weights, offsetcontra, contra_dur, 
offsetcontra_sub, contra_dur_sub, perturb_init, perturb_input, noisemean, noisescale

@author: PatriciaCooney
"""

#%% import packages
import numpy as np
#%% Define noise term according to Euler Maruyama equation
def addnoise(noisemean, noisescale, dt):
    """Add Euler Maruyama noise term so noise scale is not impacted by length of simulation"""
    return noisescale*np.random.normal(noisemean,np.sqrt(dt))
#%%sigmoid
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
#%% Simulate 2 sided WC EI eq's for multiple interconnected segments
def simulate_wc_multiseg(tau_E, b_E, theta_E, tau_I, b_I, theta_I,
                    wEEself, wEIself, wIEself, wIIself, wEEadj, wEIadj,
                    rE_init, rI_init, dt, range_t, kmax_E, kmax_I, n_segs, rest_dur, 
                    n_sides, sim_input, pulse_vals, EEadjtest, EIadjtest, contra_weights, offsetcontra, contra_dur, 
                    offsetcontra_sub, contra_dur_sub, perturb_init, 
                    perturb_input, noisemean, noisescale, gate, **otherpars):
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
    
    if gate == 1:
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
    if gate == 1:
        wEsEs, wIsEg, wEgEs, wIsIs = contra_weights
    elif gate == 0:
        wEEcontra, wEIcontra, wIEcontra, wIIcontra = contra_weights

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
                if gate == 1:  
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
                
                if gate == 0:
                    # Calculate the derivative of the I population - same update for all seg
                    drIms[k,seg,s] = dt / tau_I * (-rIms[k,seg,s] + (kmax_I - rIms[k,seg,s]) 
                                                   * G((wIIself * rIms[k,seg,s] + wIEself * rEms_sides[k,seg,s]
                                                       + wIEcontra * rEms_sides[k,seg,cs] + wIIcontra * rIms[k,seg,cs] + I_ext_I[k,seg,s]), b_I, theta_I)) + noise_s
                    I_ext_I[k,seg,s] = I_ext_I[k,seg,s] + noise_s
                    
                    if seg == n_segs-1:
                        noise_s = addnoise(noisemean, noisescale, dt)
                        #eq without +1 terms
                        # Calculate the derivative of the E population
                        drEms_sides[k,seg,s] = dt / tau_E * (-rEms_sides[k,seg,s] + (kmax_E - rEms_sides[k,seg,s]) * G(((EEadjtest*2 * rEms_sides[k,seg-1,s]) + (wEEself * rEms_sides[k,seg,s]) 
                                                                       + (EIadjtest*2 * rIms[k,seg-1,s]) + (wEIself * rIms[k,seg,s])
                                                                       + (wEEcontra * rEms_sides[k,seg,cs]) + (wEIcontra * rIms[k,seg,cs]) + I_ext_E[k,seg,s]), b_E, theta_E)) + noise_s
                        I_ext_E[k,seg,s] = I_ext_E[k,seg,s] + noise_s
                        
                    elif seg == 0:
                        noise_s = addnoise(noisemean, noisescale, dt)
                        #eq without the i-1 terms
                        # Calculate the derivative of the E population
                        drEms_sides[k,seg,s] = dt / tau_E * (-rEms_sides[k,seg,s] + (kmax_E - rEms_sides[k,seg,s]) * G(((wEEself * rEms_sides[k,seg,s]) + (EEadjtest*2 * rEms_sides[k,seg+1,s]) 
                                                                           + (wEIself * rIms[k,seg,s]) + (EIadjtest*2 * rIms[k,seg+1,s]) 
                                                                           + (wEEcontra * rEms_sides[k,seg,cs]) + (wEIcontra * rIms[k,seg,cs]) + I_ext_E[k,seg,s]), b_E, theta_E)) + noise_s
                        I_ext_E[k,seg,s] = I_ext_E[k,seg,s] + noise_s
                        
                    else: #seg a2-7
                        noise_s = addnoise(noisemean, noisescale, dt)
                        #the eq's for all mid segs
                        drEms_sides[k,seg,s] = dt / tau_E * (-rEms_sides[k,seg,s] + (kmax_E - rEms_sides[k,seg,s]) * G(((EEadjtest * rEms_sides[k,seg-1,s]) + (wEEself * rEms_sides[k,seg,s]) + (EEadjtest * rEms_sides[k,seg+1,s]) 
                                                                       + (EIadjtest * rIms[k,seg-1,s]) + (wEIself * rIms[k,seg,s]) + (EIadjtest * rIms[k,seg+1,s]) 
                                                                       + (wEEcontra * rEms_sides[k,seg,cs]) + (wEIcontra * rIms[k,seg,cs]) + I_ext_E[k,seg,s]), b_E, theta_E)) + noise_s
                        I_ext_E[k,seg,s] = I_ext_E[k,seg,s] + noise_s

                    # Update using Euler's method
                    if rEms_sides[k+1,seg,s] == 0:
                        rEms_sides[k+1,seg,s] = float(rEms_sides[k,seg,s] + drEms_sides[k,seg,s])
                    else:
                        rEms_sides[k+1,seg,s] = rEms_sides[k+1,seg,s]
                    if rIms[k+1,seg,s] == 0:
                        rIms[k+1,seg,s] = float(rIms[k,seg,s] + drIms[k,seg,s])
                    else:
                        rIms[k+1,seg,s] = rIms[k+1,seg,s]
                    rEms_gates = []
        
    return rEms_sides, rEms_gates, rIms, I_ext_E, I_ext_I