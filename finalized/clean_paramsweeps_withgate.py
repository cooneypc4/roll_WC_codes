#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 15:08:47 2023

@author: PatriciaCooney
"""
#%% import packages
import numpy as np
from WCparamsetup import default_pars
from larva_WC_fxn import simulate_wc_multiseg
from outputchecks import motor_output_check
from modelplots import heatmaps_allparams_allphis

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

gate = 1

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
                                                                    perturb_init = perturb_init, perturb_input = perturb_input, noisemean = noisemean, noisescale = noisescale, gate = gate))
  
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
        
        