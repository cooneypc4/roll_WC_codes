#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 12:43:51 2023

Run individual simulation for crawl and roll w/ specified params

@author: PatriciaCooney
"""
#%% imports
import numpy as np
from larva_WC_fxn import simulate_wc_multiseg
from WCparamsetup import default_pars
from modelplots import plotbothsidesheatmap, plot_extinput

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

#no gating mechanism
gate = 0

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

#contraEE = 0
contraEE = 3
#contraEE = 5
#contraEI = 5
contraEI = 0
contraIE = 0 
#contraIE = -5
#contraII = -5
contraII = 0
#contraII = -20
#contraII = -12

contra_weights = [contraEE,contraEI,contraIE,contraII]

#no perturbations       
perturb_init = [0]
perturb_input = [0]

#run that sim
rEms,rIms,I_ext_E,I_ext_I = simulate_wc_multiseg(**default_pars(n_sides=n_sides, sim_input=sim_input,
                                                    pulse_vals=pulse_vals, EEadjtest=wEEadjtest, EIadjtest=wEIadjtest, contra_weights=contra_weights, 
                                                    offsetcontra = offsetcontra, contra_dur = contra_dur,
                                                    offsetcontra_sub = offsetcontra_sub, contra_dur_sub = contra_dur_sub,
                                                    perturb_init = perturb_init, perturb_input = perturb_input, noisemean = noisemean, noisescale = noisescale, gate = gate))

#plot the activity heatmap
plotbothsidesheatmap(pars,n_t,rEms,pulse_vals,contra_weights,offsetcontra,contra_dur,perturb_input,[],plotti = simname[sim],ti = simname[sim] + '_GOROINPUT_testalt_EE=3_EulerMarnoise_0m_0.005*Iv_')
#plot the external input used
plot_extinput(I_ext_E,I_ext_I,n_t,pulse_vals,noisescale,pars['dt'],ti = simname[sim],savetitle = simname[sim] + '_GOROINPUT_testalt_EE=3_EulerMarnoise_0m_0.005*Iv_')