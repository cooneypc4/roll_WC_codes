#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 16:10:34 2023

@author: PatriciaCooney
"""
#%%imports
import numpy as np
from ringfxn_ringparamsetup import simtheta
import random
#%% setup parameters
###shared
dt = np.pi*0.1
dur = 50
timesteps = np.arange(0,dur,dt)
nodes = np.arange(10)

random.seed(24)
###ring x
omegax = np.random.normal(1,0.1,len(nodes))
#omegax = np.repeat(1,len(nodes))
#omegaxeach = np.linspace(0.1,np.pi,6)

###ring y 
omegay = np.random.normal(1,0.1,len(nodes))

#%% simulate the rings-uncoupled
omega = [omegax,omegay]
phi_ij = [phi_xij, phi_yij]
num_rings = np.arange(2)

un_theta, un_dtheta = simtheta(num_rings,nodes,omega,phi_ij,phi_xy,0,0,timesteps)

plotwaveovertime(nodes, timesteps, np.sin(un_theta[:,:,0]), np.sin(un_theta[:,:,1]), 'uncoupled_neighbors_rings_sine_simulated_')

#%% simulate the rings - coupled within
omega = [omegax,omegay]
phi_ij = [phi_xij, phi_yij]
num_rings = np.arange(2)

couple_within_theta, couple_within_dtheta = simtheta(num_rings,nodes,omega,phi_ij,phi_xy,a_within_sin,0,timesteps)

#%% simulate the rings - coupled across
omega = [omegax,omegay]
phi_ij = [phi_xij, phi_yij]
num_rings = np.arange(2)

couple_both_theta, couple_both_dtheta = simtheta(num_rings,nodes,omega,phi_ij,phi_xy,a_within_sin,a_between_sin,timesteps)