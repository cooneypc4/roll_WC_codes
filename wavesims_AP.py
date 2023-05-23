#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 14:35:30 2023

@author: PatriciaCooney
"""
# Imports
import matplotlib.pyplot as plt
import numpy as np

#%% simulate wave values for set wave speed and segments A8, A1
#parameters
simsteps = 100
dt = 0.1
wavespeed = 0.4
seg = np.array([0,7])

#%%equation & simulate
epsilon = np.ones([len(seg), simsteps])
for segi,segn in enumerate(seg):
    for step in np.arange(simsteps):
        epsilon[segi,step] = segn + wavespeed*step*dt

wave_plot(simsteps,epsilon)
#%%
#plot convergence of activity in isolated E or I pops over time
def wave_plot(simsteps, epsilon):
  fig,ax = plt.figure()
  plt.plot(np.arange(simsteps), epsilon[0,:], 'b', label='A1')
  plt.plot(np.arange(simsteps), epsilon[1,:], 'b', label='A8')
  ax.set_ylabel('Time')
  ax.set_xlabel('Segments')
  plt.legend(loc='best')

  plt.tight_layout()
  plt.show()
  fig.savefig('isolated_timeplot.svg', format = 'svg', dpi = 1200)
  
  
#%% once figure out how to appropriately determine the wave for A1 and A8 over time,
#use the following equations to find effective couplings for different wave speeds to occur

#%% Coupled equations to find dE, dI values for A1 and A8 over simulation run
