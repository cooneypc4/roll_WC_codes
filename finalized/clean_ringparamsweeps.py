#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 15:57:59 2023

run ring simulations over param sweep

@author: PatriciaCooney
"""
#%%imports
import numpy as np
from ringfxn_ringparamsetup import simtheta, calcinit, calcphi_ij, calcphi_xy
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

#%% run the ring sims 1000x each and replot heatmaps of phase diffs -- does initialization really matter?
#params for sim
num_rings = 2
omegaeach = [omegax, omegay]
#a_within = 0.375

#set num sims + ranges params
#a_between_range = ([-1,0,0.1])
it_range = np.arange(0,50)
a_between_range = np.arange(-0.5,0.55,0.05).round(3)
a_within_range = np.arange(0,0.8,.05).round(3)
#set the noise params = mean, var for Gauss draw to test (white noise, nm=0,nv=0.1, made 0.1 b/c 1 = too high -- dominates frequencies - this is congruent with random draw init)
nm = 0
nv = 0.1


styperange = [0,1]
ityperange = [0,1,2]

for stit in styperange:
    for itit in ityperange:
        #initialization values - set stype 0 = wave, 1 = global oscill
        stype = stit
        itype = itit #0 = in-phase init, 1 = quarter-phase init, 2 = antiphase init
        xinit,yinit = calcinit(nodes,nodes,stype,itype) 
        if stype == 0:
            ti_start = 'waveinit_tworings_'
        elif stype == 1:
            ti_start = 'oscillinit_tworings_'
        if itype == 0:
            ti_start = ti_start + '0diff_' 
        elif itype == 1:
            ti_start = ti_start + '0.5pidiff_'
        elif itype == 2:
            ti_start = ti_start + 'pidiff_'
        
        #pre-set the arrays
        meanphixy_allsims = -np.ones([len(a_within_range),len(a_between_range),len(it_range)])
        varphixy_allsims = -np.ones([len(a_within_range),len(a_between_range),len(it_range)])
        meanphix = -np.ones([len(a_within_range),len(a_between_range),len(it_range)])
        varphix = -np.ones([len(a_within_range),len(a_between_range),len(it_range)])
        meanphiy = -np.ones([len(a_within_range),len(a_between_range),len(it_range)])
        varphiy = -np.ones([len(a_within_range),len(a_between_range),len(it_range)])
        
        #make plots of example ring dynamics
        #listplotwin = [0.2, 0.25, 0.3, 0.35]
        #listplotbt = [0.05, 0.15, 0.2, 0.25]
        listplotwin = []
        listplotbt = []
        
        for it in it_range:
            for ai,a_within in enumerate(a_within_range):
                for ab,a_between in enumerate(a_between_range):
                    #run sim & calc phi's for both rings - NO interring coupling
                    rings_thetas, rings_dthetas = simtheta(num_rings,nodes,omegax,xinit,omegay,yinit,nm,nv,a_within,a_between,timesteps)
                    phi_x_ijs = calcphi_ij(rings_thetas[:,:,0])
                    phi_y_ijs = calcphi_ij(rings_thetas[:,:,1])
                    
                    phi_xy = calcphi_xy(rings_thetas[:,:,0],rings_thetas[:,:,1])
                    
                    #plot each ring's outputs
                    if a_within in listplotwin and a_between in listplotbt:
                        plotbothheatmapthrutime(rings_thetas[:,:,0],omegax,a_within,rings_thetas[:,:,1],omegay,a_between,'Both Rings','bothrings_'
                                                +ti_start+str(len(it_range))+'x_'+'discret0.1pi_'+str(len(timesteps))+'timesteps'+'_noise['+str(nm)+','+str(nv)+']'+'_couplewin_'+str(a_within)+'_couplebt_'+str(a_between)+'_')    
                        plotbothphiovertime(phi_xy,timesteps,omegaeach,a_within,a_between,'Between Rings','bothrings_'
                                            +ti_start+str(len(it_range))+'x_'+'discret0.1pi_'+ str(len(timesteps)) +'timesteps'+'_noise['+str(nm)+','+str(nv)+']'+'couple_'+str(a_within)+'_couplebt_'+str(a_between)+'_')
                    
                    #store this phi_xy and phi_ij
                    #x_ij - so wave or oscill output; y_ij
                    meanphix[ai,ab,it] = np.mean(phi_x_ijs[:,int(phi_x_ijs.shape[1]/2):]%2*np.pi)
                    varphix[ai,ab,it] = np.var(phi_x_ijs[:,int(phi_x_ijs.shape[1]/2):]%2*np.pi)
                    meanphiy[ai,ab,it] = np.mean(phi_y_ijs[:,int(phi_y_ijs.shape[1]/2):]%2*np.pi)
                    varphiy[ai,ab,it] = np.var(phi_y_ijs[:,int(phi_y_ijs.shape[1]/2):]%2*np.pi)
                    #sync or async x,y
                    meanphixy_allsims[ai,ab,it] = np.mean(phi_xy[:,int(phi_xy.shape[1]/2):]%2*np.pi)
                    varphixy_allsims[ai,ab,it] = np.var(phi_xy[:,int(phi_xy.shape[1]/2):]%2*np.pi)
        
        #summary all iterations
        #intra-ring
        meanphix_summsims = np.mean(meanphix,2)
        varphix_summsims = np.mean(varphix,2)
        meanphiy_summsims = np.mean(meanphiy,2)
        varphiy_summsims = np.mean(varphiy,2)
        #inter-ring
        meanphixy_summsims = np.mean(meanphixy_allsims,2)
        varphixy_summsims = np.mean(varphixy_allsims,2)
        
        #plot heatmap of the simulation params
        #heatmap_params_xyphis(meanphixy_summsims,varphixy_summsims,a_within_range,a_between_range,stype,itype,ti_start+str(len(it_range))+'x_'+'_noise['+str(nm)+','+str(nv)+']'+'_heatmap_params_xyphis')
        #heatmap_params_ijphis(meanphix_summsims,varphix_summsims,meanphiy_summsims,varphiy_summsims,a_within_range,a_between_range,stype,itype,ti_start+str(len(it_range))+'x_'+'_noise['+str(nm)+','+str(nv)+']'+'_heatmap_params_ijphis')
        #plot combo heatmap - intra and interring
        heatmaps_allparams_allphis(meanphix_summsims,varphix_summsims,meanphiy_summsims,varphiy_summsims,meanphixy_summsims,varphixy_summsims,a_within_range,a_between_range,stype,itype,ti_start+str(len(it_range))+'x_'+'_noise['+str(nm)+','+str(nv)+']'+'_heatmap_params_ijandxy')
  