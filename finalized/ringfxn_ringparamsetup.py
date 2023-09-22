#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 16:02:34 2023

Setup params for ring sims and fxn for running the simulations

@author: PatriciaCooney
"""
#%%imports
import numpy as np
#%% equations
#calculate thetas for each node w/ pre-defined traveling wave equation from Ermentrout & Ko, 2009
def calcfreqs_travelwave(omega,timesteps,nodes):
    freq = -np.ones([len(nodes),len(timesteps)])
    for node in nodes:
        for i,t in enumerate(timesteps):
            freq[node,i] = omega + (2*np.pi*node)/len(nodes)
    return freq

#calculate abs diff b/t nodes
def calcphi_ij(freqs):
    phi_ij = abs(np.diff(freqs,axis=0))
    phi_final = abs(freqs[-1,:] - freqs[0,:])
    phi_ij = np.vstack((phi_ij,phi_final))
    return phi_ij

#calculate phase diff phi b/t same node in the 2 rings
def calcphi_xy(x_freqs, y_freqs):
    return abs(y_freqs-x_freqs)

#calculate initialization values that should result in traveling wave solution
def calcinit(xnodes,ynodes,stype,itype):
    if ynodes==[] and stype==0: #init for wave
        return np.array([((2*np.pi*(i))/len(xnodes)) for i in xnodes])
    elif ynodes==[] and stype==1: #init for sync
        return np.random.normal(1,0.1,len(xnodes))
    elif stype==0: #init for wave
        if itype == 0: #in-phase
            return np.array([((2*np.pi*(i))/len(xnodes)) for i in xnodes]), np.array([((2*np.pi*(i))/len(ynodes)) for i in ynodes]) #offset waves
        elif itype == 1: #quarter phase
            return np.array([((2*np.pi*(i))/len(xnodes)) for i in xnodes]), np.array([((2*np.pi*(i))/len(ynodes) + np.pi/2) for i in ynodes]) #offset waves
        elif itype == 2: #antiphase
            return np.array([((2*np.pi*(i))/len(xnodes)) for i in xnodes]), np.array([((2*np.pi*(i))/len(ynodes) + np.pi) for i in ynodes]) #offset waves
    elif stype==1: #init for oscill
        if itype == 0: #in-phase
            return np.random.normal(1,0.1,len(xnodes)), np.random.normal(1,0.1,len(ynodes)) #again, offset just to be equal to wave simulation
        elif itype == 1: #quarter phase
            return np.random.normal(1,0.1,len(xnodes)), np.random.normal(1,0.1,len(ynodes)) + np.pi/2
        elif itype == 2: #antiphase
            return np.random.normal(1,0.1,len(xnodes)), np.random.normal(1,0.1,len(ynodes)) + np.pi
    
#%% ODE for simulating the ring nodes' activities
def simtheta(num_rings,nodes,omegax,xinit,omegay,yinit,nm,nv,a_within_sin,a_between_sin,timesteps):
    if num_rings>1:
        #w/ different initial conditions for each ring
        dtheta = np.zeros([len(nodes),len(timesteps),num_rings])
        theta_x = np.zeros([len(nodes),len(timesteps)])
        theta_y = np.zeros([len(nodes),len(timesteps)])
        #init values
        theta_x[:,0] = xinit
        theta_y[:,0] = yinit
        #combined
        theta = np.dstack((theta_x,theta_y))
        
        #print(theta)
    else:
        #w/ different initial conditions for each ring
        dtheta = np.zeros([len(nodes),len(timesteps)])
        theta = np.zeros([len(nodes),len(timesteps)])
        #init values
        theta[:,0] = xinit
        
    for t in range(len(timesteps)-1):
        if num_rings>1:
            for ring in np.arange(num_rings):
                if ring == 0:
                    oppring = 1
                    omega_r = omegax
                else:
                    oppring = 0
                    omega_r = omegay
                for node in nodes:
                    omega_n = omega_r[node]
                    noise_n = np.random.normal(nm,nv)
                    #print(noise_n)
                    if node == nodes[-1]:
                        dtheta[node,t,ring] = omega_n + (a_within_sin * np.sin(theta[node-1,t,ring]-theta[node,t,ring]) + a_within_sin * np.sin(theta[nodes[0],t,ring] - theta[node,t,ring])) + (a_between_sin * np.sin(theta[node,t,oppring] - theta[node,t,ring])) + noise_n
                    elif node == 0:
                        dtheta[node,t,ring] = omega_n + (a_within_sin * np.sin(theta[nodes[-1],t,ring]-theta[node,t,ring]) + a_within_sin * np.sin(theta[node+1,t,ring] - theta[node,t,ring])) + (a_between_sin * np.sin(theta[node,t,oppring] - theta[node,t,ring])) + noise_n
                    else:
                        dtheta[node,t,ring] = omega_n + (a_within_sin * np.sin(theta[node-1,t,ring]-theta[node,t,ring]) + a_within_sin * np.sin(theta[node+1,t,ring] - theta[node,t,ring])) + (a_between_sin * np.sin(theta[node,t,oppring]-theta[node,t,ring])) + noise_n
            
                    theta[node,t+1,ring] = theta[node,t,ring] + dtheta[node,t,ring]
        else:
            omega_r = omegax
            for node in nodes:
                omega_n = omega_r[node]
                noise_n = np.random.normal(nm,nv)
                if node == nodes[-1]:
                    dtheta[node,t] = omega_n + (a_within_sin * np.sin(theta[node-1,t]-theta[node,t]) + a_within_sin * np.sin(theta[nodes[0],t] - theta[node,t])) + noise_n
                elif node == 0:
                    dtheta[node,t] = omega_n + (a_within_sin * np.sin(theta[nodes[-1],t]-theta[node,t]) + a_within_sin * np.sin(theta[node+1,t] - theta[node,t])) + noise_n
                else:
                    dtheta[node,t] = omega_n + (a_within_sin * np.sin(theta[node-1,t]-theta[node,t]) + a_within_sin * np.sin(theta[node+1,t]-theta[node,t])) + noise_n
                
                theta[node,t+1] = float(theta[node,t] + dtheta[node,t])
                
    return theta, dtheta

