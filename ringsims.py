#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 15:41:30 2023

@author: PatriciaCooney
"""
# Imports
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sb
#%% plotting fxns

#plot node traces over time
def plotmodovertime(nodes,timesteps_local,x_freqs,omegax,coupling_within,coupling_between,plotti,ti):
    fig,ax = plt.subplots()
    for node in nodes:
        #subplots for each node
        plt.subplot(len(nodes), 1, node+1)
        plt.plot(timesteps_local,x_freqs[node,:]%(2*np.pi), color = "b")
        plt.ylim([0,2*np.pi])
        for i in np.arange(x_freqs.shape[1]):
            if i > 0:
                p = (x_freqs[node,i]%(2*np.pi))
                o = (x_freqs[node,i-1]%(2*np.pi))
                if i < len(timesteps_local)-1:
                    q = (x_freqs[node,i+1]%(2*np.pi))
                    if p > o and p > q:
                        plt.axvline(timesteps_local[i],0,2*np.pi,color = 'g')

        if node == 0:
            plt.title(plotti)
        if node == len(nodes)/2:
            plt.ylabel('Nodes', fontsize=10)
        if node == nodes[-1]:
            plt.xlabel('Time', fontsize=10)
    
    nodenames = ['n'+str(nodes[i]) for i in nodes]
    allstrs = [('within coupling = ' + str(coupling_within)), ('between coupling = ' + str(coupling_between))]
    for i,n in enumerate(nodenames):
        allstrs.append(n + '=' + str(omegax[i]))
    #for loop looping list - textstr, plot, add inc
    inc=0
    for l in allstrs:
        plt.gcf().text(0.92, 0.92+inc, l, fontsize = 6)
        inc = inc-0.025

    #plt.tight_layout()
    plt.show()
    fig.savefig((ti+'_activityovertime.svg'), format = 'svg', dpi = 1200)
    fig.savefig((ti+'_activityovertime.png'), format = 'png', dpi = 1200)


#plot both rings, node traces over time
def plotbothmodovertime(nodes,timesteps_local,x_freqs,y_freqs,omegaeach,coupling_within,coupling_between,plotti,ti):
    fig,ax = plt.subplots()
    for node in nodes:
        #subplots for each node
        plt.subplot(len(nodes), 1, node+1)
        plt.plot(timesteps_local,x_freqs[node,:]%(2*np.pi), color = "b")
        plt.plot(timesteps_local,y_freqs[node,:]%(2*np.pi), color = "m")
        plt.ylim([0,2*np.pi])
        for i in np.arange(x_freqs.shape[1]):
            if i > 0:
                #check x peaks
                p = (x_freqs[node,i]%(2*np.pi))
                o = (x_freqs[node,i-1]%(2*np.pi))
                if i < len(timesteps_local)-1:
                    q = (x_freqs[node,i+1]%(2*np.pi))
                    if p > o and p > q:
                        plt.axvline(timesteps_local[i],0,2*np.pi,color = 'g')
                #check y peaks
                p = (y_freqs[node,i]%(2*np.pi))
                o = (y_freqs[node,i-1]%(2*np.pi))
                if i < len(timesteps_local)-1:
                    q = (y_freqs[node,i+1]%(2*np.pi))
                    if p > o and p > q:
                        plt.axvline(timesteps_local[i],0,2*np.pi,color = 'y')

        if node == 0:
            plt.title(plotti)
        if node == len(nodes)/2:
            plt.ylabel('Nodes', fontsize=10)
        if node == nodes[-1]:
            plt.xlabel('Time', fontsize=10)
    
    nodenames = ['n'+str(nodes[i]) for i in nodes]
    allstrs = [('within coupling = ' + str(coupling_within)), ('between coupling = ' + str(coupling_between))]
    for i,n in enumerate(nodenames):
        allstrs.append(n + '=' + str(omegaeach[i]))
    #for loop looping list - textstr, plot, add inc
    inc=0
    for l in allstrs:
        plt.gcf().text(0.92, 0.92+inc, l, fontsize = 6)
        inc = inc-0.025

    #plt.tight_layout()
    plt.show()
    fig.savefig((ti+'_BOTH_activityovertime.svg'), format = 'svg', dpi = 1200)
    fig.savefig((ti+'_BOTH_activityovertime.png'), format = 'png', dpi = 1200)
    

#plot phase difference evolution    
def plotphiovertime(phi_ijs,mean_phi_ij,omega,coupling_within,coupling_between,plotti,ti):
    #one time plot but with diff colors per node pair
    fig,ax = plt.subplots()
    nodenames = ['n'+str(nodes[i]) for i in nodes]
    if phi_ijs.shape[0]<=6:
        colorsn = ['m','b','c','g','y','r']
        for node in np.arange(phi_ijs.shape[0]):
            #diff lines
            plt.plot(np.arange(len(phi_ijs[0,:])), phi_ijs[node,:]%2*np.pi, c = colorsn[node], alpha = 0.5, label = nodenames[node])
    else:
        for node in np.arange(phi_ijs.shape[0]):
            #diff lines
            plt.plot(np.arange(len(phi_ijs[0,:])), phi_ijs[node,:]%2*np.pi, c = 'k', alpha = 0.4, label = nodenames[node])
    
    plt.plot(np.arange(len(phi_ijs[0,:])), np.mean(phi_ijs,0)%2*np.pi, color = 'k', alpha = 1, label = 'Average')
    
    plt.axhline(np.pi, color='r', ls='--')
    plt.axhline(2*np.pi, color='y', ls='--')
    plt.axhline(0, color='y', ls='--')
    plt.title(plotti)

    allstrs = [('within coupling = ' + str(coupling_within)), ('between coupling = ' + str(coupling_between))]
    for i,n in enumerate(nodenames):
        allstrs.append(n + '=' + str(omega[i]))
    #for loop looping list - textstr, plot, add inc
    inc=0
    for l in allstrs:
        plt.gcf().text(0.92, 0.92+inc, l, fontsize = 6)
        inc = inc-0.025
    
    #axis labels etc
    plt.ylim([-0.2,2*np.pi+0.2])
    plt.ylabel(r"$\phi$")
    plt.xlabel('Time')
    #plt.legend(loc='upper right', fontsize='x-small')

    #plt.tight_layout()
    plt.show()
    fig.savefig((ti+'_phasediffslong_discret0.6.svg'), format = 'svg', dpi = 1200)
    fig.savefig((ti+'_phasediffslong_discret0.6.png'), format = 'png', dpi = 1200)


def plotbothphiovertime(phi_xy,timesteps,omegaeach,coupling_within,coupling_between,plotti,ti):
    #one time plot but with diff colors per node pair
    fig,ax = plt.subplots()
    nodenames = ['n'+str(nodes[i]) for i in nodes]
    if phi_ijs.shape[0]<=6:
        colorsn = ['m','b','c','g','y','r']
        for node in np.arange(phi_ijs.shape[0]):
            #diff lines
            plt.plot(np.arange(len(phi_xy[0,:])), phi_xy[node,:]%2*np.pi, c = colorsn[node], alpha = 0.5)
    
    for node in np.arange(phi_ijs.shape[0]):
        #diff lines
        plt.plot(np.arange(len(phi_xy[0,:])), phi_xy[node,:]%2*np.pi, color = 'k', alpha = 0.4)
    plt.plot(np.arange(len(phi_xy[0,:])), np.mean(phi_xy,0)%2*np.pi, color = 'k', alpha = 0.8, label = 'Average')
    
    plt.axhline(np.pi, color='r', ls='--')
    plt.axhline(2*np.pi, color='y', ls='--')
    plt.axhline(0, color='y', ls='--')
    plt.title(plotti)
    #axis labels etc
    plt.ylim([-0.2,2*np.pi+0.2])
    plt.ylabel(r"$\phi$")
    plt.xlabel('Time')
    
    nodenames = ['n'+str(nodes[i]) for i in nodes]
    allstrs = [('within coupling = ' + str(coupling_within)), ('between coupling = ' + str(coupling_between))]
    for ring in range(2):
        if ring == 0:
            inc=0
            omegar = omegax
        elif ring == 1:
            inc = inc-0.05
            omegar = omegay
            allstrs = []
        for i,n in enumerate(nodenames):
            allstrs.append(n + '=' + str(omegar[i]))
        #for loop looping list - textstr, plot, add inc
        for l in allstrs:
            plt.gcf().text(0.92, 0.92+inc, l, fontsize = 6)
            inc = inc-0.025

    #plt.tight_layout()
    plt.show()
    fig.savefig((ti+'_phasediffsBETWEEN_discret0.6.svg'), format = 'svg', dpi = 1200)
    fig.savefig((ti+'_phasediffsBETWEEN_discret0.6.png'), format = 'png', dpi = 1200)
    
    
#plot heatmap through time
def plotheatmapthrutime(x_freqs,omegax,coupling_within,coupling_between,plotti,ti):
    f,ax = plt.subplots()
    sb.heatmap(x_freqs%(2*np.pi))
    ax.set(xlabel="Timesteps", ylabel="Nodes", title = plotti)
 
    nodenames = ['n'+str(nodes[i]) for i in nodes]
    allstrs = [('within coupling = ' + str(coupling_within)), ('between coupling = ' + str(coupling_between))]
    for i,n in enumerate(nodenames):
        allstrs.append(n + '=' + str(omegax[i]))
    #for loop looping list - textstr, plot, add inc
    inc=0
    for l in allstrs:
        plt.gcf().text(0.92, 0.92+inc, l, fontsize = 6)
        inc = inc-0.025
    plt.show()
    f.savefig(ti+'_heatmap_frequency_discret0.6.svg', format = 'svg', dpi = 1200)
    f.savefig(ti+'_heatmap_frequency_discret0.6.png', format = 'png', dpi = 1200)

#plot heatmaps through time
def plotbothheatmapthrutime(x_freqs,omegax,coupling_within,y_freqs,omegay,coupling_between,plotti,ti):
    f,(ax1,ax2) = plt.subplots(nrows=2)
    sb.heatmap(x_freqs%(2*np.pi),ax=ax1,cbar=False)
    ax1.set(ylabel="Nodes", title = plotti)
    ax1.set_xticks([])
    sb.heatmap(y_freqs%(2*np.pi),ax=ax2,cbar=False)
    ax2.set(xlabel="Timesteps", ylabel="Nodes")

    nodenames = ['n'+str(nodes[i]) for i in nodes]
    allstrs = [('within coupling = ' + str(coupling_within)), ('between coupling = ' + str(coupling_between))]
    for ring in range(2):
        if ring == 0:
            inc=0
            omegar = omegax
        elif ring == 1:
            inc = inc-0.05
            omegar = omegay
            allstrs = []
        for i,n in enumerate(nodenames):
            allstrs.append(n + '=' + str(omegar[i]))
        #for loop looping list - textstr, plot, add inc
        for l in allstrs:
            plt.gcf().text(0.92, 0.92+inc, l, fontsize = 6)
            inc = inc-0.025
    #plt.tight_layout()
    plt.show()
    f.savefig(ti+'_heatmap_frequency_discret0.6.svg', format = 'svg', dpi = 1200)
    f.savefig(ti+'_heatmap_frequency_discret0.6.png', format = 'png', dpi = 1200)
    
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
def calcinit(xnodes,ynodes,stype):
    if ynodes==[] and stype==0: #init for wave
        return np.array([((2*np.pi*(i))/len(nodes)) for i in nodes])
    elif ynodes==[] and stype==1: #init for sync
        return np.random.normal(1,0.1,len(nodes))
    elif stype==0: #init for wave
        return np.array([((2*np.pi*(i))/len(nodes)) for i in nodes]), np.array([((2*np.pi*(i))/len(nodes) + np.pi/2) for i in nodes]) #offset waves by pi, antiphase
    elif stype==1: #init for sync
        return np.random.normal(1,0.1,len(nodes)), np.random.normal((np.pi/2)+1,0.1,len(nodes)) #again, offset just to be equal to wave simulation
    
#%% ODE for simulating the ring nodes' activities
def simtheta(num_rings,nodes,omegax,xinit,omegay,yinit,a_within_sin,a_between_sin,timesteps):
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
                    if node == nodes[-1]:
                        dtheta[node,t,ring] = omega_n + (a_within_sin * np.sin(theta[node-1,t,ring]-theta[node,t,ring]) + a_within_sin * np.sin(theta[nodes[0],t,ring] - theta[node,t,ring])) + (a_between_sin * np.sin(theta[node,t,oppring] - theta[node,t,ring]))
                    elif node == 0:
                        dtheta[node,t,ring] = omega_n + (a_within_sin * np.sin(theta[nodes[-1],t,ring]-theta[node,t,ring]) + a_within_sin * np.sin(theta[node+1,t,ring] - theta[node,t,ring])) + (a_between_sin * np.sin(theta[node,t,oppring] - theta[node,t,ring]))
                    else:
                        dtheta[node,t,ring] = omega_n + (a_within_sin * np.sin(theta[node-1,t,ring]-theta[node,t,ring]) + a_within_sin * np.sin(theta[node+1,t,ring] - theta[node,t,ring])) + (a_between_sin * np.sin(theta[node,t,oppring]-theta[node,t,ring]))
            
                    theta[node,t+1,ring] = theta[node,t,ring] + dtheta[node,t,ring]
        else:
            omega_r = omegax
            for node in nodes:
                omega_n = omega_r[node]
                if node == nodes[-1]:
                    dtheta[node,t] = omega_n + (a_within_sin * np.sin(theta[node-1,t]-theta[node,t]) + a_within_sin * np.sin(theta[nodes[0],t] - theta[node,t]))
                elif node == 0:
                    dtheta[node,t] = omega_n + (a_within_sin * np.sin(theta[nodes[-1],t]-theta[node,t]) + a_within_sin * np.sin(theta[node+1,t] - theta[node,t]))
                else:
                    dtheta[node,t] = omega_n + (a_within_sin * np.sin(theta[node-1,t]-theta[node,t]) + a_within_sin * np.sin(theta[node+1,t]-theta[node,t]))
                
                theta[node,t+1] = float(theta[node,t] + dtheta[node,t])
                
    return theta, dtheta

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


#%% simulate single ring with diff omega and a values to find ideal params for traveling wave
#params for sim
num_rings = 1
a_within_range = np.array([0.35,0.375,0.4,0.425,0.45,0.475,0.5,0.525])
#a_within_range = np.arange(0,0.8,0.4)

#initialization values - set stype 0 = wave, 1 = global oscill
stype = 0
xinit = calcinit(nodes,[],stype) 
yinit = []
if stype == 0:
    ti_start = 'waveinit_singlering_'
elif stype == 1:
    ti_start = 'oscillinit_singlering_'


#simulate over different values
for ia,a_within in enumerate(a_within_range):
    ringx_thetas, ringx_dthetas = simtheta(num_rings,nodes,omegax,xinit,omegay,yinit,a_within,0,timesteps)
    
    #calculate phase diffs b/t nodes
    phi_ijs = calcphi_ij(ringx_thetas)
    mean_phi_ij = np.mean(phi_ijs,1)
    diff_mean_phi = abs(np.diff(mean_phi_ij))
    
    #plot phase diffs and activity
    plotphiovertime(phi_ijs,timesteps,omegax,a_within,[],'Ring X',ti_start+str(len(timesteps)) +'timesteps'+'_nodes'+str(len(nodes))+'couple_'+str(a_within)+'_')
    plotheatmapthrutime(ringx_thetas,omegax,a_within,[],'Ring X',ti_start+str(len(timesteps)) +'timesteps'+'_nodes'+str(len(nodes))+'couple_'+str(a_within)+'_')

        
#%% simulate a 2-ring system - waves and sync global oscillations
#params for sim
num_rings = 2
omegaeach = [omegax, omegay]
a_within = 0.375

#initialization values - set stype 0 = wave, 1 = global oscill
stype = 0
xinit,yinit = calcinit(nodes,nodes,stype) 
if stype == 0:
    ti_start = 'waveinit_tworings_'
elif stype == 1:
    ti_start = 'oscillinit_tworings_'

#a_between_range = ([-1,0,0.1])
a_between_range = np.arange(-0.5,0.5,0.1)

for a_between in a_between_range:
    #run sim & calc phi's for both rings - NO interring coupling
    rings_thetas, rings_dthetas = simtheta(num_rings,nodes,omegax,xinit,omegay,yinit,a_within,a_between,timesteps)
    phi_x_ijs = calcphi_ij(rings_thetas[:,:,0])
    phi_y_ijs = calcphi_ij(rings_thetas[:,:,1])
    
    phi_xy = calcphi_xy(rings_thetas[:,:,0],rings_thetas[:,:,1])
    
    #plot each ring's outputs
    # plotmodovertime(nodes,timesteps,rings_thetas[:,:,0],omegaeach,a_within,a_between,'Ring X','xring_'+ str(len(timesteps)) +'timesteps'+'couple_'+str(a_within)+'_'+str(omegaeach)+'_')
    # plotmodovertime(nodes,timesteps,rings_thetas[:,:,1],omegaeach,a_within,a_between,'Ring Y','yring_'+ str(len(timesteps)) +'timesteps'+'couple_'+str(a_within)+'_'+str(omegaeach)+'_')
    
    #plotphiovertime(phi_x_ijs,timesteps,omegaeach,a_within,a_between,'Ring X','xring_'+ str(len(timesteps)) +'timesteps'+'couple_'+str(a_within)+'_')
    #plotphiovertime(phi_y_ijs,timesteps,omegaeach,a_within,a_between,'Ring Y','yring_'+ str(len(timesteps)) +'timesteps'+'couple_'+str(a_within)+'_')
    
    plotbothheatmapthrutime(rings_thetas[:,:,0],omegax,a_within,rings_thetas[:,:,1],omegay,a_between,'Both Rings','bothrings_initdiff0.5pi_discret0.1pi_'+str(len(timesteps))+'timesteps'+'_couplewin_'+str(a_within)+'_couplebt_'+str(a_between)+'_')
    
    # plotbothmodovertime(nodes,timesteps,rings_thetas[:,:,0],rings_thetas[:,:,1],omegaeach,a_within,a_between,'Ring X & Ring Y',
    #                     'bothrings_'+str(len(timesteps))+'timesteps'+'_couplewin_'+str(a_within)+'_couplebt_'+str(a_between)+'_'+str(omegaeach)+'_')
    
    # plotbothphiovertime(phi_xy,timesteps,omegaeach,a_within,a_between,'Between Rings',
    #                     'bothrings_'+ str(len(timesteps)) +'timesteps'+'couple_'+str(a_within)+'_couplebt_'+str(a_between)+'_'+str(omegaeach)+'_')
    
    plotbothphiovertime(phi_xy,timesteps,omegaeach,a_within,a_between,'Between Rings',
                        'bothrings_initdiff0.5pi_discret0.1pi_'+ str(len(timesteps)) +'timesteps'+'couple_'+str(a_within)+'_couplebt_'+str(a_between)+'_')


####stopped here

#then test the antiphase and sync waves - between ring coupling
#do the same for global oscill's

#generate a plot that compares a_between with a_within in terms of each other? and in terms of phase diffs -- maybe plot as heatmap or somehting or curve? fig best way to show
#thenclean code and backup all + move plots out of Docs folder
#read papers and or start writing this all up in LaTex

    
#%% PREVIOUS stuff
#%% ODE for simulating the ring nodes' activities
# def simtheta(num_rings,nodes,omega,phi_ij,phi_xy,a_within_sin,a_between_sin,timesteps):
#     dtheta = np.zeros([len(nodes),len(timesteps),len(num_rings)])
#     theta = np.zeros([len(nodes),len(timesteps),len(num_rings)])
                
#     for i in np.arange(len(timesteps)-1):
#         for ring in num_rings:
#             omega_r = omega[ring]
#             if ring == 0:
#                 oppring = 1
#             else:
#                 oppring = 0
#             for node in nodes:
#                 if node > 0:
#                     dtheta[node,i,ring] = omega_r + (2*a_within_sin * np.sin(theta[node,i,ring]-theta[node-1,i,ring])) + (a_between_sin * np.sin(theta[node,i,ring]-theta[node,i,oppring]))
#                 else:
#                     dtheta[node,i,ring] = omega_r + (2*a_within_sin * np.sin(theta[node,i,ring]-theta[node+1,i,ring])) + (a_between_sin * np.sin(theta[node,i,ring]-theta[node,i,oppring]))
                
#                 theta[node,i+1,ring] = theta[node,i,ring] + dtheta[node,i,ring]
#     return theta, dtheta
# x_freqs = calcfreqs_travelwave(omegax,timesteps,nodes)
# y_freqs = calcfreqs_travelwave(omegay,timesteps,nodes)

# plotoverspace(nodes, timesteps, np.sin(x_freqs), np.sin(y_freqs), 'uncoupled_neighbors_rings_sine_')

# #%% calculate phase diff's phi (neighbor nodes, across rings)
# phi_xij = calcphi_ij(x_freqs)
# phi_yij = calcphi_ij(y_freqs)
# phi_xy = calcphi_xy(x_freqs,y_freqs)

#%% simulate the rings-uncoupled
# omega = [omegax,omegay]
# phi_ij = [phi_xij, phi_yij]
# num_rings = np.arange(2)

# un_theta, un_dtheta = simtheta(num_rings,nodes,omega,phi_ij,phi_xy,0,0,timesteps)

#plotwaveovertime(nodes, timesteps, np.sin(un_theta[:,:,0]), np.sin(un_theta[:,:,1]), 'uncoupled_neighbors_rings_sine_simulated_')

#%% simulate the rings - coupled within
# omega = [omegax,omegay]
# phi_ij = [phi_xij, phi_yij]
# num_rings = np.arange(2)

# couple_within_theta, couple_within_dtheta = simtheta(num_rings,nodes,omega,phi_ij,phi_xy,a_within_sin,0,timesteps)

#%% simulate the rings - coupled across
# omega = [omegax,omegay]
# phi_ij = [phi_xij, phi_yij]
# num_rings = np.arange(2)

# couple_both_theta, couple_both_dtheta = simtheta(num_rings,nodes,omega,phi_ij,phi_xy,a_within_sin,a_between_sin,timesteps)