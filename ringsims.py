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
#%% setup parameters
###shared
dt = 0.1
dur = 100
timesteps = np.arange(dur)*dt
nodes = np.arange(6)

###ring x
omegax = 2*np.pi*25
#omegaxeach = np.linspace(0.1,np.pi,6)

###ring y 
omegay = 2*np.pi*27.5

###coupling
# a_within_sin = 0.3158
# a_between_sin = 0.9831
#a_within_range = np.arange(0,10,0.5)
#just trying anything order 1 for now...
#should be order 1

#%% plotting fxns
# def plotoverspace(nodes,timesteps,x_freqs,y_freqs,ti):
#     fig,ax = plt.subplots()
#     timediv = round(len(timesteps)/8)
#     timesubs = np.arange(0,len(timesteps),timediv)
#     for subi,subt in enumerate(timesubs):
#         #subplots by timepoint
#         plt.subplot(len(timesubs)+1,1,subi+1)
#         plt.plot(x_freqs[nodes,subt], color = "b", label="ring x")
#         plt.plot(y_freqs[nodes,subt], color = "r", label="ring y")
#         plt.ylim([-1.5,1.5])

#         if subt == timesubs[4]:
#             plt.ylabel('Timesteps', fontsize=10)
#         if subt == timesubs[-1]:
#             plt.xlabel('Nodes', fontsize=10)
#     plt.legend(loc='lower right')

#     #plt.tight_layout()
#     plt.show()
#     fig.savefig((ti+'_wavepositionthrutime.svg'), format = 'svg', dpi = 1200)
#     fig.savefig((ti+'_wavepositionthrutime.png'), format = 'png', dpi = 1200)


# def plotovertime(nodes,timesteps,x_freqs,y_freqs,ti):
#     fig,ax = plt.subplots()
#     for node in nodes:
#         #subplots for each node
#         plt.subplot(len(nodes),1,node+1)
#         plt.plot(x_freqs[nodes,:], color = "b", label="ring x")
#         plt.plot(y_freqs[nodes,:], color = "r", label="ring y")
#         plt.ylim([-1.5,1.5])

#         if node == len(nodes)/2:
#             plt.ylabel('Nodes', fontsize=10)
#         if node == nodes[-1]:
#             plt.xlabel('Time', fontsize=10)
#     plt.legend(loc='lower right')

#     #plt.tight_layout()
#     plt.show()
#     fig.savefig((ti+'_activityovertime.svg'), format = 'svg', dpi = 1200)
#     fig.savefig((ti+'_activityovertime.png'), format = 'png', dpi = 1200)

#plot node traces over time
def plotmodovertime(nodes,timesteps_local,x_freqs,omegaxeach,coupling_within,coupling_between,plotti,ti):
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
    
    nodenames = ['n0','n1','n2','n3','n4','n5']
    allstrs = [('within coupling = ' + str(coupling_within)), ('between coupling = ' + str(coupling_between))]
    for i,n in enumerate(nodenames):
        allstrs.append(n + '=' + str(omegaxeach[i]))
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
    
    nodenames = ['n0','n1','n2','n3','n4','n5']
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
def plotphiovertime(phi_ijs,mean_phi_ij,omegaxeach,coupling_within,coupling_between,plotti,ti):
    #one time plot but with diff colors per node pair
    fig,ax = plt.subplots()
    colorsn = ['m','b','c','g','y','r']
    nodenames = ['n0','n1','n2','n3','n4','n5']
    for node in np.arange(phi_ijs.shape[0]):
        #diff lines
        plt.plot(np.arange(len(phi_ijs[0,:])), phi_ijs[node,:]%2*np.pi, c = colorsn[node], alpha = 0.5, label = nodenames[node])
    
    plt.plot(np.arange(len(phi_ijs[0,:])), np.mean(phi_ijs,0)%2*np.pi, color = 'k', alpha = 0.4, label = 'Average')
    
    plt.axhline(np.pi, color='r', ls='--')
    plt.axhline(2*np.pi, color='y', ls='--')
    plt.axhline(0, color='y', ls='--')
    plt.title(plotti)

    allstrs = [('within coupling = ' + str(coupling_within)), ('between coupling = ' + str(coupling_between))]
    for i,n in enumerate(nodenames):
        allstrs.append(n + '=' + str(omegaxeach[i]))
    #for loop looping list - textstr, plot, add inc
    inc=0
    for l in allstrs:
        plt.gcf().text(0.92, 0.92+inc, l, fontsize = 6)
        inc = inc-0.025
    
    #axis labels etc
    plt.ylim([-0.2,2*np.pi+0.2])
    plt.ylabel(r"$\phi$")
    plt.xlabel('Time')
    plt.legend(loc='upper right', fontsize='x-small')

    #plt.tight_layout()
    plt.show()
    fig.savefig((ti+'_phasediffslong.svg'), format = 'svg', dpi = 1200)
    fig.savefig((ti+'_phasediffslong.png'), format = 'png', dpi = 1200)


def plotbothphiovertime(phi_xy,timesteps,omegaeach,coupling_within,coupling_between,plotti,ti):
    #one time plot but with diff colors per node pair
    fig,ax = plt.subplots()
    colorsn = ['m','b','c','g','y','r']
    nodenames = ['n0','n1','n2','n3','n4','n5']
    for node in np.arange(phi_ijs.shape[0]):
        #diff lines
        plt.plot(np.arange(len(phi_xy[0,:])), phi_xy[node,:]%2*np.pi, c = colorsn[node], alpha = 0.5, label = nodenames[node])
    
    plt.plot(np.arange(len(phi_xy[0,:])), np.mean(phi_xy,0)%2*np.pi, color = 'k', alpha = 0.4, label = 'Average')
    
    plt.axhline(np.pi, color='r', ls='--')
    plt.axhline(2*np.pi, color='y', ls='--')
    plt.axhline(0, color='y', ls='--')
    plt.title(plotti)

    allstrs = [('within coupling = ' + str(coupling_within)), ('between coupling = ' + str(coupling_between))]
    for i,n in enumerate(nodenames):
        allstrs.append(n + '=' + str(omegaxeach[i]))
    #for loop looping list - textstr, plot, add inc
    inc=0
    for l in allstrs:
        plt.gcf().text(0.92, 0.92+inc, l, fontsize = 6)
        inc = inc-0.025
    
    #axis labels etc
    plt.ylim([-0.2,2*np.pi+0.2])
    plt.ylabel(r"$\phi$")
    plt.xlabel('Time')
    plt.legend(loc='upper right', fontsize='x-small')

    #plt.tight_layout()
    plt.show()
    fig.savefig((ti+'_phasediffsBETWEEN.svg'), format = 'svg', dpi = 1200)
    fig.savefig((ti+'_phasediffsBETWEEN.png'), format = 'png', dpi = 1200)
    
#%% equations
#calculate thetas for each node w/ pre-defined traveling wave equation from Ermentrout & Ko, 2009
def calcfreqs_travelwave(omega,timesteps,nodes):
    freq = -np.ones([len(nodes),len(timesteps)])
    for node in nodes:
        for i,t in enumerate(timesteps):
            freq[node,i] = omega * t + (2*np.pi*node)/len(nodes)
    return freq

#calculate abs diff b/t nodes
def calcphi_ij(freqs):
    phi_ij = abs(np.diff(freqs,axis=0))
    phi_final = abs(freqs[-1,:] - freqs[0,:])
    phi_ij = np.vstack((phi_ij,phi_final))
    return phi_ij

#calculate phase diff phi b/t same node in the 2 rings
def calcphi_xy(x_freqs, y_freqs):
    phi_xy = abs(x_freqs-y_freqs)
    return phi_xy

#%% ODE for simulating the ring nodes' activities
def simtheta(num_rings,nodes,omega,a_within_sin,a_between_sin,timesteps):
    if num_rings>1:
        #w/ different initial conditions for each ring
        dtheta = np.zeros([len(nodes),len(timesteps),num_rings])
        #ring x - init at zero
        theta_x = np.zeros([len(nodes),len(timesteps)])

        #ring y - init with randints
        theta_y = np.zeros([len(nodes),len(timesteps)])
        #from prev random init values - keep using these-
        theta_y[:,0] = np.array([0.49410184, 0.88569995, 0.99252273, 0.46113613, 0.57734853, 0.75677658])
        #theta_y[:,0] = np.random.random(len(nodes))
        #print(theta_y[:,0])
        #combined
        theta = np.dstack((theta_x,theta_y))
        
        #print(theta)
    else:
        dtheta = np.zeros([len(nodes),len(timesteps)])
        theta = np.zeros([len(nodes),len(timesteps)])
        
    for t in np.arange(len(timesteps)-1):
        if num_rings>1:
            for ring in np.arange(num_rings):
                omega_r = omega
                if ring == 0:
                    oppring = 1
                else:
                    oppring = 0
                for node in nodes:
                    omega_n = omega_r[node]
                    if node == 5:
                        dtheta[node,t,ring] = omega_n + (a_within_sin * np.sin(theta[node,t,ring]-theta[node-1,t,ring]) + a_within_sin * np.sin(theta[node,t,ring] - theta[nodes[0],t,ring])) + (a_between_sin * np.sin(theta[node,t,ring] - theta[node,t,oppring]))
                    elif node == 0:
                        dtheta[node,t,ring] = omega_n + (a_within_sin * np.sin(theta[node,t,ring]-theta[nodes[-1],t,ring]) + a_within_sin * np.sin(theta[node,t,ring] - theta[node+1,t,ring])) + (a_between_sin * np.sin(theta[node,t,ring] - theta[node,t,oppring]))
                    else:
                        dtheta[node,t,ring] = omega_n + (a_within_sin * np.sin(theta[node,t,ring]-theta[node-1,t,ring]) + a_within_sin * np.sin(theta[node,t,ring] - theta[node+1,t,ring])) + (a_between_sin * np.sin(theta[node,t,ring]-theta[node,t,oppring]))
            
                    theta[node,t+1,ring] = theta[node,t,ring] + dtheta[node,t,ring]
        else:
            omega_r = omega
            for node in nodes:
                omega_n = omega_r[node]
                if node == 5:
                    dtheta[node,t] = omega_n + (a_within_sin * np.sin(theta[node,t]-theta[node-1,t]) + a_within_sin * np.sin(theta[node,t] - theta[nodes[0],t]))
                elif node == 0:
                    dtheta[node,t] = omega_n + (a_within_sin * np.sin(theta[node,t]-theta[nodes[-1],t]) + a_within_sin * np.sin(theta[node,t] - theta[node+1,t]))
                else:
                    dtheta[node,t] = omega_n + (a_within_sin * np.sin(theta[node,t]-theta[node-1,t]) + a_within_sin * np.sin(theta[node,t]-theta[node+1,t]))
                
                theta[node,t+1] = float(theta[node,t] + dtheta[node,t])
                
    return theta, dtheta

#%% TO DO - 5/15
#X use simtheta w/ different omega's for each node in the single ring
#X and set to only 1 ring being simulated
#X iterate over different values of ring to solve for their activities -- see E & Ko for value ranges...
#X need to set up vars such that loops for each node being different (ie diff omega's for each node)
#X don't use circular numbering, just keep as is and set somehow such that node 0 and node 5 do feed into each other, b/c need repeated traveling waves around the ring

#X figure out how you want to check for the wave -- just do plotting? or do some metric for measuring -- maybe an internode interval?
    ###options for metric -- could select any variations that have some phase diff of neighbor nodes that is greater than 0 and less than 2pi for starters -- plot those? and maybe also that diff of phi b/t all pairs is 0 or very small? then plot
    ###also consider -- can I just check if the theta_node = (omega_node*t) + (2pi*node)/num_nodes --- this is the equation that E & Ko write... but Ashok said that we will see wave first, then write eq to describe

#X 5/17 - great, working - see if metric works in only pulling out/plotting the solution that worked
#X then check phi's; optimize such that plot two-like soltion would be pulled out always
#X then do long plot on plot 2--stable wave? - plot phi's instead probably

#then second ring, SETUP SECOND RING WITH A DIFF INITIALIZATION
#plot two rings w/o coupling

#couple rings, use metric to pull out, plot
#%% simulate single ring with diff omega and a values to find ideal params for traveling wave
#params for sim
num_rings = 1
omegaxeach = np.linspace(0.1,0.2*np.pi,6)
#a_within_range = np.array([0,0.5])
a_within_range = np.arange(0,1.5,0.05)

#storage of sim params that match the criteria for potential wave
solution_couple = -np.ones(a_within_range.size)
solution_omegas = -np.ones(omegaxeach.size)
solution_out_theta = -np.ones([len(nodes),len(timesteps),a_within_range.size])
solution_out_phi = -np.ones([len(nodes),len(timesteps),a_within_range.size])

#simulate over different values
for ia,a_within in enumerate(a_within_range):
    ringx_thetas, ringx_dthetas = simtheta(num_rings,nodes,omegaxeach,a_within,0,timesteps)
    
    #calculate phase diffs b/t nodes
    phi_ijs = calcphi_ij(ringx_thetas)
    mean_phi_ij = np.mean(phi_ijs,1)
    diff_mean_phi = abs(np.diff(mean_phi_ij))
    
    #calc phase diffs b/t rings
    #phi_xy = calcphi_xy(ringx_thetas,ringy_thetas)
    
    #check for simulations where differences between the phase differences of neighbor pairs across the whole ring are not large; check that phase difference for all pairs is > than 0
    #if all(np.where(mean_diff_ij<2*np.pi,1,0)==1) and all(np.where(diff_mean_diffs<0.2*np.pi,1,0)==1): 
    #if a_within == 0.5:
    #plot behavior - traces + phi's
    plotmodovertime(nodes,timesteps,ringx_thetas,omegaxeach,a_within,'singlering_100timesteps'+'couple_'+str(a_within)+'_'+str(omegaxeach)+'_')
    #plotphiovertime(phi_ijs,timesteps,omegaxeach,a_within,'singlering_50timesteps_'+'couple_'+str(a_within)+'_'+str(omegaxeach)+'_')
    #store solutions
    solution_couple[ia] = a_within
    solution_omegas = omegaxeach #for now - change this when do iterations on omega
    solution_out_theta[:,:,ia] = ringx_thetas
    solution_out_phi[:,:,ia] = phi_ijs
        
#%% simulate a 2-ring system with traveling wave omegas and a_within, run independently and plot
#seed for random init vals of ring y
#random.seed(24)

#params for sim
num_rings = 2
omegaeach = np.linspace(0.1,0.2*np.pi,6)
a_within = 0.4
a_between = 0

#run sim & calc phi's for both rings - NO interring coupling
rings_thetas, rings_dthetas = simtheta(num_rings,nodes,omegaeach,a_within,a_between,timesteps)
phi_x_ijs = calcphi_ij(rings_thetas[:,:,0])
phi_y_ijs = calcphi_ij(rings_thetas[:,:,1])

#plot each ring's outputs
plotmodovertime(nodes,timesteps,rings_thetas[:,:,0],omegaeach,a_within,a_between,'Ring X','xring_'+ str(len(timesteps)) +'timesteps'+'couple_'+str(a_within)+'_'+str(omegaeach)+'_')
plotmodovertime(nodes,timesteps,rings_thetas[:,:,1],omegaeach,a_within,a_between,'Ring Y','yring_'+ str(len(timesteps)) +'timesteps'+'couple_'+str(a_within)+'_'+str(omegaeach)+'_')

plotphiovertime(phi_x_ijs,timesteps,omegaeach,a_within,a_between,'Ring X','xring_'+ str(len(timesteps)) +'timesteps'+'couple_'+str(a_within)+'_'+str(omegaeach)+'_')
plotphiovertime(phi_y_ijs,timesteps,omegaeach,a_within,a_between,'Ring Y','yring_'+ str(len(timesteps)) +'timesteps'+'couple_'+str(a_within)+'_'+str(omegaeach)+'_')

plotbothmodovertime(nodes,timesteps,rings_thetas[:,:,0],rings_thetas[:,:,1],omegaeach,a_within,a_between,'Ring X & Ring Y',
                    'bothrings_'+str(len(timesteps))+'timesteps'+'_couplewin_'+str(a_within)+'_couplebt_'+str(a_between)+'_'+str(omegaeach)+'_')

plotbothphiovertime(phi_xy,timesteps,omegaeach,a_within,a_between,'Between Rings',
                    'bothrings_'+ str(len(timesteps)) +'timesteps'+'couple_'+str(a_within)+'_couplebt_'+str(a_between)+'_'+str(omegaeach)+'_')

#%% simulate 2-ring system from above but with inter-ring coupling a_between
#iterate over different inter-ring coupling values to see what drives sync vs. antiphase
#params for sim
num_rings = 2
omegaeach = np.linspace(0.1,0.2*np.pi,6)
a_within = 0.4

a_between_range = np.arange(-1.5,1.5,0.1)

for a_between in a_between_range:
    #run sim & calc phi's for both rings - NO interring coupling
    rings_thetas, rings_dthetas = simtheta(num_rings,nodes,omegaeach,a_within,a_between,timesteps)
    phi_x_ijs = calcphi_ij(rings_thetas[:,:,0])
    phi_y_ijs = calcphi_ij(rings_thetas[:,:,1])
    phi_xy = calcphi_xy(rings_thetas[:,:,0],rings_thetas[:,:,1])
    
    #plot each ring's outputs
    plotmodovertime(nodes,timesteps,rings_thetas[:,:,0],omegaeach,a_within,a_between,'Ring X','xring_'+ str(len(timesteps)) +'timesteps'+'couple_'+str(a_within)+'_couplebt_'+str(a_between)+'_'+str(omegaeach)+'_')
    plotmodovertime(nodes,timesteps,rings_thetas[:,:,1],omegaeach,a_within,a_between,'Ring Y','yring_'+ str(len(timesteps)) +'timesteps'+'couple_'+str(a_within)+'_couplebt_'+str(a_between)+'_'+str(omegaeach)+'_')
    
    plotphiovertime(phi_x_ijs,timesteps,omegaeach,a_within,a_between,'Ring X','xring_'+ str(len(timesteps)) +'timesteps'+'couple_'+str(a_within)+'_couplebt_'+str(a_between)+'_'+str(omegaeach)+'_')
    plotphiovertime(phi_y_ijs,timesteps,omegaeach,a_within,a_between,'Ring Y','yring_'+ str(len(timesteps)) +'timesteps'+'couple_'+str(a_within)+'_couplebt_'+str(a_between)+'_'+str(omegaeach)+'_')
    
    #plot outputs together
    plotbothmodovertime(nodes,timesteps,rings_thetas[:,:,0],rings_thetas[:,:,1],omegaeach,a_within,a_between,'Ring X & Ring Y',
                        'bothrings_'+str(len(timesteps))+'timesteps'+'_couplewin_'+str(a_within)+'_couplebt_'+str(a_between)+'_'+str(omegaeach)+'_')
    
    plotbothphiovertime(phi_xy,timesteps,omegaeach,a_within,a_between,'Between Rings',
                        'bothrings_'+ str(len(timesteps)) +'timesteps'+'couple_'+str(a_within)+'_couplebt_'+str(a_between)+'_'+str(omegaeach)+'_')
        
    
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

#%% calculate traveling wave solution for rings, no coupling, plot
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