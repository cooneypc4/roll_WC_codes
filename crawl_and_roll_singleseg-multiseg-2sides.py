#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 11:15:50 2022

@author: PatriciaCooney
"""
# Imports
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt  # root-finding algorithm
import seaborn as sb

#%% option for organizing simulations
#save all params in object; send object and plots to dir

#%% functions for plotting

#include plots from NMA of activity v time, phase plane, but extend also to 
#heat matrices in crawl paper
#contraction duration in crawl paper
#additional plots like vec fields, combined phase plane multiseg, etc

#from NMA
#plot the sigmoid response function for E and I populations
def plot_FI_EI(x, FI_exc, FI_inh):
  fig = plt.figure()
  plt.plot(x, FI_exc, 'b', label='E population')
  plt.plot(x, FI_inh, 'r', label='I population')
  plt.legend(loc='lower right')
  plt.xlabel('x')
  plt.ylabel('F(x)')
  plt.title('Sigmoid Activation Functions of E and I populations')
  fig.savefig('FIcurves.svg', format = 'svg', dpi = 1200)


#plot convergence of activity in isolated E or I pops over time
def isolated_timeplot(t, rE1, rI1):#, rE2, rI2):
  fig = plt.figure()
  ax1 = plt.subplot(211)
  ax1.plot(pars['range_t'], rE1, 'b', label='E population')
  ax1.plot(pars['range_t'], rI1, 'r', label='I population')
  ax1.set_ylabel('Activity')
  ax1.legend(loc='best')

  ax2 = plt.subplot(212, sharex=ax1, sharey=ax1)
  #ax2.plot(pars['range_t'], rE2, 'b', label='E population')
  #ax2.plot(pars['range_t'], rI2, 'r', label='I population')
  ax2.set_xlabel('t (ms)')
  ax2.set_ylabel('Activity')
  ax2.legend(loc='best')

  plt.tight_layout()
  plt.show()
  fig.savefig('isolated_timeplot.svg', format = 'svg', dpi = 1200)


#plot phase plane
def plot_activity_phase(n_t,rE,rI,I_pulse):
  fig = plt.figure(figsize=(8, 5.5))
  plt.subplot(211)
  plt.plot(np.arange(0,n_t), rE[0:n_t], 'b', label=r'$r_E$')
  plt.plot(np.arange(0,n_t), rI[0:n_t], 'r', label=r'$r_I$')
  textstr = ('mag: ' + str(np.max(I_pulse)) + ' ; dur: ' + str(np.where(I_pulse>0)[0].size))
  plt.gcf().text(0.98, 0.95, textstr, fontsize = 10)
  plt.xlabel('t (ms)', fontsize=10)
  plt.ylabel('Activity', fontsize=10)
  plt.legend(loc='best', fontsize=10)

  plt.subplot(212)
  plt.plot(rE, rI, 'k')
  plt.plot(rE[n_t], rI[n_t], 'ko')
  plt.xlabel(r'$r_E$', fontsize=18, color='b')
  plt.ylabel(r'$r_I$', fontsize=18, color='r')

  plt.tight_layout()
  plt.show()
  fig.savefig('activity-phase_redoinit-zero.svg', format = 'svg', dpi = 1200)


def plot_fp_alone(rE,drE,rI,drI):
    fig = plt.figure(figsize=(8, 5.5))
    plt.subplot(211)
    plt.plot(rE, drE, 'b', label='$r_E$')
    plt.axhline(0, color='k', ls='--')
    plt.xlabel('rE', fontsize=14)
    plt.ylabel('drE/dt', fontsize=14)
    plt.legend(loc='best', fontsize=14)

    plt.subplot(212)
    plt.plot(rI, drI, 'r', label='$r_I$')
    plt.axhline(0, color='k', ls='--')
    plt.xlabel('rI', fontsize=14)
    plt.ylabel('drI/dt', fontsize=14)
    plt.legend(loc='best', fontsize=14)

    plt.tight_layout()
    plt.show()
    fig.savefig('fp_alone.svg', format = 'svg', dpi = 1200)


def plot_nullclines(E_null_rE, E_null_rI, I_null_rE, I_null_rI, *args):
    if 'x_fp' in globals():
        fp = args[0]
        I_e = args[1]
        fig = plt.figure()
        plt.scatter(fp[0], fp[1])
        plt.text(x=fp[0]+0.3, y=fp[1]+0.3, s = 'f.p. = (' + str(fp[0]) + ',' + str(fp[1]) + ')')
        plt.text(x=0.9,y=0.8,s='I_ext: '+str(I_e))
    else:
        I_e = args[0]
        plt.text(x=0.9,y=0.9,s='I_ext: '+str(I_e))
    
    fig = plt.figure()
    plt.plot(E_null_rE, E_null_rI, 'b', label='E nullcline')
    plt.plot(I_null_rE, I_null_rI, 'r', label='I nullcline')
    plt.xlabel('$r_E$')
    plt.ylabel('$r_I$')
    plt.legend(loc='lower right')
    plt.show()
    fig.savefig('plot_nullclines.svg', format = 'svg', dpi = 1200)
    
    
def plot_vecfield(E_null_rE, E_null_rI, I_null_rE, I_null_rI, x_fp, I_e, n_skip, rE, drEdt, rI, drIdt, weights):
    fig = plt.figure()
    plt.scatter(x_fp[0], x_fp[1])
    plt.text(x=x_fp[0]+0.3, y=x_fp[1]+0.3, s = 'f.p. = (' + str(round(x_fp[0],3)) + ',' + str(round(x_fp[1],3)) + ')')
    plt.text(x=0.9,y=0.8,s='I_ext: '+str(I_e))
    
    plt.plot(E_null_rE, E_null_rI, 'b', label='E nullcline')
    plt.plot(I_null_rE, I_null_rI, 'r', label='I nullcline')
    plt.xlabel('$r_E$')
    plt.ylabel('$r_I$')
    plt.legend(loc='lower right')
    
    plt.quiver(rE[::n_skip, ::n_skip], rI[::n_skip, ::n_skip],
               drEdt[::n_skip, ::n_skip], drIdt[::n_skip, ::n_skip],
               angles='xy', scale_units='xy', scale=5., facecolor='c')
    
    #plt.text(x=0.9,y=0.8,s='weights: '+str(weights))

    plt.tight_layout()
    plt.show()
    plt.savefig('vecfields_plot_'+str(weights)+str(round(x_fp[0],3))+str(round(x_fp[1],3))+'.svg')

def plot_multiseg_2s_sepaxes(n_t,rEms,pulse_vals,contra_weights,offsetcontra,contra_dur,inhib_pulse,inhib_side,perturb_input,ti):
    #look at LR waves in diff vertical subplots, diff colors
    
    #perturb_input = [1, 1, 0, 7, 1.7, rest_dur+1, 50] #yesno, E or I, ipsi contra, seg, mag, time of onset, duration of input
    fig = plt.figure(figsize=(7,4))
    
    if rEms.shape[1] > 20:
        lseginds = np.linspace(0,rEms.shape[1]/2,8,dtype=int)
        rseginds = np.linspace(0,rEms.shape[1]/2,8,dtype=int)
        psegl = np.arange(1,8*2+1,2)
        psegr = np.arange(2,8*2+2,2)
    else:
        lseginds = np.arange(0,rEms.shape[1])
        rseginds = np.arange(0,rEms.shape[1])
        psegl = np.arange(1,rEms.shape[1]*2+1,2)
        psegr = np.arange(2,rEms.shape[1]*2+2,2)
    
    cthresh = 0.3
        
    for seg in np.arange(0,len(lseginds)):
        for s in np.arange(rEms.shape[2]):
            if s == 0:
                seginds = psegl
                datinds = lseginds
                co = 'b'
                la = '$Left r_E$'
            else:
                seginds = psegr
                datinds = rseginds
                co = 'm'
                la = '$Right r_E$'
            #do subplots for each segment and side
            plt.subplot(len(lseginds)*2,1,seginds[seg])
            plt.plot(rEms[n_t,datinds[seg],s], co, label=la)
            if any(rEms[n_t,datinds[seg],s]>cthresh):
                xset = np.where(rEms[n_t,datinds[seg],s]>cthresh)[0]
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
                plt.axvline(x=perturb_input[5]+perturb_input[6],ymin=0,ymax=1,color='r')
            ax = plt.gca()
            ax.set_xticks([])
            if seginds[seg]%3 == 0:
                ax.set_yticks([])
    
            if seg == len(seginds)/2 and s == 1:
                plt.ylabel('L & R rE', fontsize=12)
            if seg == seginds[-1] and s == 1:
                plt.xlabel('seconds', fontsize=10)
                xints = np.arange(n_t[-1]+20,20)
                plt.xticks(xints)
                ax = plt.gca()
                ax.set_xticklabels(str(xi*pars['dt']) for xi in xints) 
            
    I_mag = pulse_vals[0,0]
    I_dur = pulse_vals[0,1]
    inc = -0.1
    contra_names = ['LR-EE','LR-EI','LR-IE','LR-II'] 

    allstrs = [('L mag: ' + str(I_mag)), ('L dur: ' + str(I_dur*pars['dt'])), ('R mag: ' + str(round((I_mag*offsetcontra),2))),
               ('R dur: ' + str(I_dur*contra_dur*pars['dt']))]
    #for loop looping list - textstr, plot, add inc
    for l in allstrs:
        plt.gcf().text(0.92, 0.92+inc, l, fontsize = 6)
        inc = inc-0.1
    for iw,w in enumerate(contra_weights):
        inc = inc-0.05
        textstr = contra_names[iw] + '=' + str(w)
        plt.gcf().text(0.92, 0.92+inc, textstr, fontsize = 6)
    
    #plt.tight_layout()
    plt.show()
    #fig.savefig((ti + str(n_t) + '_2side_multiseg_traces.svg'), format = 'svg', dpi = 1200)
    fig.savefig((ti + str(n_t[0]) + str(n_t[-1]) + '_2side_multiseg_traces.png'), format = 'png', dpi = 1200)



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
            plt.plot(rEms[n_t,seg,s], co, label=la)
            ax = plt.gca()
            ax.set_xticks([])
            ax.set_yticks([])
            if any(rEms[n_t,seginds[seg],s]>cthresh):
                xset = np.where(rEms[n_t,seginds[seg],s]>cthresh)[0]
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
                if perturb_input[6] > n_t[-1]:
                    plt.axvline(x=n_t[-1],ymin=0,ymax=1,color='r')
                else:
                    plt.axvline(x=perturb_input[5]+perturb_input[6],ymin=0,ymax=1,color='r')
            if seg+1 == len(seginds)/2 and s == 0:
                plt.ylabel('L & R rE', fontsize=12)
            if seg == seginds[-1] and s == 0:
                plt.xlabel('seconds', fontsize=10)
                xints = np.arange(0,n_t[-1]+20,40)
                plt.xticks(xints)
                ax = plt.gca()
                ax.set_xticklabels(str(int(xi*pars['dt'])) for xi in xints) 
    plt.legend(loc='lower right')
            
    I_mag = pulse_vals[0,0]
    I_dur = pulse_vals[0,1]
    inc = -0.05
    contra_names = ['EE','EI','IE','II'] 

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
    fig.savefig((titype + str(n_t[0]) + str(n_t[-1]) + '_multiseg_traces_singleaxesperseg.svg'), format = 'svg', dpi = 1200)
    fig.savefig((titype + str(n_t[0]) + str(n_t[-1]) + '_multiseg_traces_singleaxesperseg.png'), format = 'png', dpi = 1200)


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
        if np.nanmax(cdur_in)>0.2:
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
        if np.nanmax(isi_in)>0.2:
            offsetisi = np.nanmax(isi_in)
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
            
            if s==0 and np.nanmax(side_diff)>0.2:
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
    contra_names = ['LR-EE','LR-EI','LR-IE','LR-II'] 

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
    
    plt.ylim([-0.1,1.1])
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
    contra_names = ['EE','EI','IE','II'] 
    
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
    

#single ISI plot for all waves and all segs - left panel = contract spikes for whole time, all segs; right panel = delta spikes all waves
def jitter_contract_raw(n_t,rEms,I_pulse_in,ti):
    #plot just like multiseg traces, but single vert lines at cthresh start
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
            ax = plt.gca()
            ax.set_xticks([])
            ax.set_yticks([])
            
            if any(rEms[n_t,seginds[seg],s]>cthresh):
                xset = np.where(rEms[n_t,seginds[seg],s]>cthresh)[0]
                xends = xset[np.where(np.diff(xset)>1)]
                xends = np.append(xends,xset[-1])
                a = np.where(np.diff(xset)>1)[0] + 1
                xstarts = np.append(xset[0],xset[a])
                ystarts = np.zeros(len(xstarts))
                heights = np.repeat(1,len(xstarts))
                for p in np.arange(0,len(xstarts)):
                    plt.axvline(xstarts[p],ystarts[p],heights[p],color = co,label = la)
            if perturb_input[0] == 1 and seg == perturb_input[3] and s == perturb_input[2]: 
                if perturb_input[1] == 0:
                    colpert = 'r'
                else:
                    colpert = 'b'
                plt.axvline(x=perturb_input[5],ymin=0,ymax=1,color=colpert)
                if perturb_input[6] > n_t[-1]:
                    plt.axvline(x=n_t[-1],ymin=0,ymax=1,color=colpert)
                else:
                    plt.axvline(x=perturb_input[5]+perturb_input[6],ymin=0,ymax=1,color=colpert)
            if seg+1 == len(seginds)/2 and s == 0:
                plt.ylabel('L & R rE', fontsize=12)
            if seg == seginds[-1] and s == 0:
                plt.xlabel('seconds', fontsize=10)
                xints = np.arange(0,n_t[-1]+20,40)
                plt.xticks(xints)
                ax = plt.gca()
                ax.set_xticklabels(str(int(xi*pars['dt'])) for xi in xints) 
    plt.legend(loc='lower right')
        
    I_mag = pulse_vals[0,0]
    I_dur = pulse_vals[0,1]
    inc = -0.025
    contra_names = ['LR-EE','LR-EI','LR-IE','LR-II'] 

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
    fig.savefig((ti+'_jitter_raw.svg'), format = 'svg', dpi = 1200)
    fig.savefig((ti+'_jitter_raw.png'), format = 'png', dpi = 1200)
        
def jitter_contract_delta(segx_in,rEms,isi_in,numwaves_in,I_pulse_in,ti):
    #plot just like the LR phase difs but here delta isi's
    fig = plt.figure(figsize=(7,5))
    if numwaves_in > 30:
        selwa = np.linspace(0,numwaves_in-2,30,dtype=int)
    else:
        selwa = np.arange(0,numwaves_in-2)    
    
    #A1
    plt.scatter(selwa,isi_in[0,selwa],c ="m")
    plt.plot(selwa,isi_in[0,selwa],c ="m",label="A2-A1")
    #A3
    plt.scatter(selwa,isi_in[2,selwa],c ="b")
    plt.plot(selwa,isi_in[2,selwa],c ="b",label="A4-A3")
    #A8/9
    plt.scatter(selwa,isi_in[6,selwa],c ="g")
    plt.plot(selwa,isi_in[6,selwa],c ="g",label="A8/9-A7")
    #Avg
    plt.scatter(selwa,np.mean(isi_in[:,selwa],0),c ="k")
    plt.plot(selwa,np.mean(isi_in[:,selwa],0),c ="k",label="Average")
    #line for comparison at 0
    plt.axhline(0, color='r', ls='--')

    ylimtop = np.max(isi_in)
    ylimbott = np.min(isi_in)
    if ylimtop > 1 or ylimbott < -1:
        plt.ylim([-ylimtop*2,ylimtop*2])
    else:
        plt.ylim([-1,1])
    plt.ylabel(r"$\Delta$")
    plt.xlabel('Activity Waves')
    ax = plt.gca()
    wavect = [str(si+1) for si in selwa]
    ax.set_xticks(selwa)
    ax.set_xticklabels(wavect)
    plt.legend(loc='lower left')
            
    I_mag = pulse_vals[0,0]
    I_dur = pulse_vals[0,1]
    inc = -0.025
    contra_names = ['LR-EE','LR-EI','LR-IE','LR-II'] 

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
    fig.savefig((ti+'_jitter_delta.svg'), format = 'svg', dpi = 1200)
    fig.savefig((ti+'_jitter_delta.png'), format = 'png', dpi = 1200)
    
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
  #pars['I_ext_I'] = 0.

  # simulation parameters
  pars['T'] = 100.        # Total duration of simulation [ms]
  pars['dt'] = .1        # Simulation time step [ms]
  pars['rE_init'] = 0.017  # Initial value of E - unclear if should keep this - tutorial used 0.2 for E and I; paper does not mention initialization
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

#deriv of sigmoid
def dG(x, b, theta):
  """
  Derivative of the population activation function.

  Args:
    x     : the population input
    b     : the gain of the function
    theta : the threshold of the function

  Returns:
    dGdx  :  Derivative of the population activation function.
  """

  dGdx = b * np.exp(-b * (x - theta)) * (1 + np.exp(-b * (x - theta)))**-2

  return dGdx

#%% one segment, one side = single coupled oscillator pair, study how parameters impact fp's and limit cycle
#setup the E and I populations with equations and derivatives
#%% Plot F-I curves for E or I pops
pars = default_pars()
activ = np.arange(0, 10, .1)

# Compute the F-I curve of the excitatory population
FI_exc = G(activ, pars['b_E'], pars['theta_E'])

# Compute the F-I curve of the inhibitory population
FI_inh = G(activ, pars['b_I'], pars['theta_I'])

# Visualize
plot_FI_EI(activ, FI_exc, FI_inh)

#%%
# Simulate WC EI eq's - use Euler method to numerically solve for activity over time for E and I pop's
def simulate_wc(tau_E, b_E, theta_E, tau_I, b_I, theta_I,
                wEEself, wEIself, wIEself, wIIself, I_ext_E,
                rE_init, rI_init, dt, range_t, kmax_E, kmax_I, **other_pars):
  """
  Simulate the Wilson-Cowan equations

  Args:
    Parameters of the Wilson-Cowan model

  Returns:
    rE, rI (arrays) : Activity of excitatory and inhibitory populations
  """
  # Initialize activity arrays
  Lt = range_t.size
  #pulse = 0.002
  rE = np.append(rE_init, np.zeros(Lt - 1))
  rI = np.append(rI_init, np.zeros(Lt - 1))
  drE = np.zeros(Lt)
  drI = np.zeros(Lt)
  #I_pulse = np.concatenate((I_ext_E * np.ones(int(Lt*pulse)), np.zeros(int(Lt*(1-pulse)))))
  I_ext_E = np.concatenate((np.zeros(100), I_ext_E * np.ones(Lt-100)))

  # Simulate the Wilson-Cowan equations
  for k in range(Lt - 1):

    # Calculate the derivative of the E population
    drE[k] = dt / tau_E * (-rE[k] + (kmax_E - rE[k]) * G(wEEself * rE[k] + wEIself * rI[k] + I_ext_E[k], b_E, theta_E))

    # Calculate the derivative of the I population
    drI[k] = dt / tau_I * (-rI[k] + (kmax_I - rI[k]) * G(wIIself * rI[k] + wIEself * rE[k], b_I, theta_I))

    # Update using Euler's method
    rE[k + 1] = rE[k] + drE[k]
    rI[k + 1] = rI[k] + drI[k]

  return rE, rI, drE, drI


pars = default_pars()
#%%
# Simulate first trajectory
#pars['I_ext_E']=1

pars['rE_init']=0.019
pars['rI_init']=0.012
rE, rI, drE, drI = simulate_wc(**pars)

# # Simulate second trajectory
# rE2, rI2 = simulate_wc(**default_pars(rE_init=.33, rI_init=.15))

# #look at activity over time for e and I - plot both as lines
# isolated_timeplot(np.arange(0,100), rE, rI)#, rE2, rI2)

#show activity relationship between E and I at set timesteps
plot_activity_phase(n_t=140,rE = rE, rI = rI, I_pulse=np.ones(100)*pars['I_ext_E'])

#%% do with multiple input values - seems that <1.7 - no activity; = 3.9 saturate at 0.5!
# testp = np.arange(0.65,4.15,0.25)

# for tp in testp:
#     pars['I_ext_E'] = tp
#     rE, rI, drE, drI = simulate_wc(**pars)
#     plot_activity_phase(n_t=100,rE = rE, rI = rI, I_pulse=np.ones(100)*pars['I_ext_E'])
#%%look at phase plane of E and I -- fixed points and limit cycle
#allows us to understand where the dynamics reach a steady state (fixed point),
#or where they steadily oscillate (limit cycle),
#or where they are unstead and blow up (unstable fp)
#the single populations in Gjorgjieva et al., 2013 should demonstrate single unstable fixed point and stable limit cycle response to constant stim

#multiple ways to find f.p.'s 
#first, look individually at simulated r vs. dr/dt (how do dyn's chg based on current activity level)
#find dE/dt, dI/dt (in function above, just store both), then plot r vs. dr/dt
plot_fp_alone(rE,drE,rI,drI)

#%% fixed point analysis
#to find fixed points of 2D system, we can look at the nullclines 
#(where we set dI/dt = 0 and look at E dyn's, set dE/dt = 0 and look at I dyn's)
#fixed points ==> dI/dt = 0 and dE/dt = 0
#first, calculate rI when dE/dt = 0, and rE when dI/dt = 0

#take inverse of sigmoid nonlinearity function
def G_inv(x, b, theta):
  """
  Args:
    x         : the population input
    b         : the gain of the function
    theta     : the threshold of the function

  Returns:
    G_inverse : value of the inverse function
  """

  # Calculate Finverse (ln(x) can be calculated as np.log(x))
  G_inverse = -(1/b * np.log((x + (1 + np.exp(b*theta))**-1)**-1 - 1)) + theta

  return G_inverse


#now solve for the different nullclines
def get_E_nullcline(rE, b_E, theta_E, wEEself, wEIself, I_ext_E, kmax_E, **other_pars):
  """
  Solve for rI along the rE from drE/dt = 0.

  Args:
    rE    : response of excitatory population
    b_E, theta_E, wEE, wEI, I_ext_E : Wilson-Cowan excitatory parameters
    Other parameters are ignored

  Returns:
    rI    : values of inhibitory population along the nullcline on the rE
  """
  # calculate rI for E nullclines on rI
  rI = (1 / wEIself) * (-wEEself * rE - I_ext_E + G_inv(rE / (kmax_E - rE), b_E, theta_E))

  return rI


def get_I_nullcline(rI, b_I, theta_I, wIEself, wIIself, kmax_I, **other_pars):
  """
  Solve for E along the rI from dI/dt = 0.

  Args:
    rI    : response of inhibitory population
    a_I, theta_I, wIE, wII, I_ext_I : Wilson-Cowan inhibitory parameters
    Other parameters are ignored

  Returns:
    rE    : values of the excitatory population along the nullcline on the rI
  """
  # calculate rE for I nullclines on rI
  rE = (1 / wIEself) * (-wIIself * rI + G_inv(rI / (kmax_I - rI), b_I, theta_I))

  return rE

#%%
# # Set parameters and simulate for set time to get rE and rI for finding the nullclines
pars = default_pars()
Exc_null_rE = np.linspace(-0.1, 1.2, 100)
Inh_null_rI = np.linspace(-0.1, 1.2, 100)

# # Compute nullclines
# #repeat to find optimal I_ext_E for intersecting nullclines - how do we get fp?
# testI = np.arange(-1,2,0.8)

# for t in testI:
#pars['I_ext_E'] = t
Exc_null_rI = get_E_nullcline(Exc_null_rE, **pars)
Inh_null_rE = get_I_nullcline(Inh_null_rI, **pars)
    
#     # Visualize
#     if 'x_fp' in globals():
#         plot_nullclines(Exc_null_rE, Exc_null_rI, Inh_null_rE, Inh_null_rI, x_fp, pars['I_ext_E'])
#     else:
#         plot_nullclines(Exc_null_rE, Exc_null_rI, Inh_null_rE, Inh_null_rI, pars['I_ext_E'])
#%% check validity of the fixed point
#plug the f.p. value as rE_init and rI init into the ODE and see if get 0 drE, 0 drI
#can already tell that the fp is unstable because get close and get pushed off farther away, then oscillate
#  rEfp, rIfp, drEfp, drIfp = simulate_wc(**default_pars(rE_init=0.234848, rI_init=0.179752))

#problem: how to get the exact numerical fixed point -- need to solve for where drE/dt = 0 and drI/dt = 0
#use the NMA method, so find root of vector function using opt.root, plug initial values into deriv eq's
#then check it by seeing if the sum of the deriv's is very close to zero 
def my_fp(pars, rE_init, rI_init):
  """
  Use opt.root function to solve Equations (2)-(3) from initial values
  """

  tau_E, b_E, theta_E = pars['tau_E'], pars['b_E'], pars['theta_E']
  tau_I, b_I, theta_I = pars['tau_I'], pars['b_I'], pars['theta_I']
  wEEself, wEIself = pars['wEEself'], pars['wEIself']
  wIEself, wIIself = pars['wIEself'], pars['wIIself']
  I_ext_E = pars['I_ext_E']
  kmax_E, kmax_I = pars['kmax_E'], pars['kmax_I']

  # define the right hand of wilson-cowan equations
  def my_WCr(x):

    rE, rI = x
    # derivative of the E and I populations
    drEdt = 1 / tau_E * (-rE + (kmax_E - rE) * G(wEEself * rE + wEIself * rI + I_ext_E, b_E, theta_E))
    drIdt = 1 / tau_I * (-rI + (kmax_I - rI) * G(wIIself * rI + wIEself * rE, b_I, theta_I))   
    
    y = np.array([drEdt, drIdt])

    return y

  x0 = np.array([rE_init, rI_init])
  x_fp = opt.root(my_WCr, x0).x

  return x_fp    

#%% check and plug back in - did we actually get the fp?
x_fp = my_fp(pars, rE_init=[0.0], rI_init=[0.0])

tau_E, b_E, theta_E = pars['tau_E'], pars['b_E'], pars['theta_E']
tau_I, b_I, theta_I = pars['tau_I'], pars['b_I'], pars['theta_I']
wEEself, wEIself = pars['wEEself'], pars['wEIself']
wIEself, wIIself = pars['wIEself'], pars['wIIself']
I_ext_E = pars['I_ext_E']
kmax_E, kmax_I = pars['kmax_E'], pars['kmax_I']

rE = x_fp[0]
rI = x_fp[1]
drEdt = 1 / tau_E * (-rE + (kmax_E - rE) * G(wEEself * rE + wEIself * rI + I_ext_E, b_E, theta_E))
drIdt = 1 / tau_I * (-rI + (kmax_I - rI) * G(wIIself * rI + wIEself * rE, b_I, theta_I)) 

plot_nullclines(Exc_null_rE, Exc_null_rI, Inh_null_rE, Inh_null_rI, x_fp, pars['I_ext_E'])

#%% single segment - plot nullclines and vec field
#then systematically vary weights and observe how flow field changes

#OPTION: also try systematically vary weights vs. systematically vary input

#calculate the vector fields
def calc_vecfield(weights):
    EI_grid = np.linspace(-0.2, 0.8, 40)
    rE, rI = np.meshgrid(EI_grid, EI_grid) #fxn that allows vectorized computation by holding row constant for one matrix and col varies for other; vice versa

    wEEself = weights[0]
    wEIself = weights[1]
    wIEself = weights[2]
    wIIself = weights[3]
    
    # #vecfield
    # # derivative of the E and I populations
    drEdt = 1 / tau_E * (-rE + (kmax_E - rE) * G(wEEself * rE + wEIself * rI + I_ext_E, b_E, theta_E))
    drIdt = 1 / tau_I * (-rI + (kmax_I - rI) * G(wIIself * rI + wIEself * rE, b_I, theta_I))  
    
    n_skip = 2
    
    plot_vecfield(Exc_null_rE, Exc_null_rI, Inh_null_rE, Inh_null_rI, x_fp, pars['I_ext_E'],
                  n_skip, rE, drEdt, rI, drIdt, weights)

#idea: for checking potential for fp - run test of subtracting all vals iteratively from null clines and see if any difference is >0

#%% calculate and plot vector fields for different weights of each type
# wE_grid = np.arange(1,21,1)
# wI_grid = -wE_grid

# comb_weights = [(wEE, wEI, wII, wIE) for wEE in wE_grid for wEI in wE_grid for wII in wI_grid for wIE in wI_grid]
# for i in np.arange(0,len(comb_weights))

weights = [pars['wEEself'], pars['wEIself'], pars['wIEself'], pars['wIIself']]
calc_vecfield(weights)

# Connection strength - GJORGJIEVA CRAWL PARAMS
# pars['wEEself'] = 16.   # E to E
# pars['wEEadj'] = 20.   # E to E
# pars['wEIself'] = -12.   # I to E
# pars['wEIadj'] = -20.   # I to E
# pars['wIEself'] = 15.  # E to I
# pars['wIIself'] = -3.  # I to I


#PLAN TO VARY
#wEEself = 0 to 20
#wEIself = -20 to 0
#wIEself = 0 to 20
#wIIself = -20 to 0
#GOAL-ID cnxn weights of interest based on how vecfield impacted - what are transition/bifurcation points - plot from within each zone and bifurc zone

#execute previous fxn to calculate the nullclines and fp's


#execute new plot fxn - null clines + vec fields ; disp fp's


#execute activity and co-varying E and I plot


#%% linearize around the fixed point with Jacobian, then take eigenvals -- stability of system
# def get_eig_Jacobian(fp):
#   """Compute eigenvalues of the Wilson-Cowan Jacobian matrix at fixed point."""
  
#   tau_E, b_E, theta_E = pars['tau_E'], pars['b_E'], pars['theta_E']
#   tau_I, b_I, theta_I = pars['tau_I'], pars['b_I'], pars['theta_I']
#   wEEself, wEIself = pars['wEEself'], pars['wEIself']
#   wIEself, wIIself = pars['wIEself'], pars['wIIself']
#   I_ext_E = pars['I_ext_E']
#   kmax_E, kmax_I = pars['kmax_E'], pars['kmax_I']
  
#   # Initialization
#   rE, rI = fp
#   J = np.zeros((2, 2))

#   # Compute the four elements of the Jacobian matrix - partial derivs wrt drE and drI
#   #J[0, 0] = (-1 + G_inv(wEEself * (kmax_E - rE) + wEIself * rI + I_ext_E, b_E, theta_E)) / tau_E
#   J[0, 0] = (-rE + (kmax_E - rE) * G_inv(wEEself * rE + wEIself * rI + I_ext_E, b_E, theta_E))/ tau_E
  
#   J[0, 1] = (-wEIself * G_inv(wEEself * rE + wEIself * rI + I_ext_E, b_E, theta_E)) / tau_E
#   J[1, 0] = (-1 + G_inv(wIIself * rI + wIEself * rE, b_I, theta_I)) / tau_I
#   J[1, 1] = (-wIEself * G_inv(wIIself * rI + wIEself * rE, b_I, theta_I)) / tau_I

#   # Compute and return the eigenvalues
#   evals = np.linalg.eig(J)[0]
#   return evals


# # Compute eigenvalues of Jacobian
# eig_1 = get_eig_Jacobian(x_fp)

# print(eig_1)

#%%
#%% 

#%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% Simulate 2 sided WC EI eq's for multiple interconnected segments
def simulate_wc_multiseg(tau_E, b_E, theta_E, tau_I, b_I, theta_I,
                    wEEself, wEIself, wIEself, wIIself, wEEadj, wEIadj,
                    rE_init, rI_init, dt, range_t, kmax_E, kmax_I, n_segs, rest_dur, 
                    n_sides, sim_input, pulse_vals, contra_weights, offsetcontra, contra_dur, 
                    offsetcontra_sub, contra_dur_sub, perturb_init, 
                    perturb_input, **otherpars):
    """
    Simulate the Wilson-Cowan equations

    Args:
      Parameters of the Wilson-Cowan model
    
    Returns:
      rE1-8, rI1-8 (arrays) : Activity of excitatory and inhibitory populations
    """  
    # Initialize activity arrays
    Lt = range_t.size
    rEms, rIms, drEms, drIms, I_ext_E, I_ext_I = np.zeros([Lt+1,n_segs,n_sides]), np.zeros([Lt+1,n_segs,n_sides]), np.zeros([Lt+1,n_segs,n_sides]), np.zeros([Lt+1,n_segs,n_sides]), np.zeros([Lt,n_segs,n_sides]), np.zeros([Lt,n_segs,n_sides])
    
    #initialize E and I activity
    # if inhib_init[0] == 1 and inhib_init[1] == 0:
    #     rIms[0,:,0] = inhib_init[2] #ipsi
    # elif inhib_init[0] == 1 and inhib_init[1] == 1:
    #     rIms[0,:,1] = inhib_init[2] #contra
    # elif inhib_init[0] == 1 and inhib_init[1] == 2:
    rIms[0,:,:] = rI_init #both
    # if excit_init[0] == 1 and excit_init[1] == 0:
    #     rEms[0,:,0] = excit_init[2] #ipsi
    # elif excit_init[0] == 1 and excit_init[1] == 1:
    #     rEms[0,:,1] = excit_init[2] #contra 
    # elif excit_init[0] == 1 and excit_init[1] == 2:
    rEms[0,:,:] = rE_init #both

    if perturb_init[0] == 1:
        if perturb_init[1] == 1: #excitatory ipsi or contra
            rEms[perturb_init[5]:perturb_init[5]+perturb_init[6],perturb_init[2]] = perturb_init[4]
        else:
            rIms[perturb_init[5]:perturb_init[5]+perturb_init[6],perturb_init[3],perturb_init[2]] = perturb_init[4]
    
    #setup external input mat
    if sim_input == 0:
        #just posterior seg input - crawl
        print('crawling')
        I_ext_E[:,n_segs-1,0] = np.concatenate((np.zeros(rest_dur), pulse_vals[0,0] * np.ones(int(pulse_vals[0,1])), np.zeros(Lt-int(pulse_vals[0,1])-rest_dur))) #ipsi
    if sim_input == 0 and n_sides>1:
        I_ext_E[:,n_segs-1,1] = np.concatenate((np.zeros(rest_dur), round(pulse_vals[0,0] * offsetcontra,2) * np.ones(int(pulse_vals[0,1]*contra_dur)), 
                                                              np.zeros(Lt-int(pulse_vals[0,1]*contra_dur)-rest_dur))) #contra
    elif sim_input == 1: #simultaneous drive
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
        end = start + int(perturb_input[6])
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
        else: #inhib
            I_ext_I[:,perturb_input[3],perturb_input[2]] = np.concatenate((np.zeros(perturb_input[5]), perturb_input[4] * np.ones(perturb_input[6]), 
                                                                                                                            np.zeros(Lt-int(perturb_input[5]+perturb_input[6]))))
        
        #perturb_input = [1, 1, 1, 7, pars['I_ext_E']*2, rest_dur+110, 100] #yesno, E or I or both, ipsi contra or both, seg, mag, time of onset, duration of input
    #pull out the individual contra weights
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
                # Calculate the derivative of the I population - same update for all seg
                drIms[k,seg,s] = dt / tau_I * (-rIms[k,seg,s] + (kmax_I - rIms[k,seg,s]) 
                                               * G((wIIself * rIms[k,seg,s] + wIEself * rEms[k,seg,s]
                                                   + wIEcontra * rEms[k,seg,cs] + wIIcontra * rIms[k,seg,cs] + I_ext_I[k,seg,s]), b_I, theta_I))
                if seg == n_segs-1:
                    #eq without +1 terms
                    # Calculate the derivative of the E population
                    drEms[k,seg,s] = dt / tau_E * (-rEms[k,seg,s] + (kmax_E - rEms[k,seg,s]) * G(((wEEadj*2 * rEms[k,seg-1,s]) + (wEEself * rEms[k,seg,s]) 
                                                                   + (wEIadj*2 * rIms[k,seg-1,s]) + (wEIself * rIms[k,seg,s])
                                                                   + (wEEcontra * rEms[k,seg,cs]) + (wEIcontra * rIms[k,seg,cs]) + I_ext_E[k,seg,s]), b_E, theta_E))
                elif seg == 0:
                    #eq without the i-1 terms
                    # Calculate the derivative of the E population
                    drEms[k,seg,s] = dt / tau_E * (-rEms[k,seg,s] + (kmax_E - rEms[k,seg,s]) * G(((wEEself * rEms[k,seg,s]) + (wEEadj*2 * rEms[k,seg+1,s]) 
                                                                       + (wEIself * rIms[k,seg,s]) + (wEIadj*2 * rIms[k,seg+1,s]) 
                                                                       + (wEEcontra * rEms[k,seg,cs]) + (wEIcontra * rIms[k,seg,cs]) + I_ext_E[k,seg,s]), b_E, theta_E))
                else: #seg a2-7
                    #the eq's for all mid segs
                    drEms[k,seg,s] = dt / tau_E * (-rEms[k,seg,s] + (kmax_E - rEms[k,seg,s]) * G(((wEEadj * rEms[k,seg-1,s]) + (wEEself * rEms[k,seg,s]) + (wEEadj * rEms[k,seg+1,s]) 
                                                                   + (wEIadj * rIms[k,seg-1,s]) + (wEIself * rIms[k,seg,s]) + (wEIadj * rIms[k,seg+1,s]) 
                                                                   + (wEEcontra * rEms[k,seg,cs]) + (wEIcontra * rIms[k,seg,cs]) + I_ext_E[k,seg,s]), b_E, theta_E))
                # Update using Euler's method
                if rEms[k+1,seg,s] == 0:
                    rEms[k+1,seg,s] = float(rEms[k,seg,s] + drEms[k,seg,s])
                else:
                    rEms[k+1,seg,s] = rEms[k+1,seg,s]
                if rIms[k+1,seg,s] == 0:
                    rIms[k+1,seg,s] = float(rIms[k,seg,s] + drIms[k,seg,s])
                else:
                    rIms[k+1,seg,s] = rIms[k+1,seg,s]
        
    return rEms, rIms, I_ext_E, I_ext_I

#%%run 2-sided multiseg crawl
#2-sided crawl
# pars = default_pars()
# length = pars['range_t'].size
# n_segs = 8

# #pulse settings
# pulse = 400
# sim_input = 0 #input to all segments at once?
# mm = 1
# contra_dur = 1 #set this to be the fraction of the ipsi pulse that you want contra pulse to be
# alt = 0

# inhib_mag = 0
# inhib_dur = 0
# inhib_pulse = np.array([-pars['I_ext_E']*inhib_mag, pulse*inhib_dur])
# inhib_side = 0 # 0 = ipsi, 1 = contra
# inhib_init = [0, 0, 0] #yes no, ipsi contra, mag
# excit_init = [0, 0, 0]

# pulse_vals = np.array([[pars['I_ext_E'], pulse, alt],[pars['I_ext_E']*mm, pulse, alt]])
# rest_dur = 100

# contra_weights = [0,0,0,-5]

# offsetcontra = 1.0117647

# rEms,rIms = simulate_wc_multiseg_2s(**default_pars(n_segs=n_segs, sim_input=sim_input, rest_dur = rest_dur,
#                                                    pulse_vals=pulse_vals, contra_weights=contra_weights, 
#                                                    offsetcontra = offsetcontra, contra_dur = contra_dur,
#                                                    inhib_pulse = inhib_pulse, inhib_side = inhib_side,
#                                                    inhib_init = inhib_init, excit_init = excit_init))

# #%%plot network analyses -- crawl 2-sided
# n_t = 400
# plot_multiseg_2s(n_t,rEms,pulse_vals,contra_weights,offsetcontra,contra_dur,inhib_pulse,inhib_side,'crawl_rerun_init0_50dur_II_-5_')

#%% 2-sided roll
# pars = default_pars()
# n_sides = 2

# #pulse settings
# pulse = 300
# sim_input = 0
# contra_dur = 1 #set this to be the fraction of the ipsi pulse that you want contra pulse to be
# alt = 0 #1 = alternating goro inputs in sine wave pattern

# inhib_mag = 0#4
# inhib_dur = 0#0.1
# inhib_pulse = np.array([-pars['I_ext_E']*inhib_mag, pulse*inhib_dur])
# inhib_side = 1 # 0 = ipsi, 1 = contra

# pulse_vals = np.array([[pars['I_ext_E'], pulse, alt]])
# offsetcontra = 1.1


# contra_weights = [5,0,0,0]


# #perturbation options setup
# #perturb_init = [1, 0, 0, 'all', 0.3] #yes no, E or I, ipsi contra, seg, init_val
# perturb_init = [0, 0, 0, 0, 0] #yes no, E or I, ipsi contra, seg, init_val
# #perturb_input = [1, 1, 1, 'all', pars['I_ext_E']*0.2, rest_dur+1, pulse*0.05] #yesno, I E or both, ipsi contra or both, seg, mag, time of onset, duration of input
# #perturb_input = [1, 1, 1, 'all', pars['I_ext_E']*1, rest_dur+1+pulse*0.05, pulse - rest_dur+1+pulse*0.05]
# perturb_input = [0]

# rEms,rIms,I_ext_E,I_ext_I = simulate_wc_multiseg(**default_pars(n_sides=n_sides, sim_input=sim_input,
#                                                     pulse_vals=pulse_vals, contra_weights=contra_weights, 
#                                                     offsetcontra = offsetcontra, contra_dur = contra_dur, 
#                                                     inhib_pulse = inhib_pulse, inhib_side = inhib_side, 
#                                                     perturb_init = perturb_init, perturb_input = perturb_input))

#%% find nearest fxn to make the side and seg comparisons easier
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

#%% calculate side diffs and motor output for 2-sided models
def motor_output_check(n_t,E,pulse_vals,c_thresh,titype):
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
                    left_ws = np.concatenate((np.reshape(left_ws,[left_ws.shape[0],-1]),np.zeros([left_starts.shape[0],wdiff])),1)
                elif wdiff < 0:
                    left_starts = np.concatenate(((left_starts,np.zeros([left_starts.shape[0],wdiff]))),1)
            elif nwends > num_waves:
                totalwaves = nwends
                wdiff = totalwaves - nwends
                if wdiff > 0:
                    left_we = np.concatenate((np.reshape(left_we,[left_we.shape[0],-1]),np.zeros([left_starts.shape[0],wdiff])),1)
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
                
                if num_waves > totalwaves:
                    totalwaves = num_waves
                if num_waves > nwends:
                    wdiff = totalwaves - num_waves
                    if wdiff > 0:
                        right_ws = np.concatenate((np.reshape(right_ws,[right_ws.shape[0],-1]),np.zeros([right_starts.shape[0],wdiff])),1)
                    elif wdiff < 0:
                        right_starts = np.concatenate(((right_starts,np.zeros([right_starts.shape[0],wdiff]))),1)
                elif nwends > num_waves:
                    totalwaves = nwends
                    wdiff = totalwaves - nwends
                    if wdiff > 0:
                        right_we = np.concatenate((np.reshape(right_we,[right_we.shape[0],-1]),np.zeros([right_starts.shape[0],wdiff])),1)
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
            for seg in segx:
                for wa in np.arange(cstart.shape[1]):
                    if seg != 0:
                        comparray_ant = cstart[seg-1,:]
                        adjval = find_nearest(comparray_ant, cstart[seg,wa])
                        isi[seg-1,wa] = cstart[seg,wa] - adjval
                        if isi[seg-1,wa] > 5:
                            isi[seg-1,wa] = np.nan
                        cdur[seg-1,wa] = abs(cend[seg,wa] - cstart[seg,wa])
                    elif seg == 0:
                        comparray_end = cstart[-1,:]
                        adjval = find_nearest(comparray_end, cstart[0,wa])
                        cnorm[wa] = adjval - cstart[0,wa]
                        cdur[seg-1,wa] = abs(cend[seg,wa] - cstart[seg,wa])
        else:
            isi = np.ones([cstart.shape[0]-1,cstart.shape[1],2])*99999
            cdur = np.ones([cstart.shape[0],cstart.shape[1],2])*99999
            cnorm = np.ones([cstart.shape[1],2])*99999
            for si in side:
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

        # #plot contraction duration of all segs, interseg phase lag, peak E amp as subplots 1 fig, all segs (fig 3 and then some)
        # plot_motor_out(segx,cdur,isi,side_diff,totalwaves,pulse_vals,ti = titype + 'offsetcontra_' + str(offsetcontra) + '_' + str(contra_weights) + '_')
        
        #run new set of plots--single ISI plot for all waves and all segs - left panel = contract spikes for whole time, all segs; right panel = delta spikes all waves
        jitter_contract_delta(segx_in,rEms,isi_in,numwaves_in,I_pulse_in,ti)
        
    else:
        cstart,cend,cdur,cdurnorm,lat,isi,isinorm,totalwaves = np.nan, np.nan, np.nan*np.ones([8]), np.nan*np.ones([8]), np.nan, np.nan*np.ones([7]), np.nan*np.ones([7]), np.nan
        mean_phasediff, ant_phasediff, mid_phasediff, post_phasediff = np.nan, np.nan, np.nan, np.nan
        side_diff = np.nan
        
    return cstart, cend, cdur, cdurnorm, lat, isi, isinorm, side_diff, totalwaves, mean_phasediff, ant_phasediff, mid_phasediff, post_phasediff

###STOPPED HERE - FIGURE OUT WHY THE JITTERS AREN'T LINING UP APPROAPRIATELY
#ALSO CHECK THAT BEING PLOTTED IN CORRECT ORDER -- LOOKS FLIPPED RIGHT NOW (A8 ON TOP, A1 ON BOTTOM)
#CONSIDER SWITCHING TO NO SUBPLOTS - JUST MAIN PLOT WITH Y = SEG, X = TIMESTEP AND CALL IT A DAY.,

#%% test stability of "roll" behavior in response to perturbations

# # diff init and input roll after pause rest state and seg's to test on
# #do really long time courses to be sure of stability over time
# initloop = np.arange(0,2,0.6)
# #inputloop = np.arange(0,3.6,0.4)
# #inputloop = [3.2]
# segloop = [0,2,4,7]
# # initloop = [0]
# # inputloop = [0]
# # segloop = [0]
# # simtype = [0,1]
# n_t = 300

# #make naming options
# simname = ['crawl', 'roll']
# segname = ['A1','A3','A5','A8']

# #loop for running stability analysis on all possible simulations of interest
# pars = default_pars()
# n_sides = 2
# pulse = 550
# alt = 0 #1 = alternating goro inputs in sine wave pattern
# pulse_vals = np.array([[pars['I_ext_E'], pulse, alt]])
# contra_choose = np.array([[5,0,0,0],[0,-5,0,0]])

# inhib_mag = 0#4
# inhib_dur = 0#0.1
# inhib_pulse = np.array([-pars['I_ext_E']*inhib_mag, pulse*inhib_dur])
# inhib_side = 1 # 0 = ipsi, 1 = contra

# #try crawl or roll, 1 or 2 sides
# sim = 1
# sim_input = sim
# if n_sides==1:
#     contra_weights = [0,0,0,0]
# else:
#     contra_weights = contra_choose[sim]
# #setup crawl or roll input
# if sim == 0:
#     offsetcontra = 1.1
#     contra_dur = 1
#     contra_dur_sub = 0
#     offsetcontra_sub = 0
# else: 
#     offsetcontra_sub = 0.05
#     contra_dur_sub = 0.01
#     offsetcontra = 1.1
#     contra_dur = 1-contra_dur_sub

# #set this loop to match the perturb_init and perturb_input above
# sign = 1
# # for inputl in inputloop:
# #     initl = 0
# #     for seg,sloop in enumerate(segloop):
# #         #perturbations
# #         perturb_init = [1, sign, 0, sloop, initl] #no yes, E or I, ipsi contra, seg, init_val
# #         perturb_input = [1, sign, 0, sloop, inputl, pars['rest_dur']+1, pulse-pars['rest_dur']+1] #yesno, I E or both, ipsi contra or both, seg, mag, time of onset, duration of input
        
# #         #run that sim
# #         rEms,rIms,I_ext_E,I_ext_I = simulate_wc_multiseg(**default_pars(n_sides=n_sides, sim_input=sim_input,
# #                                                            pulse_vals=pulse_vals, contra_weights=contra_weights, 
# #                                                            offsetcontra = offsetcontra, contra_dur = contra_dur,
# #                                                            offsetcontra_sub = offsetcontra_sub, contra_dur_sub = contra_dur_sub,
# #                                                            inhib_pulse = inhib_pulse, inhib_side = inhib_side, 
# #                                                            perturb_init = perturb_init, perturb_input = perturb_input))
        
# #         plot_multiseg_sameaxes(n_t,rEms,pulse_vals,contra_weights,offsetcontra,contra_dur,inhib_pulse,inhib_side,perturb_init,perturb_input,
# #                          titype = simname[sim] +'_onesided_fpinit_rest1_' + segname[seg] +'_initpert' + str(initl) +'_excinputpert' + str(inputl))
        
# #         cstart, cend, cdur, cdurnorm, lat, isi, isinorm, side_diff, right, mean_phasediff, ant_phasediff, mid_phasediff, post_phasediff = motor_output_check(rEms,
# #              pulse_vals,c_thresh = 0.3,titype = simname[sim] +'_onesided_fpinit_rest1_' + segname[seg] +'_initpert' + str(initl) +'_excinputpert' + str(inputl))

# for initl in initloop:
#     inputl = 0
#     for seg,sloop in enumerate(segloop):
#         #perturbations
#         perturb_init = [1, sign, 0, sloop, initl] #no yes, E or I, ipsi contra, seg, init_val
#         perturb_input = [1, sign, 0, sloop, inputl, pars['rest_dur']+1, pulse-pars['rest_dur']+1] #yesno, I E or both, ipsi contra or both, seg, mag, time of onset, duration of input 
        
#         #run that sim
#         rEms,rIms,I_ext_E,I_ext_I = simulate_wc_multiseg(**default_pars(n_sides=n_sides, sim_input=sim_input,
#                                                             pulse_vals=pulse_vals, contra_weights=contra_weights, 
#                                                             offsetcontra = offsetcontra, contra_dur = contra_dur,
#                                                             offsetcontra_sub = offsetcontra_sub, contra_dur_sub = contra_dur_sub,
#                                                             inhib_pulse = inhib_pulse, inhib_side = inhib_side, 
#                                                             perturb_init = perturb_init, perturb_input = perturb_input))
        
#         plot_multiseg_sameaxes(n_t,rEms,pulse_vals,contra_weights,offsetcontra,contra_dur,inhib_pulse,inhib_side,perturb_init,perturb_input,
#                           titype = simname[sim] +'_onesided_fpinit_rest1_' + segname[seg] +'_excinitpert' + str(initl) +'_inputpert' + str(inputl))
        
#         cstart, cend, cdur, cdurnorm, lat, isi, isinorm, side_diff, right, mean_phasediff, ant_phasediff, mid_phasediff, post_phasediff = motor_output_check(rEms,
#               pulse_vals,c_thresh = 0.3,titype = simname[sim] +'_onesided_fpinit_rest1_' + segname[seg] +'_excinitpert' + str(initl) +'_inputpert' + str(inputl))


# #%%plot network analyses -- roll 2-sided
# n_t = 350
# plot_multiseg_sameaxes(n_t,rEms,pulse_vals,contra_weights,offsetcontra,contra_dur,inhib_pulse,inhib_side,perturb_input,
#                  'crawl_rest1_pulse300x1.7_contrax1.1_initnonzerofp_contra_EE_5_')

#perfect roll: 'roll_ipsi_init0.3_1dur_1.01contra_II_-10_'


#%% updated perturbation type
#single sided stability - here, excit hard code to set scalar value - single time step to start? then make multi timestep (maybe ~5 timesteps?)
#loop over segment number and loop over perturb magnitude
#do E pop and I pop

#inputloop = np.arange(0.5,12.5,2)
inputloop = [0]
initloop = np.arange(0.5,5.5,1)
segloop = [0,2,4,7]
n_t = np.arange(80,201)
#randomize time of perturbation so can see impact at different points in the behavior
#pert_starts = np.random.randint() #either od random, or consider doing range from 100 to 160? covers 2 waves?
pert_start = 120
pert_durs = np.arange(1,21,5)


#make naming options
simname = ['crawl', 'roll']
segname = ['A1','A3','A5','A8']
signs = ['inh','exc']

#loop for running stability analysis on all possible simulations of interest
pars = default_pars()
n_sides = 1
pulse = 900 #this is for your regular input pulse NOT FOR YOUR PERTURBATIONs
alt = 0 #1 = alternating goro inputs in sine wave pattern
pulse_vals = np.array([[pars['I_ext_E'], pulse, alt]])
contra_choose = np.array([[5,0,0,0],[0,-5,0,0]])

# inhib_mag = 0#4
# inhib_dur = 0#0.1
# inhib_pulse = np.array([-pars['I_ext_E']*inhib_mag, pulse*inhib_dur])
# inhib_side = 1 # 0 = ipsi, 1 = contra

#try crawl or roll, 1 or 2 sides
sim = 0
sim_input = sim
if n_sides==1:
    contra_weights = [0,0,0,0]
else:
    contra_weights = contra_choose[sim]

#setup crawl or roll input
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
for ein,eis in enumerate(signs):
    signname = eis
    for initl in initloop:
        inputl = 0
        for seg,sloop in enumerate(segloop):
            for pd in pert_durs:
                #perturbations
                perturb_init = [1, ein, 0, sloop, initl, pars['rest_dur']+pert_start, pd-pars['rest_dur']+1] # perturb_init = [1, sign, 0, sloop, initl] #no yes, E or I, ipsi contra, seg, init_val
                perturb_input = [1, ein, 0, sloop, inputl, pars['rest_dur']+pert_start, pd-pars['rest_dur']+1] #yesno, I E or both, ipsi contra or both, seg, mag, time of onset, duration of input
                
                #run that sim
                rEms,rIms,I_ext_E,I_ext_I = simulate_wc_multiseg(**default_pars(n_sides=n_sides, sim_input=sim_input,
                                                                    pulse_vals=pulse_vals, contra_weights=contra_weights, 
                                                                    offsetcontra = offsetcontra, contra_dur = contra_dur,
                                                                    offsetcontra_sub = offsetcontra_sub, contra_dur_sub = contra_dur_sub,
                                                                    perturb_init = perturb_init, perturb_input = perturb_input))
                
                plot_multiseg_sameaxes(n_t,rEms,pulse_vals,contra_weights,offsetcontra,contra_dur,perturb_init,perturb_input,
                                  titype = simname[sim] +'_onesided_fpinit_rest1_' + segname[seg] +'_initpert' + str(initl)+'_'+signname+'_inputpert' + str(inputl)+'_pertdur'+str(pd)+'_')
                
                jitter_contract_raw(n_t,rEms,pulse_vals,ti = simname[sim] +'_onesided_fpinit_rest1_' + segname[seg] +'_initpert' + str(initl)+'_'+signname+'_inputpert' + str(inputl)+'_pertdur'+str(pd)+'_')
                
                cstart, cend, cdur, cdurnorm, lat, isi, isinorm, side_diff, right, mean_phasediff, ant_phasediff, mid_phasediff, post_phasediff = motor_output_check(n_t, rEms,
                      pulse_vals,c_thresh = 0.3,titype = simname[sim] +'_onesided_fpinit_rest1_' + segname[seg] +'_initpert' + str(initl)+'_'+signname+'_inputpert' + str(inputl)+'_pertdur='+str(pd)+'_')
        
