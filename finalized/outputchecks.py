#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 14:24:07 2023

Peak-finding and phase calculation functions
removed previous code from crawl_and_roll_singleseg-multiseg-2sides.py that calc'd, stored the ISI, contractiondur
just use phase diff calc here to get interseg and LR phase diffs

@author: PatriciaCooney
"""
#%%imports 
import numpy as np
#%% closest peak find 
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

#%%find peaks and calc phase diffs
def motor_output_check(n_t,E,pulse_vals,c_thresh,titype):
    #find contraction onset and dur by checking drE vals
    segx = np.arange(E.shape[1])
    side = np.arange(E.shape[2])

    #track contraction start and end per seg and per side - catch partial waves
    if np.where(E[:,:,:] > c_thresh)[0].size>0:
        left_starts = -np.ones([E.shape[1],100])*np.inf
        left_ends = -np.ones([E.shape[1],100])*np.inf
        totalwaves = 0 #update in each step below to capture max num waves, all segs, both sides
        suprathresh_left = np.where(E[:,:,0] > c_thresh, 1, 0)
        for seg in segx:
            left_ws = np.where(np.diff(suprathresh_left[:,seg],axis=0)==1)[0]
            left_we = np.where(np.diff(suprathresh_left[:,seg],axis=0)==-1)[0]
            if left_we.size < left_ws.size:
                left_we = np.hstack([left_we,int(E.shape[0])])
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
            right_starts = -np.ones([E.shape[1],100])*np.inf
            right_ends = -np.ones([E.shape[1],100])*np.inf
            suprathresh_right = np.where(E[:,:,1] > c_thresh, 1, 0)
            for seg in segx:
                right_ws = np.where(np.diff(suprathresh_right[:,seg],axis=0)==1)[0]
                right_we = np.where(np.diff(suprathresh_right[:,seg],axis=0)==-1)[0]
                if right_we.size < right_ws.size:
                    right_we = np.hstack([right_we,int(E.shape[0])])
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

        #do phase diff for 2-sided system
        if cstart.shape[1]>1:
            if side.size>1:
                phasediff_LR = -np.ones([cstart.shape[1]-2,cstart.shape[0]-1])*np.inf
                mean_phasediff_LR = -np.ones(cstart.shape[1]-2)*np.inf
                phasediff_interseg = -np.ones([cstart.shape[1]-2,cstart.shape[0]-1,2])*np.inf
                mean_phasediff_interseg = -np.ones([cstart.shape[1]-2,2])*np.inf
            #wave
            for wa in np.arange(cstart.shape[1]-2):
                #seg
                for seg in segx:
                    if seg < cstart.shape[0]-1: 
                        #interseg comparison each side
                        for si in side:
                            #phase diff
                            wavecurr = cstart[seg,wa,si]
                            compwavenext = cstart[seg,wa+1,si]
                            comparray_seg = cstart[seg+1,:,si] #compare to all possible waves in neighboring segs
                            adjval = find_nearest(comparray_seg, wavecurr)
                            phasediff_interseg[wa,seg,si] = abs(wavecurr - adjval)/abs(wavecurr - compwavenext)
                            
                            #LR comparison
                            if si == 0:
                                comparray_side = cstart[seg,:,1] #compare to all possible waves in this seg on the contralateral side to find nearest
                                adjval = find_nearest(comparray_side, wavecurr)
                                phasediff_LR[wa,seg] = abs(wavecurr - adjval)/abs(wavecurr - compwavenext)

                #mean across segs - interseg phi and contra phi    
                mean_phasediff_interseg[wa,0] = np.nanmean(phasediff_interseg[wa,:,0]) #mean, remove the weird entries
                mean_phasediff_interseg[wa,1] = np.nanmean(phasediff_interseg[wa,:,1]) #mean, remove the weird entries
                mean_phasediff_LR[wa] = np.nanmean(phasediff_LR[wa,:]) #mean, remove the weird entries

        else:
            #no phase diff because system --- traveling front - stays elevated/saturated with activity
            mean_phasediff_LR, phasediff_LR, mean_phasediff_interseg, phasediff_interseg = np.nan, np.nan, np.ones([2])*np.nan, np.ones([2])*np.nan
        
    return cstart, cend, totalwaves, mean_phasediff_LR, phasediff_LR, mean_phasediff_interseg, phasediff_interseg
