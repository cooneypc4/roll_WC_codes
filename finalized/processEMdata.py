#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 15:38:23 2023

processing of connectome data

take the excel files, sort by NT types, fix naming, divide into L & R and take NT pair submats

@author: PatriciaCooney
"""
#%% imports
import pandas as pd
import numpy as np

#%% bring in the PMN-MN mats of interest - connectivity reg, connectivity by side?, NT type, Jpp from FNN code
Jpm = pd.read_excel('PMNs R L to MNs R L normalized v5_arefshare_1222.xlsx')
Jpp = pd.read_excel('elife-51781-supp6-v3-PMN2PMN.xlsx')
types = pd.read_excel('elife-51781-supp3-v3-PMN_NT_IDs.xlsx')

#%% Categorize the NT values

eNTs = ['Chat','Chat likely']
iNTs = ['Glut','GABA', 'GABA Likely']

NTs = pd.Series(types.iloc[:,1])
NTvals = np.zeros(len(types))

einds = np.array(types.index[types['Neurotransmitter identity'].isin(eNTs)])
iinds = np.array(types.index[types['Neurotransmitter identity'].isin(iNTs)])

NTvals[einds] = 1
NTvals[iinds] = -1


#%% fxn for fixing naming issues
def namefix(old,pom,ei):
    namesout = list()
    ex_pnamesout = list()
    in_pnamesout = list()
    reminds = list()
    if pom == 1:
        namesout.append('Unnamed: 0')
        for ind,temp in enumerate(old):
            if 'l' in temp:
                sidechar = 'l'
            elif 'r' in temp:
                sidechar = 'r'
            if 'SN' in temp:
                print('skip')
                reminds.append(temp)
            else:
                if '_' in temp:
                    endname = temp.index('_')
                    if '-' in temp:
                        temp = temp[:temp.index("-")]
                    #if '/' = 2 MNs, then store for the number strings before and after the '/'
                    if '/' in temp:
                        temp = temp.split('/')[0]
                skipmn = temp.index('N')
                namesout.append(temp[skipmn+1:endname] + sidechar)
    else:
        for ind,temp in enumerate(old):
            if '_' in temp:
                endname = temp.index('_')
                namesout.append(temp[:endname].lower().strip() + temp[endname:endname+4])
            else:
                if 'l' in temp:
                    endname = temp.index('l')
                elif 'r' in temp:
                    endname = temp.index('r')
                namesout.append(temp[:endname].lower().strip() + '_' + temp[endname])
            if ei == 1:
                if NTvals[int(ind)] == 1:
                    ex_pnamesout.append(temp[:endname].lower().strip())
                elif NTvals[int(ind)] == -1:
                    in_pnamesout.append(temp[:endname].lower().strip())

    return namesout, ex_pnamesout, in_pnamesout, reminds

#%% Fix the naming issues - two different orders
oldtypesnames = np.array(types['PMN'])
oldJpmnames = np.array(Jpm['Unnamed: 0'])
oldmnnames = np.array(Jpm.columns[1:])

pnames, ex_pnames, in_pnames, [] = namefix(oldtypesnames,0,1)
Jpmorder_pnames, [], [], [] = namefix(oldJpmnames,0,0)
mnames, [], [], reminds = namefix(oldmnnames,1,0)

#put the fixed name strings into the Jpm column directly
Jpm['Unnamed: 0'] = Jpmorder_pnames
Jpm.columns = mnames

#and fixed names of PMNs into the Jpp mat directly
Jpp['Unnamed: 0'] = pnames

pnames.insert(0,'PMNs')
Jpp.columns = pnames


#%% DLV grouping - this time 4,5,12 later than LTs; can also try with them before
dorfx = ['1','9','2','10','3','11','19','20']
latfx = ['23','22','21','8','18','25','26','27','29']
venfx = ['4','5','12','13','30','14','6','28','15','16']

longfx = dorfx + venfx
transfx = latfx

allmuscs1s = dorfx + latfx + venfx
allmuscsfx = [m + 'l' for m in allmuscs1s] + [m + 'r' for m in allmuscs1s]

#%% reorder connectome according to this muscle order
#for each PMN (row) (look at that row and the row below, idx by 2 so that we're seeing L & R copies of PMNs)
epind = 0
ipind = 0

allPMN_MNfx = -np.ones([int(Jpm.shape[1]-1),int(Jpm.shape[0])])
ePMN_MN = -np.ones([int(Jpm.shape[1]-1),len(ex_pnames)])
iPMN_MN = -np.ones([int(Jpm.shape[1]-1),len(in_pnames)])

#reconstruct FULL LR Jpm mat according to correct PMN row alphabetical, MN col order
Jpm.sort_values("Unnamed: 0",ascending = True, inplace = True)
for pindie in np.arange(Jpm.shape[0]):    
    #find the correct ind for the MN of interest
    for mi in np.arange(1,Jpm.shape[1]):
        mind = list()
        mtemp = Jpm.columns[mi]
        mind = allmuscsfx.index(mtemp)   
        #store in PMN row MN col
        allPMN_MNfx[mind,pindie] = Jpm.iloc[pindie,mi]
        
        #break into E vs. I matrices
        if NTvals[int(pindie)] == 1:
            ePMN_MN[mind,epind] = Jpm.iloc[pindie,mi]
            if mi == Jpm.shape[1]-1:
                epind = epind + 1
            
        elif NTvals[int(pindie)] == -1:
            iPMN_MN[mind,ipind] = Jpm.iloc[pindie,mi]
            if mi == Jpm.shape[1]-1:
                ipind = ipind + 1

#organize the Jpp into mats of interest
allPMN_PMN = np.array(Jpp.iloc[:,1:],dtype='float64').T
ePMN_PMN = np.array(Jpp.iloc[NTvals == 1,1:],dtype='float64').T
iPMN_PMN = np.array(Jpp.iloc[NTvals == -1,1:],dtype='float64').T

einds = [idx+1 for idx,v in enumerate(NTvals) if v == 1]
iinds = [idx+1 for idx,v in enumerate(NTvals) if v == -1]
#also Nt to NT Jpp submats
e2e_PMN_PMN = np.array(Jpp.iloc[einds,einds],dtype='float64').T
e2i_PMN_PMN = np.array(Jpp.iloc[einds,iinds],dtype='float64').T
i2e_PMN_PMN = np.array(Jpp.iloc[iinds,einds],dtype='float64').T
i2i_PMN_PMN = np.array(Jpp.iloc[iinds,iinds],dtype='float64').T

#%%subdivide into anatomical side mats
def div_matsLR(inmat):
    outmatL = inmat[:,0:int(inmat.shape[1]/2)]
    outmatR = inmat[:,int(inmat.shape[1]/2):]
    return outmatL, outmatR

#%% divide the mats
lPMN_MN, rPMN_MN = div_matsLR(allPMN_MNfx)
lPMN_PMN, rPMN_PMN = div_matsLR(allPMN_PMN)

elPMN_MN, erPMN_MN = div_matsLR(ePMN_MN)
elPMN_PMN, erPMN_PMN = div_matsLR(ePMN_PMN)

ilPMN_MN, irPMN_MN = div_matsLR(iPMN_MN)
ilPMN_PMN, irPMN_PMN = div_matsLR(iPMN_PMN)


#%%by submats of NT to NT types
e2e_lPMN_PMN, e2e_rPMN_PMN = div_matsLR(e2e_PMN_PMN)

e2i_lPMN_PMN, e2i_rPMN_PMN = div_matsLR(e2i_PMN_PMN)

i2e_lPMN_PMN, i2e_rPMN_PMN = div_matsLR(i2e_PMN_PMN)

i2i_lPMN_PMN, i2i_rPMN_PMN = div_matsLR(i2i_PMN_PMN)


#%% weighted avg for each PMN according to DLV grps, sort, and replot
#fxn for generating and sorting PMNs according to weighted DLV sums

# 1. assign locations 1-30 - muscles
# 2. take weighted average for that spatial number for each PMN
# 3. sort PMN rows by where weighted average is highest for that PMN
def wavg(mat_syn, out, orignames):
    mat_out = np.zeros(mat_syn.shape[1])
    var_out = np.zeros(mat_syn.shape[1])    
    
    np.nan_to_num(mat_syn, copy=False)
    
    for pm in np.arange(0,mat_syn.shape[1]):
        if np.sum(mat_syn[:,int(pm)] > 0):
            pmint = int(pm)
            print('ashape = '+str(out.shape))
            print('weights='+str(mat_syn[:,pmint].shape))
            mat_out[pmint] = np.average(out, weights = mat_syn[:,pmint])
            var_out[pmint] = np.average((out - mat_out[pmint])**2, weights = mat_syn[:,pmint])
        else:
            pmint = int(pm)
            mat_out[pmint] = np.nan
            var_out[pmint] = np.nan
        
    sortpmns = mat_out.argsort()
    reordp = mat_syn[:,sortpmns]
    zreord = stats.zscore(reordp, axis = 1)
    
    sortnames = [orignames[i] for i in sortpmns]
    xj = mat_out[sortpmns]
    
    #mat_out, sortpmns, reordp, zreord,
    
    return  reordp, zreord, sortpmns, sortnames

#%%calc wavg and store vars to plot for each type of mat
def calc_wavg_sortmats(inmatL,inmatR,valstarg,names):
    reordp_left, zreordp_left, sortp_left, leftsortnames = wavg(inmatL, valstarg, names)
    reordp_right, zreordp_right, sortp_right, rightsortnames = wavg(inmatR, valstarg, names)
    
    #comb_outweights = [xj_left,xj_right]
    comb_regsort = np.vstack([reordp_left,reordp_right])
    comb_zsort = np.vstack([zreordp_left,zreordp_right])
    namesout = np.vstack([leftsortnames, rightsortnames])
    sortp = [sortp_left,sortp_right]
 
    return comb_regsort, comb_zsort, sortp, namesout

#%%basic stats fxn - MWU for mult groups, report stats and Bonferroni
def mwu_grps(inputdata):
    bonfcorr = 0.05/scipy.special.binom(inputdata.shape[1],2)
    stats_out = -np.ones([inputdata.shape[1],inputdata.shape[1],2])
    for input_test in np.arange(inputdata.shape[1]):
        for input_comp in np.arange(inputdata.shape[1]):
            d1 = np.nan_to_num(inputdata[:,input_test])
            d2 = np.nan_to_num(inputdata[:,input_comp])
            stats_out[input_test,input_comp,:] = stats.mannwhitneyu(d1,d2,alternative = "two-sided",method = "exact")
    return stats_out, bonfcorr

#%%resort names such that all left pnames come before all right pnames
def pnames_LRseq(oldnames):
    if 'PMNs' in oldnames:
        oldnames.remove('PMNs')
    
    pnames_sep = oldnames[::2]
    pnames_rep = [sub.replace('l', 'r') for sub in pnames_sep]
    newnames = np.hstack([pnames_sep,pnames_rep])

    return newnames