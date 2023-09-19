#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 15:29:08 2023

@author: PatriciaCooney
"""

#%% imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sb
from scipy import stats
import scipy
import scipy.spatial as sp

#%% plots

#heatmap zscored PMN-MNs primary to MNs
#heatmap connectivity matrices
def plot_p2m_weights(mat_pm,outnum,outname,names,plotti,saveti):
    f,ax = plt.subplots()
    sb.heatmap(mat_pm,cmap = 'Blues')
    #option: sb.clustermap?
    ax.set(ylabel="MNs", xticks=np.arange(len(names)), xticklabels=names, yticks = outnum, yticklabels = outname, title = plotti)
    ax.set_xlabel('Left PMNs', loc = 'left')
    ax.set_xlabel('Right PMNs', loc = 'right')
    #plt.xticks(fontsize=10, rotation=0)
    for tick in ax.xaxis.get_majorticklabels():  # example for xaxis
        tick.set_fontsize(4)
    for tick in ax.yaxis.get_majorticklabels():  # example for xaxis
        tick.set_fontsize(6)
    #add line to show LR division of MNs and PMNs
    ax.vlines(len(names)/2, *ax.get_ylim(), color='k')
    ax.hlines(len(outname)/2, *ax.get_xlim(), color='k')


    plt.tight_layout()
    plt.show()
    f.savefig(saveti+'_LR_PtoM_weights.svg', format = 'svg', dpi = 1200)


#heatmap zscored PMN-PMNs sec and prim to primary
def plot_p2p_weights(mat_pp,outnum,outname,names,plotti,saveti):
    f,ax = plt.subplots()
    sb.heatmap(mat_pp,cmap = 'Blues')
    #option: sb.clustermap?
    ax.set(ylabel="PMNs", xticks=np.arange(len(names)), xticklabels=names, yticks = outnum, yticklabels = outname, title = plotti)
    ax.set_xlabel('Left PMNs', loc = 'left')
    ax.set_xlabel('Right PMNs', loc = 'right')
    #plt.xticks(fontsize=10, rotation=0)
    for tick in ax.xaxis.get_majorticklabels():  # example for xaxis
        tick.set_fontsize(4)
    for tick in ax.yaxis.get_majorticklabels():  # example for xaxis
        tick.set_fontsize(6)
    #add line to show LR division of MNs and PMNs
    ax.vlines(len(names)/2, *ax.get_ylim(), color='k')
    ax.hlines(len(outname)/2, *ax.get_xlim(), color='k')

    plt.tight_layout()
    plt.show()
    f.savefig(saveti+'_LR_PtoP_weights.svg', format = 'svg', dpi = 1200)


#plot hists for optional number of data groups, smushed into 3D array - iterate plot over 3rd dimension
#also iterate over # threshold lines to plot, depdneing on needs
def plot_LRwavg_dists(data,datlabs,thresh,plotti,saveti):
    f,ax = plt.subplots()
    colarr = ['c', 'm', 'b', 'r']
    for i,d in enumerate(data):
        cd = colarr[i]
        plt.hist(data[i], bins = 20, color = cd,  alpha = 0.6, label = datlabs[i])
    ax = plt.gca()
    for t in thresh:
        ax.vlines(t, *ax.get_ylim(), color='k', alpha = 0.8)
    ax.set(xlabel = 'Left vs. Right Synaptic Weight Average', ylabel = 'Frequency of PMNs')
    plt.legend()
    plt.title(plotti)
    
    f.savefig(saveti+'histogram_wavgLR.svg', format = 'svg', dpi = 1200)


#swarmplot of LR wavgs
def swarm_lrwavg(wavgs,xticklabs,plotti,saveti):
    f,ax = plt.subplots()
    sb.swarmplot(data = wavgs,palette = "rocket", dodge=True)
    #add medianlines - show the medians bc run MWU test
    sb.boxplot(showmeans=False,
            meanline=False,
            meanprops={'color': 'k', 'ls': '-', 'lw': 2},
            medianprops={'visible': True},
            whiskerprops={'visible': False},
            zorder=10,
            data=wavgs,
            showfliers=False,
            showbox=False,
            showcaps=False)
    
    ax.set(xlabel="PMN Type",xticklabels=xticklabs, ylabel="LR Preference",title = plotti)
    plt.savefig(saveti + ".svg")


#plot hists for influ metrics
def plot_influmet(data,datlabs,thresh,plotti,saveti):
    f,ax = plt.subplots()
    colarr = ['c', 'm', 'b', 'r']
    for i,d in enumerate(data):
        cd = colarr[i]
        plt.hist(data[i], bins = 20, color = cd,  alpha = 0.6, label = datlabs[i])
    ax = plt.gca()
    for t in thresh:
        ax.vlines(t, *ax.get_ylim(), color='k', alpha = 0.8)
    ax.set(xlabel = 'Influence of Secondary PMNs on MNs', ylabel = 'Frequency of PMNs')
    plt.legend()
    plt.title(plotti)
    
    f.savefig(saveti+'histogram_influ.svg', format = 'svg', dpi = 1200)


#swarmplot of influs
def swarm_influs(influs,plotti,saveti):
    f,ax = plt.subplots()
    sb.swarmplot(data = influs,palette = "rocket", dodge=True)
    #add medianlines - show the medians bc run MWU test
    sb.boxplot(showmeans=False,
            meanline=False,
            meanprops={'color': 'k', 'ls': '-', 'lw': 2},
            whiskerprops={'visible': True},
            boxprops = {'facecolor':'none'},
            zorder=10,
            data=influs,
            showfliers=False,
            showbox=False,
            showcaps=True)
    
    ax.set(xlabel="Secondary to MN Influence per PMN",xticklabels=['E-->E','E-->I','I-->E','I-->I'],ylabel="Normalized Motor Influence",title = plotti)
    plt.savefig(saveti + ".svg")


#swarmplot of LR wavgs
def swarm_contrain(data,xticklabs,ylabtype,plotti,saveti):
    f,ax = plt.subplots()
    sb.swarmplot(data = data,palette = "rocket", dodge=True)
    #add medianlines - show the medians bc run MWU test
    sb.boxplot(showmeans=False,
            meanline=False,
            meanprops={'color': 'k', 'ls': '-', 'lw': 2},
            medianprops={'visible': True},
            whiskerprops={'visible': True},
            boxprops = {'facecolor':'none'},
            zorder=10,
            data=data,
            showfliers=False,
            showbox=False,
            showcaps=True)
    
    if ylabtype == 0:
        ylabel = "Number of Contralateral Inputs"
    elif ylabtype == 1:
        ylabel = "Proportion of Contralateral Inputs"
    elif ylabtype == 2:
        ylabel = "Proportion of Contra to Ipsi Influence"
    else:
        ylabel = "Mean Influence of Contralateral Inputs"
    
    ax.set(xlabel="PMN-MN Type",xticklabels=xticklabs, ylabel=ylabel,title = plotti)
    ax.set_ylim(0.2,0.8)
    plt.savefig(saveti + ".svg")


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

#%% calc wavg of primary PMNs onto MNs - L vs. R
leftMNs = np.zeros(len(allmuscs1s))
rightMNs = np.ones(len(allmuscs1s))
muscvals = np.hstack([leftMNs, rightMNs])
muscles = np.arange(1,1+len(muscvals))

pnames = pnames_LRseq(pnames)
ex_pnames = pnames_LRseq(ex_pnames)
in_pnames = pnames_LRseq(in_pnames)

#PtoM, E, I
comb_LR_PtoM_weights, comb_LR_PtoM, zcomb_LR_PtoM, sortp2m = calc_wavg_sortmats(lPMN_MN,rPMN_MN,muscvals)#,pnames)
e_comb_LR_PtoM_weights, e_comb_LR_PtoM, ze_comb_LR_PtoM, e_sortp2m = calc_wavg_sortmats(elPMN_MN,erPMN_MN,muscvals)#,ex_pnames)
i_comb_LR_PtoM_weights, i_comb_LR_PtoM, zi_comb_LR_PtoM, i_sortp2m = calc_wavg_sortmats(ilPMN_MN,irPMN_MN,muscvals)#,in_pnames)

#%% heatmaps PtoM
plot_p2m_weights(comb_LR_PtoM,muscles,allmuscsfx,pnames,'Primary PMNs to LR MNs','all_pmn_mn_')
plot_p2m_weights(e_comb_LR_PtoM,muscles,allmuscsfx,ex_pnames,'Excitatory Primary PMNs to LR MNs','ex_pmn_mn_')
plot_p2m_weights(i_comb_LR_PtoM,muscles,allmuscsfx,in_pnames,'Inhibitory Primary PMNs to LR MNs','inh_pmn_mn_')


#%% calc wavg of secondary PMNs onto primary PMNs (anatomical LR)
leftPMNs = np.zeros(len(lPMN_PMN))
rightPMNs = np.ones(len(rPMN_PMN))
pmnvals = np.hstack([leftPMNs,rightPMNs])

#PtoP, E, I
comb_LR_PtoP_weights, comb_LR_PtoP, p2pnames, sortp2p = calc_wavg_sortmats(lPMN_PMN,rPMN_PMN,pmnvals,pnames)
e_comb_LR_PtoP_weights, e_comb_LR_PtoP, e_p2pnames, e_sortp2p = calc_wavg_sortmats(elPMN_PMN,erPMN_PMN,pmnvals,ex_pnames)
i_comb_LR_PtoP_weights, i_comb_LR_PtoP, i_p2pnames, i_sortp2p = calc_wavg_sortmats(ilPMN_PMN,irPMN_PMN,pmnvals,in_pnames)

#%% heatmap PtoP
np.nan_to_num(comb_LR_PtoP, copy=False)
np.nan_to_num(e_comb_LR_PtoP, copy=False)
np.nan_to_num(i_comb_LR_PtoP, copy=False)

plot_p2p_weights(comb_LR_PtoP,np.arange(comb_LR_PtoP.shape[0]),p2pnames,p2pnames,'Secondary PMNs to LR PMNs','all_pmn_pmn_')
plot_p2p_weights(e_comb_LR_PtoP,np.arange(comb_LR_PtoP.shape[0]),p2pnames,e_p2pnames,'Excitatory Secondary PMNs to LR PMNs','ex_pmn_pmn_')
plot_p2p_weights(i_comb_LR_PtoP,np.arange(comb_LR_PtoP.shape[0]),p2pnames,i_p2pnames,'Inhibitory Secondary PMNs to LR PMNs','inh_pmn_pmn_')


#%%do mat mult - influence of 2order PMNs onto MNs on each side L and R
#do first for all PMNs, then E PMNs and I PMNs
#just to plot heatmap and visualize that the second order PMNs do or do not show preference for L or R MNs on average -- mostly bilateral influence??
lPMN_PMN = np.where(np.isnan(lPMN_PMN),0,lPMN_PMN)
rPMN_PMN = np.where(np.isnan(rPMN_PMN),0,rPMN_PMN)

secLtoMN = allPMN_MNfx@lPMN_PMN
secRtoMN = allPMN_MNfx@rPMN_PMN
sectoMN_comb = np.concatenate([secLtoMN,secRtoMN],axis=1)

#transpose so MN is row,calc wavg,sort, make heatmap
influ_sec2MN_weights, influ_sec2MN, secp2mnames, sortsecp2m = calc_wavg_sortmats(secLtoMN,secRtoMN,muscvals,pnames)
np.nan_to_num(influ_sec2MN, copy=False)
plot_p2m_weights(influ_sec2MN,muscles,allmuscsfx,secp2mnames,'Influence of Secondary PMNs onto MNs','sec_pmn_mn_')

#%% then do E vs. I onto all PMNs --> MNs
#then do for E vs. I
lPMN_PMN = np.where(np.isnan(lPMN_PMN),0,lPMN_PMN)
rPMN_PMN = np.where(np.isnan(rPMN_PMN),0,rPMN_PMN)

secLtoMN = allPMN_MNfx@lPMN_PMN
secRtoMN = allPMN_MNfx@rPMN_PMN
sectoMN_comb = np.concatenate([secLtoMN,secRtoMN],axis=1)


#pull e
e_secLtoMN = elPMN_PMN@allPMN_MNfx
e_secRtoMN = erPMN_PMN@allPMN_MNfx
e_sectoMN_comb = np.concatenate([e_secLtoMN,e_secRtoMN],axis=1)

#transpose so MN is row, col is 2 pmn and make heatmap
e_influ_sec2MN_weights, e_influ_sec2MN, e_secp2mnames, e_sortsecp2m = calc_wavg_sortmats(e_secLtoMN,e_secRtoMN,muscvals,ex_pnames)
np.nan_to_num(e_influ_sec2MN, copy=False)
plot_p2m_weights(e_influ_sec2MN,muscles,allmuscsfx,e_secp2mnames,'Influence of Excitatory Secondary PMNs onto MNs','e_sec_pmn_mn_')

i_secLtoMN = ilPMN_PMN@allPMN_MNfx
i_secRtoMN = irPMN_PMN@allPMN_MNfx
i_sectoMN_comb = np.concatenate([i_secLtoMN,i_secRtoMN],axis=1)

#transpose so MN is row, col is 2 pmn and make heatmap
i_influ_sec2MN_weights, i_influ_sec2MN, i_secp2mnames, i_sortsecp2m = calc_wavg_sortmats(i_secLtoMN,i_secRtoMN,muscvals,ex_pnames)
np.nan_to_num(i_influ_sec2MN, copy=False)
plot_p2m_weights(i_influ_sec2MN,muscles,allmuscsfx,i_secp2mnames,'Influence of Inhibitory Secondary PMNs onto MNs','i_sec_pmn_mn_')


#%% now look at secondary PMN influence according to EE, EI, IE, II groupings
e2e_secLtoMN = e2e_lPMN_PMN@ePMN_MN
e2e_secRtoMN = e2e_rPMN_PMN@ePMN_MN
e2e_sectoMN_comb = np.concatenate([e2e_secLtoMN,e2e_secRtoMN],axis=1)

#transpose so MN is row, col is 2 pmn and make heatmap
e2e_influ_sec2MN_weights, e2e_influ_sec2MN, e2e_secp2mnames, e2e_sortsecp2m = calc_wavg_sortmats(e2e_secLtoMN,e2e_secRtoMN,muscvals,ex_pnames)
np.nan_to_num(e2e_influ_sec2MN, copy=False)
plot_p2m_weights(e2e_influ_sec2MN,muscles,allmuscsfx,e2e_secp2mnames,'Influence of E-E PMNs on MNs','e2e_sec_pmn_mn_')


#e2i
e2i_secLtoMN = e2i_lPMN_PMN@iPMN_MN
e2i_secRtoMN = e2i_rPMN_PMN@iPMN_MN
e2i_sectoMN_comb = np.concatenate([e2i_secLtoMN,e2i_secRtoMN],axis=1)

#transpose so MN is row, col is 2 pmn and make heatmap
e2i_influ_sec2MN_weights, e2i_influ_sec2MN, e2i_secp2mnames, e2i_sortsecp2m = calc_wavg_sortmats(e2i_secLtoMN,e2i_secRtoMN,muscvals,ex_pnames)
np.nan_to_num(e2i_influ_sec2MN, copy=False)
plot_p2m_weights(e2i_influ_sec2MN,muscles,allmuscsfx,e2i_secp2mnames,'Influence of E-I PMNs on MNs','e2i_sec_pmn_mn_')


#i2e
i2e_secLtoMN = i2e_lPMN_PMN@ePMN_MN
i2e_secRtoMN = i2e_rPMN_PMN@ePMN_MN
i2e_sectoMN_comb = np.concatenate([i2e_secLtoMN,i2e_secRtoMN],axis=1)

#transpose so MN is row, col is 2 pmn and make heatmap
i2e_influ_sec2MN_weights, i2e_influ_sec2MN, i2e_secp2mnames, i2e_sortsecp2m = calc_wavg_sortmats(i2e_secLtoMN,i2e_secRtoMN,muscvals,ex_pnames)
np.nan_to_num(i2e_influ_sec2MN, copy=False)
plot_p2m_weights(i2e_influ_sec2MN,muscles,allmuscsfx,i2e_secp2mnames,'Influence of I-E PMNs on MNs','i2e_sec_pmn_mn_')


#i2i
i2i_secLtoMN = i2i_lPMN_PMN@iPMN_MN
i2i_secRtoMN = i2i_rPMN_PMN@iPMN_MN
i2i_sectoMN_comb = np.concatenate([i2i_secLtoMN,i2i_secRtoMN],axis=1)

#transpose so MN is row, col is 2 pmn and make heatmap
i2i_influ_sec2MN_weights, i2i_influ_sec2MN, i2i_secp2mnames, i2i_sortsecp2m = calc_wavg_sortmats(i2i_secLtoMN,i2i_secRtoMN,muscvals,ex_pnames)
np.nan_to_num(i2i_influ_sec2MN, copy=False)
plot_p2m_weights(i2i_influ_sec2MN,muscles,allmuscsfx,i2i_secp2mnames,'Influence of I-I PMNs on MNs','i2i_sec_pmn_mn_')


#%% check to see whether the second order PMN interactions of each NT type have diferent levels of influ on MNs
tot_e2e = np.sum(e2e_sectoMN_comb)
tot_e2i = np.sum(e2i_sectoMN_comb)
tot_i2e = np.sum(i2e_sectoMN_comb)
tot_i2i = np.sum(i2i_sectoMN_comb)

avg_e2e = np.mean(e2e_sectoMN_comb,axis=0)
avg_e2i = np.mean(e2i_sectoMN_comb,axis=0)
avg_i2e = np.mean(i2e_sectoMN_comb,axis=0)
avg_i2i = np.mean(i2i_sectoMN_comb,axis=0)

comball_NT_secinflu = list([avg_e2e,avg_e2i,avg_i2e,avg_i2i])

#plot distribution of influ metrics for each NT pair type
plot_influmet(comball_NT_secinflu,['E-->E','E-->I','I-->E','I-->I'],[],plotti='Motor Influence from NT Pair Types',saveti='nttype_influ_')

#plot as swarmplot
mean_influ = np.array([avg_e2e,avg_e2i,np.concatenate([avg_i2e,np.repeat(np.nan,34)]),np.concatenate([avg_i2i,np.repeat(np.nan,34)])])

df_influs = pd.DataFrame(data=mean_influ)

swarm_influs(df_influs, plotti='Motor Influence from NT Pair Types', saveti="influ_by_ntpairs")
#%% run stats on the influence metrics by NT pair type
influ_arr = np.array(df_influs)
stats_influ,bonf_influ = mwu_grps(influ_arr)


#%%
#%%
#%%
#%%
#cosine similarity matrices
def plot_cos(cosmat,ti):
    f,ax = plt.subplots()
    sb.heatmap(cosmat,cmap = 'Blues', vmin=0, vmax=1)
    #ax.set(xlabel="Muscles D-->V", ylabel="Muscles (V --> D)",xticks=np.arange(len(muscs)), xticklabels = muscs,  yticks=np.arange(len(muscs)), yticklabels = muscs,title = ti)
    ax.set(title = ti)
    for tick in ax.xaxis.get_majorticklabels():  # example for xaxis
        tick.set_fontsize(8)
    for tick in ax.yaxis.get_majorticklabels():  # example for xaxis
        tick.set_fontsize(8)
        f.savefig(ti+'_cossim_weights.svg', format = 'svg', dpi = 1200)

#%% 8/31 - calculate the cosine similarity of specific NT and side submats
#first do agnostic to musc grp
#all the primary E mats -- make 1 ipsi, 1 contra
v, w, ipsi_left_primE_sort, t, u = wavg(elPMN_MN[:int(elPMN_MN.shape[0]/2),:int(elPMN_MN.shape[1]/2)],muscvals[:int(muscvals.shape[0]/2)])
v, w, contra_left_primE_sort, t, u = wavg(elPMN_MN[int(elPMN_MN.shape[0]/2):,int(elPMN_MN.shape[1]/2):],muscvals[int(muscvals.shape[0]/2):])
v, w, ipsi_right_primE_sort, t, u = wavg(erPMN_MN[:int(erPMN_MN.shape[0]/2),:int(erPMN_MN.shape[1]/2)],muscvals[:int(muscvals.shape[0]/2)])
v, w, contra_right_primE_sort, t, u = wavg(erPMN_MN[int(erPMN_MN.shape[0]/2):,int(erPMN_MN.shape[1]/2):],muscvals[int(muscvals.shape[0]/2):])

ipsi_primE_sort = np.mean([ipsi_left_primE_sort,ipsi_right_primE_sort],axis=0)
contra_primE_sort = np.mean([contra_left_primE_sort,contra_right_primE_sort],axis=0)

#all the secondary I2E mats -- make 1 ipsi, 1 contra
toleft_seci2e_influ = i2e_lPMN_PMN[:,:int(i2e_lPMN_PMN.shape[1]/2)]@elPMN_MN
toright_seci2e_influ = i2e_rPMN_PMN[:,:int(i2e_rPMN_PMN.shape[1]/2)]@erPMN_MN

v, w, ipsi_left_seci2e_sort, t, u = wavg(toleft_seci2e_influ[:,:int(toleft_seci2e_influ.shape[1]/2)],muscvals[:int(muscvals.shape[0]/2)])
v, w, contra_left_seci2e_sort, t, u = wavg(toleft_seci2e_influ[:,int(toleft_seci2e_influ.shape[1]/2):],muscvals[int(muscvals.shape[0]/2):])
v, w, ipsi_right_seci2e_sort, t, u = wavg(toright_seci2e_influ[:,:int(toright_seci2e_influ.shape[1]/2)],muscvals[:int(muscvals.shape[0]/2)])
v, w, contra_right_seci2e_sort, t, u = wavg(toright_seci2e_influ[:,int(toright_seci2e_influ.shape[1]/2):],muscvals[int(muscvals.shape[0]/2):])

ipsi_seci2e_sort = np.mean([ipsi_left_seci2e_sort,ipsi_right_seci2e_sort],axis=0)
contra_seci2e_sort = np.mean([contra_left_seci2e_sort,contra_right_seci2e_sort],axis=0)

#take the cosine similarity of these mats
# do these separate first for check of data quality, but then can combine just same (ipsi-ipsi + contra-contra) and diff (ipsi-contra, contra-ipsi)
cos_ipsiE_ipsiI2E = 1 - sp.distance.cdist(ipsi_primE_sort, ipsi_seci2e_sort, 'cosine')
cos_ipsiE_contraI2E = 1 - sp.distance.cdist(ipsi_primE_sort, contra_seci2e_sort, 'cosine')
cos_contraE_ipsiI2E = 1 - sp.distance.cdist(contra_primE_sort, ipsi_seci2e_sort, 'cosine')
cos_contraE_contraI2E = 1 - sp.distance.cdist(contra_primE_sort, contra_seci2e_sort, 'cosine')


#replot with muscle labels and appropriate titles
plot_cos(cos_ipsiE_ipsiI2E, 'Cosine Similarity - Ipsi Primary E-Ipsi Secondary I to E')
plot_cos(cos_ipsiE_contraI2E, 'Cosine Similarity - Ipsi Primary E-Contra Secondary I to E')
plot_cos(cos_contraE_ipsiI2E, 'Cosine Similarity - Contra Primary E-Ipsi Secondary I to E')
plot_cos(cos_contraE_contraI2E, 'Cosine Similarity - Contra Primary E-Contra Secondary I to E')

plot_cos(np.mean([cos_ipsiE_ipsiI2E, cos_contraE_contraI2E],axis=0), 'Cosine Similarity - Same Side Primary E and Secondary I to E')
plot_cos(np.mean([cos_ipsiE_contraI2E, cos_contraE_ipsiI2E],axis=0), 'Cosine Similarity - Opposite Side Primary E and Secondary I to E')


#%%
#try again , but this time with binary mats
bi_ipsi_primE = np.where(ipsi_primE_sort>0, 1, 0)
bi_contra_primE = np.where(contra_primE_sort>0, 1, 0)

bi_ipsi_seci2e = np.where(ipsi_seci2e_sort>0, 1, 0)
bi_contra_seci2e = np.where(contra_seci2e_sort>0, 1, 0)

# do these separate first for check of data quality, but then can combine just same (ipsi-ipsi + contra-contra) and diff (ipsi-contra, contra-ipsi)
bicos_ipsiE_ipsiI2E = 1 - sp.distance.cdist(bi_ipsi_primE[:10], bi_ipsi_seci2e[:10], 'cosine')
bicos_ipsiE_contraI2E = 1 - sp.distance.cdist(bi_ipsi_primE[:10], bi_contra_seci2e[:10], 'cosine')
bicos_contraE_ipsiI2E = 1 - sp.distance.cdist(bi_contra_primE[:10], bi_ipsi_seci2e[:10], 'cosine')
bicos_contraE_contraI2E = 1 - sp.distance.cdist(bi_contra_primE[:10], bi_contra_seci2e[:10], 'cosine')


#replot with muscle labels and appropriate titles
plot_cos(bicos_ipsiE_ipsiI2E, 'Cosine Similarity - Top10 Ipsi Primary E-Ipsi Secondary I to E')
plot_cos(bicos_ipsiE_contraI2E, 'Cosine Similarity - Top10 Ipsi Primary E-Contra Secondary I to E')
plot_cos(bicos_contraE_ipsiI2E, 'Cosine Similarity - Top10 Contra Primary E-Ipsi Secondary I to E')
plot_cos(bicos_contraE_contraI2E, 'Cosine Similarity - Top10 Contra Primary E-Contra Secondary I to E')

plot_cos(np.mean([bicos_ipsiE_ipsiI2E[:10], bicos_contraE_contraI2E[:10]],axis=0), 'Cosine Similarity - Top10 Same Side Primary E and Secondary I to E')
plot_cos(np.mean([bicos_ipsiE_contraI2E[:10], bicos_contraE_ipsiI2E[:10]],axis=0), 'Cosine Similarity - Top10 Opposite Side Primary E and Secondary I to E')





#%%
#%%
#%% do the same test, but check prim I and sec E to I
#first do agnostic to musc grp
#all the primary E mats -- make 1 ipsi, 1 contra
v, w, ipsi_left_primE_sort, t, u = wavg(ilPMN_MN[:int(ilPMN_MN.shape[0]/2),:int(ilPMN_MN.shape[1]/2)],muscvals[:int(muscvals.shape[0]/2)])
v, w, contra_left_primE_sort, t, u = wavg(ilPMN_MN[int(ilPMN_MN.shape[0]/2):,int(ilPMN_MN.shape[1]/2):],muscvals[int(muscvals.shape[0]/2):])
v, w, ipsi_right_primE_sort, t, u = wavg(irPMN_MN[:int(irPMN_MN.shape[0]/2),:int(irPMN_MN.shape[1]/2)],muscvals[:int(muscvals.shape[0]/2)])
v, w, contra_right_primE_sort, t, u = wavg(irPMN_MN[int(irPMN_MN.shape[0]/2):,int(irPMN_MN.shape[1]/2):],muscvals[int(muscvals.shape[0]/2):])

ipsi_primE_sort = np.mean([ipsi_left_primE_sort,ipsi_right_primE_sort],axis=0)
contra_primE_sort = np.mean([contra_left_primE_sort,contra_right_primE_sort],axis=0)

#all the secondary e2i mats -- make 1 ipsi, 1 contra
toleft_sece2i_influ = e2i_lPMN_PMN[:,:int(e2i_lPMN_PMN.shape[1]/2)]@ilPMN_MN
toright_sece2i_influ = e2i_rPMN_PMN[:,:int(e2i_rPMN_PMN.shape[1]/2)]@irPMN_MN

v, w, ipsi_left_sece2i_sort, t, u = wavg(toleft_sece2i_influ[:,:int(toleft_sece2i_influ.shape[1]/2)],muscvals[:int(muscvals.shape[0]/2)])
v, w, contra_left_sece2i_sort, t, u = wavg(toleft_sece2i_influ[:,int(toleft_sece2i_influ.shape[1]/2):],muscvals[int(muscvals.shape[0]/2):])
v, w, ipsi_right_sece2i_sort, t, u = wavg(toright_sece2i_influ[:,:int(toright_sece2i_influ.shape[1]/2)],muscvals[:int(muscvals.shape[0]/2)])
v, w, contra_right_sece2i_sort, t, u = wavg(toright_sece2i_influ[:,int(toright_sece2i_influ.shape[1]/2):],muscvals[int(muscvals.shape[0]/2):])

ipsi_sece2i_sort = np.mean([ipsi_left_sece2i_sort,ipsi_right_sece2i_sort],axis=0)
contra_sece2i_sort = np.mean([contra_left_sece2i_sort,contra_right_sece2i_sort],axis=0)

#take the cosine similarity of these mats
# do these separate first for check of data quality, but then can combine just same (ipsi-ipsi + contra-contra) and diff (ipsi-contra, contra-ipsi)
cos_ipsiE_ipsie2i = 1 - sp.distance.cdist(ipsi_primE_sort, ipsi_sece2i_sort, 'cosine')
cos_ipsiE_contrae2i = 1 - sp.distance.cdist(ipsi_primE_sort, contra_sece2i_sort, 'cosine')
cos_contraE_ipsie2i = 1 - sp.distance.cdist(contra_primE_sort, ipsi_sece2i_sort, 'cosine')
cos_contraE_contrae2i = 1 - sp.distance.cdist(contra_primE_sort, contra_sece2i_sort, 'cosine')


#replot with muscle labels and appropriate titles
plot_cos(cos_ipsiE_ipsie2i, 'Cosine Similarity - Ipsi Primary I-Ipsi Secondary E to I')
plot_cos(cos_ipsiE_contrae2i, 'Cosine Similarity - Ipsi Primary I-Contra Secondary E to I')
plot_cos(cos_contraE_ipsie2i, 'Cosine Similarity - Contra Primary I-Ipsi Secondary E to I')
plot_cos(cos_contraE_contrae2i, 'Cosine Similarity - Contra Primary I-Contra Secondary E to I')

plot_cos(np.mean([cos_ipsiE_ipsie2i, cos_contraE_contrae2i],axis=0), 'Cosine Similarity - Same Side Primary I and Secondary E to I')
plot_cos(np.mean([cos_ipsiE_contrae2i, cos_contraE_ipsie2i],axis=0), 'Cosine Similarity - Opposite Side Primary I and Secondary E to I')


#%%
#try again , but this time with binary mats
bi_ipsi_primE = np.where(ipsi_primE_sort>0, 1, 0)
bi_contra_primE = np.where(contra_primE_sort>0, 1, 0)

bi_ipsi_sece2i = np.where(ipsi_sece2i_sort>0, 1, 0)
bi_contra_sece2i = np.where(contra_sece2i_sort>0, 1, 0)

# do these separate first for check of data quality, but then can combine just same (ipsi-ipsi + contra-contra) and diff (ipsi-contra, contra-ipsi)
bicos_ipsiE_ipsie2i = 1 - sp.distance.cdist(bi_ipsi_primE[:10], bi_ipsi_sece2i[:10], 'cosine')
bicos_ipsiE_contrae2i = 1 - sp.distance.cdist(bi_ipsi_primE[:10], bi_contra_sece2i[:10], 'cosine')
bicos_contraE_ipsie2i = 1 - sp.distance.cdist(bi_contra_primE[:10], bi_ipsi_sece2i[:10], 'cosine')
bicos_contraE_contrae2i = 1 - sp.distance.cdist(bi_contra_primE[:10], bi_contra_sece2i[:10], 'cosine')


#replot with muscle labels and appropriate titles
plot_cos(bicos_ipsiE_ipsie2i, 'Cosine Similarity - Top10 Ipsi Primary I-Ipsi Secondary E to I')
plot_cos(bicos_ipsiE_contrae2i, 'Cosine Similarity - Top10 Ipsi Primary I-Contra Secondary E to I')
plot_cos(bicos_contraE_ipsie2i, 'Cosine Similarity - Top10 Contra Primary I-Ipsi Secondary E to I')
plot_cos(bicos_contraE_contrae2i, 'Cosine Similarity - Top10 Contra Primary I-Contra Secondary E to I')

plot_cos(np.mean([bicos_ipsiE_ipsie2i[:10], bicos_contraE_contrae2i[:10]],axis=0), 'Cosine Similarity - Top10 Same Side Primary I and Secondary E to I')
plot_cos(np.mean([bicos_ipsiE_contrae2i[:10], bicos_contraE_ipsie2i[:10]],axis=0), 'Cosine Similarity - Top10 Opposite Side Primary I and Secondary E to I')


#%%
#%%
#%% 
#%% pull out cnxn count for PMN to MN contra - break down by the musc grp project out to = first pass, primary PMN mats, e vs i
#binarize the mats so then can look at each PMN's outputs = if > 1 in quad LL then ipsi, etc
#then take sum per MN of contra inputs - make into array for that musc grp
econtra_bin_lPMN_MN = np.where(elPMN_MN[int(len(muscles)/2):,:]>0, 1, 0)
econtra_bin_rPMN_MN = np.where(erPMN_MN[int(len(muscles)/2):,:]>0, 1, 0)
#try this sum counts with normalization
econtra_PMN_MN = np.squeeze(np.sum([np.sum([econtra_bin_lPMN_MN,econtra_bin_rPMN_MN],axis=0)],axis=2))/len(einds)

icontra_bin_lPMN_MN = np.where(ilPMN_MN[int(len(muscles)/2):,:]>0, 1, 0)
icontra_bin_rPMN_MN = np.where(irPMN_MN[int(len(muscles)/2):,:]>0, 1, 0)
#try this sum counts with normalization
icontra_PMN_MN = np.squeeze(np.sum([np.sum([icontra_bin_lPMN_MN,icontra_bin_rPMN_MN],axis=0)],axis=2))/len(iinds)

#then pull out the musc grp inds
e_longcontra = np.hstack([econtra_PMN_MN[:len(dorfx)],econtra_PMN_MN[len(dorfx)+len(latfx):]])
i_longcontra = np.hstack([icontra_PMN_MN[:len(dorfx)],icontra_PMN_MN[len(dorfx)+len(latfx):]])

e_transcontra = econtra_PMN_MN[len(dorfx):len(dorfx)+len(latfx)]
i_transcontra = icontra_PMN_MN[len(dorfx):len(dorfx)+len(latfx)]

#then make a swarmplot
prim_contra_ei = pd.DataFrame(data = np.array([e_longcontra, i_longcontra, np.concatenate([e_transcontra,np.repeat(np.nan,len(e_longcontra)-len(e_transcontra))]), np.concatenate([i_transcontra,np.repeat(np.nan,len(i_longcontra)-len(i_transcontra))])]))
swarm_contrain(prim_contra_ei.T,['Prim E to Long','Prim I to Long','Prim E to Trans','Prim I to Trans'],0,'Contralateral Innervation of Muscle Groups','longvstrans_evsi_contra_primary_norm')


stats_contraei_prim, bonf_contraei_prim = mwu_grps(np.array(prim_contra_ei).T)

#%% do again as proportion of contra to ipsi
eipsi_bin_lPMN_MN = np.where(elPMN_MN[:int(len(muscles)/2),:]>0, 1, 0)
eipsi_bin_rPMN_MN = np.where(erPMN_MN[:int(len(muscles)/2),:]>0, 1, 0)
#try this sum counts with normalization
eipsi_PMN_MN = np.squeeze(np.sum([np.sum([eipsi_bin_lPMN_MN,eipsi_bin_rPMN_MN],axis=0)],axis=2))/len(einds)

iipsi_bin_lPMN_MN = np.where(ilPMN_MN[:int(len(muscles)/2),:]>0, 1, 0)
iipsi_bin_rPMN_MN = np.where(irPMN_MN[:int(len(muscles)/2),:]>0, 1, 0)
#try this sum counts with normalization
iipsi_PMN_MN = np.squeeze(np.sum([np.sum([iipsi_bin_lPMN_MN,iipsi_bin_rPMN_MN],axis=0)],axis=2))/len(iinds)

#then pull out the musc grp inds
e_longipsi = np.hstack([eipsi_PMN_MN[:len(dorfx)],eipsi_PMN_MN[len(dorfx)+len(latfx):]])
i_longipsi = np.hstack([iipsi_PMN_MN[:len(dorfx)],iipsi_PMN_MN[len(dorfx)+len(latfx):]])

e_transipsi = eipsi_PMN_MN[len(dorfx):len(dorfx)+len(latfx)]
i_transipsi = iipsi_PMN_MN[len(dorfx):len(dorfx)+len(latfx)]

#proportion of contra to ipsi
prop_ci_e_long = e_longcontra / (e_longcontra + e_longipsi)
prop_ci_i_long = i_longcontra / (i_longcontra + i_longipsi)
prop_ci_e_trans = e_transcontra / (e_transcontra + e_transipsi)
prop_ci_i_trans = i_transcontra / (i_transcontra + i_transipsi)

#then make a swarmplot
prim_prop_ci_ei = pd.DataFrame(data = np.array([prop_ci_e_long, prop_ci_i_long, np.concatenate([prop_ci_e_trans,np.repeat(np.nan,len(prop_ci_e_long)-len(prop_ci_e_trans))]), np.concatenate([prop_ci_i_trans,np.repeat(np.nan,len(prop_ci_i_long)-len(prop_ci_i_trans))])]))
swarm_contrain(prim_prop_ci_ei.T,['Prim E to Long','Prim I to Long','Prim E to Trans','Prim I to Trans'],1,'Contralateral Innervation of Muscle Groups','longvstrans_evsi_prop_ci_primary')


stats_propci_ei_prim, bonf_propci_ei_prim = mwu_grps(np.array(prim_prop_ci_ei).T)

#%% redo analysis with anovas b/c normally distributed data for all
kruskal_prim_contraei_p, kruskal_prim_contraei_bonf = krusk_grps(np.array(prim_prop_ci_ei).T)

#%% set up anova
def anova_grps(inputdata):
    bonfcorr = 0.05/scipy.special.binom(inputdata.shape[1],2)
    stats_out = -np.ones([inputdata.shape[1],inputdata.shape[1],2])
    for input_test in np.arange(inputdata.shape[1]):
        for input_comp in np.arange(inputdata.shape[1]):
            d1 = np.nan_to_num(inputdata[:,input_test])
            d2 = np.nan_to_num(inputdata[:,input_comp])
            stats_out[input_test,input_comp,:] = stats.f_oneway(d1,d2)
    return stats_out, bonfcorr
#%% redo anova
anova_prim_contraei_p, anova_prim_contraei_bonf = anova_grps(np.array(prim_prop_ci_ei).T)

#%%
#%%
#%%
#%% do at secondary level - plotting # and normalized counts of contra inputs 
#remove nan entries from all PMN mats
lPMN_PMN = np.where(np.isnan(lPMN_PMN),0,lPMN_PMN)
rPMN_PMN = np.where(np.isnan(rPMN_PMN),0,rPMN_PMN)

#make the general influ mats == same steps as above, but now will pull just contra + pull out inds of interst
secLtoMN = allPMN_MNfx@lPMN_PMN
secRtoMN = allPMN_MNfx@rPMN_PMN
sectoMN_comb = np.concatenate([secLtoMN,secRtoMN],axis=1)

#pull e
e_secLtoMN = allPMN_MNfx@elPMN_PMN
e_secRtoMN = allPMN_MNfx@erPMN_PMN
e_sectoMN_comb = np.concatenate([e_secLtoMN,e_secRtoMN],axis=1)
#pull i
i_secLtoMN = allPMN_MNfx@ilPMN_PMN
i_secRtoMN = allPMN_MNfx@irPMN_PMN
i_sectoMN_comb = np.concatenate([i_secLtoMN,i_secRtoMN],axis=1)

#e2e
e2e_secLtoMN = ePMN_MN@e2e_lPMN_PMN
e2e_secRtoMN = ePMN_MN@e2e_rPMN_PMN
e2e_sectoMN_comb = np.concatenate([e2e_secLtoMN,e2e_secRtoMN],axis=1)
#e2i
e2i_secLtoMN = iPMN_MN@e2i_lPMN_PMN
e2i_secRtoMN = iPMN_MN@e2i_rPMN_PMN
e2i_sectoMN_comb = np.concatenate([e2i_secLtoMN,e2i_secRtoMN],axis=1)
#i2e
i2e_secLtoMN = ePMN_MN@i2e_lPMN_PMN
i2e_secRtoMN = ePMN_MN@i2e_rPMN_PMN
i2e_sectoMN_comb = np.concatenate([i2e_secLtoMN,i2e_secRtoMN],axis=1)
#i2i
i2i_secLtoMN = iPMN_MN@i2i_lPMN_PMN
i2i_secRtoMN = iPMN_MN@i2i_rPMN_PMN
i2i_sectoMN_comb = np.concatenate([i2i_secLtoMN,i2i_secRtoMN],axis=1)


#%% pull out contra submats and make mean mats
contra_e2e = np.mean(np.mean([e2e_secLtoMN[int(len(e2e_secLtoMN)/2):,:],e2e_secRtoMN[:int(len(e2e_secRtoMN)/2),:]],axis=0),axis=1)
contra_i2e = np.mean(np.mean([i2e_secLtoMN[int(len(i2e_secLtoMN)/2):,:],i2e_secRtoMN[:int(len(i2e_secRtoMN)/2),:]],axis=0),axis=1)
contra_e2i = np.mean(np.mean([e2i_secLtoMN[int(len(e2i_secLtoMN)/2):,:],e2i_secRtoMN[:int(len(e2i_secRtoMN)/2),:]],axis=0),axis=1)
contra_i2i = np.mean(np.mean([i2i_secLtoMN[int(len(i2i_secLtoMN)/2):,:],i2i_secRtoMN[:int(len(i2i_secRtoMN)/2),:]],axis=0),axis=1)

contra_e2e = np.where(np.isnan(contra_e2e),0,contra_e2e)
contra_i2e = np.where(np.isnan(contra_i2e),0,contra_i2e)

#%% same for the ipsi to do a proportions test
ipsi_e2e = np.mean(np.mean([e2e_secLtoMN[:int(len(e2e_secLtoMN)/2),:],e2e_secRtoMN[int(len(e2e_secRtoMN)/2):,:]],axis=0),axis=1)
ipsi_i2e = np.mean(np.mean([i2e_secLtoMN[:int(len(i2e_secLtoMN)/2),:],i2e_secRtoMN[int(len(i2e_secRtoMN)/2):,:]],axis=0),axis=1)
ipsi_e2i = np.mean(np.mean([e2i_secLtoMN[:int(len(e2i_secLtoMN)/2),:],e2i_secRtoMN[int(len(e2i_secRtoMN)/2):,:]],axis=0),axis=1)
ipsi_i2i = np.mean(np.mean([i2i_secLtoMN[:int(len(i2i_secLtoMN)/2),:],i2i_secRtoMN[int(len(i2i_secRtoMN)/2):,:]],axis=0),axis=1)
ipsi_e2e = np.where(np.isnan(ipsi_e2e),0,ipsi_e2e)
ipsi_i2e = np.where(np.isnan(ipsi_i2e),0,ipsi_i2e)

#%% fxn to pull out longs vs trans
def pullmuscs(inmat,musctype):
    if musctype == 0: #longs
        if inmat.shape[1]>0:
            outmatleft = np.vstack([inmat[:len(dorfx),:],inmat[len(dorfx)+len(latfx):int(len(muscvals)/2),:]])
            outmatright = np.vstack([inmat[int(len(muscvals)/2):len(dorfx)+int(len(muscvals)/2),:],inmat[len(dorfx)+len(latfx)+int(len(muscvals)/2):,:]])
            outmat = np.vstack([outmatleft,outmatright])
        else:
            outmat = np.hstack([inmat[:len(dorfx)],inmat[len(dorfx)+len(latfx):]])
    if musctype == 1: #trans
        if inmat.shape[1]>0:    
            outmat = inmat[len(dorfx):len(dorfx)+len(latfx),:int(len(muscvals)/2),:]
        else:
            outmat = inmat[len(dorfx):len(dorfx)+len(latfx)]
    return outmat

#%% pull out longs vs trans
contra_e2e_long = pullmuscs(contra_e2e,0)
contra_i2e_long = pullmuscs(contra_i2e,0)
contra_e2i_long = pullmuscs(contra_e2i,0)
contra_i2i_long = pullmuscs(contra_i2i,0)

contra_e2e_trans = np.concatenate([pullmuscs(contra_e2e,1),np.repeat(np.nan,len(contra_e2e_long)-len(latfx))])
contra_i2e_trans = np.concatenate([pullmuscs(contra_i2e,1),np.repeat(np.nan,len(contra_i2e_long)-len(latfx))])
contra_e2i_trans = np.concatenate([pullmuscs(contra_e2i,1),np.repeat(np.nan,len(contra_e2i_long)-len(latfx))])
contra_i2i_trans = np.concatenate([pullmuscs(contra_i2i,1),np.repeat(np.nan,len(contra_i2i_long)-len(latfx))])

#plot swarmplot from these new mats
sec_contra_ei = pd.DataFrame(data = np.array([contra_e2e_long, contra_i2e_long, contra_e2i_long, contra_i2i_long, contra_e2e_trans, contra_i2e_trans, contra_e2i_trans, contra_i2i_trans]))
swarm_contrain(sec_contra_ei.T,['EE-long','EI-long','IE-long','II-long','EE-trans','EI-trans','IE-trans','II-trans'],2,'Secondary Contralateral Innervation of Muscle Groups','longvstrans_motifs_contra_secondary_norm')

stats_contraei_sec, bonf_contraei_sec = mwu_grps(np.array(sec_contra_ei).T)

#%% repeat but for proportion of contra to ipsi
ipsi_e2e_long = pullmuscs(ipsi_e2e,0)
ipsi_i2e_long = pullmuscs(ipsi_i2e,0)
ipsi_e2i_long = pullmuscs(ipsi_e2i,0)
ipsi_i2i_long = pullmuscs(ipsi_i2i,0)

ipsi_e2e_trans = np.concatenate([pullmuscs(ipsi_e2e,1),np.repeat(np.nan,len(ipsi_e2e_long)-len(latfx))])
ipsi_i2e_trans = np.concatenate([pullmuscs(ipsi_i2e,1),np.repeat(np.nan,len(ipsi_i2e_long)-len(latfx))])
ipsi_e2i_trans = np.concatenate([pullmuscs(ipsi_e2i,1),np.repeat(np.nan,len(ipsi_e2i_long)-len(latfx))])
ipsi_i2i_trans = np.concatenate([pullmuscs(ipsi_i2i,1),np.repeat(np.nan,len(ipsi_i2i_long)-len(latfx))])

#proportion contra to ipsi
prop_ci_e2e_long = contra_e2e_long / (ipsi_e2e_long + contra_e2e_long)
prop_ci_i2e_long = contra_i2e_long / (ipsi_i2e_long + contra_i2e_long)
prop_ci_e2i_long = contra_e2i_long / (ipsi_e2i_long + contra_e2i_long)
prop_ci_i2i_long = contra_i2i_long / (ipsi_i2i_long + contra_i2i_long)

prop_ci_e2e_trans = contra_e2e_trans / (ipsi_e2e_trans + contra_e2e_trans)
prop_ci_i2e_trans = contra_i2e_trans / (ipsi_i2e_trans + contra_i2e_trans)
prop_ci_e2i_trans = contra_e2i_trans / (ipsi_e2i_trans + contra_e2i_trans)
prop_ci_i2i_trans = contra_i2i_trans / (ipsi_i2i_trans + contra_i2i_trans)

#%%
#plot swarmplot from these new mats
sec_propci_ei = pd.DataFrame(data = np.array([prop_ci_e2e_long, prop_ci_i2e_long, prop_ci_e2i_long, prop_ci_i2i_long, prop_ci_e2e_trans, prop_ci_i2e_trans, prop_ci_e2i_trans, prop_ci_i2i_trans]))
swarm_contrain(sec_propci_ei.T,['EE-long','EI-long','IE-long','II-long','EE-trans','EI-trans','IE-trans','II-trans'],2,'Secondary Contralateral Innervation of Muscle Groups','longvstrans_motifs_proportion_contraipsi_secondary')

stats_contraei_sec, bonf_contraei_sec = mwu_grps(np.array(sec_propci_ei).T)


#%% concerns about stats output -- looks like normal dists for long muscles, not normal for trans
#check normality
normtest = -np.ones([sec_propci_ei.shape[0],2])
for sgrp in range(sec_propci_ei.shape[0]):
    normtest[sgrp,:] = stats.normaltest(sec_propci_ei.iloc[sgrp,:],nan_policy='omit')

#%% set up kruskal
def krusk_grps(inputdata):
    bonfcorr = 0.05/scipy.special.binom(inputdata.shape[1],2)
    stats_out = -np.ones([inputdata.shape[1],inputdata.shape[1],2])
    for input_test in np.arange(inputdata.shape[1]):
        for input_comp in np.arange(inputdata.shape[1]):
            d1 = np.nan_to_num(inputdata[:,input_test])
            d2 = np.nan_to_num(inputdata[:,input_comp])
            stats_out[input_test,input_comp,:] = stats.kruskal(d1,d2,nan_policy = 'omit')
    return stats_out, bonfcorr

#%% redo analysis with anovas b/c normally distributed data for all
kruskal_contraei_p, kruskal_contraei_bonf = krusk_grps(np.array(sec_propci_ei).T)

#%% redo anova
anova_contraei_p, anova_contraei_bonf = anova_grps(np.array(sec_propci_ei).T)



#%% fix the e2e_secLtoMN, e2i_secLtoMN etc mat sortings to make lists of pnames for longs that are more contralateral than not
#pull the subset of longs MNs rows from mat mults for e2e or e2i influ mats
#e2e_secLtoMN = e2e_lPMN_PMN@ePMN_MN
#e2e_secRtoMN = e2e_rPMN_PMN@ePMN_MN
long_muscvals = np.hstack([np.zeros(len(dorfx) + len(venfx)), np.ones(len(dorfx) + len(venfx))])

e2e_secLtoMN_long = pullmuscs(e2e_secLtoMN,0)
e2e_secRtoMN_long = pullmuscs(e2e_secRtoMN,0)

#run thru the wavg fxn so get list of strongest lat pref pmns
e2e_long_sec2MN_weights, e2e_long_sec2MN, e2e_long_sortsecp2m, e2e_long_secp2mnames = calc_wavg_sortmats(e2e_secLtoMN_long,e2e_secRtoMN_long,long_muscvals,ex_pnames)


#e2i
e2i_secLtoMN_long = pullmuscs(e2i_secLtoMN,0)
e2i_secRtoMN_long = pullmuscs(e2i_secRtoMN,0)

#run thru the wavg fxn so get list of strongest lat pref pmns
e2i_long_sec2MN_weights, e2i_long_sec2MN, e2i_long_sortsecp2m, e2i_long_secp2mnames = calc_wavg_sortmats(e2i_secLtoMN_long,e2i_secRtoMN_long,long_muscvals,ex_pnames)

#save each list into csvs -- note, the left prefer right and right prefer left names for each list should be roughly equal
#note - also want to compare the two lists
#take ~top 5-10 candidates - anything we know about each PMN type so far?
e2e_latpref_list = pd.DataFrame(e2e_long_secp2mnames.T)
e2i_latpref_list = pd.DataFrame(e2i_long_secp2mnames.T)

e2e_latpref_list.to_csv('e2e_latpref_list.csv', index = False)
e2i_latpref_list.to_csv('e2i_latpref_list.csv', index = False)
#%%
# #%%rpt for by NT pair type from secondary level
# #binarize the mats so then can look at each PMN's outputs = if > 1 in quad LL then ipsi, etc
# #then take sum per MN of contra inputs - make into array for that musc grp
# econtra_bin_lPMN_MN = np.where(elPMN_MN[int(len(muscles)/2):,:]>0, 1, 0)
# econtra_bin_rPMN_MN = np.where(erPMN_MN[int(len(muscles)/2):,:]>0, 1, 0)
# econtra_PMN_MN = np.squeeze(np.sum([np.sum([econtra_bin_lPMN_MN,econtra_bin_rPMN_MN],axis=0)],axis=2))

# icontra_bin_lPMN_MN = np.where(ilPMN_MN[int(len(muscles)/2):,:]>0, 1, 0)
# icontra_bin_rPMN_MN = np.where(irPMN_MN[int(len(muscles)/2):,:]>0, 1, 0)
# icontra_PMN_MN = np.squeeze(np.sum([np.sum([icontra_bin_lPMN_MN,icontra_bin_rPMN_MN],axis=0)],axis=2))

# #then pull out the musc grp inds
# e_longcontra = np.hstack([econtra_PMN_MN[:len(dorfx)],econtra_PMN_MN[len(dorfx)+len(latfx):]])
# i_longcontra = np.hstack([icontra_PMN_MN[:len(dorfx)],icontra_PMN_MN[len(dorfx)+len(latfx):]])

# e_transcontra = econtra_PMN_MN[len(dorfx):len(dorfx)+len(latfx)]
# i_transcontra = icontra_PMN_MN[len(dorfx):len(dorfx)+len(latfx)]

# #then make a swarmplot
# prim_contra_ei = pd.DataFrame(data = np.array([e_longcontra, i_longcontra, np.concatenate([e_transcontra,np.repeat(np.nan,len(e_longcontra)-len(e_transcontra))]), np.concatenate([i_transcontra,np.repeat(np.nan,len(i_longcontra)-len(i_transcontra))])]))
# swarm_contrain(prim_contra_ei.T,['Prim E to Long','Prim I to Long','Prim E to Trans','Prim I to Trans'],'Contralateral Innervation of Muscle Groups','longvstrans_evsi_contra_primary')


# #%%by submats of NT to NT types
# e2e_lPMN_PMN, e2e_rPMN_PMN = div_matsLR(e2e_PMN_PMN)

# e2i_lPMN_PMN, e2i_rPMN_PMN = div_matsLR(e2i_PMN_PMN)

# i2e_lPMN_PMN, i2e_rPMN_PMN = div_matsLR(i2e_PMN_PMN)

# i2i_lPMN_PMN, i2i_rPMN_PMN = div_matsLR(i2i_PMN_PMN)



#%%
#%%
#%%


#%% plotting histograms of LR wavg for 
# #all PtoM, all PtoP combined by LR wavg of MNs
# comball_PtoM_PtoP = comb_LR_PtoM_weights.copy()
# comball_PtoM_PtoP.extend(comb_LR_PtoP_weights)
# plot_LRwavg_dists(comball_PtoM_PtoP,['Left PtoM','Right PtoM','Left PtoP','Right PtoP'],[0.2,0.5,0.8],plotti = 'LR Preference, PtoP and PtoM', saveti = 'lrpref_pp_pm_comb_')

# #plot PtoM separate
# plot_LRwavg_dists(comb_LR_PtoM_weights,['Left PtoM','Right PtoM'],[0.2,0.5,0.8],plotti = 'LR Preference - Primary PMNs to MNs', saveti = 'lrpref_pm_')

# #plot PtoP separate
# plot_LRwavg_dists(comb_LR_PtoP_weights,['Left PtoP','Right PtoP'],[0.2,0.5,0.8],plotti = 'LR Preference - Secondary to Primary PMNs', saveti = 'lrpref_pp_')

# #%% stats on LRwavg by side
# combarray_LRweights = np.concatenate([np.array(comb_LR_PtoM_weights),np.array(comb_LR_PtoP_weights)]).T
# stats_lrprimsec, bonf_lrprimsec = mwu_grps(combarray_LRweights)

# #plot as swarmplot for stat comp vis
# swarm_lrwavg(combarray_LRweights,['Primary Left','Primary Right','Secondary Left','Secondary Right'],plotti='LR Preference by PMN Order',saveti='lrpref_primsec_')

# #%% repeat for E and I 
# #EXCIT all PtoM, all PtoP combined by LR wavg of MNs
# e_comball_PtoM_PtoP = e_comb_LR_PtoM_weights.copy()
# e_comball_PtoM_PtoP.extend(e_comb_LR_PtoP_weights)
# plot_LRwavg_dists(e_comball_PtoM_PtoP,['Left PtoM','Right PtoM','Left PtoP','Right PtoP'],[0.2,0.5,0.8],plotti = 'Excitatory LR Preference, PtoP and PtoM', saveti = 'excit_lrpref_pp_pm_comb_')

# #plot PtoM separate
# plot_LRwavg_dists(e_comb_LR_PtoM_weights,['Left PtoM','Right PtoM'],[0.2,0.5,0.8],plotti = 'Excitatory LR Preference - Primary PMNs to MNs', saveti = 'excit_lrpref_pm_')

# #plot PtoP separate
# plot_LRwavg_dists(e_comb_LR_PtoP_weights,['Left PtoP','Right PtoP'],[0.2,0.5,0.8],plotti = 'Excitatory LR Preference - Secondary to Primary PMNs', saveti = 'excit_lrpref_pp_')


# #INHIB all PtoM, all PtoP combined by LR wavg of MNs
# i_comball_PtoM_PtoP = i_comb_LR_PtoM_weights.copy()
# i_comball_PtoM_PtoP.extend(i_comb_LR_PtoP_weights)
# plot_LRwavg_dists(i_comball_PtoM_PtoP,['Left PtoM','Right PtoM','Left PtoP','Right PtoP'],[0.2,0.5,0.8],plotti = 'Inhibitory LR Preference, PtoP and PtoM', saveti = 'inhib_lrpref_pp_pm_comb_')

# #plot PtoM separate
# plot_LRwavg_dists(i_comb_LR_PtoM_weights,['Left PtoM','Right PtoM'],[0.2,0.5,0.8],plotti = 'Inhibitory LR Preference - Primary PMNs to MNs', saveti = 'inhib_lrpref_pm_')

# #plot PtoP separate
# plot_LRwavg_dists(i_comb_LR_PtoP_weights,['Left PtoP','Right PtoP'],[0.2,0.5,0.8],plotti = 'Inhibitory LR Preference - Secondary to Primary PMNs', saveti = 'inhib_lrpref_pp_')

# #%% stats on LRwavg E vs. I by side
# combarray_ei_LR_P2M_weights = np.concatenate([np.array(e_comb_LR_PtoM_weights),np.hstack([np.array(i_comb_LR_PtoM_weights),np.nan*np.ones([2,(e_comb_LR_PtoM_weights[0].shape[0]-i_comb_LR_PtoM_weights[0].shape[0])])])]).T
# stats_ei_lr_prim, bonf_ei_lr_prim = mwu_grps(combarray_ei_LR_P2M_weights)

# #plot as swarmplot for stat comp vis
# swarm_lrwavg(combarray_ei_LR_P2M_weights,['1$^\circ$ L Exc','1$^\circ$ R Exc','1$^\circ$ L Inh','1$^\circ$ R Inh'],plotti='Primary PMN LR Preference by NT Type',saveti='lrpref_prim_evsi_')

# #%% stats on LRwavg E vs. I by side - secondary
# combarray_ei_LR_P2P_weights = np.concatenate([np.array(e_comb_LR_PtoP_weights),np.hstack([np.array(i_comb_LR_PtoP_weights),np.nan*np.ones([2,(e_comb_LR_PtoP_weights[0].shape[0]-i_comb_LR_PtoP_weights[0].shape[0])])])]).T
# stats_ei_lr_sec, bonf_ei_lr_sec = mwu_grps(combarray_ei_LR_P2P_weights)

# #plot as swarmplot for stat comp vis
# swarm_lrwavg(combarray_ei_LR_P2P_weights,['2$^\circ$ L Exc','2$^\circ$ R Exc','2$^\circ$ L Inh','2$^\circ$ R Inh'],plotti='Secondary PMN LR Preference by NT Type',saveti='lrpref_sec_evsi_')

# #%% stats on LRwavg E vs. I by side - primary and secondary
# combarray_ei_LR_P2MandP_weights = np.hstack([combarray_ei_LR_P2M_weights,combarray_ei_LR_P2P_weights])
# stats_ei_lr_primandsec, bonf_ei_lr_sec_primandsec = mwu_grps(combarray_ei_LR_P2MandP_weights)

# #plot as swarmplot for stat comp vis
# swarm_lrwavg(combarray_ei_LR_P2MandP_weights,['1$^\circ$ L E','1$^\circ$ R E','1$^\circ$ L I','1$^\circ$ R I','2$^\circ$ L E','2$^\circ$ R E','2$^\circ$ L I','2$^\circ$ R I'],plotti='Primary and Secondary PMN LR Preference by NT Type',saveti='lrpref_primandsec_evsi_')

# #%% fxn for sep groups of primary PMNs so that can have groups for investigating hypotheses on ipsi, contra, bi at primary and secondary levels
# def sort_icb(sortindslist,wavgmatin,lp_thresh,rp_thresh):
#     left_pref = list()
#     ipsiL_pref = list()
#     contraL_pref = list()
#     right_pref = list()
#     ipsiR_pref = list()
#     contraR_pref = list()
#     bi_pref = list()
    
#     lp_thresh = 0.2
#     rp_thresh = 0.8
    
#     for s in np.arange(len(sortindslist)):
#         for i,v in enumerate(wavgmatin[s]):
#             if v < lp_thresh:
#                 left_pref.append(sortindslist[s][i])
#                 if s == 0:
#                     ipsiL_pref.append(sortindslist[s][i])
#                 elif s == 1:
#                     contraL_pref.append(sortindslist[s][i])
#             elif v >= rp_thresh:
#                 right_pref.append(sortindslist[s][i])
#                 if s == 1:
#                     ipsiR_pref.append(sortindslist[s][i])
#                 elif s == 0:
#                     contraR_pref.append(sortindslist[s][i])
#             elif v > lp_thresh and v < rp_thresh:
#                 bi_pref.append(sortindslist[s][i])

# #%%divide out the primary PMNs that have preference to L and R by weights in comb_LR_PtoM_weights
# xx = sort_icb(sortp2m,comb_LR_PtoM_weights,0.2,0.8)

# ##stopped here, figure out why sorting not working

# #also quickly move onto the matrix multiplication just to check how much influence the left or right secondary PMNs have on each side of MNs
# #ALSO - do this influence mat on the seconary PMNs but wrt the fxnal muscle groups -- could allow a hypoth test of whether E-E cnxns at sec level sync up the LTs 
# #for crawling and for rolling! -- thnk more about the muscle data on this too- could it be that the LTs are active in bend and stretch during roll?



#####
#E PtoM, I PtoM combined by LR wavg of MNs


#E PtoP, I PtoP combined by LR wavg onto anatomical side PMNs


#COMPARE WITH E PtoP, I PtoP combined by LR wavg onto FXNAL PREF of PMNs (see thresh for categorization below)



####STOPPED here --

#finish making these plots nice - the heatmap and histograms

#then, with this sorted PMN order for L and R proj's to MNs,
#assign a new score at 0.5 threshold
#also consider score at 0.2 and 0.8 to cut bilat neurons
#then do another option of L-PMNs to PMNs in the >0.2, < 0.8 range
#for 0 L-preferring PMNs, 0 R-preferring PMNs

#and run same wavg process from PMN onto PMNs w/ these sorted groups


#to plots and values, can add thresholds for ipsi, contra, bi
#ALSO -- go back and add list of pnames TO the wavg and sort code so that the list spits out which is contra, ipsi, bi







#%% pull out E vs. I, unilat vs. bilat, ipsi vs. contra
#do for primary, secondary 
#make list w/ all variables included - pd.df -- so can check with previously published indiv neurons

#%% more general metrics to plot -- 
#do counts + plot proportions -- answers question of what is most probable
#on primary PMNs
#1. E vs. I (at least w/ known NT ID)
#2. unilat vs. bilat
#3. ipsi vs. contra

#4. E vs. I, unilat vs. bilat
#5. E vs. I, ipsi vs. contra


#on secondary PMNs
#1. E vs. I onto list from #4 - E,I unilat,bilat
#2. E vs. I onto list from #5 - E,I ipsi, contra


### SEE FIGURE 5 IN WINDING ET AL -- 
#make heatmap colored by ipsi and contra?
#do histogram showing - number of neuron pairs by fraction of contralateral presynaptic sites
#syn site distrib -- make images from pymaid volume generation
#also D - the ipsi and contra partner comparison of bilateral neurons -- cosine similarity as swarmplot and groupings by cosine similarity thresholds
#reciprocal loop check - at primary and secondary levels

#see also again figure 7 - brain vnc interactions... how did they generate the brain vs. behavior projection matrix?