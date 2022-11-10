#!/usr/bin/python


from sklearn import linear_model
import sklearn.linear_model as _skl
import numpy as _N
import RPSvAI.utils.read_taisen as _rt
import scipy.io as _scio
import scipy.stats as _ss
import matplotlib.pyplot as _plt
import RPSvAI.utils.read_taisen as _rd
import RPSvAI.utils.misc as _Am
from scipy.signal import savgol_filter
from GCoh.eeg_util import unique_in_order_of_appearance, increasing_labels_mapping, rmpd_lab_trnsfrm, find_or_retrieve_GMM_labels, shift_correlated_shuffle, shuffle_discrete_contiguous_regions, mtfftc
from RPSvAI.utils.dir_util import workdirFN, datadirFN
import os
import sys
from sumojam.devscripts.cmdlineargs import process_keyval_args
import pickle
import mne.time_frequency as mtf
import GCoh.eeg_util as _eu
#import RPSvAI.rpsms as rpsms
import GCoh.preprocess_ver as _ppv

import RPSvAI.constants as _cnst
#from RPSvAI.utils.dir_util import getResultFN
import GCoh.datconfig as datconf
import RPSvAI.models.CRutils as _crut
import RPSvAI.models.empirical_ken as _emp
from sklearn.decomposition import PCA
import RPSvAI.AIRPSfeatures as _aift

import GCoh.eeg_util as _eu
import matplotlib.ticker as ticker
from statsmodels import robust

__DSUWTL__ = 0
__RPSWTL__ = 1
__DSURPS__ = 2
__ALL__    = 3

mode       = __ALL__
#mode       = __DSUWTL__
#mode       = __RPSWTL__
#mode       = __DSURPS__
_plt.ioff()
__1st__ = 0
__2nd__ = 1

_ME_WTL = 0
_ME_RPS = 1

_SHFL_KEEP_CONT  = 0
_SHFL_NO_KEEP_CONT  = 1

#  sum_sd
#  entropyL
#  isi_cv, isis_corr

def rm_outliersCC_neighbors(x, y):
    ix = x.argsort()
    iy = y.argsort()
    dsx = _N.mean(_N.diff(_N.sort(x)))
    dsy = _N.mean(_N.diff(_N.sort(y)))

    L = len(x)
    x_std = _N.std(x)
    y_std = _N.std(y)
    rmv   = []
    i = 0
    while x[ix[i+1]] - x[ix[i]] > 2.5*dsx:
        rmv.append(ix[i])
        i+= 1
    i = 0
    while x[ix[L-1-i]] - x[ix[L-1-i-1]] > 2.5*dsx:
        rmv.append(ix[L-1-i])
        i+= 1
    i = 0
    while y[iy[i+1]] - y[iy[i]] > 2.5*dsy:
        rmv.append(iy[i])
        i+= 1
    i = 0
    while y[iy[L-1-i]] - y[iy[L-1-i-1]] > 2.5*dsy:
        rmv.append(iy[L-1-i])
        i+= 1
        
    ths = _N.array(rmv)
    ths_unq = _N.unique(ths)
    interiorPts = _N.setdiff1d(_N.arange(len(x)), ths_unq)
    #print("%(ths)d" % {"ths" : len(ths)})
    return _ss.pearsonr(x[interiorPts], y[interiorPts])

def cleanISI(isi, minISI=2):
    #print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    ths = _N.where(isi[1:-1] <= minISI)[0] + 1
    #print(len(ths))
    if len(ths) > 0:
        rebuild = isi.tolist()
        for ih in ths:
            rebuild[ih-1] += minISI//2
            rebuild[ih+1] += minISI//2
        for ih in ths[::-1]:
            rebuild.pop(ih)
        isi = _N.array(rebuild)
    return isi

def rebuild_sds_array(nPartIDs, lm, name):
    arr = _N.empty((nPartIDs, 3, 3))
    for ic in range(3):
        for ia in range(3):        
            cname = "%(n)s_%(c)d%(a)d" % {"n" : name, "c" : ic, "a" : ia}
            arr[:, ic, ia] = lm[cname]
    return arr
    
def secs_as_string(sec0, sec1):
    l = []
    for s in range(sec0, sec1):
        if s < 10:
            l.append("0%d" % s)
        else:
            l.append("%d"  % s)
    return l

def show(p1, p2, shf):
    fig = _plt.figure()
    avg = _N.mean(rc_trg_avg[p1:p2, :, shf], axis=0)
    for ipid in range(p1, p2):
        _plt.plot(rc_trg_avg[ipid, :, shf], color="grey")
    _plt.plot(avg, color="black")        

def only_complete_data(partIDs, TO, label, SHF_NUM):
    pid = -1
    incomplete_data = []
    for partID in partIDs:
        pid += 1

        dmp       = depickle(workdirFN("%(rpsm)s/%(lb)d/variousCRs_%(visit)d.dmp" % {"rpsm" : partID, "lb" : label, "visit" : visit}))
        _prob_mvsDSUWTL = dmp["cond_probsDSUWTL"][SHF_NUM]
        _prob_mvsRPSWTL = dmp["cond_probsRPSWTL"][SHF_NUM]
        #_prob_mvsDSURPS = dmp["cond_probsDSURPS"][SHF_NUM]                
        __hnd_dat = dmp["all_tds"][SHF_NUM]
        _hnd_dat   = __hnd_dat[0:TO]

        if _hnd_dat.shape[0] < TO:
            incomplete_data.append(pid)
    for inc in incomplete_data[::-1]:
        #  remove from list 
        partIDs.pop(inc)
    return partIDs, incomplete_data

def depickle(s):
     import pickle
     with open(s, "rb") as f:
          lm = pickle.load(f)
     return lm
##  Then I expect wins following UPs and DOWNs to also be correlated to AQ28
look_at_AQ = True

biggest=True
top_comps=9
thrI = 1
nI=1
r1=0.4

#_plt.ioff()
process_keyval_args(globals(), sys.argv[1:])   #  For when we run from cmd line

#visit = 2
#visits= [1, 2]   #  if I want 1 of [1, 2], set this one to [1, 2]
visit = 1
visits= [1, ]   #  if I want 1 of [1, 2], set this one to [1, 2]
    
# if data == "TMB2":
#     dates = _rt.date_range(start='7/13/2021', end='12/30/2021')
#     partIDs, dats, cnstrs = _rt.filterRPSdats(data, dates, visits=visits, domainQ=(_rt._TRUE_ONLY_ if look_at_AQ else _rt._TRUE_AND_FALSE_), demographic=_rt._TRUE_AND_FALSE_, mentalState=_rt._TRUE_AND_FALSE_, min_meanIGI=500, max_meanIGI=15000, minIGI=20, maxIGI=30000, MinWinLossRat=0.35, has_useragent=True, has_start_and_end_times=True, has_constructor=True, blocks=1)
#     #partIDs, dats, cnstrs = _rt.filterRPSdats(data, dates, visits=[1], domainQ=(_rt._TRUE_ONLY_ if look_at_AQ else _rt._TRUE_AND_FALSE_), demographic=_rt._TRUE_AND_FALSE_, mentalState=_rt._TRUE_AND_FALSE_, min_meanIGI=800, max_meanIGI=8000, minIGI=200, maxIGI=30000, MinWinLossRat=0.4, has_useragent=True, has_start_and_end_times=True, has_constructor=True, blocks=1)
#     ####  use this for reliability
#     #partIDs, dats, cnstrs = _rt.filterRPSdats(data, dates, visits=visits, domainQ=(_rt._TRUE_AND_FALSE_ if look_at_AQ else _rt._TRUE_AND_FALSE_), demographic=_rt._TRUE_AND_FALSE_, mentalState=_rt._TRUE_AND_FALSE_, min_meanIGI=500, max_meanIGI=8000, minIGI=50, maxIGI=30000, MinWinLossRat=0.4, has_useragent=True, has_start_and_end_times=True, has_constructor=True, blocks=1)

A1 = []
show_shuffled = False
process_keyval_args(globals(), sys.argv[1:])
#######################################################

win_type = 2   #  window is of fixed number of games
#win_type = 1  #  window is of fixed number of games that meet condition 
win     = 3
smth    = 1
label          = win_type*100+win*10+smth
TO = 300
SHF_NUM = 0

expt = "TMB2"
if expt == "TMB2":
    lm = depickle(workdirFN("AQ28_vs_RPS_%(v)d_%(wt)d%(w)d%(s)d.dmp" % {"v" : visit, "wt" : win_type, "w" : win, "s" : smth, "wd" : os.environ["RPSWORKDIR"]}))
    #lm = depickle("predictAQ28dat/AQ28_vs_RPS_1_%(wt)d%(w)d%(s)d.dmp" % {"wt" : win_type, "w" : win, "s" : smth})
    partIDs = lm["partIDs"]
    TO = 300
elif expt == "EEG1":
    #partIDs = ["20200109_1504-32"]
    #partIDs = ["20210606_1237-17", "20210609_1230-28", "20210609_1248-16", "20210609_1321-35", "20210609_1517-23", "20210609_1747-07"]
    partIDs = ["20210606_1237-17", "20210609_1230-28", "20210609_1248-16", "20210609_1321-35", "20210609_1517-23", "20210609_1747-07", "20210526_1318-12", "20210526_1358-27", "20210526_1416-25", "20210526_1503-39"]
elif expt == "SIMHUM1":
    partIDs = []
    for sec in secs_as_string(0, 60):
        partIDs.append("20110101_0000-%s" % sec)
        lm = {}
        lm["filtdat"] = _N.arange(60)
        TO = 1000

filtdat = lm["filtdat"]


#filtdat = _N.array([8])
    
partIDs, incmp_dat = only_complete_data(partIDs, TO, label, SHF_NUM)
strtTr=0
TO -= strtTr

#fig= _plt.figure(figsize=(14, 14))

SHUFFLES = 205
extra_w = 2
t0 = -5 - extra_w
t1 = 10 + extra_w
cut = 3
all_avgs = _N.empty((len(partIDs), SHUFFLES+1, t1-t0))
l_all_avgs = []
netwins  = _N.empty(len(partIDs), dtype=_N.int)
gk = _Am.gauKer(1)
gk /= _N.sum(gk)
#gk = None

pid = 0

ts  = _N.arange(t0-2, t1-2)
signal_5_95 = _N.zeros((len(partIDs), 4, t1-t0))

hnd_dat_all = _N.zeros((len(partIDs), TO, 4), dtype=_N.int)

pfrm_change69 = _N.zeros(len(partIDs))
jump          = _N.zeros(len(partIDs))
m1s          = _N.zeros(len(partIDs))
m2s          = _N.zeros(len(partIDs))
c1s          = _N.zeros(len(partIDs))
c2s          = _N.zeros(len(partIDs))
weights      = _N.zeros(len(partIDs))

isis    = _N.empty(len(partIDs))
isis_sd    = _N.empty(len(partIDs))
isis_cv    = _N.empty(len(partIDs))
isis_kur    = _N.empty(len(partIDs))
iwis_cv    = _N.empty(len(partIDs))
itis_cv    = _N.empty(len(partIDs))
ilis_cv    = _N.empty(len(partIDs))
isis_lv    = _N.empty(len(partIDs))
isis_corr    = _N.empty(len(partIDs))

all_maxs  = []

aboves = []
belows = []

AQ28scrs  = _N.empty(len(partIDs))
soc_skils = _N.empty(len(partIDs))
rout      = _N.empty(len(partIDs))
switch    = _N.empty(len(partIDs))
imag      = _N.empty(len(partIDs))
fact_pat  = _N.empty(len(partIDs))

all_prob_mvs = []
all_prob_pcs = []
istrtend     = 0
strtend      = _N.zeros(len(partIDs)+1, dtype=_N.int)

incomplete_data = []
gkISI = _Am.gauKer(1)
gkISI /= _N.sum(gkISI)

#  DISPLAYED AS R,S,P
#  look for RR RS RP
#  look for SR SS SP
#  look for PR PS PP

L30  = 30

rc_trg_avg = _N.empty((len(partIDs), t1-t0, SHUFFLES+1))
rc_trg_avg_RPS = _N.empty((len(partIDs), t1-t0, SHUFFLES+1))
#rc_trg_avg_DSURPS = _N.empty((len(partIDs), t1-t0, SHUFFLES+1))

chg = _N.empty(len(partIDs))

n_maxes   = _N.zeros((len(partIDs), SHUFFLES+1), dtype=_N.int)

# mdl, SHUFFLES, cond, act
stds        = _N.zeros((len(partIDs), 3, SHUFFLES+1, 3, 3, ))
# mdl, 1st hlf, 2nd hlf, SHUFFLES cond, act
stds12      = _N.zeros((len(partIDs), 3, 2, SHUFFLES+1, 3, 3))

thrs = _N.empty(len(partIDs), dtype=_N.int)
#stds      = _N.zeros((len(partIDs), 3, SHUFFLES+1))
#stdsDSUWTL      = _N.zeros((len(partIDs), 3, 3, 3, SHUFFLES+1))
#stdsRPSWTL      = _N.zeros((len(partIDs), 3, 3, 3, SHUFFLES+1))
#stdsDSURPS      = _N.zeros((len(partIDs), 3, 3, 3, SHUFFLES+1))

marginalCRs = _N.empty((len(partIDs), SHUFFLES, 3, 3))

sum_sd_DSUWTL = rebuild_sds_array(len(partIDs), lm, "sum_sd_DSUWTL")
sum_sd_RPSWTL = rebuild_sds_array(len(partIDs), lm, "sum_sd_RPSWTL")
sum_sd_DSUAIRPS = rebuild_sds_array(len(partIDs), lm, "sum_sd_DSUAIRPS")

frameworks = ["DSUWTL", "RPSWTL", "DSUAIRPS", "RPSAIRPS", "RPSRPS"]
frameworks_p = ["p(DSU | WTL)", "p(RPS | WTL)", "p(DSU | AI_RPS)", "p(RPS | AI_RPS)", "p(RPS | RPS)"]

#frameworks = ["DSUWTL", "DSUAIRPS", "RPSAIRPS", "RPSRPS"]
#frameworks = ["DSUWTL", "RPSRPS"]

caksDSUWTL = _N.empty(len(partIDs))
caksRPSWTL = _N.empty(len(partIDs))
caksRPSRPS = _N.empty(len(partIDs))
caksRPSAIRPS = _N.empty(len(partIDs))
caksDSUAIRPS = _N.empty(len(partIDs))
par        = _N.empty((len(partIDs), 5))
pick_acts        = _N.empty((len(partIDs), 5))eworks
avoid_acts        = _N.empty((len(partIDs), 5))

z1s        = _N.empty((len(partIDs), 5))
z0s        = _N.empty((len(partIDs), 5))
rank       = _N.empty((len(partIDs), 5), dtype=_N.int)
cmp_z1s    = _N.empty((len(partIDs), 5, 3, 3))
cmp_z1sZ   = _N.empty((len(partIDs), 5, 3, 3))
fr_cmp_rank1       = _N.empty((len(partIDs), 5, 3, 3), dtype=_N.int)
fr_cmp_rank0       = _N.empty((len(partIDs), 5, 3, 3), dtype=_N.int)
fr_cmp_lohi_rank       = _N.zeros((len(partIDs), 5, 3, 3), dtype=_N.int)

CCs        = _N.empty((len(partIDs), 5, 5))

hi_allpcs     = []
mid_allpcs     = []
lo_allpcs     = []
iex = 1
for partID in partIDs:
    pid += 1
    print(pid)
    dmp       = depickle(workdirFN("%(rpsm)s/%(lb)d/variousCRs_%(v)d.dmp" % {"rpsm" : partID, "lb" : label, "v" : visit}))

    if expt == "TMB2":
        AQ28scrs[pid-1], soc_skils[pid-1], rout[pid-1], switch[pid-1], imag[pid-1], fact_pat[pid-1] = _rt.AQ28(datadirFN("%(data)s/%(date)s/%(pID)s/AQ29.txt" % {"date" : partIDs[pid-1][0:8], "pID" : partIDs[pid-1], "data" : expt}))

    _prob_mvsDSUWTL = dmp["cond_probsDSUWTL"][:, :, strtTr:]
    _prob_mvsRPSWTL = dmp["cond_probsRPSWTL"][:, strtTr:]
    #_prob_mvsDSURPS = dmp["cond_probsDSURPS"][:, strtTr:]
    _prob_mvsDSUAIRPS = dmp["cond_probsDSUAIRPS"][:, strtTr:]
    _prob_mvsRPSRPS = dmp["cond_probsRPSRPS"][:, strtTr:]
    _prob_mvsRPSAIRPS = dmp["cond_probsRPSAIRPS"][:, strtTr:]        
    prob_mvsDSUWTL  = _prob_mvsDSUWTL[:, :, 0:TO - win]  #  is bigger than hand by win size
    prob_mvsRPSWTL  = _prob_mvsRPSWTL[:, :, 0:TO - win]  #  is bigger than hand by win size
    #prob_mvsDSURPS  = _prob_mvsDSURPS[:, :, 0:TO - win]  #  is bigger than hand by win size
    prob_mvsDSUAIRPS  = _prob_mvsDSUAIRPS[:, :, 0:TO - win]  #  is bigger than hand by win size    
    prob_mvsRPSRPS  = _prob_mvsRPSRPS[:, :, 0:TO - win]  #  is bigger than hand by win size
    prob_mvsRPSAIRPS  = _prob_mvsRPSAIRPS[:, :, 0:TO - win]  #  is bigger than hand by win size    

    #stds_all_mdls[0] = _N.std(prob_mvs, axis=2)

    N = 300 - win
    near1DSUWTL = _N.zeros((SHUFFLES+1, N), dtype=_N.int)
    near1RPSWTL = _N.zeros((SHUFFLES+1, N), dtype=_N.int)
    near1RPSAIRPS = _N.zeros((SHUFFLES+1, N), dtype=_N.int)
    near1RPSRPS = _N.zeros((SHUFFLES+1, N), dtype=_N.int)
    near1DSUAIRPS = _N.zeros((SHUFFLES+1, N), dtype=_N.int)
    near0DSUWTL = _N.zeros((SHUFFLES+1, N), dtype=_N.int)
    near0RPSWTL = _N.zeros((SHUFFLES+1, N), dtype=_N.int)
    near0RPSAIRPS = _N.zeros((SHUFFLES+1, N), dtype=_N.int)
    near0RPSRPS = _N.zeros((SHUFFLES+1, N), dtype=_N.int)
    near0DSUAIRPS = _N.zeros((SHUFFLES+1, N), dtype=_N.int)
    
    # for SHF_NUM in range(0, SHUFFLES+1):
    #     _prob_mvsDSUWTL = dmp["cond_probsDSUWTL"][SHF_NUM][:, strtTr:]
    #     _prob_mvsRPSWTL = dmp["cond_probsRPSWTL"][SHF_NUM][:, strtTr:]
    #     _prob_mvsDSURPS = dmp["cond_probsDSURPS"][SHF_NUM][:, strtTr:]
    #     _prob_mvsDSUAIRPS = dmp["cond_probsDSUAIRPS"][SHF_NUM][:, strtTr:]
    #     _prob_mvsRPSRPS = dmp["cond_probsRPSRPS"][SHF_NUM][:, strtTr:]
    #     _prob_mvsRPSAIRPS = dmp["cond_probsRPSAIRPS"][SHF_NUM][:, strtTr:]

    #     #_prob_mvs_STSW = dmp["cond_probsSTSW"][SHF_NUM][:, strtTr:]    
    #     _hnd_dat = dmp["all_tds"][SHF_NUM][strtTr:]
        
    #     #end_strts[pid-1] = _N.mean(_hnd_dat[-1, 3] - _hnd_dat[0, 3])
    #     hdcol = 0

    #     inds =_N.arange(_hnd_dat.shape[0])
    #     hnd_dat_all[pid-1] = _hnd_dat[0:TO]

    #     cv_sum = 0
    #     dhd = _N.empty(TO)
    #     dhd[0:TO-1] = _N.diff(_hnd_dat[0:TO, 3])
    #     dhd[TO-1] = dhd[TO-2]
    #     #dhdr = dhd.reshape((20, 15))
    #     #rsp_tms_cv[pid-1] = _N.mean(_N.std(dhdr, axis=1) / _N.mean(dhdr, axis=1))

    #     #rsp_tms_cv[pid-1] = _N.std(_hnd_dat[:, 3]) / _N.mean(_hnd_dat[:, 3])
    #     #marginalCRs[pid-1] = _emp.marginalCR(_hnd_dat)
    #     prob_mvsDSUWTL  = _prob_mvsDSUWTL[:, 0:TO - win]  #  is bigger than hand by win size
    #     prob_mvsRPSWTL  = _prob_mvsRPSWTL[:, 0:TO - win]  #  is bigger than hand by win size
    #     prob_mvsDSURPS  = _prob_mvsDSURPS[:, 0:TO - win]  #  is bigger than hand by win size
    #     prob_mvsDSUAIRPS  = _prob_mvsDSUAIRPS[:, 0:TO - win]  #  is bigger than hand by win size
    #     prob_mvsRPSRPS  = _prob_mvsRPSRPS[:, 0:TO - win]  #  is bigger than hand by win size
    #     prob_mvsRPSAIRPS  = _prob_mvsRPSAIRPS[:, 0:TO - win]  #  is bigger than hand by win size                        
    #     #prob_mvs_STSW  = _prob_mvs_STSW[:, 0:TO - win]  #  is bigger than hand by win size    
    #     prob_mvsDSUWTL = prob_mvsDSUWTL.reshape((3, 3, prob_mvsDSUWTL.shape[1]))
    #     prob_mvsRPSWTL = prob_mvsRPSWTL.reshape((3, 3, prob_mvsRPSWTL.shape[1]))
    #     prob_mvsDSURPS = prob_mvsDSURPS.reshape((3, 3, prob_mvsDSURPS.shape[1]))
    #     prob_mvsDSUAIRPS = prob_mvsDSUAIRPS.reshape((3, 3, prob_mvsDSUAIRPS.shape[1]))        
    #     prob_mvsRPSRPS = prob_mvsRPSRPS.reshape((3, 3, prob_mvsRPSRPS.shape[1]))
    #     prob_mvsRPSAIRPS = prob_mvsRPSAIRPS.reshape((3, 3, prob_mvsRPSAIRPS.shape[1]))        

    #     ##  what we really mean here is that for each condition, 
    #     ##  
    #     thr1 = 0.6
    #     thr2 = 0.1

    #     ifr = -1
    #     for frmwk in frameworks:
    #         ifr += 1
    #         exec("prob_mvs = prob_mvs%s" % frmwk)
    #         exec("near1 = near1%s" % frmwk)
    #         exec("near0 = near0%s" % frmwk)

    #         for game in range(prob_mvs.shape[2]):
    #             conditional_action_known = 0
    #             conditional_action_not_taken = 0
    #             for ic in range(3):
    #                 knowNextMove = _N.where(prob_mvs[ic, :, game] > thr1)[0]
    #                 if len(knowNextMove) > 0:
    #                     conditional_action_known += 1
    #                     near1[SHF_NUM, game] = conditional_action_known
    #             for ic in range(3):
    #                 notNextMove = _N.where(prob_mvs[ic, :, game] < thr2)[0]
    #                 if len(notNextMove) == 1:
    #                     conditional_action_not_taken += 1
    #                     near0[SHF_NUM, game] = conditional_action_not_taken


    # # frames = ["DSUWTL", "RPSWTL", "RPSRPS", "DSUAIRPS", "RPSAIRPS"]
    # # for if1 in range(5):
    # #     exec("near1_1 = near1%s[0]" % frames[if1])
    # #     for if2 in range(if1+1, 5):
    # #         exec("near1_2 = near1%s[0]" % frames[if2])
    # #         pc, pv = _ss.pearsonr(near1_1, near1_2)
    # #         print("%(f1)s  %(f2)s    %(pc).3e" % {"f1" : frames[if1], "f2" : frames[if2], "pc" : pc})
    # #         CCs[pid-1, if1, if2] = pc
    # #         CCs[pid-1, if2, if1] = pc

    # #fig = _plt.figure(figsize=(8, 8))
    # ifr = -1
    # bins=_N.linspace(0, 3, 31)
    # for frmwk in frameworks:
    #     ifr += 1
    #     #fig.add_subplot(4, 3, ifr+1)
    #     exec("near1 = near1%s" % frmwk)
    #     sums = _N.mean(near1, axis=1)
    #     #_plt.title(frmwk)
    #     #_plt.hist(sums[1:], bins=bins)
    #     #_plt.axvline(x=sums[0])
    #     z1s[pid-1, ifr] = (sums[0] - _N.mean(sums[1:])) / _N.std(sums[1:])
    #     srtd = _N.sort(sums[1:])
    #     rank[pid-1, ifr] = len(_N.where(sums[0] > srtd)[0])
    # ifr = -1
    # for frmwk in frameworks:
    #     ifr += 1
    #     #fig.add_subplot(4, 3, 6+ifr+1)
    #     exec("near0 = near0%s" % frmwk)
    #     sums = _N.mean(near0, axis=1)
    #     sin = "**" if len(_N.where(filtdat == pid-1)[0]) > 0 else "  "
    #     #_plt.title("%(f)s   %(p)s" % {"f" : frmwk, "p" : sin})
    #     #_plt.hist(sums[1:], bins=bins)
    #     #_plt.axvline(x=sums[0])
    #     z0s[pid-1, ifr] = (sums[0] - _N.mean(sums[1:])) / _N.std(sums[1:])
    # #_plt.savefig("cmp_frameworks%d" % (pid-1))
    # #_plt.close()

    allDSUWTLs = dmp["cond_probsDSUWTL"].reshape((SHUFFLES+1, 3, 3, 300-win))
    allRPSWTLs = dmp["cond_probsRPSWTL"].reshape((SHUFFLES+1, 3, 3, 300-win))
    allRPSRPSs = dmp["cond_probsRPSRPS"].reshape((SHUFFLES+1, 3, 3, 300-win))
    allRPSAIRPSs = dmp["cond_probsRPSAIRPS"].reshape((SHUFFLES+1, 3, 3, 300-win))
    allDSUAIRPSs = dmp["cond_probsDSUAIRPS"].reshape((SHUFFLES+1, 3, 3, 300-win))

    ifr = -1
    print("----------------    %d" % pid)

    hi_rnk_cmpts = [[], [], [], [], []]
    mid_rnk_cmpts = [[], [], [], [], []]
    lo_rnk_cmpts = [[], [], [], [], []]

    for frmwk in frameworks:
        ifr += 1
        exec("alls = dmp['cond_probs%(fr)s'].reshape((%(shfp1)d, 3, 3, %(N)d))" % {"fr" : frmwk, "shfp1" : (SHUFFLES+1), "N" : (300-win)})
        print(frmwk)

        for ic in range(3):
            for ia in range(3):
                stds = _N.std(alls[:, ic, ia], axis=1)
                #stds = robust.mad(alls[:, ic, ia], axis=1)

                #cmp_z1s[pid-1, ifr, ic, ia] = (stds[0] - _N.mean(stds[1:])) / _N.std(stds[1:])
                #cmp_z1sZ[pid-1, ifr, ic, ia] = (stds[0] - _N.median(stds[1:])) / _N.std(stds[1:])
                cmp_z1sZ[pid-1, ifr, ic, ia] = stds[0] / _N.median(stds[1:])
                cmp_z1s[pid-1, ifr, ic, ia] = stds[0]
                rnk1 = len(_N.where(stds[0] > stds[1:])[0])
                #rnk1 = len(_N.where(stds[1] > stds[2:])[0])
                fr_cmp_rank1[pid-1, ifr, ic, ia] = rnk1
                print("%(c)d %(a)d   %(r)d" % {"c" : ic, "a" : ia, "r" : rnk1})
                if rnk1 / SHUFFLES < 0.3:
                    lo_rnk_cmpts[ifr].append(_N.array(alls[0, ic, ia]))
                elif rnk1 / SHUFFLES > 0.9:
                    hi_rnk_cmpts[ifr].append(_N.array(alls[0, ic, ia]))
                elif (rnk1 / SHUFFLES) > 0.4 and (rnk1 / SHUFFLES) < 0.85:
                    mid_rnk_cmpts[ifr].append(_N.array(alls[0, ic, ia]))

                # if rnk1/SHUFFLES > 0.995:
                #     fig = _plt.figure(figsize=(13, 2.5))
                #     for i in range(1, SHUFFLES, 20):
                #         _plt.plot(alls[i, ic, ia], color="grey")
                #     _plt.plot(alls[0, ic, ia], color="black", lw=3)
                #     _plt.suptitle("%(pid)s  fr %(fr)s %(c)d %(a)d" % {"pid" : partID, "fr" : frmwk, "c" : ic, "a" : ia})
                #     _plt.savefig("example%d.png" % iex)
                #     _plt.close()
                #     iex += 1
v                    

    ifr1 = -1
    for frmwk1 in frameworks:
        ifr1 += 1
        lo_ifLen1  = len(lo_rnk_cmpts[ifr1])
        mid_ifLen1  = len(mid_rnk_cmpts[ifr1])
        hi_ifLen1  = len(hi_rnk_cmpts[ifr1])
        ifr2 = -1
        for frmwk2 in frameworks:
            pcs_4_frmwk_pair = []
            ifr2 += 1
            lo_ifLen2  = len(lo_rnk_cmpts[ifr2])   # number of cmpts
            mid_ifLen2  = len(mid_rnk_cmpts[ifr2])   # number of cmpts
            hi_ifLen2  = len(hi_rnk_cmpts[ifr2])   # number of cmpts
            if ifr1 < ifr2:
                for ifi1 in range(hi_ifLen1):
                    for ifi2 in range(hi_ifLen2):
                        pc, pv = _ss.pearsonr(hi_rnk_cmpts[ifr1][ifi1], hi_rnk_cmpts[ifr2][ifi2])
                        hi_allpcs.append(pc)
                for ifi1 in range(lo_ifLen1):
                    for ifi2 in range(lo_ifLen2):
                        pc, pv = _ss.pearsonr(lo_rnk_cmpts[ifr1][ifi1], lo_rnk_cmpts[ifr2][ifi2])
                        lo_allpcs.append(pc)
                for ifi1 in range(mid_ifLen1):
                    for ifi2 in range(mid_ifLen2):
                        pc, pv = _ss.pearsonr(mid_rnk_cmpts[ifr1][ifi1], mid_rnk_cmpts[ifr2][ifi2])
                        mid_allpcs.append(pc)

            
            #print(_N.mean(pcs_4_frmwk_pair))
                    
            
        

        # for ic in range(3):
        #     for ia in range(3):
        #         lens1 = _N.zeros(SHUFFLES+1, dtype=_N.int)
        #         lens0 = _N.zeros(SHUFFLES+1, dtype=_N.int)
        #         for shf in range(SHUFFLES+1):
        #             lens1[shf] = len(_N.where(alls[shf, ic, ia] > thr1)[0])
        #             lens0[shf] = len(_N.where(alls[shf, ic, ia] < thr2)[0])
        #         srtd1 = _N.sort(lens1[1:])
        #         srtd0 = _N.sort(lens0[1:])
        #         rnk1 = len(_N.where(lens1[0] > srtd1)[0])
        #         rnk0 = len(_N.where(lens0[0] > srtd0)[0])
        #         fr_cmp_rank1[pid-1, ifr, ic, ia]       = rnk1
        #         fr_cmp_rank0[pid-1, ifr, ic, ia]       = rnk0
        # conds1, acts1 = _N.where(fr_cmp_rank1[pid-1, ifr] > 90)
        # conds0, acts0 = _N.where(fr_cmp_rank0[pid-1, ifr] > 90)

        # ca1     = _N.empty((conds1.shape[0], 2), dtype=_N.int)
        # ca1[:, 0] = conds1
        # ca1[:, 1] = acts1
        # ca0     = _N.empty((conds0.shape[0], 2), dtype=_N.int)
        # ca0[:, 0] = conds0
        # ca0[:, 1] = acts0
        # for i1 in range(ca1.shape[0]):
        #     for i0 in range(ca0.shape[0]):
        #         if (ca1[i1, 0] == ca0[i0, 0]) and (ca1[i1, 1] == ca0[i0, 1]):
        #             #print(ca1[i1])
        #             #print(ca0[i0])
        #             #both_big_swing.append(ca1[i1])
        #             fr_cmp_lohi_rank[pid-1, ifr, ca1[i1, 0], ca1[i1, 1]] = 1
        
        #             # if conds1 = [0, 0, 2, 2], acts1 = [0, 1, 0, 1]
        #             # and
        #             #    conds2 = [0, 0, 2, 2], acts1 = [0, 2, 0, 2]
        #             #  Then [0, 0] and [2, 0] are the components where both near cond prob 0 and cond prob 1 occur 
        


dmpout = open("out", "wb")
pickle.dump({"z1s" : z1s, "rank" : rank, "filtdat" : filtdat}, dmpout, -1)
dmpout.close()

#SHUFFLES = SHUFFLES-1


for i in range(filtdat.shape[0]):
    _plt.plot(fr_cmp_rank1[filtdat[i], 0, 2], color="black")


for ic in range(3):
    ones_kys = {}
    for i in range(filtdat.shape[0]):
        key = str(_N.where(fr_cmp_rank1[filtdat[i], 0, ic] > int(0.98*SHUFFLES))[0])
        try:
            ones_kys[key] += 1
        except KeyError:
            ones_kys[key] =  1
    print(ones_kys)


for ic in range(3):
    ones_kys = {}
    for i in range(filtdat.shape[0]):
        key = str(_N.where(fr_cmp_rank1[filtdat[i], 0, ic] > int(0.98*SHUFFLES))[0])
        try:
            ones_kys[key] += 1
        except KeyError:
            ones_kys[key] =  1
    print(ones_kys)


#  how many cond_prob components have big amplitude
fig = _plt.figure(figsize=(10, 6.5))
_plt.suptitle("# of big fluctuation components", fontsize=14)
for ifr in range(5):
    fig.add_subplot(5, 1, ifr+1)
    ipt, icn, iac = _N.where(fr_cmp_rank1[filtdat, ifr] > int(0.95*SHUFFLES))
    cnts, bins, lns  = _plt.hist(ipt, bins=_N.linspace(-0.5, 188.5, 190), color="black")
    _plt.ylim(0, 7)
    _plt.xlim(-1.5, len(filtdat)+0.5)
    _plt.title("Framework %s" % frameworks_p[ifr], fontsize=11)
    _plt.yticks([0, 3, 6], fontsize=11)
    _plt.xticks(fontsize=11)
_plt.xlabel("participant #", fontsize=14)
fig.subplots_adjust(wspace=0.5, hspace=0.85, top=0.85, left=0.08, right=0.94)
_plt.savefig("Num_of_frameworks_comps_big_%d" % win)
    
for ifr in range(5):
    print("--------   %s" % frameworks[ifr])
    for ic in range(3):
        for ia in range(3):
            print("component   %(c)d %(a)d" % {"c" : ic, "a" : ia})
            for star in ["soc_skils", "imag", "rout", "switch", "fact_pat", "AQ28scrs"]:
                exec("tar = %s" % star)
                pc, pv = _ss.pearsonr(cmp_z1s[filtdat, ifr, ic, ia], tar[filtdat])
                #pc, pv = _ss.pearsonr(fr_cmp_rank1[filtdat, ifr, ic, ia], AQ28scrs[filtdat])
                print("pc %(pc).3f   pv %(pv).3f" % {"pc" : pc, "pv" : pv})



fig = _plt.figure(figsize=(12, 3))
fig.add_subplot(1, 3, 1)
_plt.hist(lo_allpcs, bins=_N.linspace(-1, 1, 51), color="grey", edgecolor="grey", density=True)
_plt.title("components w/ small fluctuation")
_plt.grid(ls=":", color="black")
_plt.axvline(x=0, color="black", ls="--")
_plt.xlabel("correlation")
fig.add_subplot(1, 3, 2)
_plt.title("components w/ medium fluctuation")
_plt.hist(mid_allpcs, bins=_N.linspace(-1, 1, 51), color="grey", edgecolor="grey", density=True)
_plt.grid(ls=":", color="black")
_plt.axvline(x=0, color="black", ls="--")
_plt.xlabel("correlation")
fig.add_subplot(1, 3, 3)
_plt.title("components w/ large fluctuation")
_plt.hist(hi_allpcs, bins=_N.linspace(-1, 1, 51), color="grey", edgecolor="grey", density=True)
_plt.grid(ls=":", color="black")
_plt.axvline(x=0, color="black", ls="--")
_plt.xlabel("correlation")
fig.subplots_adjust(bottom=0.17)
_plt.savefig("Framework_correlations")

