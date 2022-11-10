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
import os
import sys
from sumojam.devscripts.cmdlineargs import process_keyval_args
import pickle
import mne.time_frequency as mtf
import GCoh.eeg_util as _eu
#import RPSvAI.rpsms as rpsms
import GCoh.preprocess_ver as _ppv

import RPSvAI.constants as _cnst
from RPSvAI.utils.dir_util import workdirFN, datadirFN
import GCoh.datconfig as datconf
import RPSvAI.models.CRutils as _crut
import RPSvAI.models.empirical_ken as _emp
from sklearn.decomposition import PCA
import RPSvAI.AIRPSfeatures as _aift

import GCoh.eeg_util as _eu
import matplotlib.ticker as ticker

__DSUWTL__ = 0
__RPSWTL__ = 1
__DSURPS__ = 2
__ALL__    = 3

mode       = __ALL__
#mode       = __DSUWTL__
#mode       = __RPSWTL__
#mode       = __DSURPS__

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
top_comps=3
thrI = 1
nI=1
r1=0.4

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

expt = "SIMHUM3"
#expt = "TMB2"
if expt == "TMB2":
    lm = depickle(workdirFN("AQ28_vs_RPS_%(v)d_%(wt)d%(w)d%(s)d.dmp" % {"v" : visit, "wt" : win_type, "w" : win, "s" : smth, "wd" : os.environ["RPSWORKDIR"]}))

    #lm = depickle("predictAQ28dat/AQ28_vs_RPS_1_%(wt)d%(w)d%(s)d.dmp" % {"wt" : win_type, "w" : win, "s" : smth})
    partIDs = lm["partIDs"]
    TO = 300
elif expt == "EEG1":
    #partIDs = ["20200109_1504-32"]
    #partIDs = ["20210606_1237-17", "20210609_1230-28", "20210609_1248-16", "20210609_1321-35", "20210609_1517-23", "20210609_1747-07"]
    partIDs = ["20210606_1237-17", "20210609_1230-28", "20210609_1248-16", "20210609_1321-35", "20210609_1517-23", "20210609_1747-07", "20210526_1318-12", "20210526_1358-27", "20210526_1416-25", "20210526_1503-39"]

elif expt == "SIMHUM3":
    partIDs = []
    for sec in secs_as_string(0, 60):
        partIDs.append("20110103_0000-%s" % sec)
    for sec in secs_as_string(0, 60):
        partIDs.append("20110103_0001-%s" % sec)
    for sec in secs_as_string(0, 60):
        partIDs.append("20110103_0002-%s" % sec)
    for sec in secs_as_string(0, 60):
        partIDs.append("20110103_0003-%s" % sec)
    for sec in secs_as_string(0, 60):
        partIDs.append("20110103_0004-%s" % sec)
    for sec in secs_as_string(0, 60):
        partIDs.append("20110103_0005-%s" % sec)
    lm = {}
    lm["filtdat"] = _N.arange(360)
    TO = 300

filtdat = lm["filtdat"]
#filtdat = _N.array([8])
    
partIDs, incmp_dat = only_complete_data(partIDs, TO, label, SHF_NUM)
strtTr=0
TO -= strtTr

#fig= _plt.figure(figsize=(14, 14))

SHUFFLES = 0
extra_w = 5
t0 = -5 - extra_w
t1 = 10 + extra_w
cut = 3
all_avgs = _N.zeros((len(partIDs), SHUFFLES+1, t1-t0))
l_all_avgs = []
netwins  = _N.empty(len(partIDs), dtype=_N.int)
gk = _Am.gauKer(1)
gk /= _N.sum(gk)
gk2 = _Am.gauKer(2)
gk2 /= _N.sum(gk2)
#gk2 = None

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
rc_trg_avg_DSURPS = _N.empty((len(partIDs), t1-t0, SHUFFLES+1))

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

winlosses       = _N.empty((len(partIDs), 2))
marginalCRs = _N.empty((len(partIDs), SHUFFLES, 3, 3))

# sum_sd_DSUWTL = rebuild_sds_array(len(partIDs), lm, "sum_sd_DSUWTL")
# sum_sd_RPSWTL = rebuild_sds_array(len(partIDs), lm, "sum_sd_RPSWTL")
# sum_sd_DSUAIRPS = rebuild_sds_array(len(partIDs), lm, "sum_sd_DSUAIRPS")

lm = depickle(workdirFN("shuffledCRs_5CFs_%(ex)s_%(w)d_%(v)d" % {"ex" : expt, "w" : win, "v" : visit}))
ranks_of_cmps = lm["fr_cmp_fluc_rank1"]
ranks_of_lotsof0s = lm["fr_lotsof0s"]
len1s = lm["len1s"]

has_nonzero_CR_comps = _N.zeros(len(partIDs), dtype=_N.int)

for partID in partIDs:
    pid += 1
    dmp       = depickle(workdirFN("%(rpsm)s/%(lb)d/variousCRs_%(visit)d.dmp" % {"rpsm" : partID, "lb" : label, "visit" : visit}))
    
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
    
    for SHF_NUM in range(SHUFFLES+1):
    #for SHF_NUM in range(70, 71):
        _prob_mvsDSUWTL = dmp["cond_probsDSUWTL"][SHF_NUM][:, strtTr:]
        _prob_mvsRPSWTL = dmp["cond_probsRPSWTL"][SHF_NUM][:, strtTr:]
        #_prob_mvsDSURPS = dmp["cond_probsDSURPS"][SHF_NUM][:, strtTr:]
        _prob_mvsDSUAIRPS = dmp["cond_probsDSUAIRPS"][SHF_NUM][:, strtTr:]
        _prob_mvsRPSRPS = dmp["cond_probsRPSRPS"][SHF_NUM][:, strtTr:]
        _prob_mvsRPSAIRPS = dmp["cond_probsRPSAIRPS"][SHF_NUM][:, strtTr:]

        #_prob_mvs_STSW = dmp["cond_probsSTSW"][SHF_NUM][:, strtTr:]    
        _hnd_dat = dmp["all_tds"][SHF_NUM][strtTr:]
        
        #end_strts[pid-1] = _N.mean(_hnd_dat[-1, 3] - _hnd_dat[0, 3])
        winlosses[pid-1, 0] = len(_N.where(_hnd_dat[:, 2] == 1)[0])
        winlosses[pid-1, 1] = len(_N.where(_hnd_dat[:, 2] == -1)[0])        
        hdcol = 0

        inds =_N.arange(_hnd_dat.shape[0])
        hnd_dat_all[pid-1] = _hnd_dat[0:TO]

        cv_sum = 0
        dhd = _N.empty(TO)
        dhd[0:TO-1] = _N.diff(_hnd_dat[0:TO, 3])
        dhd[TO-1] = dhd[TO-2]
        #dhdr = dhd.reshape((20, 15))
        #rsp_tms_cv[pid-1] = _N.mean(_N.std(dhdr, axis=1) / _N.mean(dhdr, axis=1))

        #rsp_tms_cv[pid-1] = _N.std(_hnd_dat[:, 3]) / _N.mean(_hnd_dat[:, 3])
        #marginalCRs[pid-1] = _emp.marginalCR(_hnd_dat)
        prob_mvsDSUWTL  = _prob_mvsDSUWTL[:, 0:TO - win]  #  is bigger than hand by win size
        prob_mvsRPSWTL  = _prob_mvsRPSWTL[:, 0:TO - win]  #  is bigger than hand by win size
        #prob_mvsDSURPS  = _prob_mvsDSURPS[:, 0:TO - win]  #  is bigger than hand by win size
        prob_mvsDSUAIRPS  = _prob_mvsDSUAIRPS[:, 0:TO - win]  #  is bigger than hand by win size
        prob_mvsRPSRPS  = _prob_mvsRPSRPS[:, 0:TO - win]  #  is bigger than hand by win size
        prob_mvsRPSAIRPS  = _prob_mvsRPSAIRPS[:, 0:TO - win]  #  is bigger than hand by win size                        
        #prob_mvs_STSW  = _prob_mvs_STSW[:, 0:TO - win]  #  is bigger than hand by win size    
        prob_mvsDSUWTL = prob_mvsDSUWTL.reshape((3, 3, prob_mvsDSUWTL.shape[1]))
        prob_mvsRPSWTL = prob_mvsRPSWTL.reshape((3, 3, prob_mvsRPSWTL.shape[1]))
        #prob_mvs_DSURPS = prob_mvsDSURPS.reshape((3, 3, prob_mvsDSURPS.shape[1]))
        prob_mvsDSUAIRPS = prob_mvsDSUAIRPS.reshape((3, 3, prob_mvsDSUAIRPS.shape[1]))        
        prob_mvsRPSRPS = prob_mvsRPSRPS.reshape((3, 3, prob_mvsRPSRPS.shape[1]))
        prob_mvsRPSAIRPS = prob_mvsRPSAIRPS.reshape((3, 3, prob_mvsRPSAIRPS.shape[1]))        

        #marginalCRs[pid-1, SHF_NUM] = _emp.marginalCR(_hnd_dat)
        N = prob_mvsDSUWTL.shape[2]
        #dbehv, behv    = _crut.get_dbehv_combined([prob_mvs_DSURPS, prob_mvs_DSUAIRPS, prob_mvs_RPS, prob_mvs], gkISI, equalize=True)
        #dbehv, behv    = _crut.get_dbehv_combined([prob_mvs_DSUAIRPS, prob_mvs_DSURPS, prob_mvs_RPS, prob_mvs], gkISI, equalize=True)
        #dbehv, behv    = _crut.get_dbehv_combined([prob_mvs_DSURPS, prob_mvsRPS, prob_mvs, prob_mvs_DSUAIRPS], gkISI, equalize=False)
        #dbehv, behv    = _crut.get_dbehv_combined([prob_mvs_DSURPS, prob_mvsRPS, prob_mvs], gkISI, equalize=False)
        #dbehv, behv    = _crut.get_dbehv_combined([prob_mvs_DSURPS, prob_mvs_RPS, prob_mvs], None, equalize=False, weight=True)
        #dbehv, behv    = _crut.get_dbehv_combined([prob_mvs, prob_mvs_DSURPS, prob_mvs_DSUAIRPS, prob_mvs, prob_mvsRPS], None, biggest=True, top_comps=9)

        ##### GOOD
        #dbehv, behv    = _crut.get_dbehv_combined([prob_mvs, prob_mvs_DSURPS, prob_mvs_DSUAIRPS, prob_mvsRPS, prob_mvsRPSRPS, prob_mvsRPSAIRPS], None, biggest=biggest, top_comps=top_comps)
        #dbehv, behv    = _crut.get_dbehv_combined([prob_mvsDSUWTL, prob_mvs_DSUAIRPS, prob_mvsRPSRPS, prob_mvsRPSAIRPS], None, biggest=biggest, top_comps=top_comps)
        #frameworks = ["DSUWTL", "RPSWTL", "DSUAIRPS", "RPSAIRPS", "RPSRPS"]

        #dbehv, behv    = _crut.get_dbehv_biggest_fluc([prob_mvsDSUWTL, prob_mvsRPSWTL, prob_mvs_DSUAIRPS, prob_mvsRPSAIRPS, prob_mvsRPSRPS], ranks[pid-1])
        #maxs = _aift.get_maxes(behv, thrs, thrI=thrI, nI=nI, r1=r1, win=3)




        #ranks_of_cmps[pid-1, 0] = 0
        #ranks_of_lotsof0s[pid-1, 0] = 0
        #behv   = _crut.get_dbehv_biggest_fluc([prob_mvsDSUWTL, prob_mvsRPSWTL, prob_mvsRPSRPS, prob_mvsDSUAIRPS, prob_mvsRPSAIRPS], gk2, ranks_of_cmps[pid-1], ranks_of_lotsof0s[pid-1], len1s[pid-1], big_percentile=0.6, min_big_comps=4, flip_choose_components=False)
        #behv   = _crut.get_dbehv_biggest_fluc([prob_mvsDSUWTL, prob_mvsRPSWTL, prob_mvsRPSRPS, prob_mvsDSUAIRPS, prob_mvsRPSAIRPS], gk2, ranks_of_cmps[pid-1], ranks_of_lotsof0s[pid-1], len1s[pid-1], big_percentile=0.5, min_big_comps=4, flip_choose_components=True)
        behv   = _crut.get_dbehv_biggest_fluc([prob_mvsRPSRPS, prob_mvsDSUAIRPS, prob_mvsRPSAIRPS], gk2, ranks_of_cmps[pid-1], ranks_of_lotsof0s[pid-1], len1s[pid-1], big_percentile=0.94, min_big_comps=3, flip_choose_components=False)
        #behv   = _crut.get_dbehv_biggest_fluc([prob_mvsDSUWTL, prob_mvsRPSWTL, prob_mvsRPSRPS, prob_mvsDSUAIRPS, prob_mvsRPSAIRPS], gk2, ranks_of_cmps[pid-1], _N.zeros((3, 3)), len1s[pid-1], big_percentile=0.95, min_big_comps=2)





        ##### GOOD
        #dbehv, behv    = _crut.get_dbehv_combined([prob_mvs, prob_mvs_DSURPS, prob_mvs_DSUAIRPS, prob_mvs, prob_mvsRPS, prob_mvsRPSRPS, prob_mvsRPSAIRPS], None, biggest=True, top_comps=7)
        #maxs = _aift.get_maxes(behv, thrs, thrI=4, nI=2, r1=0.2, win=3)

        # dbehv, behv    = _crut.get_dbehv_combined([prob_mvs, prob_mvs_DSURPS, prob_mvs_DSUAIRPS, prob_mvs, prob_mvsRPS, prob_mvsRPSRPS, prob_mvsRPSAIRPS], None, biggest=True, top_comps=3)
        # maxs = _aift.get_maxes(behv, thrs, thrI=1, nI=3, r1=0.4, win=3)
        
        #maxs = _aift.get_maxes(behv, thrs, thrI=2, nI=2, r1=0.5, win=3)
        #dbehv, behv    = _crut.get_dbehv_combined([prob_mvs, prob_mvs_RPS, prob_mvs_DSUAIRPS], None, biggest=True, top_comps=4, use_sds=[sum_sd_DSUWTL[pid-1], sum_sd_RPSWTL[pid-1], sum_sd_DSUAIRPS[pid-1]])
        #maxs = _aift.get_maxes(behv, thrs, thrI=1, nI=4, r1=0.2, win=3)
        """
        maxima = _N.where((behv[0:-3] < behv[1:-2]) & (behv[1:-2] > behv[2:-1]))[0]
        minima = _N.where((behv[0:-3] > behv[1:-2]) & (behv[1:-2] < behv[2:-1]))[0]
        nMins = len(minima)
        nMaxs = len(maxima)        
    
        start_thr = _N.sort(behv[minima + 1])[int(0.25*nMins)]  #  we don't want maxes to be below any mins
        thr_max   = 0.5*(_N.max(behv[maxima]) - _N.min(behv[maxima])) + _N.min(behv[maxima]) 

        dthr      = (thr_max - start_thr) / 30.

        bDone     = False
        i = -1
        while (not bDone) and (i < 30):
            i += 1
            max_thr = start_thr + dthr*i
            maxs = maxima[_N.where(behv[maxima+1] > max_thr)[0]] + win//2+1
            intvs = _N.diff(maxs)
            #print("!!!!!!!")
            #print(intvs)

            if len(_N.where(intvs <= 1)[0]) < 4:   #  not too many of these
                 bDone = True
                 thrs[pid-1] = i
        if not bDone:   #  didn't find it.
            max_thr = start_thr + dthr*28
            maxs = maxima[_N.where(behv[maxima+1] > max_thr)[0]] + win//2+1
            thrs[pid-1] = 28
        #####!!!!!  len(maxs) < cut means nothing to trigger average
        """

        if behv is not None:
            has_nonzero_CR_comps[pid-1] = 1
            #dbehv  = _N.diff(_N.convolve(behv, gk, mode="same")) #+ _N.diff(behv))
            dbehv  = _N.diff(behv)
            maxs = _N.where((dbehv[0:TO-11] >= 0) & (dbehv[1:TO-10] < 0))[0] + (win//2)#  3 from label71

            for sh in range(1):
                if sh > 0:
                    _N.random.shuffle(inds)
                hnd_dat = _hnd_dat[inds]

                avgs = _N.zeros((len(maxs)-2*cut, t1-t0))
                #print(maxs)


                for im in range(cut, len(maxs)-cut):
                    #print(hnd_dat[maxs[im]+t0:maxs[im]+t1, 2].shape)
                    #print("%(1)d %(2)d" % {"1" : maxs[im]+t0, "2" : maxs[im]+t1})
                    st = 0
                    en = t1-t0
                    if maxs[im] + t0 < 0:   #  just don't use this one
                        print("DON'T USE THIS ONE")
                        avgs[im-1, :] = 0
                    else:
                        try:
                            avgs[im-cut, :] = hnd_dat[maxs[im]+t0:maxs[im]+t1, 2]
                            if len(_N.where((pid-1) == filtdat)[0]) == 1:  #  in filtdat
                                l_all_avgs.append(hnd_dat[maxs[im]+t0:maxs[im]+t1, 2])
                        except ValueError:   #  trigger lags past end of games
                            print("*****  pid-1:%(pid)d   SHF_NUM: %(sh)d     t0=%(1)d  t1=%(2)d" % {"1" : maxs[im]+t0, "2" : maxs[im]+t1, "sh" : SHF_NUM, "pid" : (pid-1)})
                            #print(avgs[im-1, :].shape)
                            #print(hnd_dat[maxs[im]+t0:maxs[im]+t1, 2])
                            avgs[im-1, :] = 0                        


                all_avgs[pid-1, sh] = _N.mean(avgs, axis=0)  #  trigger average
                if _N.sum(_N.isnan(all_avgs[pid-1, sh])):
                    print("ISNAN   %(pid)d   %(sh)d" % {"sh" : SHF_NUM, "pid" : (pid-1)})
                    print(all_avgs[pid-1, sh])
                    print(avgs)
                    print(avgs.shape)
                    #print("..........    %d" % _N.sum(_N.isnan(prob_mvs)))
                #fig.add_subplot(5, 5, pid)
                #_plt.plot(_N.mean(avgs, axis=0))

            isi   = _N.diff(maxs)
            pc, pv = rm_outliersCC_neighbors(isi[0:-1], isi[1:])
            #pc, pv = _ss.pearsonr(isi[0:-1], isi[1:])
            isis_corr[pid-1] = pc
            isis[pid-1] = _N.mean(isi)        
            isis_cv[pid-1] = _N.std(isi) / isis[pid-1]

            isis_lv[pid-1] = (3/(len(isi)-1))*_N.sum((isi[0:-1] - isi[1:])**2 / (isi[0:-1] + isi[1:])**2 )
            pfrm_change69[pid-1] = _N.max(all_avgs[pid-1, 0, 5:20]) - _N.min(all_avgs[pid-1, 0, 5:20])

#             #srtd   = _N.sort(all_avgs[pid-1, 1:], axis=0)
#             #signal_5_95[pid-1, 1] = srtd[int(0.05*SHUFFLES)]
#             #signal_5_95[pid-1, 2] = srtd[int(0.95*SHUFFLES)]
#             signal_5_95[pid-1, 0] = all_avgs[pid-1, 0]
#             signal_5_95[pid-1, 3] = (signal_5_95[pid-1, 0] - signal_5_95[pid-1, 1]) / (signal_5_95[pid-1, 2] - signal_5_95[pid-1, 1])

#             #pfrm_change36[pid-1] = _N.max(signal_5_95[pid-1, 0, 3:6]) - _N.min(signal_5_95[pid-1, 0, 3:6])

#             sInds = _N.argsort(signal_5_95[pid-1, 0, 6:9])
#             #sInds = _N.argsort(signal_5_95[pid-1, 0, 5:10])
#             if sInds[2] - sInds[0] > 0:
#                 m69 = 1
#             else:
#                 m69 = -1

#             imax69 = _N.argmax(signal_5_95[pid-1, 0, 6:9])+6
#             imin69 = _N.argmin(signal_5_95[pid-1, 0, 6:9])+6
#             if SHF_NUM == 0:
#                 chg[pid-1] = _N.mean(signal_5_95[pid-1, 0, 7:11]) - _N.mean(signal_5_95[pid-1, 0, 4:8])
#                 pfrm_change69[pid-1] = signal_5_95[pid-1, 0, imax69] - signal_5_95[pid-1, 0, imin69]
#                 #pfrm_change69[pid-1] = _N.mean(signal_5_95[pid-1, 0, 7:9]) - _N.mean(signal_5_95[pid-1, 0, 5:7])


#             ###################################################
#             #fig.add_subplot(6, 6, pid)

#             netwins[pid-1] = _N.sum(_hnd_dat[:, 2])
#             # _plt.title(netwins[pid-1] )
#             # _plt.plot(ts, signal_5_95[pid-1, 0], marker=".", ms=10)
#             # _plt.plot(ts, signal_5_95[pid-1, 1])
#             # _plt.plot(ts, signal_5_95[pid-1, 2])
#             # _plt.axvline(x=-0.5, ls="--")

#             be = _N.where(signal_5_95[pid-1, 0] < signal_5_95[pid-1, 1])[0]
#             if len(be) > 0:
#                 belows.extend(be)
#             ab = _N.where(signal_5_95[pid-1, 0] > signal_5_95[pid-1, 2])[0]        
#             if len(ab) > 0:
#                 aboves.extend(ab)

#             #prob_mvs = lm["all_prob_mvsA"][ip]

#             cntr = 0
#             n_cntr = 0
#             maxp_chg_times_wtl = []

#             rc_trg_avg[pid-1, :, SHF_NUM] = signal_5_95[pid-1, 0]
#             #y_pred = obs_v_preds[byScores[igd], 0:test_sz-1, 1]

#             halfT = (t1-t0)//2
#             A1 = _N.vstack([_N.arange(halfT), _N.ones(halfT)]).T
#             A2 = _N.vstack([_N.arange(halfT+1), _N.ones(halfT+1)]).T
#             #A2 = _N.vstack([_N.arange(8), _N.ones(8)]).T
#             m1, c1 = _N.linalg.lstsq(A1, rc_trg_avg[pid-1, 0:halfT, 0], rcond=-1)[0]
#             m2, c2 = _N.linalg.lstsq(A2, rc_trg_avg[pid-1, halfT:2*halfT+1, 0], rcond=-1)[0]
#             y1 = m1*(halfT-1) + c1
#             y2 = c2
#             jump[pid-1] = y2-y1
#             m1s[pid-1] = m1
#             m2s[pid-1] = m2
#             c1s[pid-1] = c1
#             c2s[pid-1] = c2

#             isi   = cleanISI(_N.diff(maxs), minISI=3)
#             #else:
#             #    isi   = cleanISI(_N.diff(maxs_DSURPS), minISI=1)
#             #maxs = maxs_DSUWTL
#             #_aift.rulechange(_hnd_dat, signal_5_95, pfrm_change36, pfrm_change69, pfrm_change912, imax_imin_pfrm36, imax_imin_pfrm69, imax_imin_pfrm912, all_avgs, SHUFFLES, t0, t1, maxs, cut, pid)
#             #isi   = cleanISI(_N.diff(maxs), minISI=2)
#             pc, pv = rm_outliersCC_neighbors(isi[0:-1], isi[1:])
#             #pc, pv = _ss.pearsonr(isi[0:-1], isi[1:])
#             isis_corr[pid-1] = pc
#             isis[pid-1] = _N.mean(isi)        
#             isis_cv[pid-1] = _N.std(isi) / isis[pid-1]

#             isis_lv[pid-1] = (3/(len(isi)-1))*_N.sum((isi[0:-1] - isi[1:])**2 / (isi[0:-1] + isi[1:])**2 )
        
# popmn_rc_trg_avg = _N.mean(rc_trg_avg[filtdat], axis=0)        

# #  p(UP | W)
# #  a big change means lots of UP | W
# fig  = _plt.figure(figsize=(8, 4))
# for sh in range(SHUFFLES):
#     _plt.plot(popmn_rc_trg_avg[:, sh+1], color="grey", lw=1)
# _plt.plot(popmn_rc_trg_avg[:, 0], color="black", lw=3)
# _plt.xticks(_N.arange(t1-t0), _N.arange(-7, 8), fontsize=15)
# _plt.yticks(fontsize=15)
# _plt.axvline(x=7, ls=":", color="grey")
# _plt.xlabel("lags from rule change (# games)", fontsize=18)
# _plt.ylabel("p(WIN, lag) - p(LOS, lag)", fontsize=18)
# fig.subplots_adjust(bottom=0.15, left=0.15)
# _plt.xlim(0, 14)
# #_plt.ylim(-0.1, 0.1)
# _plt.savefig("Rulechange_w_shuffles")


# bf =_N.mean(rc_trg_avg[:, 5:7, 0], axis=1)
# af =_N.mean(rc_trg_avg[:, 7:9, 0], axis=1)

# save_trl_by_trl = _N.empty((len(filtdat), t1-t0))
# iii = -1
# for ifd in filtdat:
#     iii += 1
#     save_trl_by_trl[iii] = rc_trg_avg[ifd, :, 0] - _N.mean(rc_trg_avg[ifd, :, 0])
# _N.savetxt("trl_mean_rulechg_jmp.txt", save_trl_by_trl.T, fmt=("%.3f " * len(filtdat)))

# fig = _plt.figure(figsize=(5, 11))
# #_plt.plot(popmn_rc_trg_avg[:, 0], color="black", lw=3)

# all_trg_trls = _N.array(l_all_avgs)
# ys1stHalf = _N.empty((halfT, len(partIDs)))
# ys2ndHalf = _N.empty((halfT+1, len(partIDs)))

# for pid in range(len(partIDs)):
#     #_plt.plot(A1[:, 0], c1s[pid] + m1s[pid]*A1[:, 0], color="#DDDDDD")
#     ys1stHalf[:, pid] = c1s[pid] + m1s[pid]*A1[:, 0]
#     #_plt.plot(halfT+A2[:, 0], c2s[pid] + m2s[pid]*A2[:, 0], color="#DDDDDD")
#     ys2ndHalf[:, pid] = c2s[pid] + m2s[pid]*A2[:, 0]
# m_ys1stHalf = ys1stHalf - _N.mean(ys1stHalf, axis=0).reshape((1, ys1stHalf.shape[1]))
# m_ys2ndHalf = ys2ndHalf - _N.mean(ys2ndHalf, axis=0).reshape((1, ys2ndHalf.shape[1]))    
# for pid in range(len(partIDs)):
#     _plt.plot(A1[:, 0], ys1stHalf[:, pid], color="#DDDDDD")
#     _plt.plot(halfT+A2[:, 0], ys2ndHalf[:, pid], color="#DDDDDD")    

# weights = n_maxes[:, 0] / _N.sum(n_maxes[:, 0])
# mn_att = _N.mean(_N.mean(all_trg_trls, axis=0))

# _plt.plot(_N.mean(all_trg_trls, axis=0), color="black", lw=2, marker=".", ms=15)
# #_plt.plot(A1[:, 0], _N.sum(ys1stHalf[:, filtdat]*weights[filtdat], axis=1), color="orange")
# _plt.plot(A1[:, 0], _N.mean(ys1stHalf[:, filtdat], axis=1), color="orange")
# #_plt.plot(halfT+A2[:, 0], _N.sum(ys2ndHalf[:, filtdat]*weights[filtdat], axis=1), color="orange")
# _plt.plot(halfT+A2[:, 0], _N.mean(ys2ndHalf[:, filtdat], axis=1), color="orange")
# #_plt.plot(A1[:, 0], _N.sum(ys1stHalf*weights, axis=1), color="orange")
# #_plt.plot(halfT+A2[:, 0], _N.sum(ys2ndHalf*weights, axis=1), color="orange")


# tksz=15
# lblsz=17
# _plt.xticks(_N.arange(15+2*extra_w), _N.arange(-7-extra_w, 8+extra_w), fontsize=tksz)
# _plt.yticks(fontsize=tksz)
# _plt.xlim(0, 14+2*extra_w)
# _plt.xlabel("lags from rule change (#games)", fontsize=lblsz)
# _plt.ylabel("p(WIN) - p(LOSE) at lag", fontsize=lblsz)
# fig.subplots_adjust(left=0.2, right=0.98, bottom=0.1, top=0.98)
# _plt.savefig("rulechange")

# """
# fig = _plt.figure()
# A1 = _N.vstack([_N.arange(halfT), _N.ones(halfT)]).T
# A2 = _N.vstack([_N.arange(halfT+1), _N.ones(halfT+1)]).T
# #A2 = _N.vstack([_N.arange(8), _N.ones(8)]).T
# m1, c1 = _N.linalg.lstsq(A1, rc_trg_avg[filtdat[0], 0:halfT, 0], rcond=-1)[0]
# m2, c2 = _N.linalg.lstsq(A2, rc_trg_avg[filtdat[0], halfT:2*halfT+1, 0], rcond=-1)[0]
# y1 = m1*(halfT-1) + c1
# y2 = c2
# y1 = c1 + m1*A1[:, 0]
# y2 = c2 + m2*A2[:, 0]
# _plt.plot(A1[:, 0], y1)
# _plt.plot(A2[:, 0]+halfT, y2)
# _plt.plot(rc_trg_avg[filtdat[0], :, 0])
# """


# sths = _N.array(filtdat)
# print("interval statistics")

# dmp_dat    = {}
# data       = {}
# dmp_dat["prms_%(tc)d_%(thrI)d_%(nI)d_%(r1).2f" % {"tc" : top_comps, "thrI" : thrI, "nI" : nI, "r1" : r1}] = data

# data["avg"] = _N.mean(all_trg_trls, axis=0)
# print(".................. all_trg_trls")
# print(data["avg"])
nzfiltdat = _N.intersect1d(_N.where(has_nonzero_CR_comps)[0], filtdat)

"""
for sud in ["isis", "isis_corr", "isis_cv", "isis_lv", "pfrm_change69"]:
    #data[sud] = _N.empty((6, 2))
    print("int stat------   %s" % sud)
    exec("ist_ud = %s" % sud)
    ist = -1
    for star in ["AQ28scrs", "soc_skils", "imag", "rout", "switch", "fact_pat"]:
        ist += 1
        exec("tar = %s" % star)
        print("!!!!!  %s" % star)
        pc, pv = _ss.pearsonr(ist_ud[nzfiltdat], tar[nzfiltdat])
        #if _N.abs(pc) > 0.15:
        #data[sud][ist] = pc, pv
        print("%(pc).3f  %(pv).3f" % {"pc" : pc, "pv" : pv})
"""

# if os.access("Results_231/RC_all_combos.dmp", os.F_OK):
#     lm = depickle("Results_231/RC_all_combos.dmp")
#     for ky in lm.keys():
#         dmp_dat[ky] = lm[ky]

# dmpout = open("Results_231/RC_all_combos.dmp", "wb")    
# pickle.dump(dmp_dat, dmpout, -1)
# dmpout.close()

fig = _plt.figure(figsize=(7, 4))
#for inz in range(len(nzfiltdat)):
#    _plt.plot(all_avgs[nzfiltdat[inz], 0] - _N.mean(all_avgs[nzfiltdat[inz], 0]), color="grey")
ts = _N.arange(-(t1-t0)//2+1, (t1-t0)//2+1)
mnsig = _N.mean(all_avgs[nzfiltdat, 0], axis=0)
_plt.suptitle(len(nzfiltdat))
_plt.plot(ts, mnsig, color="black", lw=3)
#_plt.plot(ts, mnsig, color="orange", lw=3)
_plt.grid()
_plt.axvline(x=0, ls="--", color="grey")
_plt.xlabel("lagged games from rule change", fontsize=12)
_plt.xticks(fontsize=11)
_plt.yticks(fontsize=11)
_plt.ylabel("win prob. - lose prob.", fontsize=12)
_plt.ylim(-0.15, 0.05)
_plt.savefig("Rule-change_%(exp)s_%(w)d" % {"exp" : expt, "w" : win}, transparent=True)
