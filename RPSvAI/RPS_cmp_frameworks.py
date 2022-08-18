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
from RPSvAI.utils.dir_util import workdirFN
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
        _prob_mvsDSURPS = dmp["cond_probsDSURPS"][SHF_NUM]                
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

SHUFFLES = 0
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

sum_sd_DSUWTL = rebuild_sds_array(len(partIDs), lm, "sum_sd_DSUWTL")
sum_sd_RPSWTL = rebuild_sds_array(len(partIDs), lm, "sum_sd_RPSWTL")
sum_sd_DSUAIRPS = rebuild_sds_array(len(partIDs), lm, "sum_sd_DSUAIRPS")

sumdetDSUWTL = _N.empty(len(partIDs))
sumdetRPSWTL = _N.empty(len(partIDs))
sumdetRPSAIRPS = _N.empty(len(partIDs))
sumdetRPSRPS = _N.empty(len(partIDs))
sumdetDSUAIRPS = _N.empty(len(partIDs))
for partID in partIDs:
    pid += 1
    dmp       = depickle(workdirFN("%(rpsm)s/%(lb)d/WTL_%(v)d.dmp" % {"rpsm" : partID, "lb" : label, "v" : visit}))

    #if expt == "TMB2":
    #    AQ28scrs[pid-1], soc_skils[pid-1], rout[pid-1], switch[pid-1], imag[pid-1], fact_pat[pid-1] = _rt.AQ28("/Users/arai/Sites/taisen/DATA/%(data)s/%(date)s/%(pID)s/AQ29.txt" % {"date" : partIDs[pid-1][0:8], "pID" : partIDs[pid-1], "data" : expt})

    _prob_mvsDSUWTL = dmp["cond_probsDSUWTL"][:, :, strtTr:]
    _prob_mvsRPSWTL = dmp["cond_probsRPSWTL"][:, strtTr:]
    _prob_mvsDSURPS = dmp["cond_probsDSURPS"][:, strtTr:]
    _prob_mvsDSUAIRPS = dmp["cond_probsDSUAIRPS"][:, strtTr:]
    _prob_mvsRPSRPS = dmp["cond_probsRPSRPS"][:, strtTr:]
    _prob_mvsRPSAIRPS = dmp["cond_probsRPSAIRPS"][:, strtTr:]        
    prob_mvsDSUWTL  = _prob_mvsDSUWTL[:, :, 0:TO - win]  #  is bigger than hand by win size
    prob_mvsRPSWTL  = _prob_mvsRPSWTL[:, :, 0:TO - win]  #  is bigger than hand by win size
    prob_mvsDSURPS  = _prob_mvsDSURPS[:, :, 0:TO - win]  #  is bigger than hand by win size
    prob_mvsDSUAIRPS  = _prob_mvsDSUAIRPS[:, :, 0:TO - win]  #  is bigger than hand by win size    
    prob_mvsRPSRPS  = _prob_mvsRPSRPS[:, :, 0:TO - win]  #  is bigger than hand by win size
    prob_mvsRPSAIRPS  = _prob_mvsRPSAIRPS[:, :, 0:TO - win]  #  is bigger than hand by win size    

    #stds_all_mdls[0] = _N.std(prob_mvs, axis=2)
    
    for SHF_NUM in range(SHUFFLES+1):
    #for SHF_NUM in range(70, 71):
        _prob_mvsDSUWTL = dmp["cond_probsDSUWTL"][SHF_NUM][:, strtTr:]
        _prob_mvsRPSWTL = dmp["cond_probsRPSWTL"][SHF_NUM][:, strtTr:]
        _prob_mvsDSURPS = dmp["cond_probsDSURPS"][SHF_NUM][:, strtTr:]
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
        prob_mvsDSURPS  = _prob_mvsDSURPS[:, 0:TO - win]  #  is bigger than hand by win size
        prob_mvsDSUAIRPS  = _prob_mvsDSUAIRPS[:, 0:TO - win]  #  is bigger than hand by win size
        prob_mvsRPSRPS  = _prob_mvsRPSRPS[:, 0:TO - win]  #  is bigger than hand by win size
        prob_mvsRPSAIRPS  = _prob_mvsRPSAIRPS[:, 0:TO - win]  #  is bigger than hand by win size                        
        #prob_mvs_STSW  = _prob_mvs_STSW[:, 0:TO - win]  #  is bigger than hand by win size    
        prob_mvsDSUWTL = prob_mvsDSUWTL.reshape((3, 3, prob_mvsDSUWTL.shape[1]))
        prob_mvsRPSWTL = prob_mvsRPSWTL.reshape((3, 3, prob_mvsRPSWTL.shape[1]))
        prob_mvsDSURPS = prob_mvsDSURPS.reshape((3, 3, prob_mvsDSURPS.shape[1]))
        prob_mvsDSUAIRPS = prob_mvsDSUAIRPS.reshape((3, 3, prob_mvsDSUAIRPS.shape[1]))        
        prob_mvsRPSRPS = prob_mvsRPSRPS.reshape((3, 3, prob_mvsRPSRPS.shape[1]))
        prob_mvsRPSAIRPS = prob_mvsRPSAIRPS.reshape((3, 3, prob_mvsRPSAIRPS.shape[1]))        

        #marginalCRs[pid-1, SHF_NUM] = _emp.marginalCR(_hnd_dat)
        N = prob_mvsDSUWTL.shape[2]

        sum_deterministicDSUWTL = _N.zeros((3, 3), dtype=_N.int)        
        sum_deterministicRPSWTL = _N.zeros((3, 3), dtype=_N.int)        
        sum_deterministicRPSAIRPS = _N.zeros((3, 3), dtype=_N.int)        
        sum_deterministicDSUAIRPS = _N.zeros((3, 3), dtype=_N.int)        
        sum_deterministicRPSRPS = _N.zeros((3, 3), dtype=_N.int)        


        for game in range(prob_mvsDSUWTL.shape[2]):

            for ic in range(3):
                for ia in range(3):
                    if (prob_mvsDSUWTL[ic, ia, game] > 0.9) or (prob_mvsDSUWTL[ic, ia, game] < 0.1):
                        sum_deterministicDSUWTL[ic, ia] += 1
                    if (prob_mvsRPSWTL[ic, ia, game] > 0.9) or (prob_mvsRPSWTL[ic, ia, game] < 0.1):
                        sum_deterministicRPSWTL[ic, ia] += 1
                    if (prob_mvsRPSAIRPS[ic, ia, game] > 0.9) or (prob_mvsRPSAIRPS[ic, ia, game] < 0.1):
                        sum_deterministicRPSAIRPS[ic, ia] += 1
                    if (prob_mvsRPSRPS[ic, ia, game] > 0.9) or (prob_mvsRPSRPS[ic, ia, game] < 0.1):
                        sum_deterministicRPSRPS[ic, ia] += 1
                    if (prob_mvsDSUAIRPS[ic, ia, game] > 0.9) or (prob_mvsDSUAIRPS[ic, ia, game] < 0.1):
                        sum_deterministicDSUAIRPS[ic, ia] += 1

        print("%(DSUWTL)d  %(RPSWTL)d  %(RPSAIRPS)d" % {"DSUWTL" : _N.sum(sum_deterministicDSUWTL), "RPSWTL" : _N.sum(sum_deterministicRPSWTL), "RPSAIRPS" : _N.sum(sum_deterministicRPSAIRPS)})
        sumdetDSUWTL[pid-1] = _N.sum(sum_deterministicDSUWTL)
        sumdetRPSRPS[pid-1] = _N.sum(sum_deterministicRPSRPS)
        sumdetRPSWTL[pid-1] = _N.sum(sum_deterministicRPSWTL)
        sumdetRPSAIRPS[pid-1] = _N.sum(sum_deterministicRPSAIRPS)
        sumdetDSUAIRPS[pid-1] = _N.sum(sum_deterministicDSUAIRPS)

frameworks = ["DSUWTL", "RPSWTL", "DSUAIRPS", "RPSAIRPS", "RPSRPS"]
for fr1 in range(5):
    for fr2 in range(5):
        pr1 = frameworks[fr1]
        pr2 = frameworks[fr2]
        if fr1 > fr2:
            exec("sums1 = sumdet%s" % pr1)
            exec("sums2 = sumdet%s" % pr2)
            fig = _plt.figure()
            _plt.scatter(sums1, sums2)
            _plt.plot([300, 1400], [300, 1400])
            _plt.xlabel(pr1)
            _plt.ylabel(pr2)
            
            _plt.savefig("%(f1)s_%(f2)s_cmp_framwork" % {"f1" : pr1, "f2" : pr2})
