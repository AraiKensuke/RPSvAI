
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
#from sumojam.devscripts.cmdlineargs import process_keyval_args
import pickle
import mne.time_frequency as mtf
import GCoh.eeg_util as _eu
#import RPSvAI.rpsms as rpsms
import GCoh.preprocess_ver as _ppv

from RPSvAI.utils.dir_util import workdirFN, datadirFN
import GCoh.datconfig as datconf
import RPSvAI.models.CRutils as _crut
import RPSvAI.models.empirical_ken as _emp
from sklearn.decomposition import PCA
import RPSvAI.AIRPSfeatures as _aift

import GCoh.eeg_util as _eu
import matplotlib.ticker as ticker

__1st__ = 0
__2nd__ = 1

_ME_WTL = 0
_ME_RPS = 1

_SHFL_KEEP_CONT  = 0
_SHFL_NO_KEEP_CONT  = 1

flip_HUMAI = False

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

        sFlipped  = "_flipped" if flip_HUMAI else ""
        dmp       = depickle(workdirFN("%(rpsm)s/%(lb)d/variousCRs%(flp)s_%(v)d.dmp" % {"rpsm" : partID, "lb" : label, "v" : visit, "flp" : sFlipped}))

        _prob_mvsDSUWTL = dmp["cond_probsDSUWTL"][SHF_NUM]
        _prob_mvsRPSWTL = dmp["cond_probsRPSWTL"][SHF_NUM]
        #_prob_mvsDSURPS = dmp["cond_probsDSURPS"][SHF_NUM]                
        __hnd_dat = dmp["all_tds"][SHF_NUM]
        _hnd_dat   = __hnd_dat[0:TO]

        if _hnd_dat.shape[0] < TO:
            incomplete_data.append(pid)
            print("appending")
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

#process_keyval_args(globals(), sys.argv[1:])   #  For when we run from cmd line

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
#process_keyval_args(globals(), sys.argv[1:])
#######################################################

win_type = 2   #  window is of fixed number of games
#win_type = 1  #  window is of fixed number of games that meet condition 
win     = 3
smth    = 1
label          = win_type*100+win*10+smth
SHF_NUM = 0

svisits =str(visits).replace(" ", "").replace("[", "").replace("]", "")
detected_rulechange_triggered = []
#expt = "SIMHUM45"
#expt = "SIMHUM2"
#expt = "SIMHUM2"
expt  = "WPI"
expt  = "TMB2"
#expt = "CogWeb"
know_gt = False  #
#expt = "SIMHUM3"
#know_gt = True  #

if expt == "TMB2":
    lm = depickle(workdirFN("TMB2_AQ28_vs_RPS_features_%(v)d_of_%(vs)s_%(wt)d%(w)d%(s)d.dmp" % {"v" : visit, "wt" : win_type, "w" : win, "s" : smth, "wd" : os.environ["RPSWORKDIR"], "vs" : svisits}))
    #lm = depickle(workdirFN("TMB2_AQ28_vs_RPS_features%(flp)s_1_of_1_%(lb)s.dmp" % {"lb" : label, "flp" : sFlipped}))


    #lm = depickle("predictAQ28dat/AQ28_vs_RPS_1_%(wt)d%(w)d%(s)d.dmp" % {"wt" : win_type, "w" : win, "s" : smth})
    partIDs_okgames = lm["partIDs_okgames"]
    partIDs = lm["partIDs"]
    
    TO = 300
if expt == "CogWeb":
    lm = {}
    dates = _rt.date_range(start='2/25/2024', end='4/30/2029')
    partIDs, dats, cnstrs, has_domainQs, has_domainQs_wkeys = _rt.filterRPSdats(expt, dates, visits=visits, domainQ=(_rt._TRUE_AND_FALSE_), demographic=_rt._TRUE_AND_FALSE_, mentalState=_rt._TRUE_AND_FALSE_, min_meanIGI=600, max_meanIGI=20000, minIGI=10, maxIGI=50000, MinWinLossRat=0.3, has_useragent=True, has_start_and_end_times=True, has_constructor=True, blocks=1)
    lm["filtdat"] = _N.arange(len(partIDs))
    TO = 300
elif expt == "KEN":
    #partIDs, TO = ["20200109_1504-32", "20200108_1703-13"], 440
    partIDs, TO = ["20200108_1703-13"], 440
    partIDs, TO = ["20200109_1504-32"], 570    
    lm = {}
    lm["filtdat"] = _N.arange(len(partIDs))
    #
elif expt == "WPI":
    #partIDs = ["20210609_1230-28", "20210609_1248-16", "20210609_1321-35", "20210609_1517-23", "20210609_1747-07"]
    partIDs = ["20210609_1230-28", "20210609_1248-16", "20210609_1517-23", "20210609_1747-07"]
    partIDs = ["20210526_1318-12",]
    TO = 300
    lm = {}
    lm["filtdat"] = _N.arange(len(partIDs))
    #TO = 440

elif expt == "EEG1":
    #partIDs = ["20200109_1504-32"]
    #partIDs = ["20210606_1237-17", "20210609_1230-28", "20210609_1248-16", "20210609_1321-35", "20210609_1517-23", "20210609_1747-07"]
    partIDs = ["20210606_1237-17", "20210609_1230-28", "20210609_1248-16", "20210609_1321-35", "20210609_1517-23", "20210609_1747-07", "20210526_1318-12", "20210526_1358-27", "20210526_1416-25", "20210526_1503-39"]
    lm = {}
    lm["filtdat"] = _N.arange(len(partIDs))
elif expt[0:6] == "SIMHUM":
    partIDs = []

    nSIMHUM=int(expt[6:])
    syr    = "201101%s" % ("0%d" % nSIMHUM if nSIMHUM < 10 else str(nSIMHUM))
    yr_dir    = datadirFN("%(e)s/%(syr)s" % {"e" : expt, "syr" : syr})

    candidate_dirs = os.listdir(yr_dir)

    for i in range(len(candidate_dirs)):
        if candidate_dirs[i][0:8] == syr:
            partIDs.append(candidate_dirs[i])
        
    lm = {}
    lm["filtdat"] = _N.arange(len(candidate_dirs))
    TO = 300
    
filtdat = lm["filtdat"]
if expt=="TMB2":
    #filtdat = lm["filtdat_okgames"]
    filtdat = lm["filtdat"]

partIDs, incmp_dat = only_complete_data(partIDs, TO, label, SHF_NUM)
strtTr=0
TO -= strtTr

#fig= _plt.figure(figsize=(14, 14))

SHUFFLES = 0
extra_w = 7
t0  = -12
t1  = 16
#t0 = -5 - extra_w
#t1 = 10 + extra_w
cut = 2
all_avgs = _N.zeros((len(partIDs), SHUFFLES+1, t1-t0))
l_all_avgs = []
netwins  = _N.empty(len(partIDs), dtype=_N.int32)
gk = _Am.gauKer(1)
gk /= _N.sum(gk)
gk2 = _Am.gauKer(2)
gk2 /= _N.sum(gk2)
#gk2 = None

#gk = None

pid = 0

ts  = _N.arange(t0-2, t1-2)
signal_5_95 = _N.zeros((len(partIDs), 4, t1-t0))

hnd_dat_all = _N.zeros((len(partIDs), TO, 4), dtype=_N.int32)

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
rule_changes_in_context = _N.zeros((len(partIDs), TO), dtype=_N.int32)
detected_rule_changes_in_context = _N.zeros((len(partIDs), TO), dtype=_N.int32)

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
strtend      = _N.zeros(len(partIDs)+1, dtype=_N.int32)

incomplete_data = []
gkISI = _Am.gauKer(1)
gkISI /= _N.sum(gkISI)

#  DISPLAYED AS R,S,P
#  look for RR RS RP
#  look for SR SS SP
#  look for PR PS PP

big_percentile=0.985
#big_percentile=0.5
min_big_comps  =2

rc_trg_avg = _N.empty((len(partIDs), t1-t0, SHUFFLES+1))
rc_trg_avg_RPS = _N.empty((len(partIDs), t1-t0, SHUFFLES+1))
rc_trg_avg_DSURPS = _N.empty((len(partIDs), t1-t0, SHUFFLES+1))

chg = _N.empty(len(partIDs))

n_maxes   = _N.zeros((len(partIDs), SHUFFLES+1), dtype=_N.int32)

# mdl, SHUFFLES, cond, act
stds        = _N.zeros((len(partIDs), 3, SHUFFLES+1, 3, 3, ))
# mdl, 1st hlf, 2nd hlf, SHUFFLES cond, act
stds12      = _N.zeros((len(partIDs), 3, 2, SHUFFLES+1, 3, 3))

thrs = _N.empty(len(partIDs), dtype=_N.int32)
#stds      = _N.zeros((len(partIDs), 3, SHUFFLES+1))
#stdsDSUWTL      = _N.zeros((len(partIDs), 3, 3, 3, SHUFFLES+1))
#stdsRPSWTL      = _N.zeros((len(partIDs), 3, 3, 3, SHUFFLES+1))
#stdsDSURPS      = _N.zeros((len(partIDs), 3, 3, 3, SHUFFLES+1))

winlosses       = _N.empty((len(partIDs), 2))
marginalCRs = _N.empty((len(partIDs), SHUFFLES, 3, 3))

# sum_sd_DSUWTL = rebuild_sds_array(len(partIDs), lm, "sum_sd_DSUWTL")
# sum_sd_RPSWTL = rebuild_sds_array(len(partIDs), lm, "sum_sd_RPSWTL")
# sum_sd_DSUAIRPS = rebuild_sds_array(len(partIDs), lm, "sum_sd_DSUAIRPS")

sFlipped  = "_flipped" if flip_HUMAI else ""
lm = depickle(workdirFN("shuffledCRs_5CFs%(flp)s_%(ex)s_%(w)d_%(v)d_%(vs)s" % {"ex" : expt, "w" : win, "v" : visit, "vs" : svisits, "flp" : sFlipped}))
ranks_of_cmps = lm["fr_cmp_fluc_rank2"]
cmpZs = lm["z1s"]
ranks_of_lotsof0s = lm["fr_lotsof0s"]
#ranks_of_lotsof0s = 0.5*(lm["fr_clumped0s"]+lm["fr_lotsof0s"])
ranks_of_lotsof1s = lm["fr_lotsof1s"]
len1s = lm["len1s"]
#cv_onrule_rank = lm["cv_onrule_rank"]
#cv_offrule_rank = lm["cv_offrule_rank"]

has_nonzero_CR_comps = _N.zeros(len(partIDs), dtype=_N.int32)
n_CR_comps = _N.zeros(len(partIDs), dtype=_N.int32)

lags = 12

decr1 = []  #  start from end of ON, trace until start of OFF
incr1 = []  #  start from start of ON, trace back until end of OFF
decr2 = []  #  start from start of OFF, trace until end of ON
incr2 = []  #  start from end of OFF, trace until start of ON
ccs0s1s = _N.empty(len(partIDs))

nRuleChanges = _N.ones(len(partIDs), dtype=_N.int32)
ICIs  = _N.ones(len(partIDs)) *-1   #  inter-change-interval
#for partID in partIDs[0:25]:
pid = 0
for partID in partIDs:
    pid += 1

    dmp       = depickle(workdirFN("%(rpsm)s/%(lb)d/variousCRs%(flp)s_%(v)d.dmp" % {"rpsm" : partID, "lb" : label, "v" : visit, "flp" : sFlipped}))
    
    # if expt == "TMB2":
    #     AQ28scrs[pid-1], soc_skils[pid-1], rout[pid-1], switch[pid-1], imag[pid-1], fact_pat[pid-1] = _rt.AQ28(datadirFN("%(data)s/%(date)s/%(pID)s/AQ29.txt" % {"date" : partIDs[pid-1][0:8], "pID" : partIDs[pid-1], "data" : expt}))

    _prob_mvsDSUWTL = dmp["cond_probsDSUWTL"][:, :, strtTr:]
    _prob_mvsRPSWTL = dmp["cond_probsRPSWTL"][:, strtTr:]
    _prob_mvsDSURPS = dmp["cond_probsDSURPS"][:, strtTr:]
    _prob_mvsDSUAIRPS = dmp["cond_probsDSUAIRPS"][:, strtTr:]
    #_prob_mvsRPSRPS = dmp["cond_probsRPSRPS"][:, strtTr:]
    #_prob_mvsRPSAIRPS = dmp["cond_probsRPSAIRPS"][:, strtTr:]
    _prob_mvsLCBRPS = dmp["cond_probsLCBAIRPS"][:, strtTr:]
    _prob_mvsLCBAIRPS = dmp["cond_probsLCBRPS"][:, strtTr:]        
    
    prob_mvsDSUWTL  = _prob_mvsDSUWTL[:, :, 0:TO - win]  #  is bigger than hand by win size
    prob_mvsRPSWTL  = _prob_mvsRPSWTL[:, :, 0:TO - win]  #  is bigger than hand by win size
    prob_mvsDSURPS  = _prob_mvsDSURPS[:, :, 0:TO - win]  #  is bigger than hand by win size
    prob_mvsDSUAIRPS  = _prob_mvsDSUAIRPS[:, :, 0:TO - win]  #  is bigger than hand by win size    
    #prob_mvsRPSRPS  = _prob_mvsRPSRPS[:, :, 0:TO - win]  #  is bigger than hand by win size
    #prob_mvsRPSAIRPS  = _prob_mvsRPSAIRPS[:, :, 0:TO - win]  #  is bigger than hand by win size    
    prob_mvsLCBRPS  = _prob_mvsLCBRPS[:, :, 0:TO - win]  #  is bigger than hand by win size
    prob_mvsLCBAIRPS  = _prob_mvsLCBAIRPS[:, :, 0:TO - win]  #  is bigger than hand by win size    

    #stds_all_mdls[0] = _N.std(prob_mvs, axis=2)

    td0, start_time0, end_time0, UA0, cnstr0, inp_meth0, ini_percep0, fin_percep0, gt_dump0 = _rt.return_hnd_dat(partID, has_useragent=True, has_start_and_end_times=True, has_constructor=True, expt=expt, visit=visit, know_gt=know_gt)

    
    decr = []
    incr = []    
    for SHF_NUM in range(SHUFFLES+1):
    #for SHF_NUM in range(70, 71):
        _prob_mvsDSUWTL = dmp["cond_probsDSUWTL"][SHF_NUM][:, strtTr:]
        _prob_mvsRPSWTL = dmp["cond_probsRPSWTL"][SHF_NUM][:, strtTr:]
        _prob_mvsDSURPS = dmp["cond_probsDSURPS"][SHF_NUM][:, strtTr:]
        _prob_mvsDSUAIRPS = dmp["cond_probsDSUAIRPS"][SHF_NUM][:, strtTr:]
        #_prob_mvsRPSRPS = dmp["cond_probsRPSRPS"][SHF_NUM][:, strtTr:]
        #_prob_mvsRPSAIRPS = dmp["cond_probsRPSAIRPS"][SHF_NUM][:, strtTr:]
        _prob_mvsLCBRPS = dmp["cond_probsLCBRPS"][SHF_NUM][:, strtTr:]
        _prob_mvsLCBAIRPS = dmp["cond_probsLCBAIRPS"][SHF_NUM][:, strtTr:]

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
        #prob_mvsRPSRPS  = _prob_mvsRPSRPS[:, 0:TO - win]  #  is bigger than hand by win size
        #prob_mvsRPSAIRPS  = _prob_mvsRPSAIRPS[:, 0:TO - win]  #  is bigger than hand by win size
        prob_mvsLCBRPS  = _prob_mvsLCBRPS[:, 0:TO - win]  #  is bigger than hand by win size
        prob_mvsLCBAIRPS  = _prob_mvsLCBAIRPS[:, 0:TO - win]  #  is bigger than hand by win size                        
        
        #prob_mvs_STSW  = _prob_mvs_STSW[:, 0:TO - win]  #  is bigger than hand by win size    
        prob_mvsDSUWTL = prob_mvsDSUWTL.reshape((3, 3, prob_mvsDSUWTL.shape[1]))
        prob_mvsRPSWTL = prob_mvsRPSWTL.reshape((3, 3, prob_mvsRPSWTL.shape[1]))
        prob_mvsDSURPS = prob_mvsDSURPS.reshape((3, 3, prob_mvsDSURPS.shape[1]))
        prob_mvsDSUAIRPS = prob_mvsDSUAIRPS.reshape((3, 3, prob_mvsDSUAIRPS.shape[1]))        
        #prob_mvsRPSRPS = prob_mvsRPSRPS.reshape((3, 3, prob_mvsRPSRPS.shape[1]))
        #prob_mvsRPSAIRPS = prob_mvsRPSAIRPS.reshape((3, 3, prob_mvsRPSAIRPS.shape[1]))        
        prob_mvsLCBRPS = prob_mvsLCBRPS.reshape((3, 3, prob_mvsLCBRPS.shape[1]))
        prob_mvsLCBAIRPS = prob_mvsLCBAIRPS.reshape((3, 3, prob_mvsLCBAIRPS.shape[1]))        

        #all_frmwks = [prob_mvsDSUWTL, prob_mvsRPSWTL, prob_mvsRPSRPS, prob_mvsDSUAIRPS, prob_mvsRPSAIRPS, prob_mvsLCBRPS]
        all_frmwks = [prob_mvsDSUWTL, prob_mvsRPSWTL,
                      prob_mvsDSURPS, prob_mvsLCBRPS,
                      #prob_mvsRPSRPS, prob_mvsLCBRPS,
                      prob_mvsDSUAIRPS, prob_mvsLCBAIRPS]
        #marginalCRs[pid-1, SHF_NUM] = _emp.marginalCR(_hnd_dat)
        N = prob_mvsDSUWTL.shape[2]

        #behv, rules, nCR   = _crut.get_dbehv_biggest_fluc_seprules(all_frmwks, [2, 3, 4, 5], gk2, ranks_of_cmps[pid-1], ranks_of_lotsof0s[pid-1], len1s[pid-1], big_percentile=big_percentile, min_big_comps=min_big_comps, flip_choose_components=False)
        #behv, rules, n_CR_comps[pid-1], lohis   = _crut.get_dbehv_biggest_fluc_seprules(all_frmwks, [2, 3, 4, 5], gk2, ranks_of_lotsof0s[pid-1], ranks_of_lotsof1s[pid-1], big_percentile=big_percentile, min_big_comps=min_big_comps, flip_choose_components=False)
        #behv, rules, n_CR_comps[pid-1]   = _crut.get_dbehv_biggest_fluc_seprules(all_frmwks, [2, 3, 4, 5], gk2, ranks_of_cmps[pid-1], ranks_of_lotsof0s[pid-1], len1s[pid-1], big_percentile=big_percentile, min_big_comps=min_big_comps, flip_choose_components=False)        
        #behv, rules, n_CR_comps[pid-1]   = _crut.get_dbehv_biggest_fluc_seprules(all_frmwks, [0, 1, 2, 3, 4, 5], gk2, ranks_of_cmps[pid-1], ranks_of_lotsof0s[pid-1], len1s[pid-1], big_percentile=big_percentile, min_big_comps=min_big_comps, flip_choose_components=False)
        behv, rules, n_CR_comps[pid-1], lohis   = _crut.get_dbehv_biggest_fluc_seprules(all_frmwks, [0, 1, 2, 3, 4, 5], gk2, ranks_of_lotsof0s[pid-1], ranks_of_lotsof1s[pid-1], big_percentile=big_percentile, min_big_comps=min_big_comps, flip_choose_components=False, SHUFFLES=400)
        print(behv)
        #behv, rules, n_CR_comps[pid-1]   = _crut.get_dbehv_biggest_fluc_seprules(all_frmwks, [0, 1, 2, 3, 4, 5], gk2, ranks_of_cmps[pid-1], ranks_of_lotsof0s[pid-1], len1s[pid-1], big_percentile=big_percentile, min_big_comps=min_big_comps, flip_choose_components=False)

        ccs0s1s[pid-1], pv = _ss.pearsonr(ranks_of_lotsof0s[pid-1].flatten(), ranks_of_lotsof1s[pid-1].flatten())
        if behv is not None:
            
            #dbehv  = _N.diff(_N.convolve(behv, gk, mode="same")) #+ _N.diff(behv))
            """ 
            dbehv  = _N.diff(behv)
            dbehv5  = _N.diff(behv5)            
            maxs = _N.where((dbehv[0:TO-11] >= 0) & (dbehv[1:TO-10] < 0))[0] + (win//2)#  3 from label71
            maxs5 = _N.where((dbehv5[0:TO-11] >= 0) & (dbehv5[1:TO-10] < 0))[0] + (win//2)#  3 from label71
            """

            #  behv is |dP/dt|  

            RCtms = []
            RCtmsM = []            
            
            for ib in range(len(behv)):
                #  detect when ON/OFF periods
                # if lohis[ib] == 0:
                #     hiTh = 0.6
                #     loTh = 0.05
                # elif lohis[ib] == 1:
                #     hiTh = 0.95
                #     loTh = 0.3
                # else:
                #     hiTh = 0.9
                #     loTh = 0.1

                hiTh = 0.8
                loTh = 0.2
                if lohis[ib] == 0:
                    ONrounds  = _N.where(rules[ib] > hiTh)[0]
                    OFFrounds = _N.where(rules[ib] < loTh)[0]
                    
                ONrounds  = _N.where(rules[ib] > hiTh)[0]
                OFFrounds = _N.where(rules[ib] < loTh)[0]

                ONend  = ONrounds[_N.where(_N.diff(ONrounds) > 1)[0]]
                ONstrt  = ONrounds[_N.where(_N.diff(ONrounds) > 1)[0]+1]
                if len(ONrounds) > 0:
                    if ONrounds[0] > 0:
                        ONstrt = _N.array([ONrounds[0]] + ONstrt.tolist())
                    if ONrounds[-1] < len(rules[ib]):
                        ONend  = _N.array(ONend.tolist() + [ONrounds[-1]])

                OFFend  = OFFrounds[_N.where(_N.diff(OFFrounds) > 1)[0]]
                OFFstrt  = OFFrounds[_N.where(_N.diff(OFFrounds) > 1)[0]+1]

                if len(OFFrounds) > 0:
                    if OFFrounds[0] > 0:
                        OFFstrt = _N.array([OFFrounds[0]] + OFFstrt.tolist())
                    if OFFrounds[-1] < len(rules[ib]):
                        OFFend  = _N.array(OFFend.tolist() + [OFFrounds[-1]])

                """
                fig = _plt.figure(figsize=(4, 7))
                fig.add_subplot(2, 1, 1)
                _plt.plot(rules[0])
                for strt in ONstrt:                       
                    _plt.plot([strt, strt], [0, 1], color="black")
                for end in ONend:                       
                    _plt.plot([end, end], [0, 1], color="blue")
                fig.add_subplot(2, 1, 2)
                _plt.plot(rules[0])
                for strt in OFFstrt:
                    _plt.plot([strt, strt], [0, 1], color="black")
                for end in OFFend:                       
                    _plt.plot([end, end], [0, 1], color="blue")
                """
                #  rule change timing
                #  for the ONs, go back in time, does the rule

                radius = 18
                for strt in ONstrt:
                    #  Look at start of ON state, trace back to OFF state
                    fnd = False
                    si = -1
                    while (not fnd) and (si < radius) and (strt - si > 0):
                        si += 1
                        if rules[ib][strt-si] < loTh+0.05:
                            fnd = True
                            RCtms.append(strt-si)
                            RCtmsM.append(strt-si//2)
                            incr1.append(rules[ib][strt-si:strt])

                for end in ONend:
                    fnd = False
                    ei = -1
                    while (not fnd) and (ei < radius) and (end + ei < len(rules[ib])-1):
                        ei += 1
                        if rules[ib][end+ei] < loTh+0.05:
                            fnd = True
                            RCtms.append(end)
                            RCtmsM.append(end+ei//2)
                            decr1.append(rules[ib][end:end+ei])

                for strt in OFFstrt:
                    fnd = False
                    si = -1
                    while (not fnd) and (si < radius) and (strt - si > 0):
                        si += 1
                        if rules[ib][strt-si] > hiTh-0.05:
                            fnd = True
                            RCtms.append(strt-si)
                            RCtmsM.append(strt-si//2)
                            decr2.append(rules[ib][strt-si:strt])

                for end in OFFend:
                    fnd = False
                    ei = -1
                    while (not fnd) and (ei < radius) and (end + ei < len(rules[ib])-1):
                        ei += 1
                        if rules[ib][end+ei] > hiTh-0.05:
                            fnd = True
                            RCtms.append(end)
                            RCtmsM.append(end+ei//2)
                            incr2.append(rules[ib][end:end+ei])

            maxsM = _N.sort(_N.unique(RCtmsM))  #  mid-point of rule change 
            maxs  = _N.sort(_N.unique(RCtms))   #  start of rule change
            maxsUse = maxs

            #####  for ib in range(len(behv)):
            #maxs = _N.unique(_N.sort(maxs_from_comps))
            maxs5= maxsUse
            nRuleChanges[pid-1] = len(maxs5)

            if len(maxs5) > 0:
                detected_rule_changes_in_context[pid-1, maxs5] = 1

            if know_gt:
                rch01  = _N.zeros(TO, dtype=_N.int32)
                rch01[gt_dump0["rule_change_times"]]= 1
                rule_changes_in_context[pid-1, gt_dump0["rule_change_times"]] = 1
                #maxs = gt_dump0["rule_change_times"]  #  trigger on GT rule-change times
                for im in range(1, len(maxs5)-1):
                    iRCT = maxs5[im]
                    if (iRCT-lags > 0) and (iRCT+lags) < TO:
                        detected_rulechange_triggered.append(rch01[iRCT-lags:iRCT+lags+1])
            
            for sh in range(1):
                if sh > 0:
                    _N.random.shuffle(inds)
                hnd_dat = _hnd_dat[inds]

                if len(maxsUse) > 2*cut:
                    #print(maxs)

                    okCPs = []
                    for im in range(cut, len(maxsUse)-cut):
                        if (maxsUse[im]+t0 >= 0) and (maxsUse[im]+t1 < hnd_dat.shape[0]):
                            okCPs.append(im)
                    avgs = _N.zeros((len(okCPs), t1-t0))
                    ICIs[pid-1] = len(okCPs)

                    #for im in range(cut, len(maxsUse)-cut):
                    for im in okCPs:
                        #print(hnd_dat[maxs[im]+t0:maxs[im]+t1, 2].shape)
                        #print("%(1)d %(2)d" % {"1" : maxs[im]+t0, "2" : maxs[im]+t1})
                        st = 0
                        en = t1-t0
                        if maxsUse[im] + t0 < 0:   #  just don't use this one
                            print("DON'T USE THIS ONE")
                            avgs[im-1, :] = 0
                        else:
                            try:
                                avgs[im-okCPs[0], :] = hnd_dat[maxsUse[im]+t0:maxsUse[im]+t1, 2]
                                if len(_N.where((pid-1) == filtdat)[0]) == 1:  #  in filtdat
                                    #print("-----  looking for hnd_dat[%(1)d:%(2)d], 2]" % {"1" : maxsUse[im]+t0, "2" : maxsUse[im]+t1})
                                    pidInFiltdat = _N.intersect1d(filtdat, pid-1)
                                    if len(pidInFiltdat) > 0:
                                        l_all_avgs.append(hnd_dat[maxsUse[im]+t0:maxsUse[im]+t1, 2])
                            except ValueError:   #  trigger lags past end of games
                                print("-----  looking for hnd_dat[%(1)d:%(2)d], 2]" % {"1" : maxsUse[im]+t0, "2" : maxsUse[im]+t1})
                                #print(avgs[im-1, :].shape)
                                #print(hnd_dat[maxs[im]+t0:maxs[im]+t1, 2])
                                avgs[im-1, :] = 0

                    all_avgs[pid-1, sh] = _N.mean(avgs, axis=0)  #  trigger average
                    has_nonzero_CR_comps[pid-1] = len(maxsUse)-2*cut
                    
                    if _N.sum(_N.isnan(all_avgs[pid-1, sh])):
                        print("ISNAN   %(pid)d   %(sh)d" % {"sh" : SHF_NUM, "pid" : (pid-1)})
                        #print("..........    %d" % _N.sum(_N.isnan(prob_mvs)))
                                

            #print("............................................")
            #print(workdirFN("%(pID)s/RuleChanges_%(visit)d.dmp" % {"pID" : partID, "visit" : visit}))
            dmp = open(workdirFN("%(pID)s/RuleChanges_%(visit)d.dmp" % {"pID" : partID, "visit" : visit}), "wb")
            pickle.dump([maxs, maxsUse, td0], dmp, -1)
            dmp.close()
        #if behv is not None:            
    # fig = _plt.figure()
    # sss = _N.zeros(TO)
    # sss[maxs] = 1
    # _plt.acorr(sss - _N.mean(sss), maxlags=20, usevlines=False, linestyle="-")
    # _plt.axhline(y=0, ls=":", color="black")

            

######  has_nonzero_CR_comps   is number of rule changes during game
nzfiltdat = _N.intersect1d(_N.where(has_nonzero_CR_comps > 3)[0], filtdat)
#nzfiltdat = _N.intersect1d(_N.where(has_nonzero_CR_comps > 2)[0], filtdat)

fig = _plt.figure(figsize=(5, 3.3))
# for inz in range(len(nzfiltdat)):
#     _plt.plot(all_avgs[nzfiltdat[inz], 0] - _N.mean(all_avgs[nzfiltdat[inz], 0]), color="grey")
ts = _N.arange(-(t1-t0)//2+1, (t1-t0)//2+1)
#mnsig = _N.mean(all_avgs[nzfiltdat, 0], axis=0)
all_avgs_each_trigger = _N.array(l_all_avgs)
mnsig_et = _N.mean(all_avgs_each_trigger, axis=0)
#sdsig_et = _N.std(all_avgs_each_trigger, axis=0)

mnlevel = _N.round(_N.mean(mnsig_et)*100)/100


_plt.suptitle("frac games %.2f" % (len(nzfiltdat) / len(filtdat)))
#_plt.plot(ts, mnsig, color="black", lw=3)
_plt.plot(ts, mnsig_et, color="black", lw=3)
#_plt.plot(ts, mnsig_et-sdsig_et, color="black", lw=3)
#_plt.plot(ts, mnsig_et+sdsig_et, color="black", lw=3)
#if expt == "SIMHUM1":
#_plt.ylim(-0.06, 0.06)
#elif expt== "SIMHUM6":
#_plt.ylim(-0.12, 0.)
_plt.xticks([-10, -5, 0, 5, 10], ["-10", "-5", "0", "+5", "+10"])
_plt.ylim(mnlevel-0.11, mnlevel+0.11)
#_plt.ylim(mnlevel-0.3, mnlevel+0.3)
#_plt.ylim(mnlevel-0.1, mnlevel+0.12)
_plt.yticks([mnlevel-0.1, mnlevel, mnlevel+0.1])
_N.savetxt("Rule-change_%(exp)s_%(w)d" % {"exp" : expt, "w" : win}, mnsig_et, fmt="%.5f")
    
_plt.grid()
_plt.axvline(x=0, ls="--", color="grey", lw=4)
_plt.xlabel("lag from detected rule change (rounds)", fontsize=15)
_plt.xticks(fontsize=13)
_plt.yticks(fontsize=13)
_plt.xlim(-lags, lags)
_plt.ylabel("mean win - lose rate", fontsize=15)
fig.subplots_adjust(bottom=0.2, left=0.2, right=0.98)
#_plt.ylim(-0.12, 0.04)
#_plt.ylim(-0.2, -0.04)
#_plt.ylim(-0.3, -0.04)
_plt.savefig("Rule-change_%(exp)s_%(w)d" % {"exp" : expt, "w" : win}, transparent=True)

if know_gt:
    A = _N.array(detected_rulechange_triggered)
    num_GT_RCs = _N.sum(A, axis=0)
    fig = _plt.figure(figsize=(5, 3.7))
    #fig = _plt.figure(figsize=(9, 4))    
    fig.add_subplot(1, 1, 1)
    _plt.bar(_N.arange(-lags, lags+1), num_GT_RCs, color="black")
    _plt.axvline(x=0, ls="--")
    _plt.xlabel("lag from actual rulechange (rounds)", fontsize=15)
    _plt.ylabel("# detected rulechanges at lag", fontsize=15)
    _plt.xticks([-10, -5, 0, 5, 10], ["-10", "-5", "0", "+5", "+10"], fontsize=13)
    _plt.yticks(fontsize=13)        
    _plt.ylim(0, _N.max(num_GT_RCs)*1.1)
    _plt.xlim(-lags - .5, lags + .5)
    fig.subplots_adjust(bottom=0.2, right=0.98, left=0.2)
    _plt.savefig("Rule-change_detect_%(exp)s_%(w)d" % {"exp" : expt, "w" : win}, transparent=False)
    fig = _plt.figure(figsize=(13, 3.))
    ax  = fig.add_subplot(1, 1, 1)
    ax.set_facecolor("#BBBBBB")
    for ig in range(1):
        nRCts = _N.where(rule_changes_in_context[ig] == 1)[0]
        nDRCts = _N.where(detected_rule_changes_in_context[ig] == 1)[0]
        for i in range(len(nRCts)):
            _plt.plot([nRCts[i], nRCts[i]], [ig, ig+0.8], color="black", lw=6)
        for i in range(len(nDRCts)):
            _plt.plot([nDRCts[i], nDRCts[i]], [ig+.2, ig+0.6], color="white", lw=4)
    _plt.xlim(-0.5, TO + .5)
    _plt.xlabel("game #", fontsize=22)
    #_plt.ylabel("tournament #", fontsize=18)
    #_plt.yticks([0.5, 1.5, 2.5, 3.5], [1, 2, 3, 4], fontsize=16)
    #_plt.yticks([0.5, 1.5, 2.5, 3.5], [1], fontsize=16)
    _plt.yticks([])
    _plt.xticks(fontsize=19)
    fig.subplots_adjust(bottom=0.26, left=0.05, right=0.97)
    _plt.savefig("Rule-change_in_context_%(exp)s_%(w)d" % {"exp" : expt, "w" : win}, transparent=False)

both01s = _N.empty(len(partIDs), dtype=_N.int32)
for i in range(len(partIDs)):
    both01s[i] = len(_N.where((ranks_of_lotsof0s[i].flatten() > 200) & (ranks_of_lotsof1s[i].flatten() > 200))[0])

icf1 = -1

# #probslist = [prob_mvsDSUWTL, prob_mvsRPSWTL, prob_mvsRPSRPS, prob_mvsLCBRPS, prob_mvsDSUAIRPS, prob_mvsLCBAIRPS]
# probslist = [prob_mvsDSUWTL, prob_mvsRPSWTL, prob_mvsDSURPS, prob_mvsLCBRPS, prob_mvsDSUAIRPS, prob_mvsLCBAIRPS]
# for icf1 in range(6):
#     cf1 = probslist[icf1]
#     for icf2 in range(icf1+1, 6):
#         cf2 = probslist[icf2]        
#         if icf1 != icf2:
#             for ic1 in range(3):
#                 for ia1 in range(3):    
#                     for ic2 in range(3):
#                         for ia2 in range(3):    
#                             pc, pv = _ss.pearsonr(cf1[ic1, ia1], cf2[ic2, ia2])
#                             if _N.abs(pc) > 0.7:
#                                 print("%(1)d  %(2)d" % {"1" : icf1, "2" : icf2})
#                                 print(pc)

# gt0 = _N.where(ICIs > 0)[0]
# these = _N.intersect1d(gt0, filtdat)
# for tar in [soc_skils, imag, rout, switch, fact_pat, AQ28scrs]:
#     pc, pv = _ss.pearsonr(tar[these], ICIs[these])
#     print(pc)
#print("expt=%(e)s   mean # of rule changes %(mnr).1f   std # of rule changes %(snr).1f" % {"mnr" : _N.mean(nRuleChanges[filtdat]), "snr" : _N.std(nRuleChanges[filtdat]), "e" : expt})

fig = _plt.figure(figsize=(5, 3.3))
# for inz in range(len(nzfiltdat)):
#     _plt.plot(all_avgs[nzfiltdat[inz], 0] - _N.mean(all_avgs[nzfiltdat[inz], 0]), color="grey")
ts = _N.arange(-(t1-t0)//2+1, (t1-t0)//2+1)
mnSTNWs = _N.mean(hnd_dat_all[:, :, 2], axis=1)
mnSTNWs = mnSTNWs.reshape((mnSTNWs.shape[0],1))
#for ig in nzfiltdat:
#    _plt.plot(all_avgs[ig, 0] - mnSTNWs, color="grey")
zrmn_all_avgs = all_avgs[nzfiltdat, 0] - mnSTNWs[nzfiltdat]
srtd_zrmn_all_avgs = _N.sort(zrmn_all_avgs, axis=0)
pct5     = srtd_zrmn_all_avgs[int(len(nzfiltdat)*0.05)]
pct95    = srtd_zrmn_all_avgs[int(len(nzfiltdat)*0.95)]
mn_zrmn = _N.mean(zrmn_all_avgs, axis=0)
#sd_zrmn = _N.std(zrmn_all_avgs, axis=0)
_plt.fill_between(ts, pct5, pct95, color="#CCCCFF", alpha=0.3)
_plt.plot(ts, mn_zrmn, color="black", lw=3)
_plt.xticks([-10, -5, 0, 5, 10], ["-10", "-5", "0", "+5", "+10"], fontsize=13)
_plt.yticks(fontsize=13)
_plt.axvline(x=0, ls="--", color="grey", lw=4)
_plt.ylim(-0.5, 0.5)
_plt.xlim(-lags, lags)
_plt.xlabel("lag from detected rule change (rounds)", fontsize=15)
_plt.ylabel("(W-L) - <W-L>, per game", fontsize=15)
fig.subplots_adjust(bottom=0.2, left=0.2, right=0.98)
_plt.savefig("RPS_rulechange_bygame_%s" % expt)
