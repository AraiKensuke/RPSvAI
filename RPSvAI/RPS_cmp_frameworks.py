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
#from sumojam.devscripts.cmdlineargs import process_keyval_args
import pickle
import mne.time_frequency as mtf
import GCoh.eeg_util as _eu
#import RPSvAI.rpsms as rpsms
import GCoh.preprocess_ver as _ppv

#from RPSvAI.utils.dir_util import getResultFN
import GCoh.datconfig as datconf
import RPSvAI.models.CRutils as _crut
import RPSvAI.models.empirical_ken as _emp
from sklearn.decomposition import PCA
import RPSvAI.AIRPSfeatures as _aift

import GCoh.eeg_util as _eu
import matplotlib.ticker as ticker
#from statsmodels import robust

flip_HUMAI=False
#_plt.ioff()
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

        sFlipped  = "_flipped" if flip_HUMAI else ""
        dmp       = depickle(workdirFN("%(rpsm)s/%(lb)d/variousCRs%(flp)s_%(visit)d.dmp" % {"rpsm" : partID, "lb" : label, "visit" : visit, "flp" : sFlipped}))
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
    print("len of partIDs %d" % len(partIDs))
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
#process_keyval_args(globals(), sys.argv[1:])   #  For when we run from cmd line

visit = 1
visits= [1, ]   #  if I want 1 of [1, 2], set this one to [1, 2]

A1 = []
show_shuffled = False
#process_keyval_args(globals(), sys.argv[1:])
#######################################################

win_type = 2   #  window is of fixed number of games
#win_type = 1  #  window is of fixed number of games that meet condition 
win     = 3
smth    = 1
label          = win_type*100+win*10+smth
TO = 300
SHF_NUM = 0

expt = "TMB2"

svisits =str(visits).replace(" ", "").replace("[", "").replace("]", "")
filtdat_okgames = None
if expt == "TMB2":
    lm = depickle(workdirFN("TMB2_AQ28_vs_RPS_features_%(v)d_of_%(vs)s_%(wt)d%(w)d%(s)d.dmp" % {"v" : visit, "wt" : win_type, "w" : win, "s" : smth, "wd" : os.environ["RPSWORKDIR"], "vs" : svisits}))
    #lm = depickle("predictAQ28dat/AQ28_vs_RPS_1_%(wt)d%(w)d%(s)d.dmp" % {"wt" : win_type, "w" : win, "s" : smth})
    partIDs_okgames = lm["partIDs"]
    partIDs         = lm["partIDs"]
    filtdat_okgames = lm["filtdat"]
    TO = 300
if expt == "TMBCW":
    lm = depickle(workdirFN("TMBCW_AQ28_vs_RPS_features_%(v)d_of_%(vs)s_%(wt)d%(w)d%(s)d.dmp" % {"v" : visit, "wt" : win_type, "w" : win, "s" : smth, "wd" : os.environ["RPSWORKDIR"], "vs" : svisits}))
    #lm = depickle("predictAQ28dat/AQ28_vs_RPS_1_%(wt)d%(w)d%(s)d.dmp" % {"wt" : win_type, "w" : win, "s" : smth})
    partIDs_okgames = lm["partIDs_okgames"]
    partIDs         = lm["partIDs_okgames"]
    filtdat_okgames = lm["filtdat_okgames"]
    TO = 300

filtdat = lm["filtdat"]


#filtdat = _N.array([8])
    
partIDs, incmp_dat = only_complete_data(partIDs, TO, label, SHF_NUM)
strtTr=0
TO -= strtTr

#fig= _plt.figure(figsize=(14, 14))

#SHUFFLES = 205
SHUFFLES = 400
extra_w = 2
t0 = -5 - extra_w
t1 = 10 + extra_w
cut = 3
all_avgs = _N.empty((len(partIDs), SHUFFLES+1, t1-t0))
l_all_avgs = []
netwins  = _N.empty(len(partIDs), dtype=_N.int32)
gk = _Am.gauKer(1)
gk /= _N.sum(gk)
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

L30  = 30

rc_trg_avg = _N.empty((len(partIDs), t1-t0, SHUFFLES+1))
rc_trg_avg_RPS = _N.empty((len(partIDs), t1-t0, SHUFFLES+1))
#rc_trg_avg_DSURPS = _N.empty((len(partIDs), t1-t0, SHUFFLES+1))

chg = _N.empty(len(partIDs))

n_maxes   = _N.zeros((len(partIDs), SHUFFLES+1), dtype=_N.int32)

# mdl, SHUFFLES, cond, act
stds        = _N.zeros((len(partIDs), 3, SHUFFLES+1, 3, 3, ))
# mdl, 1st hlf, 2nd hlf, SHUFFLES cond, act
stds12      = _N.zeros((len(partIDs), 3, 2, SHUFFLES+1, 3, 3))

thrs = _N.empty(len(partIDs), dtype=_N.int32)

marginalCRs = _N.empty((len(partIDs), SHUFFLES, 3, 3))

frameworks   = ["DSUWTL", "RPSWTL",
                "DSURPS", "LCBRPS",
                "DSUAIRPS", "LCBAIRPS"]
frameworks_p = ["p(DSU | WTL)", "p(RPS | WTL)",
                "p(DSU | RPS)", "p(LCB | RPS)",
                "p(DSU | AI_RPS)", "p(LCB | AI_RPS)"]

nFrameworks= 6
par        = _N.empty((len(partIDs), nFrameworks))
pick_acts        = _N.empty((len(partIDs), nFrameworks))
avoid_acts        = _N.empty((len(partIDs), nFrameworks))

z1s        = _N.empty((len(partIDs), nFrameworks))
z0s        = _N.empty((len(partIDs), nFrameworks))
rank       = _N.empty((len(partIDs), nFrameworks), dtype=_N.int32)
cmp_z1s    = _N.empty((len(partIDs), nFrameworks, 3, 3))
cmp_z1sZ   = _N.empty((len(partIDs), nFrameworks, 3, 3))
fr_cmp_fluc_rank1       = _N.empty((len(partIDs), nFrameworks, 3, 3), dtype=_N.int32)
cv_onrule_rank       = _N.empty((len(partIDs), nFrameworks, 3, 3), dtype=_N.int32)
cv_offrule_rank       = _N.empty((len(partIDs), nFrameworks, 3, 3), dtype=_N.int32)

#fr_cmp_fluc_rank2       = _N.empty((len(partIDs), nFrameworks, 3, 3), dtype=_N.int)
fr_cmp_fluc_rank2       = _N.empty((len(partIDs), nFrameworks, 3, 3))
fr_lotsof0s       = _N.empty((len(partIDs), nFrameworks, 3, 3), dtype=_N.int32)
fr_lotsof1s       = _N.empty((len(partIDs), nFrameworks, 3, 3), dtype=_N.int32)
fr_lotsof1zs       = _N.empty((len(partIDs), nFrameworks, 3, 3))

fr_clumped0s       = _N.empty((len(partIDs), nFrameworks, 3, 3), dtype=_N.int32)
fr_clumped1s       = _N.empty((len(partIDs), nFrameworks, 3, 3), dtype=_N.int32)

fr_cmp_rank0       = _N.empty((len(partIDs), nFrameworks, 3, 3), dtype=_N.int32)
fr_cmp_lohi_rank       = _N.zeros((len(partIDs), nFrameworks, 3, 3), dtype=_N.int32)
len1s       = _N.zeros((len(partIDs), nFrameworks, 3, 3), dtype=_N.int32)

CCs        = _N.empty((len(partIDs), nFrameworks, nFrameworks))

hi_allpcs     = []
mid_allpcs     = []
lo_allpcs     = []
iex = 1
pid=0
for partID in partIDs:#_okgames:
    pid += 1
    #print(pid)
    sFlipped  = "_flipped" if flip_HUMAI else ""
    dmp       = depickle(workdirFN("%(rpsm)s/%(lb)d/variousCRs%(flp)s_%(v)d.dmp" % {"rpsm" : partID, "lb" : label, "v" : visit, "flp" : sFlipped}))

    # if expt == "TMB2":
    #     AQ28scrs[pid-1], soc_skils[pid-1], rout[pid-1], switch[pid-1], imag[pid-1], fact_pat[pid-1] = _rt.AQ28(datadirFN("%(data)s/%(date)s/%(pID)s/AQ29.txt" % {"date" : partIDs[pid-1][0:8], "pID" : partIDs[pid-1], "data" : expt}))

    N = TO - win
    near1DSUWTL = _N.zeros((SHUFFLES+1, N), dtype=_N.int32)
    near1RPSWTL = _N.zeros((SHUFFLES+1, N), dtype=_N.int32)
    near1RPSAIRPS = _N.zeros((SHUFFLES+1, N), dtype=_N.int32)
    #near1RPSRPS = _N.zeros((SHUFFLES+1, N), dtype=_N.int32)
    near1DSURPS = _N.zeros((SHUFFLES+1, N), dtype=_N.int32)
    near1DSUAIRPS = _N.zeros((SHUFFLES+1, N), dtype=_N.int32)
    #near1LCBAIRPS = _N.zeros((SHUFFLES+1, N), dtype=_N.int32)
    near1LCBRPS = _N.zeros((SHUFFLES+1, N), dtype=_N.int32)    
    near0DSUWTL = _N.zeros((SHUFFLES+1, N), dtype=_N.int32)
    near0RPSWTL = _N.zeros((SHUFFLES+1, N), dtype=_N.int32)
    near0RPSAIRPS = _N.zeros((SHUFFLES+1, N), dtype=_N.int32)
    #near0RPSRPS = _N.zeros((SHUFFLES+1, N), dtype=_N.int32)
    near0DSURPS = _N.zeros((SHUFFLES+1, N), dtype=_N.int32)
    near0DSUAIRPS = _N.zeros((SHUFFLES+1, N), dtype=_N.int32)
    #near0LCBAIRPS = _N.zeros((SHUFFLES+1, N), dtype=_N.int32)
    near0LCBRPS = _N.zeros((SHUFFLES+1, N), dtype=_N.int32)    

    Ngames = dmp["cond_probsDSUWTL"].shape[2]
    if Ngames == TO-win:
        print(dmp["cond_probsDSUWTL"].shape)
        allDSUWTLs = dmp["cond_probsDSUWTL"][:, :, 0:TO-win].reshape((SHUFFLES+1, 3, 3, TO-win))
        allRPSWTLs = dmp["cond_probsRPSWTL"][:, :, 0:TO-win].reshape((SHUFFLES+1, 3, 3, TO-win))
        #allRPSRPSs = dmp["cond_probsRPSRPS"][:, :, 0:TO-win].reshape((SHUFFLES+1, 3, 3, TO-win))
        allDSURPSs = dmp["cond_probsDSURPS"][:, :, 0:TO-win].reshape((SHUFFLES+1, 3, 3, TO-win))
        #allRPSAIRPSs = dmp["cond_probsRPSAIRPS"].reshape((SHUFFLES+1, 3, 3, 300-win))
        allDSUAIRPSs = dmp["cond_probsDSUAIRPS"][:, :, 0:TO-win].reshape((SHUFFLES+1, 3, 3, TO-win))
        allLCBAIRPSs = dmp["cond_probsLCBAIRPS"][:, :, 0:TO-win].reshape((SHUFFLES+1, 3, 3, TO-win))
        allLCBRPSs = dmp["cond_probsLCBRPS"][:, :, 0:TO-win].reshape((SHUFFLES+1, 3, 3, TO-win))

        
        ifr = -1
        print("----------------    %d" % pid)

        hi_rnk_cmpts = [[], [], [], [], [], []]
        mid_rnk_cmpts = [[], [], [], [], [], []]
        lo_rnk_cmpts = [[], [], [], [], [], []]

        len0  = _N.zeros(SHUFFLES+1, dtype=_N.int32)
        len1  = _N.zeros(SHUFFLES+1, dtype=_N.int32)
        clumped0  = _N.zeros(SHUFFLES+1, dtype=_N.int32)
        clumped1  = _N.zeros(SHUFFLES+1, dtype=_N.int32)

        hnd_dat = dmp["all_tds"][:, 0:TO]
        cv_onrule, cv_offrule = _crut.deterministic_rule(hnd_dat, TO=TO)
        for frmwk in frameworks:
            ifr += 1
            exec("alls = dmp['cond_probs%(fr)s'][:, :, 0:TO-win].reshape((%(shfp1)d, 3, 3, %(N)d))" % {"fr" : frmwk, "shfp1" : (SHUFFLES+1), "N" : (TO-win)})
            print(frmwk)

            for ic in range(3):
                for ia in range(3):
                    nCV_on = len(_N.where(cv_onrule[0, ifr, ic, ia] > cv_onrule[1:, ifr, ic, ia])[0])
                    nCV_off = len(_N.where(cv_offrule[0, ifr, ic, ia] > cv_offrule[1:, ifr, ic, ia])[0])                

                    cv_offrule_rank[pid-1, ifr, ic, ia] = nCV_off
                    cv_onrule_rank[pid-1, ifr, ic, ia] = nCV_on

                    stds = _N.std(alls[:, ic, ia], axis=1)
                    #stds = robust.mad(alls[:, ic, ia], axis=1)

                    #cmp_z1s[pid-1, ifr, ic, ia] = (stds[0] - _N.mean(stds[1:])) / _N.std(stds[1:])
                    cmp_z1sZ[pid-1, ifr, ic, ia] = (stds[0] - _N.mean(stds[1:])) / _N.std(stds[1:])
                    #cmp_z1sZ[pid-1, ifr,a ic, ia] = stds[0] / _N.mean(stds[1:])
                    cmp_z1s[pid-1, ifr, ic, ia] = stds[0] 
                    rnk1 = len(_N.where(stds[0] > stds[1:])[0])

                    SHUFFLES2 = SHUFFLES//2
                    for ist in range(SHUFFLES+1):
                        len0[ist] = len(_N.where(alls[ist, ic, ia] == 0)[0])
                        len1[ist] = len(_N.where(alls[ist, ic, ia] > 0.8)[0])

                        dths0 = _N.diff(_N.where(alls[ist, ic, ia] > 0.01)[0])
                        dths1 = _N.diff(_N.where(alls[ist, ic, ia] > 0.8)[0])

                        clumped0[ist] = 0
                        clumped1[ist] = 0                    
                        if len(dths0) > 2:
                            srtd = _N.sort(dths0)
                            clumped0[ist] = _N.mean(srtd[-3:])
                        if len(dths1) > 2:
                            clumped1[ist] = _N.std(dths1)/_N.mean(dths1)
                        # fig.add_subplot(3, 1, ia+1)
                        # _plt.title(srtd[-1])
                        # _plt.plot(alls[ist, ic, ia])

                    #len1[0] = len(_N.where(alls[0, ic, ia] > 0.8)[0])
                    len1[0] = len(_N.where(alls[0, ic, ia] > 0.76)[0])
                    rnk2 = len(_N.where(len1[0] > len1[1:])[0])
                    fr_lotsof0s[pid-1, ifr, ic, ia] = len(_N.where(len0[0] > len0[1:])[0])
                    fr_lotsof1s[pid-1, ifr, ic, ia] = len(_N.where(len1[0] > len1[1:])[0])
                    hlf = _N.mean(len1[1+SHUFFLES2:1+SHUFFLES2+10])
                    fr_lotsof1zs[pid-1, ifr, ic, ia] = len1[0] / hlf
                    fr_clumped0s[pid-1, ifr, ic, ia] = len(_N.where(clumped0[0] > clumped0[1:])[0])
                    fr_clumped1s[pid-1, ifr, ic, ia] = len(_N.where(clumped1[0] > clumped1[1:])[0])                

                    len1s[pid-1, ifr, ic, ia]       = len1[0]

                    #rnk1 = len(_N.where(stds[1] > stds[2:])[0])
                    fr_cmp_fluc_rank1[pid-1, ifr, ic, ia] = rnk1
                    fr_cmp_fluc_rank2[pid-1, ifr, ic, ia] = rnk2

                    print("%(c)d %(a)d   %(r)d" % {"c" : ic, "a" : ia, "r" : rnk1})
                    if rnk1 / SHUFFLES < 0.1:
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

            
sFlipped = "_flipped" if flip_HUMAI else ""
dmpout = open(workdirFN("shuffledCRs_5CFs%(flp)s_%(ex)s_%(w)d_%(v)d_%(vs)s" % {"ex" : expt, "w" : win, "v" : visit, "vs" : svisits, "flp" : sFlipped}), "wb")
if filtdat_okgames is not None:
    pickle.dump({"z1s" : cmp_z1sZ, "cv_onrule_rank" : cv_onrule_rank, "cv_offrule_rank" : cv_offrule_rank, "fr_clumped1s" : fr_clumped1s, "fr_clumped0s" : fr_clumped0s, "fr_cmp_fluc_rank1" : fr_cmp_fluc_rank1, "fr_cmp_fluc_rank2" : fr_cmp_fluc_rank2, "filtdat" : filtdat, "filtdat_okgames" : filtdat_okgames, "SHUFFLES" : SHUFFLES, "partIDs" : partIDs, "fr_lotsof0s" : fr_lotsof0s, "fr_lotsof1s" : fr_lotsof1s, "fr_lotsof1zs" : fr_lotsof1zs, "len1s" : len1s}, dmpout, -1)
else:
    pickle.dump({"z1s" : cmp_z1sZ, "cv_onrule_rank" : cv_onrule_rank, "cv_offrule_rank" : cv_offrule_rank, "fr_clumped1s" : fr_clumped1s, "fr_clumped0s" : fr_clumped0s, "fr_cmp_fluc_rank1" : fr_cmp_fluc_rank1, "fr_cmp_fluc_rank2" : fr_cmp_fluc_rank2, "filtdat" : filtdat, "SHUFFLES" : SHUFFLES, "partIDs" : partIDs, "fr_lotsof0s" : fr_lotsof0s, "fr_lotsof1s" : fr_lotsof1s, "fr_lotsof1zs" : fr_lotsof1zs, "len1s" : len1s}, dmpout, -1)
dmpout.close()

#SHUFFLES = SHUFFLES-1

print(fr_cmp_fluc_rank1.shape)
for i in range(filtdat.shape[0]):
    _plt.plot(fr_cmp_fluc_rank1[filtdat[i], 0, 2], color="black")


for ic in range(3):
    ones_kys = {}
    for i in range(filtdat.shape[0]):
        key = str(_N.where(fr_cmp_fluc_rank1[filtdat[i], 0, ic] > int(0.98*SHUFFLES))[0])
        try:
            ones_kys[key] += 1
        except KeyError:
            ones_kys[key] =  1
    print(ones_kys)


for ic in range(3):
    ones_kys = {}
    for i in range(filtdat.shape[0]):
        key = str(_N.where(fr_cmp_fluc_rank1[filtdat[i], 0, ic] > int(0.98*SHUFFLES))[0])
        try:
            ones_kys[key] += 1
        except KeyError:
            ones_kys[key] =  1
    print(ones_kys)


#  how many cond_prob components have big amplitude
fig = _plt.figure(figsize=(10, 9.5))
_plt.suptitle("# of big fluctuation components", fontsize=14)
for ifr in range(nFrameworks):
    fig.add_subplot(nFrameworks, 1, ifr+1)
    ipt, icn, iac = _N.where(fr_cmp_fluc_rank1[filtdat, ifr] > int(0.95*SHUFFLES))
    #cnts, bins, lns  = _plt.hist(ipt, bins=_N.linspace(-0.5, 188.5, 190), color="black")
    _plt.ylim(0, 7)
    _plt.xlim(-1.5, len(filtdat)+0.5)
    _plt.title("Framework %s" % frameworks_p[ifr], fontsize=11)
    _plt.yticks([0, 3, 6], fontsize=11)
    _plt.xticks(fontsize=11)
_plt.xlabel("participant #", fontsize=14)
fig.subplots_adjust(wspace=0.5, hspace=0.85, top=0.85, left=0.08, right=0.94)
_plt.savefig("Num_of_frameworks_comps_big_%d" % win)




