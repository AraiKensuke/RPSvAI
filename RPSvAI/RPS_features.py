#!/usr/bin/python

import numpy as _N
import RPSvAI.utils.read_taisen as _rt
import scipy.io as _scio
import scipy.stats as _ss
import matplotlib.pyplot as _plt
import RPSvAI.utils.read_taisen as _rd
import RPSvAI.utils.misc as _Am
import os
import sys
import pickle

from RPSvAI.utils.dir_util import workdirFN

import RPSvAI.models.CRutils as _crut
import RPSvAI.models.empirical_ken as _emp

import AIRPSfeatures as _aift

flip_HUMAI = False
sFlipped   = "_flipped" if flip_HUMAI else ""
# __1st__ = 0
# __2nd__ = 1
# __ALL__ = 2

# _SHFL_KEEP_CONT  = 0
# _SHFL_NO_KEEP_CONT  = 1

#  sum_sd
#  entropyL
#  isi_cv, isis_corr

def only_complete_data(partIDs, TO, label, SHF_NUM):
    pid = -1
    incomplete_data = []
    print(partIDs)
    for partID in partIDs:
        pid += 1

        dmp       = depickle(workdirFN("%(rpsm)s/%(lb)d/variousCRs%(flp)s_%(visit)d.dmp" % {"rpsm" : partID, "lb" : label, "visit" : visit, "flp" : sFlipped}))
        _prob_mvs = dmp["cond_probsDSUWTL"][SHF_NUM]
        _prob_mvsRPS = dmp["cond_probsRPSWTL"][SHF_NUM]
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


mouseOffset = 400
##  Then I expect wins following UPs and DOWNs to also be correlated to AQ28
look_at_AQ = True
data   = "TMB2"

visit = 1
visits= [1, ]   #  if I want 1 of [1, 2], set this one to [1, 2]
#visit = 2
#visits = [1]
#visits= [1, 2]   #  if I want 1 of [1, 2], set this one to [1, 2]
svisits =str(visits).replace(" ", "").replace("[", "").replace("]", "")    
if data == "TMB2":
    dates = _rt.date_range(start='7/13/2021', end='12/30/2021')
    #partIDs, dats, cnstrs, has_domainQs, has_domainQs_wkeys = _rt.filterRPSdats(data, dates, visits=visits, domainQ=(_rt._TRUE_ONLY_ if look_at_AQ else _rt._TRUE_AND_FALSE_), demographic=_rt._TRUE_AND_FALSE_, mentalState=_rt._TRUE_AND_FALSE_, min_meanIGI=500, max_meanIGI=20000, minIGI=10, maxIGI=50000, MinWinLossRat=0.2, has_useragent=True, has_start_and_end_times=True, has_constructor=True, blocks=1)

    partIDs, dats, cnstrs, has_domainQs, has_domainQs_wkeys = _rt.filterRPSdats(data, dates, visits=visits, domainQ=(_rt._TRUE_ONLY_ if look_at_AQ else _rt._TRUE_AND_FALSE_), demographic=_rt._TRUE_AND_FALSE_, mentalState=_rt._TRUE_AND_FALSE_, min_meanIGI=600, max_meanIGI=20000, minIGI=10, maxIGI=30000, MinWinLossRat=0.2, has_useragent=True, has_start_and_end_times=True, has_constructor=True, blocks=1)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    partIDs_okgames, dats_okgames, cnstrs_okgames, has_domainQs_okgames, has_domainQs_wkeys_okgames = _rt.filterRPSdats(data, dates, visits=visits, domainQ=_rt._TRUE_AND_FALSE_, demographic=_rt._TRUE_AND_FALSE_, mentalState=_rt._TRUE_AND_FALSE_, min_meanIGI=600, max_meanIGI=20000, minIGI=10, maxIGI=30000, MinWinLossRat=0.2, has_useragent=True, has_start_and_end_times=True, has_constructor=True, blocks=1)

if data == "TMBCW":
    dates = _rt.date_range(start='7/13/2021', end='12/30/2029')
    #partIDs, dats, cnstrs, has_domainQs, has_domainQs_wkeys = _rt.filterRPSdats(data, dates, visits=visits, domainQ=(_rt._TRUE_ONLY_ if look_at_AQ else _rt._TRUE_AND_FALSE_), demographic=_rt._TRUE_AND_FALSE_, mentalState=_rt._TRUE_AND_FALSE_, min_meanIGI=500, max_meanIGI=20000, minIGI=10, maxIGI=50000, MinWinLossRat=0.2, has_useragent=True, has_start_and_end_times=True, has_constructor=True, blocks=1)

    partIDs, dats, cnstrs, has_domainQs, has_domainQs_wkeys = _rt.filterRPSdats(data, dates, visits=visits, domainQ=(_rt._TRUE_ONLY_ if look_at_AQ else _rt._TRUE_AND_FALSE_), demographic=_rt._TRUE_AND_FALSE_, mentalState=_rt._TRUE_AND_FALSE_, min_meanIGI=600, max_meanIGI=20000, minIGI=10, maxIGI=30000, MinWinLossRat=0.2, has_useragent=True, has_start_and_end_times=True, has_constructor=True, blocks=1)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    partIDs_okgames, dats_okgames, cnstrs_okgames, has_domainQs_okgames, has_domainQs_wkeys_okgames = _rt.filterRPSdats(data, dates, visits=visits, domainQ=_rt._TRUE_AND_FALSE_, demographic=_rt._TRUE_AND_FALSE_, mentalState=_rt._TRUE_AND_FALSE_, min_meanIGI=600, max_meanIGI=20000, minIGI=10, maxIGI=30000, MinWinLossRat=0.2, has_useragent=True, has_start_and_end_times=True, has_constructor=True, blocks=1)
    
elif data == "CogWeb":
    dates = _rt.date_range(start='2/25/2024', end='10/30/2029')
    partIDs, dats, cnstrs, has_domainQs, has_domainQs_wkeys = _rt.filterRPSdats(data, dates, visits=visits, domainQ=(_rt._TRUE_ONLY_ if look_at_AQ else _rt._TRUE_AND_FALSE_), demographic=_rt._TRUE_AND_FALSE_, mentalState=_rt._TRUE_AND_FALSE_, min_meanIGI=600, max_meanIGI=20000, minIGI=10, maxIGI=30000, MinWinLossRat=0.2, has_useragent=True, has_start_and_end_times=True, has_constructor=True, blocks=1)
    partIDs_okgames = partIDs

    
A1 = []
show_shuffled = False
#######################################################

win_type = 2   #  window is of fixed number of games
#win_type = 1  #  window is of fixed number of games that meet condition 
win     = 3
smth    = 1
#smth    = 3
label          = win_type*100+win*10+smth

TO = 300
SHF_NUM = 0

print(partIDs)
partIDs, incmp_dat = only_complete_data(partIDs, TO, label, SHF_NUM)
#partIDs_okgames, incmp_dat = only_complete_data(partIDs_okgames, TO, label, SHF_NUM)
print(partIDs)
strtTr=0
TO -= strtTr

#fig= _plt.figure(figsize=(14, 14))

SHUFFLES = 1
#nMimics = _N.empty(len(partIDs), dtype=_N.int)
t0 = -5
t1 = 10
trigger_temp = _N.empty(t1-t0)
cut = 1
all_avgs = _N.empty((len(partIDs), SHUFFLES+1, t1-t0))
netwins  = _N.empty(len(partIDs), dtype=_N.int32)

gk = _Am.gauKer(1)
gk /= _N.sum(gk)
gk2 = _Am.gauKer(2)
gk2 /= _N.sum(gk2)

#gk = None

ts  = _N.arange(t0-2, t1-2)
hnd_dat_all = _N.zeros((len(partIDs), TO, 4), dtype=_N.int32)

resp_t_stds = _N.empty(len(partIDs))
resp_t_cvs = _N.empty(len(partIDs))

ages      = _N.empty(len(partIDs))
gens      = _N.empty(len(partIDs))
Engs      = _N.empty(len(partIDs))

time_aft_los = _N.empty(len(partIDs))
time_aft_tie  = _N.empty(len(partIDs))
time_aft_win = _N.empty(len(partIDs))
time_b4aft_los_mn = _N.empty(len(partIDs))
time_b4aft_tie_mn  = _N.empty(len(partIDs))
time_b4aft_win_mn = _N.empty(len(partIDs))
time_b4aft_los_sd = _N.empty(len(partIDs))
time_b4aft_tie_sd  = _N.empty(len(partIDs))
time_b4aft_win_sd = _N.empty(len(partIDs))

score  = _N.empty(len(partIDs))
maxCs  = _N.empty(len(partIDs))
pcW_UD  = _N.empty(len(partIDs))
pcT_UD  = _N.empty(len(partIDs))
pcL_UD  = _N.empty(len(partIDs))

certWs = _N.empty(len(partIDs))
certTs = _N.empty(len(partIDs))
certLs = _N.empty(len(partIDs))
certAIRs = _N.empty(len(partIDs))
certAISs = _N.empty(len(partIDs))
certAIPs = _N.empty(len(partIDs))
certRs = _N.empty(len(partIDs))
certSs = _N.empty(len(partIDs))
certPs = _N.empty(len(partIDs))

AIcerts= _N.empty((len(partIDs), 9))

nFrameworks  = 6
all_CR_corrs = _N.empty((len(partIDs), int(0.5*(9*nFrameworks)*(9*nFrameworks-1))))
all_CR_corrs_pairID = _N.ones(int(0.5*(9*nFrameworks)*(9*nFrameworks-1)), dtype=_N.int32) * -1
all_CR_corrs_trms = _N.ones((int(0.5*(9*nFrameworks)*(9*nFrameworks-1)), 6), dtype=_N.int32)

FR_conds= _N.zeros((int(9*nFrameworks), 3))

ifca = -1
for fr in range(nFrameworks):
    for ic in range(3):
        for ia in range(3):
            ifca += 1
            FR_conds[ifca, 0] = fr
            FR_conds[ifca, 1] = ic
            FR_conds[ifca, 2] = ia
all_CR_sds = _N.empty((len(partIDs), nFrameworks, 3, 3))
all_CRs   = _N.empty((len(partIDs), 9*nFrameworks, TO-win))
all_CRs_F = _N.empty((len(partIDs), 9*nFrameworks), dtype=_N.int32)
all_fr01s = _N.empty((len(partIDs), nFrameworks))
    
up_or_dn     = _N.empty(len(partIDs))
#  triggered outcomes

WTL_aft_WTL  = _N.empty((len(partIDs), 3, 3))
WTL_aft_LCB  = _N.empty((len(partIDs), 3, 3))
RPS_aft_WTL  = _N.empty((len(partIDs), 3, 3))
AIRPS_aft_WTL  = _N.empty((len(partIDs), 3, 3))
RPS_aft_AIRPS  = _N.empty((len(partIDs), 3, 3))
wtlStreaks  = _N.empty((len(partIDs), 3))

win_aft_win  = _N.empty(len(partIDs))
win_aft_los  = _N.empty(len(partIDs))
win_aft_tie  = _N.empty(len(partIDs))
tie_aft_win  = _N.empty(len(partIDs))
tie_aft_los  = _N.empty(len(partIDs))
tie_aft_tie  = _N.empty(len(partIDs))
los_aft_win  = _N.empty(len(partIDs))
los_aft_los  = _N.empty(len(partIDs))
los_aft_tie  = _N.empty(len(partIDs))

win_aft_AIR  = _N.empty(len(partIDs))
win_aft_AIS  = _N.empty(len(partIDs))
win_aft_AIP  = _N.empty(len(partIDs))
tie_aft_AIR  = _N.empty(len(partIDs))
tie_aft_AIS  = _N.empty(len(partIDs))
tie_aft_AIP  = _N.empty(len(partIDs))
los_aft_AIR  = _N.empty(len(partIDs))
los_aft_AIS  = _N.empty(len(partIDs))
los_aft_AIP  = _N.empty(len(partIDs))

win_aft_R  = _N.empty(len(partIDs))
win_aft_S  = _N.empty(len(partIDs))
win_aft_P  = _N.empty(len(partIDs))
tie_aft_R  = _N.empty(len(partIDs))
tie_aft_S  = _N.empty(len(partIDs))
tie_aft_P  = _N.empty(len(partIDs))
los_aft_R  = _N.empty(len(partIDs))
los_aft_S  = _N.empty(len(partIDs))
los_aft_P  = _N.empty(len(partIDs))

win_aft_L  = _N.empty(len(partIDs))
tie_aft_L  = _N.empty(len(partIDs))
los_aft_L  = _N.empty(len(partIDs))
win_aft_C  = _N.empty(len(partIDs))
tie_aft_C  = _N.empty(len(partIDs))
los_aft_C  = _N.empty(len(partIDs))
win_aft_B  = _N.empty(len(partIDs))
tie_aft_B  = _N.empty(len(partIDs))
los_aft_B  = _N.empty(len(partIDs))

win_aft_D  = _N.empty(len(partIDs))
tie_aft_D  = _N.empty(len(partIDs))
los_aft_D  = _N.empty(len(partIDs))
win_aft_S  = _N.empty(len(partIDs))
tie_aft_S  = _N.empty(len(partIDs))
los_aft_S  = _N.empty(len(partIDs))
win_aft_U  = _N.empty(len(partIDs))
tie_aft_U  = _N.empty(len(partIDs))
los_aft_U  = _N.empty(len(partIDs))


R_aft_win  = _N.empty(len(partIDs))
R_aft_los  = _N.empty(len(partIDs))
R_aft_tie  = _N.empty(len(partIDs))
P_aft_win  = _N.empty(len(partIDs))
P_aft_los  = _N.empty(len(partIDs))
P_aft_tie  = _N.empty(len(partIDs))
S_aft_win  = _N.empty(len(partIDs))
S_aft_los  = _N.empty(len(partIDs))
S_aft_tie  = _N.empty(len(partIDs))

u_or_d_res   = _N.empty(len(partIDs))
u_or_d_tie   = _N.empty(len(partIDs))
s_res        = _N.empty(len(partIDs))
s_tie        = _N.empty(len(partIDs))

up_res   = _N.empty(len(partIDs))
dn_res   = _N.empty(len(partIDs))
stay_res         = _N.empty(len(partIDs))
stay_tie         = _N.empty(len(partIDs))

AQ28scrs  = _N.empty(len(partIDs))
soc_skils = _N.empty(len(partIDs))
rout      = _N.empty(len(partIDs))
switch    = _N.empty(len(partIDs))
imag      = _N.empty(len(partIDs))
fact_pat  = _N.empty(len(partIDs))
ans_soc_skils = _N.empty((len(partIDs), 7), dtype=_N.int32)
ans_rout      = _N.empty((len(partIDs), 4), dtype=_N.int32)
ans_switch    = _N.empty((len(partIDs), 4), dtype=_N.int32)
ans_imag      = _N.empty((len(partIDs), 8), dtype=_N.int32)
ans_fact_pat  = _N.empty((len(partIDs), 5), dtype=_N.int32)

end_strts     = _N.empty(len(partIDs))

all_AI_weights = _N.empty((len(partIDs), TO+1, 3, 3, 2))
all_AI_preds = _N.empty((len(partIDs), TO+1, 3))
AIfeatsWTL   = _N.empty((len(partIDs), 3, 3))
AIfeatsRPS   = _N.empty((len(partIDs), 3, 3))
AIfeatsAIRPS = _N.empty((len(partIDs), 3, 3))
#
AIfeatsWTL_m1   = _N.empty((len(partIDs), 3, 3))
AIfeatsRPS_m1   = _N.empty((len(partIDs), 3, 3))
AIfeatsAIRPS_m1 = _N.empty((len(partIDs), 3, 3))
#
AIfeatsWTLm2   = _N.empty((len(partIDs), 3, 3))
AIfeatsRPSm2   = _N.empty((len(partIDs), 3, 3))
AIfeatsAIRPSm2 = _N.empty((len(partIDs), 3, 3))
#
AIfeatsWTL_m3   = _N.empty((len(partIDs), 3, 3))
AIfeatsRPS_m3   = _N.empty((len(partIDs), 3, 3))
AIfeatsAIRPS_m3 = _N.empty((len(partIDs), 3, 3))

start_T  = 0


n_copies = _N.empty(len(partIDs), dtype=_N.int32)

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

#ths = _N.where((AQ28scrs > 35))[0]
#ths = _N.where((AQ28scrs > 0))[0]
resp_times_OK = []
resp_times_OK_okgames = []
L30  = 30

not_outliers = []
not_outliers_okgames = []
notmany_repeats = []
notmany_repeats_okgames = []
correct_ngames = []
correct_ngames_okgames = []

sFlipped="_flipped" if flip_HUMAI else ""
#lm = depickle(workdirFN("shuffledCRs_5CFs%(flp)s_%(ex)s_%(w)d_%(v)d_%(vs)s" % {"ex" : data, "w" : win, "v" : visit, "vs" : svisits, "flp" : sFlipped}))
#ranks_of_cmps1 = lm["fr_cmp_fluc_rank1"]
#ranks_of_cmps2 = lm["fr_cmp_fluc_rank2"]

#ranks_of_lotsof0s = lm["fr_lotsof0s"]
#ranks_of_lotsof1s = lm["fr_lotsof1s"]
#len1s = lm["len1s"]

not_outliers_okgames = _N.arange(len(partIDs_okgames))
time_aft_los_okgames = _N.empty(len(partIDs_okgames))
time_aft_tie_okgames  = _N.empty(len(partIDs_okgames))
time_aft_win_okgames = _N.empty(len(partIDs_okgames))
resp_t_stds_okgames = _N.empty(len(partIDs_okgames))
resp_t_cvs_okgames = _N.empty(len(partIDs_okgames))
win_rat_okgames   = _N.empty(len(partIDs_okgames), dtype=_N.int32)
win_rat   = _N.empty(len(partIDs), dtype=_N.int32)

pid = 0
for partID in partIDs_okgames:
    pid += 1

    dmp       = depickle(workdirFN("%(rpsm)s/%(lb)d/variousCRs%(flp)s_%(visit)d.dmp" % {"rpsm" : partID, "lb" : label, "visit" : visit, "flp" : sFlipped}))
    inp_meth = dmp["inp_meth"]
    _hnd_dat = dmp["all_tds"][SHF_NUM][strtTr:]
    srtd_resp_ts = _N.sort(_N.diff(_hnd_dat[:, 3]))
    print("----------------------------------")
    print(srtd_resp_ts[0:5])
    print(srtd_resp_ts[-5:])
    resp_t_stds_okgames[pid-1] =_N.std(_N.diff(_hnd_dat[:, 3]))
    resp_t_cvs_okgames[pid-1] =_N.std(_N.diff(_hnd_dat[:, 3]))/_N.mean(_N.diff(_hnd_dat[:, 3]))
    win_rat_okgames[pid-1] = _N.sum(_hnd_dat[:, 2])

    hdcol = 0

    repeats = _N.sort(_rt.repeated_keys(_hnd_dat))[::-1]

    print("repeats %(id)s ----------   %(1)d %(2)d" % {"1" : repeats[0], "2" : repeats[1], "id" : partID})
    if not ((repeats[0] > 15) or ((repeats[0] > 10) and (repeats[1] > 10))):
        notmany_repeats_okgames.append(pid-1)
    n_mouse, n_keys, mouse_resp_t, key_resp_t, resp_time_all = _aift.resptime_aft_wtl(_hnd_dat, TO, pid, inp_meth, time_aft_win_okgames, time_aft_tie_okgames, time_aft_los_okgames)

    if (n_mouse > 0) and (n_keys > 0) and (key_resp_t) < 200:
    #if (n_mouse > 0) and (n_keys > 0) and (n_keys > 50) and (key_resp_t) < 300:
        print("keyboard really short")
    else:
        resp_times_OK_okgames.append(pid-1)
    if _hnd_dat.shape[0] == 300:
        correct_ngames_okgames.append(pid-1)
        
        
pid=0
for partID in partIDs:
    pid += 1
    print("!!!!!!!!!!!!!!!!!!!!   %(pid)d  partID  %(ID)s" % {"ID" : partID, "pid" : pid})
    #if (partID != "20210801_0015-00") and (partID != "20210801_0020-00") and (partID != "20211021_0130-00") and (partID != "20210924_0025-00") and (partID != "20211004_0015-00") and (partID != "20211018_0010-00"):
    #if (partID != "20210801_0015-00") and (partID != "20210801_0020-00"):# and (partID != "20211021_0130-00") and (partID != "20210924_0025-00") and (partID != "20211004_0015-00") and (partID != "20211018_0010-00"):
    #if (partID != "20211021_0130-00") and (partID != "20210924_0025-00") and (partID != "20211004_0015-00") and (partID != "20211018_0010-00"):
    if data == "TMB2":
        not_outliers.append(pid-1)
        #print("nope")
    if (data == "CogWeb") and (partID != "20240301_0010-00") and (partID != "20240301_0000-00"):
        not_outliers.append(pid-1)
        #print("nope")


    dmp       = depickle(workdirFN("%(rpsm)s/%(lb)d/variousCRs%(flp)s_%(visit)d.dmp" % {"rpsm" : partID, "lb" : label, "visit" : visit, "flp" : sFlipped}))
    _prob_mvsDSUWTL = dmp["cond_probsDSUWTL"][SHF_NUM]
    _prob_mvsRPSWTL = dmp["cond_probsRPSWTL"][SHF_NUM]
    #_prob_mvsRPSRPS = dmp["cond_probsRPSRPS"][SHF_NUM][:, strtTr:]
    _prob_mvsDSURPS = dmp["cond_probsDSURPS"][SHF_NUM][:, strtTr:]    
    _prob_mvsLCBRPS = dmp["cond_probsLCBRPS"][SHF_NUM][:, strtTr:]    
    #_prob_mvsDSURPS = _N.array(_prob_mvsRPSRPS)
    #_prob_mvsDSURPS[_N.array([0, 1, 2, 3, 4, 5, 6, 7, 8])] = _prob_mvsRPSRPS[_N.array([2, 1, 0, 3, 4, 5, 7, 8, 6])]
    #_prob_mvsRPSAIRPS = dmp["cond_probsRPSAIRPS"][SHF_NUM][:, strtTr:]
    _prob_mvsDSUAIRPS = dmp["cond_probsDSUAIRPS"][SHF_NUM][:, strtTr:]    
    _prob_mvsLCBAIRPS = dmp["cond_probsLCBAIRPS"][SHF_NUM][:, strtTr:]

    resp_t_stds[pid-1] =_N.std(_N.diff(_hnd_dat[:, 3]))
    resp_t_cvs[pid-1] =_N.std(_N.diff(_hnd_dat[:, 3]))/_N.mean(_N.diff(_hnd_dat[:, 3]))
    win_rat[pid-1] = _N.sum(_hnd_dat[:, 2])
    

    inp_meth = dmp["inp_meth"]
    _hnd_dat = dmp["all_tds"][SHF_NUM][strtTr:]
    end_strts[pid-1] = _N.mean(_hnd_dat[-1, 3] - _hnd_dat[0, 3])

    hdcol = 0

    hnd_dat_all[pid-1] = _hnd_dat[0:TO]
    netwins[pid-1] = _N.sum(_hnd_dat[0:TO, 2])
    repeats = _N.sort(_rt.repeated_keys(_hnd_dat))[::-1]

    print("repeats %(id)s ----------   %(1)d %(2)d" % {"1" : repeats[0], "2" : repeats[1], "id" : partID})
    if not ((repeats[0] > 15) or ((repeats[0] > 10) and (repeats[1] > 10))):
        notmany_repeats.append(pid-1)

    all_AI_weights[pid-1] = dmp["AI_weights"][0:TO+1]
    all_AI_preds[pid-1] = dmp["AI_preds"][0:TO+1]

    if look_at_AQ:
        #ans_soc_skils[pid-1], ans_rout[pid-1], ans_switch[pid-1], ans_imag[pid-1], ans_fact_pat[pid-1] = _rt.AQ28ans("%(datadir)s/%(data)s/%(date)s/%(pID)s/AQ29.txt" % {"date" : partIDs[pid-1][0:8], "pID" : partIDs[pid-1], "data" : data, "datadir" : os.environ["TAISENDATA"]})
        if (data == "TMB2") or (data == "TMBCW"):
            AQ28scrs[pid-1], soc_skils[pid-1], rout[pid-1], switch[pid-1], imag[pid-1], fact_pat[pid-1] = _rt.AQ28("%(datadir)s/%(data)s/%(date)s/%(pID)s/AQ29.txt" % {"date" : partIDs[pid-1][0:8], "pID" : partIDs[pid-1], "data" : data, "datadir" : os.environ["TAISENDATA"]})
            ages[pid-1], gens[pid-1], Engs[pid-1] = _rt.Demo("%(datadir)s/%(data)s/%(date)s/%(pID)s/DQ1.txt" % {"date" : partIDs[pid-1][0:8], "pID" : partIDs[pid-1], "data" : data, "datadir" : os.environ["TAISENDATA"]})
        elif data == "CogWeb":
            AQ28scrs[pid-1], soc_skils[pid-1], rout[pid-1], switch[pid-1], imag[pid-1], fact_pat[pid-1] = _rt.AQ28("%(datadir)s/%(data)s/%(date)s/%(pID)s/AQ29.txt" % {"date" : partIDs[pid-1][0:8], "pID" : partIDs[pid-1], "data" : data, "datadir" : os.environ["TAISENDATA"]})
            ages[pid-1], gens[pid-1], Engs[pid-1] = _rt.Demo("%(datadir)s/%(data)s/%(date)s/%(pID)s/DQ1.txt" % {"date" : partIDs[pid-1][0:8], "pID" : partIDs[pid-1], "data" : data, "datadir" : os.environ["TAISENDATA"]})            
    
    n_mouse, n_keys, mouse_resp_t, key_resp_t, resp_time_all = _aift.resptime_aft_wtl(_hnd_dat, TO, pid, inp_meth, time_aft_win, time_aft_tie, time_aft_los)
    _aift.resptime_b4aft_wtl(_hnd_dat, TO, pid, inp_meth, time_b4aft_win_mn, time_b4aft_win_sd, time_b4aft_tie_mn, time_b4aft_tie_sd, time_b4aft_los_mn, time_b4aft_los_sd)

    if (n_mouse > 0) and (n_keys > 0) and (key_resp_t) < 200:
    #if (n_mouse > 0) and (n_keys > 0) and (n_keys > 50) and (key_resp_t) < 300:
        print("keyboard really short")
    else:
        resp_times_OK.append(pid-1)
    if _hnd_dat.shape[0] == 300:
        correct_ngames.append(pid-1)

    _aift.wtl_after_wtl(_hnd_dat, TO, pid, win_aft_win, tie_aft_win, los_aft_win, win_aft_tie, tie_aft_tie, los_aft_tie, win_aft_los, tie_aft_los, los_aft_los, R_aft_win, S_aft_win, P_aft_win, R_aft_tie, S_aft_tie, P_aft_tie, R_aft_los, S_aft_los, P_aft_los, win_aft_AIR, tie_aft_AIR, los_aft_AIR, win_aft_AIS, tie_aft_AIS, los_aft_AIS, win_aft_AIP, tie_aft_AIP, los_aft_AIP)

    
    
    WTL_aft_WTL[pid-1, 0, 0] = win_aft_win[pid-1]
    WTL_aft_WTL[pid-1, 0, 1] = tie_aft_win[pid-1]
    WTL_aft_WTL[pid-1, 0, 2] = los_aft_win[pid-1]
    WTL_aft_WTL[pid-1, 1, 0] = win_aft_tie[pid-1]
    WTL_aft_WTL[pid-1, 1, 1] = tie_aft_tie[pid-1]
    WTL_aft_WTL[pid-1, 1, 2] = los_aft_tie[pid-1]
    WTL_aft_WTL[pid-1, 2, 0] = win_aft_los[pid-1]
    WTL_aft_WTL[pid-1, 2, 1] = tie_aft_los[pid-1]
    WTL_aft_WTL[pid-1, 2, 2] = los_aft_los[pid-1]

    _aift.wtl_after_lcb(_hnd_dat, TO, pid,
                        win_aft_L, tie_aft_L, los_aft_L,
                        win_aft_C, tie_aft_C, los_aft_C,
                        win_aft_B, tie_aft_B, los_aft_B,
                        win_aft_D, tie_aft_D, los_aft_D,
                        win_aft_S, tie_aft_S, los_aft_S,
                        win_aft_U, tie_aft_U, los_aft_U)
                        

    WTL_aft_LCB[pid-1, 0, 0] = win_aft_L[pid-1]
    WTL_aft_LCB[pid-1, 0, 1] = tie_aft_L[pid-1]
    WTL_aft_LCB[pid-1, 0, 2] = los_aft_L[pid-1]
    WTL_aft_LCB[pid-1, 1, 0] = win_aft_C[pid-1]
    WTL_aft_LCB[pid-1, 1, 1] = tie_aft_C[pid-1]
    WTL_aft_LCB[pid-1, 1, 2] = los_aft_C[pid-1]
    WTL_aft_LCB[pid-1, 2, 0] = win_aft_B[pid-1]
    WTL_aft_LCB[pid-1, 2, 1] = tie_aft_B[pid-1]
    WTL_aft_LCB[pid-1, 2, 2] = los_aft_B[pid-1]

    WTL_aft_LCB[pid-1, 0, 0] = win_aft_D[pid-1]
    WTL_aft_LCB[pid-1, 0, 1] = tie_aft_D[pid-1]
    WTL_aft_LCB[pid-1, 0, 2] = los_aft_D[pid-1]
    WTL_aft_LCB[pid-1, 1, 0] = win_aft_S[pid-1]
    WTL_aft_LCB[pid-1, 1, 1] = tie_aft_S[pid-1]
    WTL_aft_LCB[pid-1, 1, 2] = los_aft_S[pid-1]
    WTL_aft_LCB[pid-1, 2, 0] = win_aft_U[pid-1]
    WTL_aft_LCB[pid-1, 2, 1] = tie_aft_U[pid-1]
    WTL_aft_LCB[pid-1, 2, 2] = los_aft_U[pid-1]


    winBlock = False
    tieBlock = False
    losBlock = False
    iWin     = 0
    iTie     = 0
    iLos     = 0
    winStreak = 0
    tieStreak = 0
    losStreak = 0    
    for i in range(TO):
        if _hnd_dat[i, 2] == 1:
            if not winBlock:
                winBlock = True
                iWin = 0
            if winBlock:
                iWin     += 1
        else:
            if winBlock:
                if iWin >= 4:
                    winStreak += 1
            winBlock = False    
    for i in range(TO):
        if _hnd_dat[i, 2] == 0:
            if not tieBlock:
                tieBlock = True
                iTie = 0
            if tieBlock:
                iTie     += 1
        else:
            if tieBlock:
                if iTie >= 4:
                    tieStreak += 1
            tieBlock = False    
    for i in range(TO):
        if _hnd_dat[i, 2] == -1:
            if not losBlock:
                losBlock = True
                iLos = 0
            if losBlock:
                iLos     += 1
        else:
            if losBlock:
                if iLos >= 4:
                    losStreak += 1
            losBlock = False
    if iWin >= 3:
        winStreak += 1
    if iTie >= 3:
        tieStreak += 1
    if iLos >= 3:
        losStreak += 1
    wtlStreaks[pid-1, 0] = winStreak / len(_N.where(_hnd_dat[:, 2] == 1)[0])
    wtlStreaks[pid-1, 1] = tieStreak / len(_N.where(_hnd_dat[:, 2] == 0)[0])
    wtlStreaks[pid-1, 2] = losStreak / len(_N.where(_hnd_dat[:, 2] == -1)[0])


    RPS_aft_AIRPS[pid-1, 0, 0] = len(_N.where((_hnd_dat[0:-1, 1] == 1) & (_hnd_dat[1:, 0] == 1))[0])
    RPS_aft_AIRPS[pid-1, 0, 1] = len(_N.where((_hnd_dat[0:-1, 1] == 1) & (_hnd_dat[1:, 0] == 2))[0])
    RPS_aft_AIRPS[pid-1, 0, 2] = len(_N.where((_hnd_dat[0:-1, 1] == 1) & (_hnd_dat[1:, 0] == 3))[0])
    RPS_aft_AIRPS[pid-1, 1, 0] = len(_N.where((_hnd_dat[0:-1, 1] == 2) & (_hnd_dat[1:, 0] == 1))[0])
    RPS_aft_AIRPS[pid-1, 1, 1] = len(_N.where((_hnd_dat[0:-1, 1] == 2) & (_hnd_dat[1:, 0] == 2))[0])
    RPS_aft_AIRPS[pid-1, 1, 2] = len(_N.where((_hnd_dat[0:-1, 1] == 2) & (_hnd_dat[1:, 0] == 3))[0])
    RPS_aft_AIRPS[pid-1, 2, 0] = len(_N.where((_hnd_dat[0:-1, 1] == 3) & (_hnd_dat[1:, 0] == 1))[0])
    RPS_aft_AIRPS[pid-1, 2, 1] = len(_N.where((_hnd_dat[0:-1, 1] == 3) & (_hnd_dat[1:, 0] == 2))[0])
    RPS_aft_AIRPS[pid-1, 2, 2] = len(_N.where((_hnd_dat[0:-1, 1] == 3) & (_hnd_dat[1:, 0] == 3))[0])
    
    cv_sum = 0

    prob_mvsDSUWTL  = _prob_mvsDSUWTL[:, 0:TO - win]
    prob_mvsRPSWTL  = _prob_mvsRPSWTL[:, 0:TO - win]
    #prob_mvsRPSRPS  = _prob_mvsRPSRPS[:, 0:TO - win]
    prob_mvsDSURPS  = _prob_mvsDSURPS[:, 0:TO - win]
    prob_mvsLCBRPS  = _prob_mvsLCBRPS[:, 0:TO - win]
    prob_mvsDSUAIRPS  = _prob_mvsDSUAIRPS[:, 0:TO - win]        
    prob_mvsLCBAIRPS  = _prob_mvsLCBAIRPS[:, 0:TO - win]


    
    prob_mvsDSUWTL = prob_mvsDSUWTL.reshape((3, 3, prob_mvsDSUWTL.shape[1]))
    prob_mvsRPSWTL = prob_mvsRPSWTL.reshape((3, 3, prob_mvsRPSWTL.shape[1]))
    prob_mvsLCBRPS = prob_mvsLCBRPS.reshape((3, 3, prob_mvsLCBRPS.shape[1]))        
    #prob_mvsRPSRPS = prob_mvsRPSRPS.reshape((3, 3, prob_mvsRPSRPS.shape[1]))
    prob_mvsDSURPS = prob_mvsDSURPS.reshape((3, 3, prob_mvsDSURPS.shape[1]))
    prob_mvsDSUAIRPS = prob_mvsDSUAIRPS.reshape((3, 3, prob_mvsDSUAIRPS.shape[1]))    
    prob_mvsLCBAIRPS = prob_mvsLCBAIRPS.reshape((3, 3, prob_mvsLCBAIRPS.shape[1]))


        
    

    #prob_mvsFRMWKs = [prob_mvsDSUWTL, prob_mvsRPSWTL, prob_mvsRPSRPS, prob_mvsDSUAIRPS, prob_mvsRPSAIRPS, prob_mvsLCBRPS]
    #prob_mvsFRMWKs = [prob_mvsDSUWTL, prob_mvsRPSWTL, prob_mvsRPSRPS, prob_mvsLCBRPS, prob_mvsDSUAIRPS, prob_mvsLCBAIRPS]
    prob_mvsFRMWKs = [prob_mvsDSUWTL, prob_mvsRPSWTL, prob_mvsDSURPS, prob_mvsLCBRPS, prob_mvsDSUAIRPS, prob_mvsLCBAIRPS]
    #  RPSRPS is the same as DSURPS
    #  LCBAIRPS  is the DASAUARPS

    for fr in range(nFrameworks):
        for ic in range(3):
            for ia in range(3):
                all_CRs[pid-1, 9*fr+ic*3+ia] = prob_mvsFRMWKs[fr][ic, ia]
                if _N.sum(all_CRs[pid-1, 9*fr+ic*3+ia] == 0) == TO-win:
                    all_CRs[pid-1, 9*fr+ic*3+ia] += 0.0001*_N.random.rand(TO-win)
    iii = -1

    # if pid == 4:
    #     fppp = open("CRout", "w")
    #     for fr in range(6):
    #         for ic in range(3):
    #             for ia in range(3):
    #                 #if _N.sum(prob_mvsFRMWKs[fr][ic, ia]) == 0:
    #                 fppp.write(str(prob_mvsFRMWKs[fr][ic, ia]))
    #                 fppp.write("\n")
    #                 #print("%(fr)d   %(ic)d %(ia)d" % {"fr" : fr, "ic" : ic, "ia" : ia})
    #     fppp.write("**********************\n")                    
    #     for ii in range(nFrameworks*9):
    #         fppp.write(str(all_CRs[pid-1, ii]))
    #         fppp.write("\n")
    #     fppp.close()
                        
    for i1 in range(nFrameworks*9):
        for i2 in range(i1+1, nFrameworks*9):
            iii += 1
            pc, pv = _ss.pearsonr(all_CRs[pid-1, i1], all_CRs[pid-1, i2])
            if pid == 4:
                if _N.isnan(pc):
                    print("%(i1)d  %(i2)d" % {"i1" : i1, "i2" : i2})
            all_CR_corrs[pid-1, iii] = pc            

            ifr1 = FR_conds[i1, 0]
            ifr2 = FR_conds[i2, 0]
            ic1  = FR_conds[i1, 1]
            ic2  = FR_conds[i2, 1]            
            ia1  = FR_conds[i1, 2]
            ia2  = FR_conds[i2, 2]            

            all_CR_corrs_trms[iii, 0] = ifr1
            all_CR_corrs_trms[iii, 1] = ic1
            all_CR_corrs_trms[iii, 2] = ia1
            all_CR_corrs_trms[iii, 3] = ifr2
            all_CR_corrs_trms[iii, 4] = ic2
            all_CR_corrs_trms[iii, 5] = ia2
            
            if ifr1 == ifr2:
                if ic1 == ic2:     # same FR, same cond
                    all_CR_corrs_pairID[iii] = 0
                elif ic1 != ic2:  # same FR, diff cond
                    all_CR_corrs_pairID[iii] = 1
            else:
                #all_CR_corrs_wPairID[pid-1, iii, 1] = 2            #  diff FR
                #  cond 0, 1  same
                #  cond 2
                #  cond 3, 4  same
                #  DSUWTL and RPSWTL
                ####  
                if (((ifr1 == 0) and (ifr2 == 1))) or (((ifr1 == 1) and (ifr2 == 0))) or (((ifr1 == 3) and (ifr2 == 4)) or ((ifr1 == 4) and (ifr2 == 3))):  # diff FRs but same conds
                    if ic1 == ic2:  # diff FR but same cond
                        all_CR_corrs_pairID[iii] = 2
                    else:           # diff FR but same cond
                        all_CR_corrs_pairID[iii] = 3
                else:               # diff FR cond set differeent
                        all_CR_corrs_pairID[iii] = 4
    
    tMv = _N.diff(_hnd_dat[:, 3])
    succ = _hnd_dat[1:, 2]

    preds = all_AI_preds[pid-1]

    PCS=3
    prob_Mimic            = _N.empty((3, prob_mvsDSUWTL.shape[2]))
    prob_Mimic_v2            = _N.empty((3, prob_mvsDSUWTL.shape[2]))    
    #t00 = 3
    #t01 = prob_mvsDSUWTL.shape[2]-3
    #ctprob_mvsDSUWTL          = prob_mvsDSUWTL[:, :, t00:t01]

    #  Since

    sdsDSUWTL = _N.std(prob_mvsDSUWTL, axis=2)# /_N.mean(prob_mvsDSUWTL, axis=2)
    sdsRPSWTL = _N.std(prob_mvsRPSWTL, axis=2)# / _N.mean(prob_mvsRPSWTL, axis=2)
    sdsDSURPS = _N.std(prob_mvsDSURPS, axis=2)# / _N.mean(prob_mvsRPSRPS, axis=2)
    sdsLCBRPS = _N.std(prob_mvsLCBRPS, axis=2)# /  _N.mean(prob_mvsLCBRPS, axis=2)
    sdsDSUAIRPS = _N.std(prob_mvsDSUAIRPS, axis=2)# / _N.mean(prob_mvsDSUAIRPS, axis=2)
    sdsLCBAIRPS = _N.std(prob_mvsLCBAIRPS, axis=2)# /   _N.mean(prob_mvsLCBAIRPS, axis=2)
    #meanSDS  = sdsDSUWTL+ sdsRPSWTL+sdsRPSRPS+sdsLCBRPS+sdsDSUAIRPS+sdsLCBAIRPS
    meanSDS  = sdsDSUWTL+ sdsRPSWTL+sdsDSURPS+sdsLCBRPS+sdsDSUAIRPS+sdsLCBAIRPS

    all_CR_sds[pid-1, 0]= sdsDSUWTL# / meanSDS#_N.mean(sdsDSUWTL)
    all_CR_sds[pid-1, 1]= sdsRPSWTL# / meanSDS#_N.mean(sdsRPSWTL)
    #all_CR_sds[pid-1, 2]= sdsRPSRPS# / meanSDS#_N.mean(sdsRPSRPS)
    all_CR_sds[pid-1, 2]= sdsDSURPS# / meanSDS#_N.mean(sdsRPSRPS)
    all_CR_sds[pid-1, 3]= sdsLCBRPS# / meanSDS#_N.mean(sdsLCBRPS)
    all_CR_sds[pid-1, 4]= sdsDSUAIRPS# / meanSDS#_N.mean(sdsDSUAIRPS)
    all_CR_sds[pid-1, 5]= sdsLCBAIRPS# / meanSDS#_N.mean(sdsLCBAIRPS)
    #lotsof01s_by_frmwk = _N.sum(_N.sum(ranks_of_lotsof0s[pid-1], axis=2), axis=1) + _N.sum(_N.sum(ranks_of_lotsof1s[pid-1], axis=2), axis=1)
    #all_fr01s[pid-1] = lotsof01s_by_frmwk
    

    """
    it0t1 = -1
    sdsDSUWTL_pcs   = _N.empty((5, 3, 3))
    sdsRPSWTL_pcs   = _N.empty((5, 3, 3))
    sdsRPSRPS_pcs   = _N.empty((5, 3, 3))
    sdsLCBRPS_pcs   = _N.empty((5, 3, 3))    
    sdsDSUAIRPS_pcs   = _N.empty((5, 3, 3))
    sdsLCBAIRPS_pcs   = _N.empty((5, 3, 3))        
    for t0t1 in [[0, 148], [37, 185], [74, 222], [111, 259], [148, 297]]:
        it0t1 += 1
        t0 = t0t1[0]
        t1 = t0t1[1]        
        sdsDSUWTL_pcs[it0t1] = _N.std(prob_mvsDSUWTL[:, :, t0:t1], axis=2)
        sdsRPSWTL_pcs[it0t1] = _N.std(prob_mvsRPSWTL[:, :, t0:t1], axis=2)
        sdsRPSRPS_pcs[it0t1] = _N.std(prob_mvsRPSRPS[:, :, t0:t1], axis=2)
        sdsLCBRPS_pcs[it0t1] = _N.std(prob_mvsLCBRPS[:, :, t0:t1], axis=2)
        sdsDSUAIRPS_pcs[it0t1] = _N.std(prob_mvsDSUAIRPS[:, :, t0:t1], axis=2)                       
        sdsLCBAIRPS_pcs[it0t1] = _N.std(prob_mvsLCBAIRPS[:, :, t0:t1], axis=2)                    
        
    all_CR_sds[pid-1, 0]= _N.max(sdsDSUWTL_pcs, axis=0)
    all_CR_sds[pid-1, 1]= _N.max(sdsRPSWTL_pcs, axis=0)
    all_CR_sds[pid-1, 2]= _N.max(sdsRPSRPS_pcs, axis=0)
    all_CR_sds[pid-1, 3]= _N.max(sdsLCBRPS_pcs, axis=0)
    all_CR_sds[pid-1, 4]= _N.max(sdsDSUAIRPS_pcs, axis=0)
    all_CR_sds[pid-1, 5]= _N.max(sdsLCBAIRPS_pcs, axis=0)
    """
    score[pid-1] = _N.sum(_hnd_dat[:, 2])# / _hnd_dat.shape[0] 



    ####################################
    t_offset = 1
    ties = _N.where(hnd_dat_all[pid-1, start_T:-1, 2] == 0)[0] + t_offset +start_T
    wins = _N.where(hnd_dat_all[pid-1, start_T:-1, 2] == 1)[0] + t_offset + start_T
    loss = _N.where(hnd_dat_all[pid-1, start_T:-1, 2] == -1)[0] + t_offset + start_T
    Rs = _N.where(hnd_dat_all[pid-1, start_T:-1, 0] == 1)[0] + t_offset +start_T
    Ss = _N.where(hnd_dat_all[pid-1, start_T:-1, 0] == 2)[0] + t_offset + start_T
    Ps = _N.where(hnd_dat_all[pid-1, start_T:-1, 0] == 3)[0] + t_offset + start_T
    AIRs = _N.where(hnd_dat_all[pid-1, start_T:-1, 1] == 1)[0] + t_offset +start_T
    AISs = _N.where(hnd_dat_all[pid-1, start_T:-1, 1] == 2)[0] + t_offset + start_T
    AIPs = _N.where(hnd_dat_all[pid-1, start_T:-1, 1] == 3)[0] + t_offset + start_T

    ################################################################
    #  find the variance of prediction values over time for R, S, P
    #
    #  all_AI_preds   214 x 301 x 3
    preds_aft_wins = _N.std(all_AI_preds[pid-1,wins], axis=0)    #  for each subject, 3 numbers - variability of R, P and S across time in after given condition
    preds_aft_ties = _N.std(all_AI_preds[pid-1,ties], axis=0)    
    preds_aft_loss = _N.std(all_AI_preds[pid-1,loss], axis=0)    
    preds_aft_Rs   = _N.std(all_AI_preds[pid-1,Rs], axis=0)    
    preds_aft_Ss   = _N.std(all_AI_preds[pid-1,Ss], axis=0)    
    preds_aft_Ps   = _N.std(all_AI_preds[pid-1,Ps], axis=0)    
    preds_aft_AIRs   = _N.std(all_AI_preds[pid-1,AIRs], axis=0)    
    preds_aft_AISs   = _N.std(all_AI_preds[pid-1,AISs], axis=0)    
    preds_aft_AIPs   = _N.std(all_AI_preds[pid-1,AIPs], axis=0)    

    preds_aft_wtl = [preds_aft_wins, preds_aft_ties, preds_aft_loss]
    for iwtl in range(3):
        bottom = _N.mean(preds_aft_wtl[iwtl])   #  mean of the 3 perceptrons.  1 number
        for irps in range(3):        #  How sure is R perceptron after W?  How sure is P perceptron after W?  etc.
            AIfeatsWTL[pid-1, iwtl, irps] = preds_aft_wtl[iwtl][irps]# / bottom
    preds_aft_rps = [preds_aft_Rs, preds_aft_Ss, preds_aft_Ps]
    for irps1 in range(3):
        bottom = _N.mean(preds_aft_rps[irps1])
        for irps2 in range(3):        
            AIfeatsRPS[pid-1, irps1, irps2] = preds_aft_rps[irps1][irps2]# / bottom
    preds_aft_airps = [preds_aft_AIRs, preds_aft_AISs, preds_aft_AIPs]
    for iairps in range(3):
        bottom = _N.mean(preds_aft_airps[iairps])
        for irps in range(3):        
            AIfeatsAIRPS[pid-1, iairps, irps] = preds_aft_airps[iairps][irps]# / bottom

    ############################
    srtd_AI_preds = _N.sort(all_AI_preds[pid-1], axis=1)
    #certainty = (srtd_AI_preds[:, 2] - srtd_AI_preds[:, 1]) / (srtd_AI_preds[:, 2] - srtd_AI_preds[:, 0])

    certainty = (srtd_AI_preds[:, 2] - srtd_AI_preds[:, 1]) /  _N.mean(srtd_AI_preds[:, 2] - srtd_AI_preds[:, 0])   ##  same as AIfeatsRPS
    tttt=0
    AIcerts[pid-1, 0] = _N.std(certainty[wins+tttt])
    AIcerts[pid-1, 1] = _N.std(certainty[ties+tttt])
    AIcerts[pid-1, 2] = _N.std(certainty[loss+tttt])
    AIcerts[pid-1, 3] = _N.std(certainty[AIRs+tttt])
    AIcerts[pid-1, 4] = _N.std(certainty[AISs+tttt])
    AIcerts[pid-1, 5] = _N.std(certainty[AIPs+tttt])
    AIcerts[pid-1, 6] = _N.std(certainty[Rs+tttt])
    AIcerts[pid-1, 7] = _N.std(certainty[Ss+tttt])
    AIcerts[pid-1, 8] = _N.std(certainty[Ps+tttt])
            

AIfeats = _N.empty((len(partIDs), 3, 3, 3))
AIfeats[:, 0] = AIfeatsWTL
AIfeats[:, 1] = AIfeatsRPS
AIfeats[:, 2] = AIfeatsAIRPS


#ths = _N.where((AQ28scrs > 35))[0]
if look_at_AQ:
    ths = _N.where((AQ28scrs < 111))[0]
else:
    ths = _N.arange(len(partIDs_okgames))
even_play = _N.where(resp_t_cvs < 4)[0]
#ths = _N.where((AQ28scrs > 0))[0]
good = _N.intersect1d(ths, resp_times_OK)
#good = _N.arange(len(partIDs))
good = _N.intersect1d(not_outliers, good)
good = _N.intersect1d(notmany_repeats, good)
good = _N.intersect1d(correct_ngames, good)
#good = _N.intersect1d(even_play, good)
filtdat = good

# #  filtdat_okgames goes up to 362

# ths  = _N.arange(len(partIDs_okgames))
# even_play = _N.where(resp_t_cvs_okgames < 4)[0]
# good = _N.intersect1d(ths, resp_times_OK_okgames)
# good = _N.intersect1d(not_outliers_okgames, good)
# good = _N.intersect1d(correct_ngames_okgames, good)
# #good = _N.intersect1d(even_play, good)
# filtdat_okgames = good

dmp_dat = {}
dmp_dat["all_CR_corrs"] = all_CR_corrs
dmp_dat["all_CR_corrs_pairID"] = all_CR_corrs_pairID
dmp_dat["all_CR_corrs_trms"] = all_CR_corrs_trms
dmp_dat["all_fr01s"] = all_fr01s
dmp_dat["all_CR_sds"] = all_CR_sds
dmp_dat["WTL_aft_WTL"] = WTL_aft_WTL
dmp_dat["RPS_aft_AIRPS"] = RPS_aft_AIRPS
dmp_dat["wtlStreaks"] = wtlStreaks
dmp_dat["AIfeats"] = AIfeats
dmp_dat["AIcerts"] = AIcerts
dmp_dat["AQ28scrs"]    = AQ28scrs
#dmp_dat["all_AI_weights"]    = all_AI_weights
#dmp_dat["all_AI_preds"]    = all_AI_preds
dmp_dat["soc_skils"] = soc_skils
dmp_dat["imag"] = imag
dmp_dat["rout"] = rout
dmp_dat["switch"] = switch
dmp_dat["fact_pat"] = fact_pat
dmp_dat["ans_soc_skils"] = ans_soc_skils
dmp_dat["ans_imag"] = ans_imag
dmp_dat["ans_rout"] = ans_rout
dmp_dat["ans_switch"] = ans_switch
dmp_dat["ans_fact_pat"] = ans_fact_pat
dmp_dat["label"] = label
dmp_dat["win"] = win
dmp_dat["time_aft_los"] = time_aft_los
dmp_dat["time_aft_tie"] = time_aft_tie
dmp_dat["time_aft_win"] = time_aft_win
dmp_dat["smth"] = smth
dmp_dat["netwins"] = netwins
dmp_dat["ages"] = ages
dmp_dat["gens"] = gens
dmp_dat["Engs"] = Engs
dmp_dat["partIDs"] = partIDs
dmp_dat["partIDs_okgames"] = partIDs_okgames
dmp_dat["hnd_dat_all"] = hnd_dat_all
dmp_dat["filtdat"] = filtdat
#dmp_dat["filtdat_okgames"] = filtdat_okgames
dmpout = open(workdirFN("%(dat)s_AQ28_vs_RPS_features%(flp)s_%(v)d_of_%(vs)s_%(wt)d%(w)d%(s)d.dmp" % {"v" : visit, "vs" : svisits, "wt" : win_type, "w" : win, "s" : smth, "wd" : os.environ["RPSWORKDIR"], "flp" : sFlipped, "dat" : data}), "wb")
pickle.dump(dmp_dat, dmpout, -1)
dmpout.close()
    

# # # for star in ["AQ28scrs", "soc_skils", "imag", "rout", "switch", "fact_pat"]:
# # #     exec("tar = %s" % star)
# # #     print(";;;;;;;;;;;;;;;;;;;;;;;;;;")
# # #     pc, pv = _ss.pearsonr(AIcerts[filtdat, 0], tar[filtdat])
# # #     print("% .3f" % pc)
# # #     pc, pv = _ss.pearsonr(AIcerts[filtdat, 1], tar[filtdat])
# # #     print("% .3f" % pc)
# # #     pc, pv = _ss.pearsonr(AIcerts[filtdat, 2], tar[filtdat])    
# # #     print("% .3f" % pc)
# # #     pc, pv = _ss.pearsonr(AIcerts[filtdat, 3], tar[filtdat])        
# # #     print("% .3f" % pc)
# # #     pc, pv = _ss.pearsonr(AIcerts[filtdat, 4], tar[filtdat])            
# # #     print("% .3f" % pc)
# # #     pc, pv = _ss.pearsonr(AIcerts[filtdat, 5], tar[filtdat])                
# # #     print("% .3f" % pc)
# # #     pc, pv = _ss.pearsonr(AIcerts[filtdat, 6], tar[filtdat])                    
# # #     print("% .3f" % pc)
# # #     pc, pv = _ss.pearsonr(AIcerts[filtdat, 7], tar[filtdat])                        
# # #     print("% .3f" % pc)
# # #     pc, pv = _ss.pearsonr(AIcerts[filtdat, 8], tar[filtdat])                            
# # #     print("% .3f" % pc)
    

# # # ################  DEMOGRAPHICS
# # # #  age distribution (sex just give a %)
# # # #  win distribution
# # # #  subscore distribution
# # # #
# # # fig = _plt.figure(figsize=(4, 2.8))
# # # _plt.hist(ages[filtdat], bins=_N.arange(0.5, 17.5, 1), color="black")
# # # _plt.xticks(_N.arange(1, 17), ["< 18", "18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64","65-69", "70-74", "75-79", "80-84", "85-89", "> 90"], rotation=60, fontsize=13)
# # # _plt.yticks(fontsize=13)
# # # _plt.xlabel("age", fontsize=15)
# # # _plt.xlim(0.5, 16.5)
# # # _plt.ylabel("# participants", fontsize=15)
# # # fig.subplots_adjust(bottom=0.32, left=0.2, top=0.98, right=0.98)
# # # _plt.savefig("age_dist")

# # # fig = _plt.figure(figsize=(4, 2.2))
# # # _plt.hist(gens, bins=[-1.5, -0.5, 0.5, 1.5, 2.5], color="black")
# # # _plt.xlabel("Gender", fontsize=15)
# # # _plt.xticks([-1, 0, 1, 2], ["N/A", "M", "F", "Non-binary"], fontsize=14)
# # # _plt.yticks(fontsize=14)
# # # _plt.ylabel("# participants", fontsize=15)
# # # fig.subplots_adjust(bottom=0.24, left=0.2)
# # # _plt.savefig("GenderDist")

# # # fig = _plt.figure(figsize=(4, 4.1))
# # # si = -1
# # # for star in ["soc_skils", "imag", "rout", "switch", "fact_pat", "AQ28scrs", ]:
# # #     si += 1
# # #     fig.add_subplot(6, 1, si+1)
# # #     exec("factor = %s" % star)
# # #     low=_N.min(factor[filtdat])
# # #     hig=_N.max(factor[filtdat])
# # #     _plt.text(70, 28, star)
# # #     _plt.hist(factor[filtdat], bins=_N.arange(-0.5, 96.5, 1), color="black")
# # #     #_plt.ylim(0, 40)
# # #     _plt.ylim(0, 40)
# # #     _plt.yticks([0, 15, 30])
# # #     if si == 5:
# # #         _plt.xticks(_N.arange(0, 100, 10))
# # #         _plt.xlabel("Score")
# # #     else:
# # #         _plt.xticks([])        
# # # fig.subplots_adjust(hspace=0.18, top=0.98)
# # # _plt.savefig("subfactor_score_dist")

# # # fig = _plt.figure(figsize=(3.3, 3))
# # # _plt.scatter(dur[filtdat], netwins[filtdat], s=3, color="black")
# # # _plt.xlabel("mean response time (s)", fontsize=14)
# # # _plt.ylabel("net wins", fontsize=14)
# # # _plt.xticks(_N.arange(0, 14, 2), fontsize=12)
# # # _plt.yticks(fontsize=12)
# # # _plt.grid(ls="--")
# # # fig.subplots_adjust(bottom=0.18, left=0.25, right=0.98, top=0.98)
# # # _plt.savefig("pace_score")
# # """
# # for star in ["soc_skils", "imag", "rout", "switch", "fact_pat", "AQ28scrs", ]:
# #     exec("tar = %s" % star)
# #     print(star)
# #     for ifr in range(6):
# #         for ic in range(3):
# #             for ia in range(3):
# #                 pc, pv = _ss.pearsonr(tar[filtdat], lrs[filtdat, ifr, ic, ia])
# #                 if _N.abs(pc) > 0.2:
# #                     print(pc)
# # """

# # # ranks_of_lotsof0s = lm["fr_lotsof0s"]
# # # ranks_of_lotsof1s = lm["fr_lotsof1s"]
# # # sdsfr=_N.sum(ranks_of_cmps1+ranks_of_cmps2, axis=3)
# # # sds=ranks_of_cmps2
# # # sds = _N.sum(ranks_of_lotsof0s-2*ranks_of_lotsof1s, axis=3)
# # # sds = _N.sum(ranks_of_lotsof1s/(ranks_of_lotsof0s+1), axis=2)
# # # sds = ranks_of_lotsof1s/(ranks_of_lotsof0s+1)
# # # sdsfr=_N.sum(_N.sum(ranks_of_cmps1+ranks_of_cmps2, axis=3), axis=2)
# # # sds=all_CR_sds
# # # for star in ["soc_skils", "imag", "rout", "switch", "fact_pat", "AQ28scrs", ]:
# # #     exec("tar = %s" % star)
# # #     for ifr in range(6):
# # #         for ic in range(3):
# # #             for ia in range(3):            
# # #             pc, pv = _ss.pearsonr(tar[filtdat], sds[filtdat, ifr, ic])
# # #             if _N.abs(pc) > 0.2:
# # #                 print("%(ifr)d   %(ic)d %(ia)d    %(pc).3f" % {"ic" : ic, "ia" : ia, "pc" : pc, "ifr": ifr})


# # # 0   2 0    0.212
# # # 3   2 2    0.212
# # # 0   0 1    -0.239
# # # 3   0 2    -0.239


# # # fig = _plt.figure(figsize=(10, 5))
# # # _plt.suptitle(partID)
# # # fig.add_subplot(6, 1, 1)
# # # for ic in range(3):
# # #     for ia in range(3):
# # #         #_plt.plot(sds[:, 0, ic, ia])
# # #         _plt.plot(prob_mvsDSUWTL[ic, ia])
# # # fig.add_subplot(6, 1, 2)
# # # for ic in range(3):
# # #     for ia in range(3):
# # #         _plt.plot(prob_mvsRPSWTL[ic, ia])        
# # #         #_plt.plot(sds[:, 3, ic, ia])

# # # fig.add_subplot(6, 1, 3)
# # # for ic in range(3):
# # #     for ia in range(3):
# # #         #_plt.plot(sds[:, 0, ic, ia])
# # #         _plt.plot(prob_mvsLCBRPS[ic, ia])
# # # fig.add_subplot(6, 1, 4)
# # # for ic in range(3):
# # #     for ia in range(3):
# # #         _plt.plot(prob_mvsRPSRPS[ic, ia])        
# # #         #_plt.plot(sds[:, 3, ic, ia])

# # # fig.add_subplot(6, 1, 5)
# # # for ic in range(3):
# # #     for ia in range(3):
# # #         #_plt.plot(sds[:, 0, ic, ia])
# # #         _plt.plot(prob_mvsLCBAIRPS[ic, ia])
# # # fig.add_subplot(6, 1, 6)
# # # for ic in range(3):
# # #     for ia in range(3):
# # #         _plt.plot(prob_mvsDSUAIRPS[ic, ia])        
# # #         #_plt.plot(sds[:, 3, ic, ia])

# sfiltdat = _N.array(filtdat)
# #_N.random.shuffle(sfiltdat)
# pcs = _N.empty(1431 + 54 + 9 + 9)
# for star in ["AQ28scrs"]:
#     exec("tar = %s" % star)
#     for ic in range(1431):            
#         pc, pv = _ss.pearsonr(tar[sfiltdat], all_CR_corrs[filtdat, ic])
#         pcs[ic] = pc
#         if _N.abs(pc) > 0.35:
#             print(all_CR_corrs_trms[ic])
#             print("%(ic)d     %(pc) .3f   %(pv).1e" % {"ic" : ic, "pc" : pc, "pv" : pv})
#             # fig = _plt.figure()
#             # _plt.suptitle("%(ic)d     pc=%(pc) .3f   pv=%(pv).1e" % {"ic" : ic, "pc" : pc, "pv" : pv})
#             # _plt.scatter(tar[sfiltdat], all_CR_corrs[filtdat, ic])

            
# iii = -1
# for star in ["AQ28scrs"]:
#     exec("tar = %s" % star)
#     for ic in range(3):
#         for ia in range(3):
#             iii += 1
#             pc, pv = _ss.pearsonr(tar[filtdat], AIfeatsWTL[filtdat, ic, ia])
#             pcs[1431+iii] = pc            
#             #if _N.abs(pc) > 0.3:
#             #    print("%(ic)d     %(pc) .3f   %(pv).1e" % {"ic" : ic, "ia" : ia, "pc" : pc, "pv" : pv})

# iii = -1
# for star in ["AQ28scrs"]:
#     exec("tar = %s" % star)
#     for ic in range(3):
#         for ia in range(3):
#             iii += 1
#             pc, pv = _ss.pearsonr(tar[filtdat], WTL_aft_WTL[filtdat, ic, ia])
#             pcs[1431+9+iii] = pc            
#             #if _N.abs(pc) > 0.3:
#             #    print("%(ic)d     %(pc) .3f   %(pv).1e" % {"ic" : ic, "ia" : ia, "pc" : pc, "pv" : pv})
            

# sfiltdat = _N.array(filtdat)
# iii = -1
# #_N.random.shuffle(sfiltdat)
# for star in ["AQ28scrs"]:
#     exec("tar = %s" % star)
#     for ifr in range(6):
#         for ic in range(3):
#             for ia in range(3):
#                 iii+=1
#                 pc, pv = _ss.pearsonr(tar[sfiltdat], all_CR_sds[filtdat, ifr, ic, ia])
#                 pcs[1431+18+iii] = pc
#                 if _N.abs(pc) > 0.3:
#                     print("%(ic)d     %(pc) .3f   %(pv).1e" % {"ic" : ic, "ia" : ia, "pc" : pc, "pv" : pv})

            
# _N.savetxt("%s_pcs.txt" % data, pcs, fmt="%.5f")
