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

# __1st__ = 0
# __2nd__ = 1
# __ALL__ = 2

# _ME_WTL = 0
# _ME_RPS = 1

# _SHFL_KEEP_CONT  = 0
# _SHFL_NO_KEEP_CONT  = 1

#  sum_sd
#  entropyL
#  isi_cv, isis_corr

def depickle(s):
     import pickle
     with open(s, "rb") as f:
          lm = pickle.load(f)
     return lm

lm = depickle(workdirFN("AIout_1_231.dmp"))

AQ28scrs  = lm[0]
soc_skils = lm[1]
imag      = lm[2]
switch    = lm[3]
rout      = lm[4]
fact_pat  = lm[5]
all_AI_weights = lm[6]
all_AI_preds   = lm[7]
hnd_dat_all    = lm[8]
partIDs        = lm[9]
filtdat        = lm[10]

featsWTL   = _N.empty((len(partIDs), 3, 3))
featsRPS   = _N.empty((len(partIDs), 3, 3))
featsAIRPS = _N.empty((len(partIDs), 3, 3))
feat1W_R = _N.empty(len(partIDs))
feat1T_R = _N.empty(len(partIDs))
feat1L_R = _N.empty(len(partIDs))
feat1W_S = _N.empty(len(partIDs))
feat1T_S = _N.empty(len(partIDs))
feat1L_S = _N.empty(len(partIDs))
feat1W_P = _N.empty(len(partIDs))
feat1T_P = _N.empty(len(partIDs))
feat1L_P = _N.empty(len(partIDs))
feat1R_R = _N.empty(len(partIDs))
feat1S_R = _N.empty(len(partIDs))
feat1P_R = _N.empty(len(partIDs))
feat1R_S = _N.empty(len(partIDs))
feat1S_S = _N.empty(len(partIDs))
feat1P_S = _N.empty(len(partIDs))
feat1R_P = _N.empty(len(partIDs))
feat1S_P = _N.empty(len(partIDs))
feat1P_P = _N.empty(len(partIDs))
feat1AIR_R = _N.empty(len(partIDs))
feat1AIS_R = _N.empty(len(partIDs))
feat1AIP_R = _N.empty(len(partIDs))
feat1AIR_S = _N.empty(len(partIDs))
feat1AIS_S = _N.empty(len(partIDs))
feat1AIP_S = _N.empty(len(partIDs))
feat1AIR_P = _N.empty(len(partIDs))
feat1AIS_P = _N.empty(len(partIDs))
feat1AIP_P = _N.empty(len(partIDs))

t_offset = 1
start_T  = 10

fig = _plt.figure(figsize=(13, 3))
ifeat = -1

corr_pairs = []
corr_pairs_pc    = []
corr_pairs_pv    = []

for pid in range(len(partIDs)):
    ties = _N.where(hnd_dat_all[pid, start_T:-1, 2] == 0)[0] + t_offset +start_T
    wins = _N.where(hnd_dat_all[pid, start_T:-1, 2] == 1)[0] + t_offset + start_T
    loss = _N.where(hnd_dat_all[pid, start_T:-1, 2] == -1)[0] + t_offset + start_T
    Rs = _N.where(hnd_dat_all[pid, start_T:-1, 1] == 1)[0] + t_offset +start_T
    Ss = _N.where(hnd_dat_all[pid, start_T:-1, 1] == 2)[0] + t_offset + start_T
    Ps = _N.where(hnd_dat_all[pid, start_T:-1, 1] == 3)[0] + t_offset + start_T
    AIRs = _N.where(hnd_dat_all[pid, start_T:-1, 0] == 1)[0] + t_offset +start_T
    AISs = _N.where(hnd_dat_all[pid, start_T:-1, 0] == 2)[0] + t_offset + start_T
    AIPs = _N.where(hnd_dat_all[pid, start_T:-1, 0] == 3)[0] + t_offset + start_T
    
    #  find the variance of prediction values over time for R, S, P
    preds_aft_wins = _N.std(all_AI_preds[pid,wins], axis=0)    
    preds_aft_ties = _N.std(all_AI_preds[pid,ties], axis=0)    
    preds_aft_loss = _N.std(all_AI_preds[pid,loss], axis=0)    
    preds_aft_Rs   = _N.std(all_AI_preds[pid,Rs], axis=0)    
    preds_aft_Ss   = _N.std(all_AI_preds[pid,Ss], axis=0)    
    preds_aft_Ps   = _N.std(all_AI_preds[pid,Ps], axis=0)    
    preds_aft_AIRs   = _N.std(all_AI_preds[pid,AIRs], axis=0)    
    preds_aft_AISs   = _N.std(all_AI_preds[pid,AISs], axis=0)    
    preds_aft_AIPs   = _N.std(all_AI_preds[pid,AIPs], axis=0)    

    preds_aft_wtl = [preds_aft_wins, preds_aft_ties, preds_aft_loss]
    for iwtl in range(3):
        bottom = _N.mean(preds_aft_wtl[iwtl])
        for irps in range(3):        
            featsWTL[pid, iwtl, irps] = preds_aft_wtl[iwtl][irps] / bottom
    preds_aft_rps = [preds_aft_Rs, preds_aft_Ss, preds_aft_Ps]
    for irps1 in range(3):
        bottom = _N.mean(preds_aft_rps[irps1])
        for irps2 in range(3):        
            featsRPS[pid, irps1, irps2] = preds_aft_rps[irps1][irps2] / bottom
    preds_aft_airps = [preds_aft_AIRs, preds_aft_AISs, preds_aft_AIPs]
    for iairps in range(3):
        bottom = _N.mean(preds_aft_airps[iairps])
        for irps in range(3):        
            featsAIRPS[pid, iairps, irps] = preds_aft_airps[iairps][irps] / bottom
            
    # feat1W_R[pid] = preds_aft_wins[0] / _N.mean(preds_aft_wins)
    # feat1W_S[pid] = preds_aft_wins[1] / _N.mean(preds_aft_wins)
    # feat1W_P[pid] = preds_aft_wins[2] / _N.mean(preds_aft_wins)
    # feat1T_R[pid] = preds_aft_ties[0] / _N.mean(preds_aft_ties)
    # feat1T_S[pid] = preds_aft_ties[1] / _N.mean(preds_aft_ties)
    # feat1T_P[pid] = preds_aft_ties[2] / _N.mean(preds_aft_ties)
    # feat1L_R[pid] = preds_aft_loss[0] / _N.mean(preds_aft_loss)
    # feat1L_S[pid] = preds_aft_loss[1] / _N.mean(preds_aft_loss)
    # feat1L_P[pid] = preds_aft_loss[2] / _N.mean(preds_aft_loss)

    # feat1R_R[pid] = preds_aft_Rs[0] / _N.mean(preds_aft_Rs)
    # feat1R_S[pid] = preds_aft_Rs[1] / _N.mean(preds_aft_Rs)
    # feat1R_P[pid] = preds_aft_Rs[2] / _N.mean(preds_aft_Rs)
    # feat1S_R[pid] = preds_aft_Ss[0] / _N.mean(preds_aft_Ss)
    # feat1S_S[pid] = preds_aft_Ss[1] / _N.mean(preds_aft_Ss)
    # feat1S_P[pid] = preds_aft_Ss[2] / _N.mean(preds_aft_Ss)
    # feat1P_R[pid] = preds_aft_Ps[0] / _N.mean(preds_aft_Ps)
    # feat1P_S[pid] = preds_aft_Ps[1] / _N.mean(preds_aft_Ps)
    # feat1P_P[pid] = preds_aft_Ps[2] / _N.mean(preds_aft_Ps)

    # feat1AIR_R[pid] = preds_aft_AIRs[0] / _N.mean(preds_aft_AIRs)
    # feat1AIR_S[pid] = preds_aft_AIRs[1] / _N.mean(preds_aft_AIRs)
    # feat1AIR_P[pid] = preds_aft_AIRs[2] / _N.mean(preds_aft_AIRs)
    # feat1AIS_R[pid] = preds_aft_AISs[0] / _N.mean(preds_aft_AISs)
    # feat1AIS_S[pid] = preds_aft_AISs[1] / _N.mean(preds_aft_AISs)
    # feat1AIS_P[pid] = preds_aft_AISs[2] / _N.mean(preds_aft_AISs)
    # feat1AIP_R[pid] = preds_aft_AIPs[0] / _N.mean(preds_aft_AIPs)
    # feat1AIP_S[pid] = preds_aft_AIPs[1] / _N.mean(preds_aft_AIPs)
    # feat1AIP_P[pid] = preds_aft_AIPs[2] / _N.mean(preds_aft_AIPs)


for SHUFFLE in range(1):
    sfiltdat = _N.array(filtdat)
    if SHUFFLE > 0:
        _N.random.shuffle(sfiltdat)
    ifeat = -1

    for feats in [featsWTL, featsRPS, featsAIRPS]:
        for icond in range(3):
            for irsp in range(3):
                #for sfeat in ["feat1W_R", "feat1W_S", "feat1W_P", "feat1T_R", "feat1T_S", "feat1T_P", "feat1L_R", "feat1L_S", "feat1L_P", "feat1R_R", "feat1R_S", "feat1R_P", "feat1S_R", "feat1S_S", "feat1S_P", "feat1P_R", "feat1P_S", "feat1P_P", "feat1AIR_R", "feat1AIR_S", "feat1AIR_P", "feat1AIS_R", "feat1AIS_S", "feat1AIS_P", "feat1AIP_R", "feat1AIP_S", "feat1AIP_P"]:
                pcs = []

                ifeat += 1
                #exec("feat = %s" % sfeat)
                #print(sfeat)
                for star in ["soc_skils", "imag", "switch", "rout", "AQ28scrs"]:
                    exec("tar = %s" % star)
                    pc, pv = _ss.pearsonr(feats[sfiltdat, icond,irsp], tar[filtdat])
                    if _N.abs(pc) > 0.22:
                        print("SHUFFLE    %d" % SHUFFLE)
                        print("------  %(tar)10s   %(pc) .3f"%{"tar" : star, "pc" : pc})
                    pcs.append(pc)
                    corr_pairs.append([_N.array(feats[filtdat, icond, irsp]), _N.array(tar[filtdat])])
                    corr_pairs_pc.append(_N.abs(pc))
                    corr_pairs_pv.append(_N.abs(pv))

                # if SHUFFLE == 0:
                #     fig.add_subplot(1, 3, ifeat+1)
                #     _plt.plot(_N.linspace(0, 1, len(pcs)), _N.sort(_N.abs(pcs)), marker=".", ls="")
                #     _plt.ylim(0,0.35)
                #     _plt.grid()
                #     _plt.suptitle("offset %d" % t_offset)

