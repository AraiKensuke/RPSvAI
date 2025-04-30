#!/usr/bin/python

import numpy as _N
import RPSvAI.utils.read_taisen as _rt
import scipy.io as _scio
import scipy.stats as _ss
import matplotlib.pyplot as _plt
import RPSvAI.utils.read_taisen as _rd
import os
import sys
import pickle
from RPSvAI.utils.dir_util import workdirFN, outdirFN
_plt.rcParams['text.usetex'] = True

__1st__ = 0
__2nd__ = 1
__ALL__ = 2

_ME_WTL = 0
_ME_RPS = 1

_SHFL_KEEP_CONT  = 0
_SHFL_NO_KEEP_CONT  = 1

def depickle(s):
    import pickle
    with open(s, "rb") as f:
        lm = pickle.load(f)
    return lm

id = 0

win_type = 2   #  window is of fixed number of games
#win_type = 1  #  window is of fixed number of games that meet condition 
win     = 3
#win     = 4
smth    = 1
#smth    = 3

win_type = 2  #  window is of fixed number of games that meet condition 
win     = 3
smth    = 1
#win_type = 2   #  window is of fixed number of games
win_type = 2  #  window is of fixed number of games that meet condition 
win     = 3
smth    = 1
label=str(win_type*100 + win*10 + smth)
lblsz=14
tksz=12

if os.access("reliabilityDB.dmp", os.F_OK):
    relPIK = depickle("reliabilityDB.dmp")
    reliabilityDB = relPIK["database"]
else:
    reliabilityDB = {}
    
outdir = "Results_%(wt)d%(w)d%(s)d" % {"wt" : win_type, "w" : win, "s" : smth}

lm1 = depickle(workdirFN("TMB2_AQ28_vs_RPS_features_1_of_1,2_%(wt)d%(w)d%(s)d.dmp" % {"wt" : win_type, "w" : win, "s" : smth}))
lm2 = depickle(workdirFN("TMB2_AQ28_vs_RPS_features_2_of_1,2_%(wt)d%(w)d%(s)d.dmp" % {"wt" : win_type, "w" : win, "s" : smth}))

lmsh1 = depickle(workdirFN("shuffledCRs_5CFs_TMB2_%(w)d_1_1,2" % {"w" : win}))
lmsh2 = depickle(workdirFN("shuffledCRs_5CFs_TMB2_%(w)d_2_1,2" % {"w" : win}))

filtdat1 = lm1["filtdat_okgames"]
filtdat2 = lm2["filtdat_okgames"]
filtdatBoth = _N.intersect1d(filtdat1, filtdat2)


all_CR_sds1    = lm1["all_CR_sds"]      # N x nFR x 3 x 3
all_CR_sds2    = lm2["all_CR_sds"]      # N x nFR x 3 x 3
fr1s = lmsh1["fr_cmp_fluc_rank2"]
fr2s = lmsh2["fr_cmp_fluc_rank2"]
all_CR_corrs1    = lm1["all_CR_corrs"]  # N x 9*nFR*(9*nFR-1)//2
all_CR_corrs2    = lm2["all_CR_corrs"]  # N x 9*nFR*(9*nFR-1)//2
all_CR_corrs_pairID    = lm1["all_CR_corrs_pairID"]
all_CR_corrs_trms    = lm1["all_CR_corrs_trms"]  # N x 9*nFR*(9*nFR-1)//2

AIfeats1    = lm1["AIfeats"]         #  N x 3 x 3 x 3
AIfeats2    = lm2["AIfeats"]         #  N x 3 x 3 x 3
AIcerts1    = lm1["AIcerts"]         #  N x 3 x 3 x 3
AIcerts2    = lm2["AIcerts"]         #  N x 3 x 3 x 3


WTL_aft_WTL1    = lm1["WTL_aft_WTL"]
WTL_aft_WTL2    = lm2["WTL_aft_WTL"]
#WTL_aft_WTL    = lm["RPS_aft_AIRPS"]

nRuleForms    = 6
SHUFFLES = 50
_all_feats1     = _N.ones((all_CR_sds1.shape[0], nRuleForms*9 + (nRuleForms*9)*(nRuleForms*9-1)//2 + 27 + 9))
_all_feats2     = _N.ones((all_CR_sds2.shape[0], nRuleForms*9 + (nRuleForms*9)*(nRuleForms*9-1)//2 + 27 + 9))

ths_feats      = []
all_feats_label= []
idatind = -1
iCorrInd = -1

mn_md = _N.median
SD    = 0.06
pcthr = 0.23

all_shinds = _N.empty((SHUFFLES+1, len(filtdat1)), dtype=_N.int)
shinds0 = _N.arange(len(filtdat1))
for shf in range(SHUFFLES+1):
    shinds = _N.arange(len(filtdat1))
    if shf > 0:
        done = False

        while not done:
            _N.random.shuffle(shinds)            
            if len(_N.where(shinds == shinds0)[0]) == 0:
                done = True
    all_shinds[shf] = shinds

fig = _plt.figure(figsize=(12, 4.4))

iGroup = -1
#########################################  SDS
for ifr in range(6):
    iGroup += 1

    shf_pcs = _N.empty(SHUFFLES+1)
    for shf in range(SHUFFLES+1):
        shinds = all_shinds[shf]
        ij = -1
        pcs = []
        for i in range(3):
            for j in range(3):
                ij += 1

                #pc, pv = _ss.pearsonr(all_CR_sds1[:, ifr, i, j], all_CR_sds2[shinds, ifr, i, j])
                pc, pv = _ss.pearsonr(fr1s[:, ifr, i, j], fr2s[shinds, ifr, i, j])
                pcs.append(pc)
        shf_pcs[shf] = mn_md(pcs)

    _plt.scatter(iGroup + SD*_N.random.randn(SHUFFLES), shf_pcs[1:], color="#FFDDDD", s=4)
    _plt.plot([iGroup-0.3, iGroup+0.3], [shf_pcs[0], shf_pcs[0]], color="red", lw=2)
    #_plt.plot([iGroup-0.3, iGroup+0.3], [0, 0], color="blue", lw=2)        

iGroup += 1
iGroup += 1
#########################################  CORR
for icat in range(5):
    iGroup += 1

    shf_pcs = _N.empty(SHUFFLES+1)
    for shf in range(SHUFFLES+1):
        shinds = all_shinds[shf]
        cat_i = _N.where(all_CR_corrs_pairID == icat)[0]
        pcs = _N.empty(cat_i.shape[0])

        for  i in range(cat_i.shape[0]):
            pcs[i], pv = _ss.pearsonr(all_CR_corrs1[:, cat_i[i]], all_CR_corrs2[shinds, cat_i[i]])
        shf_pcs[shf] = mn_md(pcs)

    _plt.scatter(iGroup + SD*_N.random.randn(SHUFFLES), shf_pcs[1:], color="#FFDDDD", s=4)
    _plt.plot([iGroup-0.3, iGroup+0.3], [shf_pcs[0], shf_pcs[0]], color="red", lw=2)
    #_plt.plot([iGroup-0.3, iGroup+0.3], [0, 0], color="blue", lw=2)    
    

################################  AI 
iGroup += 1
iGroup += 1
for icat in range(3):
    iGroup += 1
    AI1 = AIfeats1[:, icat]   #  
    AI2 = AIfeats2[:, icat]

    shf_pcs = _N.empty(SHUFFLES+1)
    for shf in range(SHUFFLES+1):
        shinds = all_shinds[shf]
    
        ij = -1
        pcs      = []

        for i in range(3):
            for j in range(3):
                ij += 1
                pc, pv = _ss.pearsonr(AI1[:, i, j], AI2[shinds, i, j])
                pcs.append(pc)
        shf_pcs[shf] = mn_md(pcs)

    _plt.scatter(iGroup + SD*_N.random.randn(SHUFFLES), shf_pcs[1:], color="#FFDDDD", s=4)
    _plt.plot([iGroup-0.3, iGroup+0.3], [shf_pcs[0], shf_pcs[0]], color="red", lw=2)
    #_plt.plot([iGroup-0.3, iGroup+0.3], [0, 0], color="blue", lw=2)        


################################  AI certs
iGroup += 1
iGroup += 1
iGroup += 1

AI1 = AIcerts1
AI2 = AIcerts2

shf_pcs = _N.empty(SHUFFLES+1)
for shf in range(SHUFFLES+1):
    shinds = all_shinds[shf]
    
    ij = -1
    pcs      = []

    for i in range(9):
        ij += 1
        pc, pv = _ss.pearsonr(AI1[:, ij], AI2[shinds, ij])
        pcs.append(pc)
    shf_pcs[shf] = mn_md(pcs)

_plt.scatter(iGroup + SD*_N.random.randn(SHUFFLES), shf_pcs[1:], color="#FFDDDD", s=4)
_plt.plot([iGroup-0.3, iGroup+0.3], [shf_pcs[0], shf_pcs[0]], color="red", lw=2)
    #_plt.plot([iGroup-0.3, iGroup+0.3], [0, 0], color="blue", lw=2)        

    
# ################################  AI

# # iGroup += 1
# # for icat in range(3):
# #     iGroup += 1
# #     AI1_m1 = AIfeats1_m1[:, icat]   #  
# #     AI2_m1 = AIfeats2_m1[:, icat]

# #     ij = -1
# #     pcs      = []
# #     for i in range(3):
# #         for j in range(3):
# #             ij += 1
# #             pc, pv = _ss.pearsonr(AI1_m1[:, i, j], AI2_m1[shinds, i, j])
# #             pcs.append(pc)
# #     _plt.scatter(iGroup + SD*_N.random.randn(len(pcs)), pcs, color="#FFDDDD", s=4)
# #     _plt.plot([iGroup-0.3, iGroup+0.3], [mn_md(pcs), mn_md(pcs)], color="red", lw=2)
# #     _plt.plot([iGroup-0.3, iGroup+0.3], [0, 0], color="blue", lw=2)
# #     _plt.ylim(-0.8, 0.8)
    
# #     print("AI%(n)d    %(pc).3f" % {"n" : icat, "pc" : mn_md(pcs)})

# #     ################################  AI

# # iGroup += 1
# # for icat in range(3):
# #     iGroup += 1

# #     AI1_m3 = AIfeats1_m3[:, icat]   #  
# #     AI2_m3 = AIfeats2_m3[:, icat]

# #     ij = -1
# #     pcs = []
# #     for i in range(3):
# #         for j in range(3):
# #             ij += 1
# #             pc, pv = _ss.pearsonr(AI1_m3[:, i, j], AI2_m3[shinds, i, j])
# #             pcs.append(pc)

# #     _plt.scatter(iGroup + SD*_N.random.randn(len(pcs)), pcs, color="#FFDDDD", s=4)
# #     _plt.plot([iGroup-0.3, iGroup+0.3], [mn_md(pcs), mn_md(pcs)], color="red", lw=2)
# #     _plt.plot([iGroup-0.3, iGroup+0.3], [0, 0], color="blue", lw=2)
# #     _plt.ylim(-0.8, 0.8)
    
# #     print("AI%(n)d    %(pc).3f" % {"n" : icat, "pc" : mn_md(pcs)})


# ################################  AI
# #pcs = _N.empty(9)
# iGroup += 1
# iGroup += 1
# iGroup += 1
# #non_feat_pcs = []


# shf_pcs = _N.empty(SHUFFLES+1)
# for shf in range(SHUFFLES+1):
#     shinds = all_shinds[shf]

#     pcs      = []

#     for ij in range(9):
#         pc, pv = _ss.pearsonr(AIcerts1[:, ij], AIcerts2[shinds, ij])
#         pcs.append(pc)
#     shf_pcs[shf] = mn_md(pcs)
        
# _plt.scatter(iGroup + SD*_N.random.randn(SHUFFLES), shf_pcs[1:], color="#FFDDDD", s=4)
# _plt.plot([iGroup-0.3, iGroup+0.3], [shf_pcs[0], shf_pcs[0]], color="red", lw=2)
# #_plt.plot([iGroup-0.3, iGroup+0.3], [0, 0], color="blue", lw=2)    
# #print("AIcerts    %(pc).3f" % {"pc" : mn_md(all_pcs)})
    
# # pcs = []#_N.empty(9)
# iGroup += 1
# iGroup += 1
# ################################   WTL_aft_WTL

iGroup += 1
iGroup += 1

iGroup += 1
shf_pcs = _N.empty(SHUFFLES+1)
for shf in range(SHUFFLES+1):
    shinds = all_shinds[shf]

    ij = -1
    pcs      = []
    for i in range(3):
        for j in range(3):
            ij += 1

            pc, pv = _ss.pearsonr(WTL_aft_WTL1[:, i, j], WTL_aft_WTL2[shinds, i, j])
            pcs.append(pc)
    shf_pcs[shf] = mn_md(pcs)

_plt.scatter(iGroup + SD*_N.random.randn(SHUFFLES), shf_pcs[1:], color="#FFDDDD", s=4)
_plt.plot([iGroup-0.3, iGroup+0.3], [shf_pcs[0], shf_pcs[0]], color="red", lw=2)
#_plt.plot([iGroup-0.3, iGroup+0.3], [0, 0], color="blue", lw=2)
_plt.plot([-0.3, iGroup+0.3], [0, 0], color="blue", lw=1, ls="--")    

ticks = ["IR   DCU$|wtl$",
         "IR   RPS$|wtl$",
         "IR   DCU$|rps$",
         "IR   DCU$|r_{\\rm A}p_{\\rm A}s_{\\rm A}$",
         "IR   $\\mbox{D}_{\\rm A}\\mbox{C}_{\\rm A}\\mbox{U}_{\\rm A}|rps$",
         "IR   $\\mbox{D}_{\\rm A}\\mbox{C}_{\\rm A}\\mbox{U}_{\\rm A}|r_{\\rm A}p_{\\rm A}s_{\\rm A}$",
         "", "",
         "corr cat 1",
         "corr cat 2",
         "corr cat 3",
         "corr cat 4",
         "corr cat 5",
         "", "",
         "var of AI\nconfidence wtl",
         "var of AI\nconfidence rps",
         "var of AI confidence\nWTL $r_{\\rm A}p_{\\rm A}s_{\\rm A}$",
         "", "",
         "AIcerts",
         "", "",         
         "WTL aft wtl"]
_plt.xticks(_N.arange(len(ticks)), ticks, rotation=70, fontsize=tksz)
#_plt.xticks(_N.arange(20), ["corr cat 1", "corr cat 2", "corr cat 3", "corr cat 4", "corr cat 5", "", "IR   DKU|wtl", "IR   RPS|wtl", "IR   RPS|rps", "IR   LCB|rps", "IR   DKU|rApAsA", "IR   LCB|rApAsA", "", "Percep var WTL", "Percep var RPS", "Percep var AIRPS", "", "var of AI confidence", "", "WTL aft WTL"], rotation=45, fontsize=tksz)
# #_plt.xticks([0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14, 15, 17, 19], ["corr cat 1", "corr cat 2", "corr cat 3", "corr cat 4", "corr cat 5", "var of frm 1", "var of frm 2", "var of frm 3", "var of frm 4", "var of frm 5", "var of frm 6", "Percep var WTL", "Percep var RPS", "Percep var AIRPS", "var of AI confidence", "WTL aft WTL"], rotation=70, fontsize=tksz)
_plt.xlabel("Feature categories", fontsize=lblsz)
_plt.ylabel("Category Mean of\nfeature measurement\ncorrelation between\nrounds 1 \& 2", fontsize=lblsz)
fig.subplots_adjust(bottom=0.41, left=0.15, right=0.98, top=0.92)        
_plt.ylim(-0.8, 0.8)
_plt.yticks([-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75], fontsize=lblsz)
_plt.savefig(outdirFN("TMB2_reliability_shuf", label))


pcs   = _N.empty((SHUFFLES+1, len(filtdat1)))
pcFRs = _N.empty((SHUFFLES+1, len(filtdat1)))
pcZs  = _N.empty((SHUFFLES+1, len(filtdat1)))
pcZRs  = _N.empty((SHUFFLES+1, len(filtdat1)))
#  Show me CC of pattern of rules

for shf in range(SHUFFLES+1):
    shinds = all_shinds[shf]
    for i in range(len(filtdat1)):
        #fig = _plt.figure(figsize=(13, 14))
        fr1 = lmsh1["fr_cmp_fluc_rank2"][i].flatten()
        fr2 = lmsh2["fr_cmp_fluc_rank2"][shinds[i]].flatten()
        z1  = _N.zeros(54)
        z2  = _N.zeros(54)
        z1[_N.where(fr1 > 190)[0]] = 1
        z2[_N.where(fr2 > 190)[0]] = 1    
        pc, pv = _ss.pearsonr(fr1, fr2)
        fr1r = fr1.reshape((18, 3))
        fr2r = fr2.reshape((18, 3))
        z1r = z1.reshape((18, 3))
        z2r = z2.reshape((18, 3))

        pcFR, pv = _ss.pearsonr(_N.sum(fr1r, axis=1), _N.sum(fr2r, axis=1))
        pc, pv = _ss.pearsonr(fr1, fr2)
        pcZ, pv = _ss.pearsonr(z1, z2)
        pcZR, pv = _ss.pearsonr(_N.mean(z1r, axis=1), _N.mean(z2r, axis=1))    

        print("%(pcFR) .3f   %(pc) .3f" % {"pcFR" : pcFR, "pc" : pc})

        pcs[shf, i] = pc
        pcFRs[shf, i] = pcFR
        pcZs[shf, i] =pcZ
        pcZRs[shf, i] =pcZR

fig = _plt.figure()
for shf in range(SHUFFLES, -1, -1):
    if shf == 0:
        _plt.scatter(_N.arange(len(filtdat1)), pcZs[shf], color="black", s=10)
    else:
        _plt.scatter(_N.arange(len(filtdat1)) + 0.05*_N.random.randn(len(filtdat1)), pcZs[shf], color="#FFDDDD", s=3)
_plt.ylim(-0.8, 0.8)

# # fr1s = lmsh1["fr_cmp_fluc_rank2"]
# # fr2s = lmsh2["fr_cmp_fluc_rank2"]

# # pc_by_FR = _N.empty((6, 3, 3))
# # pc_by_CR = _N.empty((6, 3, 3))
# # for ifr in range(6):
# #     for ic in range(3):
# #         for ia in range(3):        
# #             pcFR, pv = _ss.pearsonr(fr1s[:, ifr,  ic, ia], fr2s[:, ifr,  ic, ia])
# #             pcCR, pv = _ss.pearsonr(all_CR_sds1[:, ifr,  ic, ia], all_CR_sds2[:, ifr,  ic, ia])
# #             pc_by_FR[ifr, ic, ia] = pcFR
# #             pc_by_CR[ifr, ic, ia] = pcCR

