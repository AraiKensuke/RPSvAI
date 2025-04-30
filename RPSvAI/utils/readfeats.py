######
######
###### compared to v3, we're calling
######

import numpy as _N
import pickle
import os
from RPSvAI.utils.dir_util import workdirFN, outdirFN
import re

def standardize(y):
    ys = y - _N.mean(y)
    ys /= _N.std(ys)
    return ys

def depickle(s):
    import pickle
    with open(s, "rb") as f:
        lm = pickle.load(f)
    return lm

def translate_name(featname, latex=True):
    if latex:
        fr_acts = [["D", "C", "U"], ["R", "P", "S"],
                   ["D", "C", "U"], ["D", "C", "U"],
                   ["\mbox{D}_{\small{\mbox{A}}}", "\mbox{C}_{\small{\mbox{A}}}", "\mbox{U}_{\small{\mbox{A}}}"],
                   ["\mbox{D}_{\small{\mbox{A}}}", "\mbox{C}_{\small{\mbox{A}}}", "\mbox{U}_{\small{\mbox{A}}}"]]
    else:
        fr_acts = [["D", "C", "U"], ["R", "P", "S"],
                   ["D", "C", "U"], ["D", "C", "U"],
                   ["DA", "CA", "UA"],
                   ["DA", "CA", "UA"]]                   
    if latex:
            fr_conds = [["\mbox{w}", "\mbox{t}", "\mbox{l}"],
                        ["\mbox{w}", "\mbox{t}", "\mbox{l}"],
                        ["\mbox{r}", "\mbox{p}", "\mbox{s}"],
                        ["\mbox{r}_{\small{\mbox{A}}}", "\mbox{p}_{\small{\mbox{A}}}", "\mbox{s}_{\small{\mbox{A}}}"],                
                        ["\mbox{r}", "\mbox{p}", "\mbox{s}"],
                        ["\mbox{r}_{\small{\mbox{A}}}", "\mbox{p}_{\small{\mbox{A}}}", "\mbox{s}_{\small{\mbox{A}}}"]]
    else:
            fr_conds = [["w", "t", "l"],
                        ["w", "t", "l"],
                        ["r", "p", "s"],
                        ["rA", "pA", "sA"],
                        ["r", "p", "s"],
                        ["rA", "pA", "sA"]]

    print(featname)
    if featname[0:3] == "SDS":
        frmwk = int(featname[4])
        conds = int(featname[6])
        acts  = int(featname[8])        
        feat = r"fluc $%(a)s|%(c)s$" % {"c" : fr_conds[frmwk][conds], "a" : fr_acts[frmwk][acts]}
        return feat
    elif featname[0:4] == "corr":
        frmwk1 = int(featname[5])
        conds1 = int(featname[7])
        acts1  = int(featname[9])        
        frmwk2 = int(featname[11])
        conds2 = int(featname[13])
        acts2  = int(featname[15])        
        feat = r"temporal corr $%(a1)s|%(c1)s,%(a2)s|%(c2)s$" % {"c1" : fr_conds[frmwk1][conds1], "a1" : fr_acts[frmwk1][acts1], "c2" : fr_conds[frmwk2][conds2], "a2" : fr_acts[frmwk2][acts2]}
        return feat        
    elif featname[2:5] == "aft":
        outcomes  = ["win", "tie", "lose"]
        conds = ["w", "t", "l"]        
        out    = int(featname[0])
        cnd    = int(featname[6])
        return r"p(%(o)s aft %(c)s)" % {"o" : outcomes[out], "c" : conds[cnd]}
    elif featname[0:6] == "percep":
        eventClass = int(featname[6])   #  WTL class, RPS class, AIRPS class
        afterEvt   = int(featname[13])  #  event type WTL, RPS or AIRPS
        AIactions  = int(featname[15])  #  R, P or S
        eventClasses = ["WTL", "RPS", "AIRPS"]
        aftEvts = [["w", "t", "l"],
                   ["r", "s", "p"],
                   ["r_A", "s_A", "p_A"]]
        actions = ["R", "S", "P"]

        #return "AI variab of %(a)s aft %(e)s" % {"a": actions[AIactions],"e" : aftEvts[eventClass][afterEvt]}
        return "%(a)s prcptron variability aft %(e)s" % {"a": actions[AIactions],"e" : aftEvts[eventClass][afterEvt]}
    elif featname[0:6] == "percep":
        eventClass = int(featname[6])   #  WTL class, RPS class, AIRPS class
        return "prcptroncert%d" % eventClass
    
    

def returnFeatures(win_type, win, smth):
    label=str(win_type*100 + win*10 + smth)

    lm = depickle(workdirFN("TMB2_AQ28_vs_RPS_features_1_of_1_%s.dmp" % label))

    AQ28scores = ["AQ28scrs", "soc_skils", "imag", "rout", "switch", "fact_pat"]
    AQ28scores_ab = ["AQ28", "SS", "IM", "RT", "SW", "FP"]

    #  7+8+4+4+5 =
    soc_skils_use = _N.array([0, 1, 2, 3, 4, 5, 6])
    imag_use      = _N.array([0, 1, 2, 3, 4, 5, 6, 7])
    rout_use      = _N.array([0, 1, 2, 3])
    switch_use    = _N.array([0, 1, 2, 3])
    fact_pat_use  = _N.array([0, 1, 2, 3, 4])

    look_at_AQ    = True
    AQ28scrs      = lm["AQ28scrs"]
    soc_skils     = lm["soc_skils"]
    imag          = lm["imag"]
    rout          = lm["rout"]
    switch        = lm["switch"]
    fact_pat      = lm["fact_pat"]
    label         = lm["label"]
    win           = lm["win"]
    partIDs       = lm["partIDs"]
    filtdat           = lm["filtdat"]
    nDAT = len(partIDs)
    show_mn = True
    mn_mode = 2
    allInds = _N.arange(nDAT)

    all_CR_sds    = lm["all_CR_sds"]      # N x nFR x 3 x 3
    all_CR_corrs    = lm["all_CR_corrs"]  # N x 9*nFR*(9*nFR-1)//2
    all_CR_corrs_pairID    = lm["all_CR_corrs_pairID"]
    all_CR_corrs_trms    = lm["all_CR_corrs_trms"]  # N x 9*nFR*(9*nFR-1)//2
    AIfeats    = lm["AIfeats"]         #  N x 3 x 3 x 3
    AIcerts    = lm["AIcerts"]         #  N x 3 x 3 x 3    
    WTL_aft_WTL    = lm["WTL_aft_WTL"]
    wtlStreaks     = lm["wtlStreaks"]
    nRuleForms    = 6
    _all_feats     = _N.ones((all_CR_sds.shape[0], nRuleForms*9 + (nRuleForms*9)*(nRuleForms*9-1)//2 + 27 + 27 + 27 + 9+9))
    #_all_feats_s     = _N.empty((all_CR_sds.shape[0], nRuleForms*9 + (nRuleForms*9)*(nRuleForms*9-1)//2 + 27 + 9))
    _all_feats_s     = _N.empty((all_CR_sds.shape[0], nRuleForms*9 + (nRuleForms*9)*(nRuleForms*9-1)//2 + 27 + 27 + 27 + 9+9))
    ths_feats      = []
    all_feats_label= []
    idatind = -1
    iCorrInd = -1
    for ifr in range(6):
        for ic in range(3):
            for ia in range(3):
                idatind += 1
                all_feats_label.append("SDS_%(fr)d_%(c)d_%(a)d" % {"fr" : ifr, "c" : ic, "a" : ia})
                _all_feats[:, idatind]= all_CR_sds[:, ifr, ic, ia]
                _all_feats_s[:, idatind]= standardize(all_CR_sds[:, ifr, ic, ia])
                ths_feats.append(idatind)

    for i in range(all_CR_corrs_trms.shape[0]):
        iCorrInd += 1
        if True:
            idatind += 1
            all_feats_label.append("corr_%(fr1)d_%(c1)d_%(a1)d_%(fr2)d_%(c2)d_%(a2)d" % {"fr1" : all_CR_corrs_trms[i, 0], "c1" : all_CR_corrs_trms[i, 1], "a1" : all_CR_corrs_trms[i, 2], "fr2" : all_CR_corrs_trms[i, 3], "c2" : all_CR_corrs_trms[i, 4], "a2" : all_CR_corrs_trms[i, 5]})
            _all_feats[:, idatind]= all_CR_corrs[:, i]
            _all_feats_s[:, idatind]= standardize(all_CR_corrs[:, i])
            ths_feats.append(idatind)

    for iAftEvent in range(3):  # event WTL, RPS or AIRPS
        for ipercep in range(3):  #  perceptron for R, P or S
            for i3Events in range(3):  #  one of the 3 types of events
                idatind += 1
                all_feats_label.append("percep%(ip)d_post_%(iae)d,%(i3)d" % {"ip" : ipercep, "iae" : iAftEvent, "i3" : i3Events})
                _all_feats[:, idatind]= AIfeats[:, iAftEvent, ipercep, i3Events]
                _all_feats_s[:, idatind]= standardize(AIfeats[:, iAftEvent, ipercep, i3Events])
                ths_feats.append(idatind)

    for iAftEvent in range(9):  # event WTL, RPS or AIRPS
        idatind += 1
        all_feats_label.append("percep%d" % iAftEvent)
        _all_feats[:, idatind]= AIcerts[:, iAftEvent]
        _all_feats_s[:, idatind]= standardize(AIcerts[:, iAftEvent])
        ths_feats.append(idatind)
                
    for ic in range(3):  # event WTL, RPS or AIRPS
        for io in range(3):  #  perceptron for R, P or S
            idatind += 1
            all_feats_label.append("%(io)d_aft_%(ic)d" % {"io" : io, "ic": ic})
            _all_feats[:, idatind]= WTL_aft_WTL[:, ic, io]
            _all_feats_s[:, idatind]= standardize(WTL_aft_WTL[:, ic, io])
            ths_feats.append(idatind)        

    all_feats       = _N.ones((all_CR_sds.shape[0], len(ths_feats)))
    all_feats_s     = _N.ones((all_CR_sds.shape[0], len(ths_feats)))
    all_feats       = _all_feats[:, _N.array(ths_feats)]
    all_feats_s     = _all_feats_s[:, _N.array(ths_feats)]

    return filtdat, all_feats, all_feats_s, all_feats_label, AQ28scrs, soc_skils, imag, rout, switch, fact_pat 
