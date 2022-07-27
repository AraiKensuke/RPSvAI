import RPSvAI.models.empirical_ken as empirical
import RPSvAI.utils.misc as _Am
import numpy as _N
import matplotlib.pyplot as _plt
import GCoh.eeg_util as _eu
import RPSvAI.utils.read_taisen as _rt
from RPSvAI.utils.dir_util import workdirFN
import os
import pickle
import RPSvAI.constants as _AIconst
import scipy.stats as _ss
import glob
import RPSvAI.simulation.simulate_prcptrn as sim_prc

win_type = 1   #  window is of fixed number of games
win_type = 2  #  window is of fixed number of games that meet condition 
wins= 3
gk_w = 1

visit = 1
visits=[1]

#  These are ParticipantIDs.

expt="TMB2"

datetms = []
#fp = open("%(f)sfns_%(v)d.txt" % {"f" : expt, "v" : visit}, "r")
if expt=="TMB2":
    fp = open("%(e)sfns_%(v)s.txt" % {"e" : expt, "v" : str(visits)}, "r")
    contents = fp.readlines()
    for fn in contents:
        datetms.append(fn.rstrip())
    fp.close()

if expt[0:6] == "SIMHUM":
    nSIMHUM=int(expt[6:])
    syr    = "201101%s" % ("0%d" % nSIMHUM if nSIMHUM < 10 else str(nSIMHUM))
    yr_dir    = "DATA/%(e)s/%(syr)s" % {"e" : expt, "syr" : syr}

    candidate_dirs = os.listdir(yr_dir)

    datetms = []
    for i in range(len(candidate_dirs)):
        if candidate_dirs[i][0:8] == syr:
            datetms.append(candidate_dirs[i])
    
pcs_123 = _N.zeros((len(datetms), 3))
id = -1


for datetm in datetms:
    id += 1

    print("...............   %s" % datetm)
    flip_human_AI = False

    #fig = _plt.figure(figsize=(11, 11))

    sran = ""

    SHUFFLES = 5
    a_s = _N.zeros((len(datetms), SHUFFLES+1))
    acs = _N.zeros((len(datetms), SHUFFLES+1, 61))

    if gk_w > 0:
        gk = _Am.gauKer(gk_w)
        gk /= _N.sum(gk)
    sFlip = "_flip" if flip_human_AI else ""

    label="%(win_t)d%(wins)d%(gkw)d" % {"wins" : wins, "gkw" : gk_w, "win_t" : win_type}

    out_dir = workdirFN("%(dfn)s" % {"dfn" : datetm})

    if not os.access(out_dir, os.F_OK):
        os.mkdir(out_dir)
    out_dir = workdirFN("%(dfn)s/%(lbl)s" % {"dfn" : datetm, "lbl" : label})
    if not os.access(out_dir, os.F_OK):
        os.mkdir(out_dir)

    #  Get raw game-by-game RPS data.
    #  Also gets things like changing AI parameters
    td, start_time, end_time, UA, cnstr, inp_meth, ini_percep, fin_percep, gt_dump = _rt.return_hnd_dat(datetm, has_useragent=True, has_start_and_end_times=True, has_constructor=True, expt=expt, visit=visit)

    weights = None
    if (ini_percep is not None) and (expt[0:6] != "SIMHUM"):
        weights, preds, iw = sim_prc.recreate_percep_istate(td, ini_percep, fin_percep)

    if win_type == 1:
        ngsDSUWTL, ngsRPSWTL, ngsDSURPS, ngsDSUAIRPS, all_tds, TGames  = empirical.empirical_NGS(datetm, win=wins, SHUF=SHUFFLES, flip_human_AI=flip_human_AI, expt=expt, visit=visit, dither_unobserved=False)
    elif win_type == 2:
        ngsDSUWTL, ngsRPSWTL, ngsDSURPS, ngsDSUAIRPS, ngsRPSRPS, ngsRPSAIRPS, all_tds, TGames  = empirical.empirical_NGS_concat_conds(datetm, win=wins, SHUF=SHUFFLES, flip_human_AI=flip_human_AI, expt=expt, visit=visit)
        #ngs, ngsRPS, ngsDSURPS, ngsDSUAIRPS, all_tds, TGames  = empirical.empirical_NGS_concat_conds(datetm, win=wins, SHUF=SHUFFLES, flip_human_AI=flip_human_AI, expt=expt, visit=visit)
        #ngsDSURPS, all_tds, TGames  = empirical.empirical_NGS_concat_conds_DSURPS(datetm, win=wins, SHUF=SHUFFLES, flip_human_AI=flip_human_AI, expt=expt, visit=visit)

    if ngsDSUWTL is not None:
        fNGSDSUWTL = _N.empty((SHUFFLES+1, ngsDSUWTL.shape[1], ngsDSUWTL.shape[2]), dtype=_N.float16)
        fNGSRPSWTL = _N.empty((SHUFFLES+1, ngsDSUWTL.shape[1], ngsDSUWTL.shape[2]), dtype=_N.float16)
        fNGSDSURPS = _N.empty((SHUFFLES+1, ngsDSUWTL.shape[1], ngsDSUWTL.shape[2]), dtype=_N.float16)
        fNGSDSUAIRPS = _N.empty((SHUFFLES+1, ngsDSUWTL.shape[1], ngsDSUWTL.shape[2]), dtype=_N.float16)
        fNGSRPSRPS = _N.empty((SHUFFLES+1, ngsDSUWTL.shape[1], ngsDSUWTL.shape[2]), dtype=_N.float16)                        
        fNGSRPSAIRPS = _N.empty((SHUFFLES+1, ngsDSUWTL.shape[1], ngsDSUWTL.shape[2]), dtype=_N.float16)            
        #fNGSSTSW = _N.empty((SHUFFLES+1, ngsSTSW.shape[1], ngsSTSW.shape[2]))            
        t_ms = _N.mean(_N.diff(all_tds[0, :, 3]))
        for sh in range(SHUFFLES+1):
            for i in range(9):
                if gk_w > 0:
                    fNGSDSUWTL[sh, i] = _N.convolve(ngsDSUWTL[sh, i], gk, mode="same")
                    fNGSRPSWTL[sh, i] = _N.convolve(ngsRPSWTL[sh, i], gk, mode="same")
                    fNGSDSURPS[sh, i] = _N.convolve(ngsDSURPS[sh, i], gk, mode="same")
                    fNGSDSUAIRPS[sh, i] = _N.convolve(ngsDSUAIRPS[sh, i], gk, mode="same")
                    fNGSRPSRPS[sh, i] = _N.convolve(ngsRPSRPS[sh, i], gk, mode="same")
                    fNGSRPSAIRPS[sh, i] = _N.convolve(ngsRPSAIRPS[sh, i], gk, mode="same")                                                                        
                else:
                    fNGSDSUWTL[sh, i] = ngsDSUWTL[sh, i]
                    fNGSRPS[sh, i] = ngsRPSWTL[sh, i]
                    fNGSDSURPS[sh, i] = ngsDSURPS[sh, i]
                    fNGSDSUAIRPS[sh, i] = ngsDSUAIRPS[sh, i]
                    fNGSRPSRPS[sh, i] = ngsRPSRPS[sh, i]
                    fNGSRPSAIRPS[sh, i] = ngsRPSAIRPS[sh, i]                                        
        # for sh in range(SHUFFLES+1):
        #     for i in range(6):
        #         if gk_w > 0:
        #             fNGSSTSW[sh, i] = _N.convolve(ngsSTSW[sh, i], gk, mode="same")
        #         else:
        #             fNGSSTSW[sh, i] = ngsSTSW[sh, i]

        pklme = {}
        pklme["cond_probsDSUWTL"] = fNGSDSUWTL
        pklme["cond_probsRPSWTL"] = fNGSRPSWTL
        pklme["cond_probsDSURPS"] = fNGSDSURPS
        pklme["cond_probsDSUAIRPS"] = fNGSDSUAIRPS
        pklme["cond_probsRPSRPS"] = fNGSRPSRPS
        pklme["cond_probsRPSAIRPS"] = fNGSRPSAIRPS            
        pklme["inp_meth"] = _N.array(inp_meth.split(" "), dtype=_N.int)
        pklme["all_tds"] = all_tds

        pklme["start_time"] = start_time
        pklme["end_time"] = end_time
        if weights is not None:
            pklme["AI_weights"] = weights[iw]
            pklme["AI_preds"] = preds[iw]

        dmp = open("%(dir)s/variousCRs_%(visit)d.dmp" % {"dir" : out_dir, "visit" : visit}, "wb")
        pickle.dump(pklme, dmp, -1)
        dmp.close()


