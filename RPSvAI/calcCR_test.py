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
#expt="SIMHUM17"

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
    
    #datetms = ["20110101_0000-00", "20110101_0000-01", "20110101_0000-02", "20110101_0000-03", "20110101_0000-04", 
               # "20110101_0000-05", "20110101_0000-06", "20110101_0000-07", "20110101_0000-08", "20110101_0000-09",
               # "20110101_0000-10", "20110101_0000-11", "20110101_0000-12", "20110101_0000-13", "20110101_0000-14",
               # "20110101_0000-15", "20110101_0000-16", "20110101_0000-17", "20110101_0000-18", "20110101_0000-19",
               # "20110101_0000-20", "20110101_0000-21", "20110101_0000-22", "20110101_0000-23", "20110101_0000-24",
               # "20110101_0000-25", "20110101_0000-26", "20110101_0000-27", "20110101_0000-28", "20110101_0000-29",
               # "20110101_0000-30", "20110101_0000-31", "20110101_0000-32", "20110101_0000-33", "20110101_0000-34",
               # "20110101_0000-35", "20110101_0000-36", "20110101_0000-37", "20110101_0000-38", "20110101_0000-39",
               # "20110101_0000-40", "20110101_0000-41", "20110101_0000-42", "20110101_0000-43", "20110101_0000-44",
               # "20110101_0000-45", "20110101_0000-46", "20110101_0000-47", "20110101_0000-48", "20110101_0000-49",
               # "20110101_0000-50", "20110101_0000-51", "20110101_0000-52", "20110101_0000-53", "20110101_0000-54",
               # "20110101_0000-55", "20110101_0000-56", "20110101_0000-57","20110101_0000-58", "20110101_0000-59",]
pcs_123 = _N.zeros((len(datetms), 3))
id = -1


for datetm in datetms:
    id += 1

    print("...............   %s" % datetm)
    flip_human_AI = False

    #fig = _plt.figure(figsize=(11, 11))

    for cov in [_AIconst._WTL]:#, _AIconst._HUMRPS, _AIconst._AIRPS]:
        scov = _AIconst.sCOV[cov]
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

        td, start_time, end_time, UA, cnstr, inp_meth, ini_percep, fin_percep, gt_dump = _rt.return_hnd_dat(datetm, has_useragent=True, has_start_and_end_times=True, has_constructor=True, expt=expt, visit=visit)

        weights = None
        if (ini_percep is not None) and (expt[0:6] != "SIMHUM"):
            weights, preds, iw = sim_prc.recreate_percep_istate(td, ini_percep, fin_percep)

        if win_type == 1:
            ngs, ngsRPS, ngsDSURPS, ngsDSUAIRPS, all_tds, TGames  = empirical.empirical_NGS(datetm, win=wins, SHUF=SHUFFLES, flip_human_AI=flip_human_AI, expt=expt, visit=visit, dither_unobserved=False)
        elif win_type == 2:
            ngs, ngsRPS, ngsDSURPS, ngsDSUAIRPS, ngsRPSRPS, ngsRPSAIRPS, all_tds, TGames  = empirical.empirical_NGS_concat_conds(datetm, win=wins, SHUF=SHUFFLES, flip_human_AI=flip_human_AI, expt=expt, visit=visit)
            #ngs, ngsRPS, ngsDSURPS, ngsDSUAIRPS, all_tds, TGames  = empirical.empirical_NGS_concat_conds(datetm, win=wins, SHUF=SHUFFLES, flip_human_AI=flip_human_AI, expt=expt, visit=visit)
            #ngsDSURPS, all_tds, TGames  = empirical.empirical_NGS_concat_conds_DSURPS(datetm, win=wins, SHUF=SHUFFLES, flip_human_AI=flip_human_AI, expt=expt, visit=visit)

        if ngs is not None:
            fNGS = _N.empty((SHUFFLES+1, ngs.shape[1], ngs.shape[2]), dtype=_N.float16)
            fNGSRPS = _N.empty((SHUFFLES+1, ngs.shape[1], ngs.shape[2]), dtype=_N.float16)
            fNGSDSURPS = _N.empty((SHUFFLES+1, ngs.shape[1], ngs.shape[2]), dtype=_N.float16)
            fNGSDSUAIRPS = _N.empty((SHUFFLES+1, ngs.shape[1], ngs.shape[2]), dtype=_N.float16)
            fNGSRPSRPS = _N.empty((SHUFFLES+1, ngs.shape[1], ngs.shape[2]), dtype=_N.float16)                        
            fNGSRPSAIRPS = _N.empty((SHUFFLES+1, ngs.shape[1], ngs.shape[2]), dtype=_N.float16)            
            #fNGSSTSW = _N.empty((SHUFFLES+1, ngsSTSW.shape[1], ngsSTSW.shape[2]))            
            t_ms = _N.mean(_N.diff(all_tds[0, :, 3]))
            for sh in range(SHUFFLES+1):
                for i in range(9):
                    if gk_w > 0:
                        fNGS[sh, i] = _N.convolve(ngs[sh, i], gk, mode="same")
                        fNGSRPS[sh, i] = _N.convolve(ngsRPS[sh, i], gk, mode="same")
                        fNGSDSURPS[sh, i] = _N.convolve(ngsDSURPS[sh, i], gk, mode="same")
                        fNGSDSUAIRPS[sh, i] = _N.convolve(ngsDSUAIRPS[sh, i], gk, mode="same")
                        fNGSRPSRPS[sh, i] = _N.convolve(ngsRPSRPS[sh, i], gk, mode="same")
                        fNGSRPSAIRPS[sh, i] = _N.convolve(ngsRPSAIRPS[sh, i], gk, mode="same")                                                                        
                    else:
                        fNGS[sh, i] = ngs[sh, i]
                        fNGSRPS[sh, i] = ngsRPS[sh, i]
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
            pklme["cond_probs"] = fNGS
            pklme["cond_probsRPS"] = fNGSRPS            
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

            dmp = open("%(dir)s/%(cov)s_%(visit)d.dmp" % {"cov" : scov, "dir" : out_dir, "visit" : visit}, "wb")
            pickle.dump(pklme, dmp, -1)
            dmp.close()


