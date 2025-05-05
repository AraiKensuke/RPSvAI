import RPSvAI.models.empirical as empirical
import RPSvAI.utils.misc as _Am
import numpy as _N
import matplotlib.pyplot as _plt
import RPSvAI.utils.read_taisen as _rt
from RPSvAI.utils.dir_util import workdirFN, datadirFN
import os
import pickle
import scipy.stats as _ss
import glob
import RPSvAI.simulation.simulate_prcptrn as sim_prc
import time
#

win_type = 1   #  window is of fixed number of games
win_type = 2  #  window is of fixed number of games that meet condition 
wins= 3
gk_w = 1

visit = 1     
visits=[1]   #  visits=[1, 2] = data with same paticipant doing 2 games, in this case visit=1 or 2 to calculate for 

#  These are ParticipantIDs.

flip_HUMAI=False
expt="TMB2"
datetms = []
#fp = open("%(f)sfns_%(v)d.txt" % {"f" : expt, "v" : visit}, "r")
if expt=="TMB2":
    fp = open("%(e)sfns_%(v)s.txt" % {"e" : expt, "v" : str(visits)}, "r")
    contents = fp.readlines()
    for fn in contents:
        datetms.append(fn.rstrip())
    fp.close()
if expt=="CogWeb":
    fp = open("%(e)sfns_%(v)s.txt" % {"e" : expt, "v" : str(visits)}, "r")
    contents = fp.readlines()
    for fn in contents:
        datetms.append(fn.rstrip())
    fp.close()

if expt[0:6] == "SIMHUM":
    nSIMHUM=int(expt[6:])
    syr    = "201101%s" % ("0%d" % nSIMHUM if nSIMHUM < 10 else str(nSIMHUM))
    yr_dir    = datadirFN("%(e)s/%(syr)s" % {"e" : expt, "syr" : syr})

    candidate_dirs = os.listdir(yr_dir)

    datetms = []
    for i in range(len(candidate_dirs)):
        if candidate_dirs[i][0:8] == syr:
            datetms.append(candidate_dirs[i])

def calc_for_date(idat):
    datetm = datetms[idat]
#for datetm in datetms:
    print("...............   %s" % datetm)
    flip_human_AI = False

    #fig = _plt.figure(figsize=(11, 11))

    sran = ""

    SHUFFLES = 400
    a_s = _N.zeros((len(datetms), SHUFFLES+1))

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
    td, start_time, end_time, UA, cnstr, inp_meth, ini_percep, fin_percep, gt_dump = _rt.return_hnd_dat(datetm, has_useragent=True, has_start_and_end_times=True, has_constructor=True, expt=expt, visit=visit, flip_human_AI=flip_HUMAI)

    weights = None
    if (ini_percep is not None) and (expt[0:6] != "SIMHUM"):
        weights, preds, iw = sim_prc.recreate_percep_istate(td, ini_percep, fin_percep)

    t1 = time.time()
    if win_type == 1:
        ngsDSUWTL, ngsRPSWTL, ngsDSURPS, ngsDSUAIRPS, all_tds, TGames  = empirical.empirical_NGS(datetm, win=wins, SHUF=SHUFFLES, flip_human_AI=flip_HUMAI, expt=expt, visit=visit, dither_unobserved=False)
    elif win_type == 2:
        ngsDSUWTL, ngsRPSWTL, ngsDSURPS, ngsLCBRPS, ngsDSUAIRPS, ngsLCBAIRPS, all_tds, TGames  = empirical.empirical_NGS_concat_conds(datetm, win=wins, SHUF=SHUFFLES, flip_human_AI=flip_HUMAI, expt=expt, visit=visit, atmidwin=True)
        #ngsDSURPS= _N.array(ngsRPSRPS)
        # R|r -> S|r      P|r -> U|r       S|r -> D|r
        # R|p -> D|p      P|p -> S|p       S|p -> U|p
        # R|s -> U|s      P|s -> D|s       S|s -> S|s
        # [0,0]=[0,1]     [0,1]=[0,2]      [0,2]=[0,0]
        # [1,0]=[1,0]     [1,1]=[1,1]      [1,2]=[1,2]
        # [2,0]=[2,2]     [2,1]=[2,0]      [2,2]=[2,1]

        # ngsDSURPS[0,0] = ngsRPSRPS[0,2]
        # ngsDSURPS[0,1] = ngsRPSRPS[0,0]
        # ngsDSURPS[0,2] = ngsRPSRPS[0,1]  # ngsDSURPS[1] same 
        # ngsDSURPS[2,0] = ngsRPSRPS[2,1]
        # ngsDSURPS[2,1] = ngsRPSRPS[2,2]
        # ngsDSURPS[2,2] = ngsRPSRPS[2,0]

        if showDebug and not useMP:
            fig = _plt.figure(figsize=(11, 10))
            _plt.suptitle("expt = %s" % expt)
            _plt.subplot2grid((4, 2), (0, 0), rowspan=1)
            _plt.plot(all_tds[0, :, 0], marker=".", ms=2)
            _plt.plot(all_tds[0, :, 1]+2.5, marker=".", ms=2)
            _plt.plot(all_tds[0, :, 2]+7, marker=".", ms=2)        
            _plt.ylim(0.5, 9)
            _plt.xticks([])
            _plt.xlim(0, ngsDSUWTL.shape[2])        
            _plt.subplot2grid((4, 2), (0, 1), rowspan=1)
            _plt.plot(all_tds[0, :, 0], marker=".", ms=5)
            _plt.plot(all_tds[0, :, 1]+2.5, marker=".", ms=2)
            _plt.plot(all_tds[0, :, 2]+7, marker=".", ms=2)                
            _plt.ylim(0.5, 9)
            _plt.xticks([])
            _plt.xlim(0, ngsDSUWTL.shape[2])
            print(all_tds.shape)

            isc = -1
            for sCR in ["DSUWTL", "RPSWTL", "DSURPS", "LCBRPS", "DSUAIRPS", "LCBAIRPS"]:
                isc += 1
                if sCR == "DSUWTL":
                    CR = ngsDSUWTL
                elif sCR == "RPSWTL":
                    CR = ngsRPSWTL
                elif sCR == "DSURPS":
                    CR = ngsDSURPS
                elif sCR == "LCBRPS":
                    CR = ngsLCBRPS
                elif sCR == "DSUAIRPS":
                    CR = ngsDSUAIRPS
                elif sCR == "LCBAIRPS":
                    CR = ngsLCBAIRPS
                    
                _plt.subplot2grid((4, 2), (1 + isc//2, isc%2), rowspan=1)
                _plt.title(sCR)
                running = 0
                for i in range(9):
                    _plt.plot(CR[0, i] + running)
                    running += 1.2
                    if (i % 3 == 2):                                                
                        running += 0.5
                    _plt.xticks([])
                    _plt.xlim(0, CR.shape[2])
                
            _plt.savefig("Debug_NGS_%s" % expt)
        
        #ngs, ngsRPS, ngsDSURPS, ngsDSUAIRPS, all_tds, TGames  = empirical.empirical_NGS_concat_conds(datetm, win=wins, SHUF=SHUFFLES, flip_human_AI=flip_human_AI, expt=expt, visit=visit)
        #ngsDSURPS, all_tds, TGames  = empirical.empirical_NGS_concat_conds_DSURPS(datetm, win=wins, SHUF=SHUFFLES, flip_human_AI=flip_human_AI, expt=expt, visit=visit)
    t2 = time.time()

    if ngsDSUWTL is not None:
        #############################################333
        fNGSDSUWTL = _N.empty((SHUFFLES+1, ngsDSUWTL.shape[1], ngsDSUWTL.shape[2]), dtype=_N.float16)
        fNGSRPSWTL = _N.empty((SHUFFLES+1, ngsDSUWTL.shape[1], ngsDSUWTL.shape[2]), dtype=_N.float16)
        #############################################333        
        fNGSRPSRPS = _N.empty((SHUFFLES+1, ngsDSUWTL.shape[1], ngsDSUWTL.shape[2]), dtype=_N.float16)
        fNGSLCBRPS = _N.empty((SHUFFLES+1, ngsDSUWTL.shape[1], ngsDSUWTL.shape[2]), dtype=_N.float16)                            
        #############################################333        
        fNGSDSUAIRPS = _N.empty((SHUFFLES+1, ngsDSUWTL.shape[1], ngsDSUWTL.shape[2]), dtype=_N.float16)
        fNGSLCBAIRPS = _N.empty((SHUFFLES+1, ngsDSUWTL.shape[1], ngsDSUWTL.shape[2]), dtype=_N.float16)
        ############################################# UNUSED        
        fNGSDSURPS = _N.empty((SHUFFLES+1, ngsDSUWTL.shape[1], ngsDSUWTL.shape[2]), dtype=_N.float16)
        fNGSRPSAIRPS = _N.empty((SHUFFLES+1, ngsDSUWTL.shape[1], ngsDSUWTL.shape[2]), dtype=_N.float16)
        
        
        #fNGSSTSW = _N.empty((SHUFFLES+1, ngsSTSW.shape[1], ngsSTSW.shape[2]))            
        t3 = time.time()
        t_ms = _N.mean(_N.diff(all_tds[0, :, 3]))
        print(all_tds.shape)
        for sh in range(SHUFFLES+1):
            for i in range(9):
                if gk_w > 0:
                    fNGSDSUWTL[sh, i] = _N.convolve(ngsDSUWTL[sh, i], gk, mode="same")
                    fNGSRPSWTL[sh, i] = _N.convolve(ngsRPSWTL[sh, i], gk, mode="same")
                    fNGSDSURPS[sh, i] = _N.convolve(ngsDSURPS[sh, i], gk, mode="same")
                    fNGSDSUAIRPS[sh, i] = _N.convolve(ngsDSUAIRPS[sh, i], gk, mode="same")
                    fNGSLCBAIRPS[sh, i] = _N.convolve(ngsLCBAIRPS[sh, i], gk, mode="same")                    
                    #fNGSRPSRPS[sh, i] = _N.convolve(ngsRPSRPS[sh, i], gk, mode="same")
                    #fNGSRPSAIRPS[sh, i] = _N.convolve(ngsRPSAIRPS[sh, i], gk, mode="same")
                    fNGSLCBRPS[sh, i] = _N.convolve(ngsLCBRPS[sh, i], gk, mode="same")                                                                                                                
                else:
                    fNGSDSUWTL[sh, i] = ngsDSUWTL[sh, i]
                    fNGSRPSWTL[sh, i] = ngsRPSWTL[sh, i]
                    fNGSDSURPS[sh, i] = ngsDSURPS[sh, i]
                    fNGSDSUAIRPS[sh, i] = ngsDSUAIRPS[sh, i]
                    fNGSRPSRPS[sh, i] = ngsRPSRPS[sh, i]
                    fNGSRPSAIRPS[sh, i] = ngsRPSAIRPS[sh, i]
                    fNGSLCBAIRPS[sh, i] = ngsLCBAIRPS[sh, i]
                    fNGSLCBRPS[sh, i] = ngsLCBRPS[sh, i]                                
        t4 = time.time()
        pklme = {}
        pklme["cond_probsDSUWTL"] = fNGSDSUWTL
        pklme["cond_probsRPSWTL"] = fNGSRPSWTL
        #######################################################        
        pklme["cond_probsDSUAIRPS"] = fNGSDSUAIRPS
        pklme["cond_probsLCBAIRPS"] = fNGSLCBAIRPS        
        #######################################################
        #pklme["cond_probsRPSAIRPS"] = fNGSRPSAIRPS
        pklme["cond_probsRPSRPS"] = fNGSRPSRPS
        pklme["cond_probsDSURPS"] = fNGSDSURPS                
        pklme["cond_probsLCBRPS"] = fNGSLCBRPS
        pklme["inp_meth"] = _N.array(inp_meth.split(" "), dtype=_N.int16)
        pklme["all_tds"] = all_tds

        pklme["start_time"] = start_time
        pklme["end_time"] = end_time
        if weights is not None:
            pklme["AI_weights"] = weights[iw]
            pklme["AI_preds"] = preds[iw]

        sFlipped = "_flipped" if flip_HUMAI else ""
        dmp_fn= "%(dir)s/variousCRs%(flp)s_%(visit)d.dmp" % {"dir" : out_dir, "visit" : visit, "flp" : sFlipped}
        print(out_dir)
        dmp = open(dmp_fn, "wb")
            
        pickle.dump(pklme, dmp, -1)
        dmp.close()
        t5 = time.time()
        print("%.3f" % (t2-t1))
        print("%.3f" % (t3-t2))
        print("%.3f" % (t4-t3))
        print("%.3f" % (t5-t4))
        print("DONE    %s" % str(datetm))

        

#for proj in projs:
# Initiate variables for multiprocess. (Process, Pipe)
p = []
res = {}

useMP = True
showDebug=False
if useMP:
    from multiprocessing import Pool    
    p = Pool(14)   #  add max number of processes here 
    for iad in range(len(datetms)):
        p.apply_async(calc_for_date, args=(iad,))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
else:
    showDebug=True    
    for iad in range(1):
        calc_for_date(iad)
