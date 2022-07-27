import RPSvAI.models.empirical_ken as empirical
import RPSvAI.utils.misc as _Am
import numpy as _N
import matplotlib.pyplot as _plt
import GCoh.eeg_util as _eu
import RPSvAI.utils.read_taisen as _rt
from RPSvAI.utils.dir_util import workdirFN, datadirFN
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

visit = 1    #  which visit number data do we want?
visits=[1]   #  if [1], look for data that has at least 1 web page visit to game
             #  if [1, 2] look for data that has at least 2 web page visit to game - ie they played at least 2 x 300 games.  

#  These are ParticipantIDs.

expt="TMB2"  #  experiment name

datetms = []

if expt=="TMB2":   #  data collected online against AI
    fp = open("%(e)sfns_%(v)s.txt" % {"e" : expt, "v" : str(visits)}, "r")
    contents = fp.readlines()
    for fn in contents:
        datetms.append(fn.rstrip())
    fp.close()

if expt[0:6] == "SIMHUM":#  "simulated human" for AI to play against
    nSIMHUM=int(expt[6:])
    syr    = "201101%s" % ("0%d" % nSIMHUM if nSIMHUM < 10 else str(nSIMHUM))
    yr_dir    = "DATA/%(e)s/%(syr)s" % {"e" : expt, "syr" : syr}

    candidate_dirs = os.listdir(yr_dir)

    datetms = []
    for i in range(len(candidate_dirs)):
        if candidate_dirs[i][0:8] == syr:
            datetms.append(candidate_dirs[i])
    
pcs_123 = _N.zeros((len(datetms), 3))
pid = -1


for datetm in datetms[0:1]:
    pid += 1

    print("...............   %s" % datetm)

    #  Get raw game-by-game RPS data.
    #  Also gets things like changing AI parameters
    td, start_time, end_time, UA, cnstr, inp_meth, ini_percep, fin_percep, gt_dump = _rt.return_hnd_dat(datetm, has_useragent=True, has_start_and_end_times=True, has_constructor=True, expt=expt, visit=visit)

    #  look at td.  Col 0 human, col 1 AI, col 2 WTL, col 3 time since web page load
    #  td[:, 0]  is either 1=ROCK, 2=SCISSOR, 3=PAPER
    #  td[:, 1]  is either 1=ROCK, 2=SCISSOR, 3=PAPER
    #  td[:, 2]  is either -1=LOSE, 0=TIE, 1=WIN  (from perspective of human)
    
    #  inp_meth  is 0 (keyboard) or 1 (on-screen RPS icon)
    #  ini_percep    initial perceptron parameter (before 1st game played)
    #  fin_percep    final perceptron parameter (after last game played)
    #  gt_dump       for simulated human, the ground truth conditional response rules at all times used to generate generated moves

    ######  A REALLY simple feature might be "net number of wins"
    #  netwins = _N.sum(td[:, 2])  # Just a sum of the WTL column

    #  recreate the AI weight parameters (evolution of AI during games)
    if ini_percep is not None:
        #  weights  model weights
        #  preds    the prediction values of next move (AI will choose move with largest value of this)
        weights, preds, iw = sim_prc.recreate_percep_istate(td, ini_percep, fin_percep)

    #  ans_* gives the raw answers (1-4 Likert scale) players gave
    ans_soc_skils, ans_rout, ans_switch, ans_imag, ans_fact_pat = _rt.AQ28ans(datadirFN("%(data)s/%(date)s/%(pID)s/AQ29.txt" % {"date" : datetm[0:8], "pID" : datetm, "data" : expt}))
    #  sum(ans_soc_skils) = soc_skils   #  convenience, summed version
    #  soc_skils + rout + switch + imag + fact_pat = AQ28scrs    
    AQ28scrs, soc_skils, rout, switch, imag, fact_pat = _rt.AQ28(datadirFN("%(data)s/%(date)s/%(pID)s/AQ29.txt" % {"date" : datetm[0:8], "pID" : datetm, "data" : expt}))
        


