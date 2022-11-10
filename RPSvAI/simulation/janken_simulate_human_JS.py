#import RPSvAI.simulation.prcptrn2dw as prcptrn
import RPSvAI.simulation.prcptrnJS as prcptrn
#import RPSvAI.simulation.samk as samk
import numpy as _N
import time as _tm
import RPSvAI.simulation.janken_switch_hands_multi as _jsh
import matplotlib.pyplot as _plt
import datetime
import pickle
import RPSvAI.utils.read_taisen as _rt
from RPSvAI.utils.dir_util import datadirFN, workdirFN
import os

_NME = 0
_MC1 = 1
_MC2 = 2
_PRC = 3

month_str = ["Jan","Feb", "Mar", "Apr", "May", "Jun",
             "Jul","Aug", "Sep", "Oct", "Nov", "Dec"]

nohist_crats   = _N.array([0, 0.3333333333, 0.6666666666, 1])

###  [Win_stay, Lose_switch, ]
#T0           = build_T([3., 0.3], [3., .3], [3., 0.3])
#   T0 basically says if I lose, don't switch
cyclic_strat_chg = False
cyclic_jmp       = -1   #  or -1

####  WEAK rules
#     stay, change, change   -->
#     change, change, change   -->  If perc. knows my current move, it is quite certain next move is going to be different
#    change, change, stay
####  STRONG rules
#     stay, change, change
#  [win_stay, win_change]    [tie_stay, tie_change],   [lose_stay, lose_change]

#  [p(stay | win) p(go_to_weaker | win) p(go_to_stronger | win)]
#  [p(stay | tie) p(go_to_weaker | tie) p(go_to_stronger | tie)]
#  [p(stay | los) p(go_to_weaker | los) p(go_to_stronger | los)]

e = 0.08

DNe = [1-2*e, e, e] #  prob stay, down, up | COND
STe = [e, 1-2*e, e]
UPe = [e, e, 1-2*e]

Trepertoire = [[DNe, DNe, UPe],
               [DNe, UPe, UPe],
               [DNe, STe, UPe],
               [DNe, DNe, STe],
               [UPe, DNe, UPe],
               [UPe, STe, UPe],
               [STe, DNe, UPe],
               [STe, DNe, STe],
               [STe, UPe, STe]]

#Trepertoire = [[[1/3, 1/3, 1/3], [1/3, 1/3, 1/3], [1/3, 1/3, 1/3]]]

#  next_hand(T, wtl, last_hand)


max_hands  = 500

vs_human   = False     #  is there dynamics in the human hand selection?
hist_dep_hands = False   # if vs_human is False, does AI play random or vs rule
comp       = _PRC  #  _MC1, _MC2, _PRC
mc_decay   = 0.1

#   percept vs human
#   percept vs computer (hist_dep)
#   percept vs computer (not hist_dep)
#   Nash_eq vs human
#   Nash_eq vs computer (hist_dep)
#   Nash_eq vs computer (not hist_dep)


REPS       = 60

chg        = _N.zeros(REPS)
fws        = _N.zeros((REPS, 3), dtype=_N.int)

obs_go_prbs = _N.ones((max_hands+1, 3))*-1

now     = datetime.datetime.now()
day     = "%02d" % now.day
mnthStr = month_str[now.month-1]
year    = "%d" % (now.year-2000)
hour    = "%02d" % now.hour
minute  = "%02d" % now.minute
second  = "%02d" % now.second
jh_fn_mod = "rpsm_%(yr)s%(mth)s%(dy)s-%(hr)s%(min)s-%(sec)s" % {"yr" : year, "mth" : mnthStr, "dy" : day, "hr" : hour, "min" : minute, "sec" : second}

nRules = 6
iCurrStrat = 0

expt = "SIMHUM17"
nSIMHUM=int(expt[6:])
syr    = "201101%s" % ("0%d" % nSIMHUM if nSIMHUM < 10 else str(nSIMHUM))
expt_dir  = datadirFN(expt)
yr_dir    = datadirFN("%(e)s/%(syr)s" % {"e" : expt, "syr" : syr})

if not os.access(expt_dir, os.F_OK):
    os.mkdir(expt_dir)
if not os.access(yr_dir, os.F_OK):        
    os.mkdir(yr_dir)
else:
    os.system("rm -rf %s/*" % yr_dir)

switch_T_shrt0 = 10
switch_T_long0 = 17

XTs1   = 1-0.1*_N.random.rand(REPS) 
XTs2   = 1+0.5*_N.random.rand(REPS)

for rep in range(REPS):
    switch_T_shrt = int(XTs1[rep]*switch_T_shrt0)
    switch_T_long = int(XTs2[rep]*switch_T_long0)

    strt_chg_intvs    = _N.random.randint(switch_T_shrt, switch_T_long, size=max_hands)
    nChgL = _N.random.randint(3, 7)
    nPCS = len(strt_chg_intvs) // nChgL

    if rep % 2 == 0:  #  correlated rule change lengths
        up = 0
        for i in range(nPCS):
            if up == 1:
                strt_chg_intvs[i*nChgL:(i+1)*nChgL] = _N.sort(strt_chg_intvs[i*nChgL:(i+1)*nChgL])
                up = 0
            else:
                strt_chg_intvs[i*nChgL:(i+1)*nChgL] = _N.sort(strt_chg_intvs[i*nChgL:(i+1)*nChgL])[::-1]
                up = 1
            
    strt_chg_times    = _N.cumsum(strt_chg_intvs)
    uptohere          = len(_N.where(strt_chg_times < max_hands)[0])
    strt_chg_times01  = _N.zeros(max_hands+1, dtype=_N.int)
    strt_chg_times01[strt_chg_times[0:uptohere]] = 1


    Ts_timeseries = []

    sec    = rep % 60
    srep = "0%d" % sec if sec < 10 else "%d" % sec

    minute = rep // 60
    smin = "0%d" % minute if minute < 10 else "%d" % minute
    
    date    = "%(yr)s_00%(mn)s-%(srep)s" % {"srep" : srep, "mn" : smin, "yr" : syr}
    par_dir = "%(yrd)s/%(date)s" % {"date" : date, "e" : expt, "yrd" : yr_dir}
    out_dir = "%s/1" % par_dir    
    if not os.access(par_dir, os.F_OK):
        os.mkdir(par_dir)
    if not os.access(out_dir, os.F_OK):        
        os.mkdir(out_dir)


    rules = _N.random.choice(len(Trepertoire), size=nRules, replace=False)
    lTs = []
    for nr in range(nRules):
        lTs.append(Trepertoire[nr])
    Ts = _N.array(lTs)
    ######  int main 
    #int i,pred,m,v[3],x[3*N+1],w[9*N+3],fw[3];

    N = 2
    #  initialize
    # v = _N.zeros(3)                  # inputs into predictive units
    # x = _N.zeros(3*N+1)              # past moves by player 
    # x[3*N]=-1                        # threshold      x[3*N] never changes
    # w = _N.zeros(9*N+3)              # weights
    fw= _N.zeros(3, dtype=_N.int)    #  cum win, draw, lose

    HAL9000 = prcptrn.perceptronJS(N)

    m=1
    pairs = []

    quit  = False
    hds   = -1

    t00    = _tm.time()

    prev_w = True
    prev_m = 1

    iCurrStrat = 0
    N_strategies = Ts.shape[0]

    prevM = 1
    prevWTL = 1

    #prev_gcp_wtl = _N.zeros((9, 1), dtype=_N.int)
    prev_gcp_wtl = _N.zeros(9, dtype=_N.int)  #  prev goo chok paa win tie los
    prev_gcp_wtl_unob = _N.zeros(9, dtype=_N.int)

    iSinceLastStrChg = 0

    switch_ts = []

    initial = _N.random.choice(['1','2','3'])    
    pair = "11"

    while not quit:
        hds += 1

        switched = 0
        #  first, perceptron prediction of player move

        #pred=_N.random.randint(1, 4) if vs_NME else HAL9000.predict(m,x,w,v, update=(hds%20==0))   # pred is prediction of user move (1,2,3) */

        pred=_N.int(HAL9000.predict(int(pair[1])))   # pred is prediction of user move (1,2,3) */)

        #pred=_N.random.randint(1, 4) if vs_NME else HAL9000.predict(m,x,w,v)   # pred is prediction of user move (1,2,3) */
        #pred=_N.random.randint(1, 4) if vs_NME else HAL9000.predict(m,x,w,v, update=True, uw=0.0001)   # pred is prediction of user move (1,2,3) */

        if N_strategies > 1:
            if strt_chg_times01[hds] == 1:
                candidate = _N.random.randint(0, Ts.shape[0])
                while candidate == iCurrStrat:
                    candidate = _N.random.randint(0, Ts.shape[0])
                iCurrStrat = candidate
                switch_ts.append(hds)

        #print("prevM  %d" % prevM)
        prev_gcp_wtl[:] = 0
        #  if prev was goo     did we win tie lose
        #  if prev was choki   did we win tie lose
        prev_gcp_wtl[(prevM-1)*3+prevWTL-1] = 1  #
        #  the latent state is p(change | win), p(change | tie), p(change | lose)
        #  p(change | win) = p(change | win)  == 
        #  p(R | S, W) + p(P | S, W)
        #  p(S | R, W) + p(P | R, W)
        #  p(S | P, W) + p(R | P, W)

        #  p(change | tie)

        if hist_dep_hands:
            #  the probs is 
            m = _jsh.next_hand(Ts[iCurrStrat], prevWTL, prevM)  #  m is 1, 2, 3     #  prob of 3 hands
            #m = hds%3+1

            Ts_timeseries.append(Ts[iCurrStrat])
        else:  #  not hist_dep_hands
            rnd = _N.random.rand()
            m = _N.where((rnd >= nohist_crats[0:-1]) & (rnd < nohist_crats[1:]))[0][0]+1
        prevM = m                        
        #print("m: %(m)d    currStr %(cs)d" % {"m" : m, "cs" : iCurrStrat})

        if hds >= max_hands:
            quit = True
        pair="%(prd)d%(m)d" % {"prd" : int(pred), "m" : m}

        if not quit:
            # show perceptron's move

            #print("<->%d:   " % ((pred+1)%3+1)) 

            #   m = (pred+1)%3+1 would beat the perceptron's prediction
            #   pred=1(goo)   :(pred+1)%3+1=3(paa)
            #   pred=2(choki) :(pred+1)%3+1=1(goo)
            #   pred=3(paa)   :(pred+1)%3+1=2(choki)*/

            #  who won, lost or tied?
            
            if pred==m:  #  human move predicted, so perceptron wins
                fw[2] += 1
                prevWTL = 3
                #pairs.append([m, (pred+1)%3+1, -1, switched])
                pairs.append([m, (pred+1)%3+1, -1, iCurrStrat])
            elif (pred%3) == (m-1):
                fw[0] += 1
                prevWTL = 1
                #  human player wins because
                #   pred%3=0(paa predicted) so percep outputs choki, but player goo
                #   pred%3=1(goo predicted) so percep outputs paa, but player choki
                #   pred%3=2(choki predicted) so percep outputs goo, but player paa
                #   Toda comment 2003/05/28 */
                pairs.append([m, (pred+1)%3+1, 1, iCurrStrat])
            else:
                fw[1] += 1
                pairs.append([m, (pred+1)%3+1, 0, iCurrStrat])
                prevWTL = 2            



    t01 = _tm.time()
    #print("game duration  %.1f" % (t01-t00))    

    fws[rep] = fw
    #print("    PER %(pc)d,  TIE %(tie)d, [[HUMAN]] %(hum)d   UpOrDown %(updn)d" % {"pc" : fw[2], "tie" : fw[1], "hum" : fw[0], "updn" : (fw[0] - fw[2])})

    file_nm = "/Users/arai/nctc/Workspace/RPSvAI_SimDAT/%s.dat" % jh_fn_mod
    gt_file_nm = "/Users/arai/nctc/Workspace/RPSvAI_SimDAT/%s_GT.dat" % jh_fn_mod
    #u_fnm = uniqFN("SimDAT/%s" % file_nm, serial=True)
    #u_fnm_gt = uniqFN("SimDAT/%s" % gt_file_nm, serial=True)
    hnd_dat = _N.array(pairs, dtype=_N.int)
    #_N.savetxt(file_nm, hnd_dat, fmt="%d %d % d %d")
    #print("janken match data: %s" % file_nm)
    #if hist_dep_hands and (not vs_human):
    
    pklme = {}
    pklme["hnd_dat"] = hnd_dat
    if hist_dep_hands:
        tts = _N.array(Ts_timeseries)[0:max_hands]
        tempW = _N.array(tts[:, 0, 0])
        tempT = _N.array(tts[:, 1, 0])
        tempL = _N.array(tts[:, 2, 0])
        tts[:, 0, 0] = tts[:, 0, 1]
        tts[:, 0, 1] = tempW
        tts[:, 1, 0] = tts[:, 1, 1]
        tts[:, 1, 1] = tempT
        tts[:, 2, 0] = tts[:, 2, 1]
        tts[:, 2, 1] = tempL
    
        pklme["Ts_timeseries"] = tts
    
    print(_N.sum(hnd_dat[:, 2]))
    print("-----------------   %s" % out_dir)
    dmp = open("%(od)s/block1_AI.dmp" % {"od" : out_dir}, "wb")
    pickle.dump(pklme, dmp, -1)
    dmp.close()
    #_rt.write_hnd_dat(hnd_dat, "/Users/arai/Sites/taisen/DATA/SIMHUM/20110101/20110101_0000-55/1", "block1_AI")
    _rt.write_hnd_dat(hnd_dat, out_dir, "block1_AI")

