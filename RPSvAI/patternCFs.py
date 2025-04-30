import numpy as _N
import scipy.stats as _ss
import matplotlib.pyplot as _plt
import pickle
from RPSvAI.utils.dir_util import workdirFN, datadirFN, outdirFN

#_plt.rcParams['text.usetex'] = True

def depickle(s):
     import pickle
     with open(s, "rb") as f:
          lm = pickle.load(f)
     return lm
win_type=2
win = 3
gk_w=1
visit = 1
svisits ="1"

flip_HUMAI=False
sFlipped = "_flipped" if flip_HUMAI else ""

label=str(win_type*100 + win*10 + gk_w)
#expt = "SIMHUM"
ylim01=False
#expt="SIMHUM47"
expt="TMB2"
#expt="TMBCW"
#expt="CogWeb"
#expt="SIMHUM80"
#expt="SIMHUM51"
lm = depickle(workdirFN("shuffledCRs_5CFs%(flp)s_%(ex)s_%(w)d_%(v)d_%(vs)s" % {"ex" : expt, "w" : win, "v" : visit, "vs" : svisits, "flp" : sFlipped}))
lmF = depickle(workdirFN("%(ex)s_AQ28_vs_RPS_features%(flp)s_1_of_1_%(lb)s.dmp" % {"lb" : label, "flp" : sFlipped, "ex" : expt}))

#filtdat = lmF["filtdat_okgames"]
filtdat = lmF["filtdat"]
print(filtdat)

ranks0s = lm["fr_lotsof0s"]
ranks1s = lm["fr_lotsof1s"]

ranksclumped0s = lm["fr_clumped0s"]
ranksclumped1s = lm["fr_clumped1s"]


cmps    = 0.01*(95*ranks1s+5*ranksclumped0s)
cmps    = 0.01*(20*ranksclumped1s+75*ranks1s+5*ranksclumped0s)
cmps    = ranks1s

SHUFFLES=400
big_percentile=0.8
OFFs = (ranks0s / SHUFFLES > big_percentile)
ONs  = (ranks1s / SHUFFLES > big_percentile)

bigflucs = OFFs & ONs
all_cnts = _N.zeros(9*6, dtype=_N.int32)

#for ifr in range(6):
ifr = -1
for iifr in [0, 1, 2, 4, 3, 5]:
    ifr += 1
    #iparticipant, iconds, iacts = _N.where(bigflucs[filtdat, ifr])#_N.where(cmps[filtdat, ifr] > 200)
    iparticipant, iconds, iacts = _N.where(cmps[filtdat, iifr] > int(SHUFFLES*0.98))
    iBinsFlat = iconds*3 + iacts
    cnts, bins = _N.histogram(iBinsFlat, bins=(_N.arange(10)-0.5))
    all_cnts[ifr*9:(ifr+1)*9] = cnts

colspan=20
emptycol=15
#fig = _plt.figure(figsize=(11, 6))
#fig = _plt.figure(figsize=(21, 1.5))
fig = _plt.figure(figsize=(4.5, 9.2))
lblsz=20
tksz=17
#_plt.suptitle("Trigger", fontsize=lblsz)

conds      = [["$w$", "$t$", "$l$"], ["$w$", "$t$", "$l$"],
              ["$r$", "$p$", "$s$"], ["$r_A$", "$p_A$", "$s_A$"],
              ["$r$", "$p$", "$s$"], ["$r_A$", "$p_A$", "$s_A$"]]
acts       = [["D", "C", "U"], ["R", "P", "S"],
              ["D", "C", "U"], ["D", "C", "U"],
              ["D$_A$", "C$_A$", "U$_A$"], ["D$_A$", "C$_A$", "U$_A$"]]

# frameworks = ["p(DCU | wtl)", "p(RPS | wtl)",
#               "p(RSP | rsp)", "p(LCB | rsp)",
#               "p(DCU | AIrsp)", "p($\\mbox{D}_A\\mbox{K}_A\\mbox_{U}_A$ | AIrsp)"]
# conds      = [["$w$", "$t$", "$l$"], ["$w$", "$t$", "$l$"],
#               ["$r$", "$s$", "$p$"], ["$r$", "$s$", "$p$"],
#               ["$r_A$", "$s_A$", "$p_A$"], ["$r_A$", "$s_A$", "$p_A$"]]
# acts       = [["D", "C", "U"], ["R", "S", "P"],
#               ["D", "C", "U"], ["D$_A$", "C$_A$", "U$_A$"],
#               ["D", "C", "U"], ["D$_A$", "C$_A$", "U$_A$"]]

if expt == "SIMHUM3":
    maxY = 36  #  for SIMHUM3
else:
    maxY = int(_N.max(all_cnts)*1.08)
dY   = maxY // 3
maxticky = dY*3

rowspan  = 16
emptyrow = 7

Cmp1     = ["black", "grey", "grey"]
Cmp2     = ["grey", "black", "grey"]
Cmp3     = ["grey", "grey", "black"]
Cmp12    = ["black", "black", "grey"]
Cmp13    = ["black", "grey", "black"]
Cmp23    = ["grey", "black", "black"]
Cmp123   = ["black", "black", "black"]

Rnd      = ["grey", "grey", "grey"]
RndU     = ["blue", "blue", "blue"]
barcolors= [[RndU, RndU, RndU],
            [RndU, RndU, RndU],
            [RndU, RndU, RndU],
            [RndU, RndU, RndU],
            [RndU, RndU, RndU],
            [RndU, RndU, RndU]]

if expt == "SIMHUM20":
    #Frmwks      = [1, 2, 3, 4]
    barcolors   = [[Rnd, Rnd, Rnd],
                   [Cmp1, Cmp1, Cmp3], #  
                   [Cmp2, Cmp2, Rnd],  #  
                   [Cmp3, Rnd, Cmp1],  #  
                   [Cmp3, Cmp1, Cmp2],
                   [Rnd, Rnd, Rnd]]
if expt == "SIMHUM30":
    #Frmwks      = [1, 2, 3, 4]
    barcolors   = [[Rnd, Rnd, Rnd],
                   [Rnd, Rnd, Rnd],
                   [Rnd, Rnd, Rnd],
                   [Rnd, Rnd, Rnd],
                   [Rnd, Rnd, Rnd],
                   [Rnd, Rnd, Rnd]]                   
if expt == "SIMHUM21":
    #Frmwks      = [0, 3, 4, 5]
    barcolors = [[Cmp3, Cmp2, Cmp1], #
                 [Rnd, Rnd, Rnd],
                 [Rnd, Rnd, Rnd],
                 [Cmp2, Rnd, Cmp3],  #  
                 [Cmp1, Rnd, Cmp1],  #  
                 [Cmp2, Cmp2, Cmp2]]  #

#if expt == "SIMHUM22":
if (expt == "SIMHUM2") or (expt == "SIMHUM5"):
    #Frmwks      = [1, 2, 4, 5]
    barcolors = [[Rnd, Rnd, Rnd],
                 [Cmp2, Cmp3, Cmp1], #  
                 [Cmp2, Rnd, Cmp3],  #
                 [Rnd, Rnd, Rnd],
                 [Cmp3, Rnd, Cmp2],  #  
                 [Cmp2, Cmp3, Cmp3]]  # 

#if expt == "SIMHUM23":
if (expt == "SIMHUM3") or (expt == "SIMHUM6"):
    #Frmwks      = [0, 1, 4, 5]
    barcolors = [[Cmp2, Cmp3, Cmp1], #  
                 [Rnd, Cmp3, Cmp2],  #
                 [Rnd, Rnd, Rnd],
                 [Rnd, Rnd, Rnd],
                 [Cmp3, Cmp1, Rnd],  #  
                 [Cmp3, Cmp1, Cmp2]]  #
# if expt == "SIMHUM10":
#     #Frmwks      = [0, 1, 4, 5]
#     barcolors = [[Cmp23, Cmp13, Cmp12], #  
#                  [Rnd, Cmp3, Cmp2],  #
#                  [Cmp12, Cmp13, Cmp12],
#                  [Rnd, Rnd, Rnd],
#                  [Cmp3, Cmp1, Rnd],  #  
#                  [Cmp3, Cmp1, Cmp2]]  #
    


fwrow = 5
ifrboost = 0
ifr = -1
for iifr in [0, 1, 2, 4, 3, 5]:
     ifr += 1
     iparticipant, iconds, iacts = _N.where(cmps[filtdat, iifr] > int(SHUFFLES*0.98))
     #iparticipant, iconds, iacts = _N.where(bigflucs[filtdat, ifr])#_N.where(cmps[filtdat, ifr] > 200)
     iBinsFlat = iconds*3 + iacts
     cnts, bins = _N.histogram(iBinsFlat, bins=(_N.arange(10)-0.5))

     if (ifr == 1) or (ifr == 2) or (ifr == 4):
          ifrboost += fwrow
     for icond in range(3):
          #_plt.subplot2grid((1, 18*colspan + 5*emptycol), (0, (3*ifr+icond)*colspan + ifr*emptycol), colspan=(colspan-2))
          _plt.subplot2grid((6*rowspan + 5*emptyrow + 3*fwrow, 3), (ifr*rowspan + ifr*emptyrow + ifrboost, icond), rowspan=(rowspan-2))
            
          #s = "%s\n" % frameworks[ifr] if icond == 1 else "\n"
          s = "%s" % conds[ifr][icond]
          _plt.title(r"%s" % s, fontsize=lblsz)
          _plt.bar(_N.arange(3), cnts[3*icond:3*(icond+1)] / len(filtdat), color=barcolors[ifr][icond])
          if ylim01:
               _plt.ylim(0, 1)
          else:
               _plt.ylim(0, 0.5)

          _plt.xticks(_N.arange(3), acts[ifr], fontsize=tksz)
          #_plt.xlabel("actions", fontsize=lblsz)
          at_ticks = [0, 0.5, 1] if ylim01 else [0, 0.1, 0.2, 0.3, ]
          disp_ticks = [0, 50, 100] if ylim01 else [0, 10, 20, 30, ]    
                      
          if icond > 0:
               if ylim01:
                    _plt.yticks(at_ticks, ["", "", ""], fontsize=tksz)
               else:
                    _plt.yticks(at_ticks, ["", "", "", ""], fontsize=tksz)
                    #_plt.yticks(at_ticks, ["", "", "", "", ""], fontsize=tksz) 
          else:
               _plt.yticks(at_ticks, disp_ticks, fontsize=tksz)
                                         
          if icond == 0:
               _plt.ylabel("% games", fontsize=(lblsz-1))
               #_plt.yticks(_N.arange(0, maxticky+1, dY), fontsize=tksz)       
               #_plt.ylabel("# subjects", fontsize=lblsz)
          #else:
               #_plt.yticks(_N.arange(0, maxticky+1, dY), ["", "", "", ""])
fig.subplots_adjust(left=0.24, right=0.96, bottom=0.03, top=0.96, wspace=0.3)


_plt.savefig(outdirFN("components_look_like_rules%(flp)s_%(ex)s_%(v)d_%(wt)d%(w)d%(s)d" % {"ex" : expt, "wt" : win_type, "w" : win, "v" : visit, "s" : gk_w, "flp" : sFlipped}, label=label))
