#!/usr/bin/python
import numpy as _N
import RPSvAI.models.empirical_ken as _emp
import RPSvAI.utils.read_taisen as _rt
from RPSvAI.utils.dir_util import workdirFN
import matplotlib.pyplot as _plt
import os
import scipy.stats as _ss
import AIRPSfeatures as _aift
from sklearn.cluster import AgglomerativeClustering as AggCl
from sklearn.manifold import TSNE, LocallyLinearEmbedding, Isomap


def depickle(s):
     import pickle
     with open(s, "rb") as f:
          lm = pickle.load(f)
     return lm

compnames={"DSUWTL" : ["D|w", "S|w", "U|w",
                       "D|t", "S|t", "U|t",
                       "D|l", "S|l", "U|l"],
           "RPSWTL" : ["R|w", "S|w", "P|w",
                       "R|t", "S|t", "P|t",
                       "R|l", "S|l", "P|l"],
           "DSURPS" : ["D|r", "S|r", "U|r",
                       "D|s", "S|s", "U|s",
                       "D|p", "S|p", "U|p"],
           # "RPSRPS" : ["R|r", "S|r", "P|r",
           #             "R|s", "S|s", "P|s",
           #             "R|p", "S|p", "P|p"],
           "LCBRPS" : ["L|r", "C|r", "B|r",
                       "L|s", "C|s", "B|s",
                       "L|p", "C|p", "B|p"],
           "LCBAIRPS" : ["L|air", "C|air", "B|air",
                         "L|ais", "C|ais", "B|ais",
                         "L|aip", "C|aip", "B|aip"],
           "DSUAIRPS" : ["D|air", "S|air", "U|air",
                         "D|ais", "S|ais", "U|ais",
                         "D|aip", "S|aip", "U|aip"]}

conds      = [["$w$", "$t$", "$l$"], ["$w$", "$t$", "$l$"],
              ["$r$", "$s$", "$p$"], ["$r_A$", "$s_A$", "$p_A$"],
              ["$r$", "$s$", "$p$"], ["$r_A$", "$s_A$", "$p_A$"]]
acts       = [["D", "C", "U"], ["R", "S", "P"],
              ["D", "C", "U"], ["D", "C", "U"],
              ["D$_A$", "C$_A$", "U$_A$"], ["D$_A$", "C$_A$", "U$_A$"]]

visit = 1
visits = [1]
win_type = 2   #  window is of fixed number of games
win     = 3
smth    = 1
label          = win_type*100+win*10+smth
svisits =str(visits).replace(" ", "").replace("[", "").replace("]", "")    

expt = "TMB2"
if expt == "TMB2":
    lm = depickle(workdirFN("TMB2_AQ28_vs_RPS_features_%(v)d_of_%(vs)s_%(wt)d%(w)d%(s)d.dmp" % {"v" : visit, "wt" : win_type, "w" : win, "s" : smth, "wd" : os.environ["RPSWORKDIR"], "vs" : svisits}))
    partIDs = lm["partIDs"]
    partIDs_okgames = lm["partIDs_okgames"]    
    TO = 300
if expt == "CogWeb":
    lm = {}
    dates = _rt.date_range(start='2/25/2024', end='4/30/2024')
    partIDs, dats, cnstrs, has_domainQs, has_domainQs_wkeys = _rt.filterRPSdats(expt, dates, visits=visits, domainQ=(_rt._TRUE_AND_FALSE_), demographic=_rt._TRUE_AND_FALSE_, mentalState=_rt._TRUE_AND_FALSE_, min_meanIGI=500, max_meanIGI=20000, minIGI=10, maxIGI=50000, MinWinLossRat=0.2, has_useragent=True, has_start_and_end_times=True, has_constructor=True, blocks=1)
    lm["filtdat"] = _N.arange(len(partIDs))
    TO = 300
filtdat = lm["filtdat"]
filtdat_okgames = lm["filtdat_okgames"]

time_aft_los = _N.zeros(len(partIDs_okgames))
time_aft_tie  = _N.zeros(len(partIDs_okgames))
time_aft_win = _N.zeros(len(partIDs_okgames))

vals = ["FF", "BB", "77"]
colors  = []
for nR in range(3):
    for nG in range(3):
        for nB in range(3):
            colors.append("#%(r)s%(g)s%(b)s" % {"r" : vals[nR], "g" : vals[nG], "b" : vals[nB]})
_N.random.shuffle(colors[1:])

markers=["o", "v", "<", ">", "s", "p", "*", "X", "+"]
nClusts=8
allMargCRs = _N.empty((len(partIDs_okgames), 6, 3, 3))
pid = -1
t_btwn_rounds  = _N.empty((3, len(partIDs_okgames)))
flip_HUMAI = False
sFlipped   = "_flipped" if flip_HUMAI else ""

inds = _N.arange(300)

for partID in partIDs_okgames:
    pid += 1
    im = -1
    for model in ["DSUWTL", "RPSWTL", "DSURPS", "LCBRPS", "DSUAIRPS", "LCBAIRPS"]:
        im += 1
        allMargCRs[pid, im] = _emp.marginalCR(partID, expt=expt, visit=1, block=1, hnd_dat=None, model=model, shuffle_inds=False)
        #allCRs[pid, im] = CRs        
        
    dmp       = depickle(workdirFN("%(rpsm)s/%(lb)d/variousCRs%(flp)s_%(visit)d.dmp" % {"rpsm" : partID, "lb" : label, "visit" : visit, "flp" : sFlipped}))


    hnd_dat, start_time, end_time, UA, cnstr, input_meth, ini_percep, fin_percep, gt_dmp           = _rt.return_hnd_dat(partID, has_useragent=True, has_start_and_end_times=True, has_constructor=True, visit=1, expt=expt, block=1)

    input_meth=dmp["inp_meth"]
        
    n_mouse, n_keys, mouse_resp_t, key_resp_t, resp_time_all = _aift.resptime_aft_wtl(hnd_dat, TO, pid, input_meth, time_aft_win, time_aft_tie, time_aft_los)

    lost = _N.where(hnd_dat[0:299, 2] == -1)[0]
    tie = _N.where(hnd_dat[0:299, 2] == 0)[0]    
    won  = _N.where(hnd_dat[0:299, 2] == 1)[0]

    #t_aft_loss=_N.mean(time_aft_los)#_N.mean(hnd_dat[lost+1, 3]-hnd_dat[lost, 3])
    #t_aft_tie=_N.mean(time_aft_tie)#_N.mean(hnd_dat[tie+1, 3]-hnd_dat[tie, 3])    
    #t_aft_win=_N.mean(time_aft_win)#_N.mean(hnd_dat[won+1, 3]-hnd_dat[won, 3])
    
    t_btwn_rounds[0, pid] = time_aft_los[pid-1]#t_aft_loss
    t_btwn_rounds[1, pid] = time_aft_tie[pid-1]#t_aft_tie
    t_btwn_rounds[2, pid] = time_aft_win[pid-1]#t_aft_win    

"""
for mnsd in range(2):
    fig = _plt.figure(figsize=(4, 12))
    filtdat = lm["filtdat"]
    mns = _N.mean(allMargCRs[filtdat], axis=0)
    std = _N.std(allMargCRs[filtdat], axis=0)
    if mnsd == 0:
        ymax = _N.max(mns)*1.15
    else:
        ymax = _N.max(std)*1.15

    for im in range(6):
        for ic in range(3):
            fig.add_subplot(6, 3, im*3+ic + 1)
            if mnsd == 0:
                _plt.bar(_N.arange(3), mns[im, ic], width=0.5)                
            else:
                _plt.bar(_N.arange(3), std[im, ic], width=0.5)                

            _plt.ylim(0, ymax)
"""


fig = _plt.figure(figsize=(4, 10.2))
filtdat = lm["filtdat"]
filtdat_okgames = lm["filtdat_okgames"]
mns = _N.mean(allMargCRs[filtdat_okgames], axis=0)
std = _N.std(allMargCRs[filtdat_okgames], axis=0)
mnmax = (_N.max(mns)+0.15)*1.07

#  3*6 = 18
figH   = 28
spcH   = 5
grY = 0
lbsz=16
tksz=15
for im in range(6):
    if (im % 2 == 0) and (im > 0):
         grY += spcH
    for ic in range(3):
        #fig.add_subplot(6, 3, im*3+ic + 1)
        _plt.subplot2grid((6*figH + 2*spcH, 3), (grY, ic), rowspan=(figH-10))
        _plt.title(conds[im][ic], fontsize=lbsz)
        _plt.bar(_N.arange(3), mns[im, ic], yerr=std[im, ic], width=0.6, color="grey", ecolor="black", capsize=2)
        _plt.yticks([0.16666666, 0.33333], ["1/6", "1/3"], fontsize=tksz)
        if ic > 0:
            _plt.yticks([0.166666666, 0.33333], ["", ""])
        _plt.xticks(_N.arange(3), acts[im], fontsize=tksz)
        _plt.axhline(y=0.333333, ls=":", color="blue")
        _plt.ylim(0.16666, mnmax)
        _plt.xlim(-0.8, 2.8)
    grY += figH

fig.subplots_adjust(bottom=0.02, top=0.96, left=0.11, right=0.97)
_plt.savefig("CRvariability_%(e)s.png" % {"e" : expt})

#    t_btwn_rounds[0, pid] = t_aft_loss
#    t_btwn_rounds[1, pid] = t_aft_tie
#    t_btwn_rounds[2, pid] = t_aft_win    

lbsz=22
tksz=19
#  histogram of time after win, time after loss
#fig = _plt.figure(figsize=(8, 2.5))
fig = _plt.figure(figsize=(4.5, 8))
################################################
fig.add_subplot(3, 1, 1)
_plt.title("following win", fontsize=lbsz)
#_plt.hist(t_btwn_rounds[0, filtdat], bins=_N.linspace(0, 5000, 41), color="black")
_plt.hist(t_btwn_rounds[2, filtdat_okgames], bins=_N.linspace(0, 3, 31), color="black", density=True)
_plt.ylim(0, 1.5)
_plt.ylabel("density", fontsize=lbsz)
_plt.xticks(_N.arange(0, 4), fontsize=tksz)
_plt.yticks(_N.arange(0, 1.6, 0.5), fontsize=tksz)
_plt.axvline(x=_N.mean(t_btwn_rounds[2, filtdat_okgames]), color="orange", ls=":", lw=3)
_plt.axvline(x=_N.median(t_btwn_rounds[2, filtdat_okgames]), color="blue", ls=":", lw=3)
################################################
fig.add_subplot(3, 1, 2)
_plt.title("following tie", fontsize=lbsz)
#_plt.hist(t_btwn_rounds[0, filtdat], bins=_N.linspace(0, 5000, 41), color="black")
_plt.hist(t_btwn_rounds[1, filtdat_okgames], bins=_N.linspace(0, 3, 31), color="black", density=True)
_plt.ylim(0, 1.5)
_plt.ylabel("density", fontsize=lbsz)
_plt.xticks(_N.arange(0, 4), fontsize=tksz)
_plt.yticks(_N.arange(0, 1.6, 0.5), fontsize=tksz)
_plt.axvline(x=_N.mean(t_btwn_rounds[1, filtdat_okgames]), color="orange", ls=":", lw=3)
_plt.axvline(x=_N.median(t_btwn_rounds[1, filtdat_okgames]), color="blue", ls=":", lw=3)
################################################
fig.add_subplot(3, 1, 3)
_plt.title("following lose", fontsize=lbsz)
#_plt.hist(t_btwn_rounds[2, filtdat], bins=_N.linspace(0, 5000, 41), color="black")
_plt.hist(t_btwn_rounds[0, filtdat_okgames], bins=_N.linspace(0, 3, 31), color="black", density=True)
_plt.ylim(0, 1.5)
_plt.ylabel("density", fontsize=lbsz)
_plt.xticks(_N.arange(0, 4), fontsize=tksz)
_plt.yticks(_N.arange(0, 1.6, 0.5), fontsize=tksz)
_plt.xlabel("response time (s)", fontsize=lbsz)
_plt.axvline(x=_N.mean(t_btwn_rounds[0, filtdat_okgames]), color="orange", ls=":", lw=3)
_plt.axvline(x=_N.median(t_btwn_rounds[0, filtdat_okgames]), color="blue", ls=":", lw=3)

fig.subplots_adjust(bottom=0.09, hspace=0.44, left=0.25, right=0.96,top=0.95)
_plt.savefig("RT_after_w_l")

#for ip in range(len(filtdat)):
#     _N.where(t_btwn_rounds[0, ip] 



rnksfltd = allMargCRs[filtdat_okgames].reshape((len(filtdat_okgames), 54))
similarities     = _N.zeros((filtdat_okgames.shape[0], filtdat_okgames.shape[0]))
for i in range(filtdat_okgames.shape[0]):
    for j in range(i+1, filtdat_okgames.shape[0]):
        similarities[i, j], pv = _ss.pearsonr(rnksfltd[i], rnksfltd[j])
        similarities[j, i] = similarities[i, j]

# targets=["AQ28scrs", "soc_skils", "imag", "rout", "switch", "fact_pat"]

#for linkage in ["ward", "average", "complete"]:
ac = AggCl(n_clusters=nClusts, linkage="ward").fit(similarities)

map2d= TSNE(n_components=2)
mp2d = map2d.fit_transform(rnksfltd)

fig = _plt.figure(figsize=(3, 3))
ax = _plt.subplot2grid((1, 1), (0, 0))
ax.set_facecolor("#333333")

for ic in range(nClusts):
     these = _N.where(ac.labels_ == ic)[0]
     _plt.scatter(mp2d[these, 0], mp2d[these, 1], marker=markers[ic], color=colors[ic], s=17)
_plt.xticks([])
_plt.xlabel("TSNE dim 1", fontsize=16)
_plt.yticks([])
_plt.ylabel("TSNE dim 2", fontsize=16)
_plt.savefig("cluster_CR54")


#_plt.title(sFrmwks[ifr], fontsize=20)
# _plt.subplot2grid((5*6+spcskp*2+2-1, 1), (ifr*5+spc, 0), rowspan=4)
# _plt.title("%(a)s|%(c)s"  % {"a" : list2str(acts[ifr]), "c" : list2str(conds[ifr])})
#colors=["black", "grey", "blue", "red", "orange", "green", "brown", "yellow", "pink"]
#colors=["black", "blue", "red", "orange", "green", "brown", "yellow", "pink", "grey"]
markers=["o", "v", "<", ">", "s", "p", "*", "X", "+"]

# for ic in range(nClusts):
#      these = _N.where(ac.labels_ == ic)[0]
#      _plt.scatter(mp2d[these, 0], mp2d[these, 1], marker=markers[ic], color=colors[ic], s=17)
     
# _plt.xticks([])
# _plt.yticks([])

# ax = _plt.subplot2grid((1, 6), (0, 1), colspan=5)
# ax.set_facecolor("#333333")
# for ic in range(nClusts):
#      ths = _N.where(ac.labels_ == ic)[0]
#      nBig = _N.zeros(54)
#      ths = _N.where(ac.labels_ == ic)[0]
#      for act in range(54):
#           sz = (len(_N.where(rnksfltd[ths, act] > 180)[0])/len(ths))*150
#           _plt.scatter([act], [ic], s=int(sz), color=colors[ic])
# #_plt.xticks(_N.arange(54), xticks, rotation=90, fontsize=17)
# _plt.yticks(_N.arange(nClusts), _N.arange(1, nClusts+1), fontsize=15)

# _plt.xlim(-1, 54)
# _plt.axvline(x=2.5, ls=":" , color="#FFFFFF")
# _plt.axvline(x=5.5, ls=":", color="#FFFFFF")     
# _plt.axvline(x=8.5, ls="-" , color="#9999FF")
# #-----------------
# _plt.axvline(x=11.5, ls=":" , color="#FFFFFF")
# _plt.axvline(x=14.5, ls=":", color="#FFFFFF")     
# _plt.axvline(x=17.5, ls="-" , color="#9999FF")
# #-----------------
# _plt.axvline(x=20.5, ls=":" , color="#FFFFFF")
# _plt.axvline(x=23.5, ls=":", color="#FFFFFF")     
# _plt.axvline(x=26.5, ls="-" , color="#9999FF")
# #-----------------
# _plt.axvline(x=29.5, ls=":" , color="#FFFFFF")
# _plt.axvline(x=32.5, ls=":", color="#FFFFFF")     
# _plt.axvline(x=35.5, ls="-" , color="#9999FF")
# #-----------------
# _plt.axvline(x=38.5, ls=":" , color="#FFFFFF")
# _plt.axvline(x=41.5, ls=":", color="#FFFFFF")     
# _plt.axvline(x=44.5, ls="-" , color="#9999FF")
# #-----------------
# _plt.axvline(x=47.5, ls=":" , color="#FFFFFF")
# _plt.axvline(x=50.5, ls=":", color="#FFFFFF")     


# #_plt.hist2d(mp2d[:, 0], mp2d[:, 1], bins=40)

# fig.subplots_adjust(left=0.03, right=0.97, bottom=0.25, top=0.92)
# #_plt.savefig("clusterSDS_TMB2")
     
