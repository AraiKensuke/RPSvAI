import numpy as _N
import matplotlib.pyplot as _plt
import scipy.stats as _ss
from sklearn.cluster import AgglomerativeClustering as AggCl
#from sklearn.manifold import TSNE, LocallyLinearEmbedding, Isomap
from sklearn.manifold import TSNE, LocallyLinearEmbedding, Isomap
import umap
import RPSvAI.utils.read_taisen as _rt
import AIRPSfeatures as _aift
from RPSvAI.utils.dir_util import workdirFN

_plt.ioff()

def depickle(s):
     import pickle
     with open(s, "rb") as f:
          lm = pickle.load(f)
     return lm

visit = 1
visits = [1]
win_type = 2   #  window is of fixed number of games
win     = 3
smth    = 1
label          = win_type*100+win*10+smth
svisits =str(visits).replace(" ", "").replace("[", "").replace("]", "")
TO=300

flip_HUMAI = False
sFlipped   = "_flipped" if flip_HUMAI else ""
lmTMBCW = depickle("/home_local/arai/Projects/RPSvAI/WorkDir/shuffledCRs_6CFs_TMBCW_3_1_1")
lmF = depickle(workdirFN("TMBCW_AQ28_vs_RPS_features%(flp)s_1_of_1_%(lb)s.dmp" % {"lb" : label, "flp" : sFlipped}))
lmScrs = depickle("/home_local/arai/Projects/RPSvAI/WorkDir/TMBCW_AQ28_vs_RPS_features%(flp)s_1_of_1_231.dmp" % {"flp" : sFlipped})
filtdat     = lmF["filtdat"]
print(filtdat)
#fr_lotsof0s = lm["fr_lotsof0s"] / 205)**2
#fr_lotsof1s = lm["fr_lotsof1s"] / 205)**2
#TMB2_fr_lotsof0s  = lmTMB2["fr_clumped0s"]
TMBCW_fr_lotsof0s  = lmTMBCW["fr_lotsof0s"]
print(TMBCW_fr_lotsof0s.shape)



TMBCW_fr_lotsof1s = lmTMBCW["fr_lotsof1s"]

fr_lotsof0s = _N.empty((TMBCW_fr_lotsof0s.shape[0], 6, 3, 3), dtype=int)
#fr_lotsof0s = _N.empty((TMB2_fr_lotsof0s.shape[0], 6, 3, 3), dtype=int)
fr_lotsof1s = _N.empty((TMBCW_fr_lotsof0s.shape[0], 6, 3, 3), dtype=int)
#fr_lotsof1s = _N.empty((TMB2_fr_lotsof0s.shape[0], 6, 3, 3), dtype=int)

partIDs = lmTMBCW["partIDs"]

time_aft_los = _N.zeros(len(partIDs))
time_aft_tie  = _N.zeros(len(partIDs))
time_aft_win = _N.zeros(len(partIDs))
pid = -1
t_btwn_rounds  = _N.empty((3, len(partIDs)))

"""
for partID in partIDs:
    pid += 1
    dmp       = depickle(workdirFN("%(rpsm)s/%(lb)d/variousCRs%(flp)s_%(visit)d.dmp" % {"rpsm" : partID, "lb" : label, "visit" : visit, "flp" : sFlipped}))
    hnd_dat, start_time, end_time, UA, cnstr, input_meth, ini_percep, fin_percep, gt_dmp           = _rt.return_hnd_dat(partID, has_useragent=True, has_start_and_end_times=True, has_constructor=True, visit=1, expt="TMB2", block=1)

    input_meth=dmp["inp_meth"]
        
    n_mouse, n_keys, mouse_resp_t, key_resp_t, resp_time_all = _aift.resptime_aft_wtl(hnd_dat, TO, pid, input_meth, time_aft_win, time_aft_tie, time_aft_los)

    lost = _N.where(hnd_dat[0:299, 2] == -1)[0]
    tie = _N.where(hnd_dat[0:299, 2] == 0)[0]    
    won  = _N.where(hnd_dat[0:299, 2] == 1)[0]
    
    t_btwn_rounds[0, pid] = time_aft_los[pid-1]#t_aft_loss
    t_btwn_rounds[1, pid] = time_aft_tie[pid-1]#t_aft_tie
    t_btwn_rounds[2, pid] = time_aft_win[pid-1]#t_aft_win    
t_btwn_rounds_filt = t_btwn_rounds[:, filtdat]
"""
fr_lotsof0s[0:TMBCW_fr_lotsof0s.shape[0]] = TMBCW_fr_lotsof0s
fr_lotsof1s[0:TMBCW_fr_lotsof1s.shape[0]] = TMBCW_fr_lotsof1s

i1s, i2s, i3s, i4s = _N.where(_N.isinf(fr_lotsof1s) == 1)
fr_lotsof1s[i1s, i2s, i3s, i4s] = 0
i1s, i2s, i3s, i4s = _N.where(_N.isnan(fr_lotsof1s) == 1)
fr_lotsof1s[i1s, i2s, i3s, i4s] = 0

#fr_lotsof0s = _N.random.randint(0, 205, size=(214, 6, 3, 3))
#fr_lotsof1s = _N.random.randint(0, 205, size=(214, 6, 3, 3))

#fr_lotsof0s = _N.array(lm["fr_lotsof0s"] > 170, dtype=int)
#fr_lotsof1s = _N.array(lm["fr_lotsof1s"] > 170, dtype=int)

srtd_0s      = _N.sort(fr_lotsof0s.reshape((fr_lotsof0s.shape[0], 6, 9)), axis=2)
srtd_1s      = _N.sort(fr_lotsof1s.reshape((fr_lotsof1s.shape[0], 6, 9)), axis=2)
#rnks         = _N.sum(srtd_0s, axis=2) + _N.sum(srtd_1s, axis=2)
rnks0         = fr_lotsof0s.reshape((fr_lotsof0s.shape[0], 54))
rnks1         = fr_lotsof1s.reshape((fr_lotsof1s.shape[0], 54))

rnks = rnks1
print(rnks.shape)

#rnks          = _N.empty((254, 108), dtype=int)
#rnks[:, 0:54] = rnks0
#rnks[:, 54:]  = rnks1

# xticks = [["D$|w$", "C$|w$", "U$|w$",
#            "D$|t$", "C$|t$", "U$|t$",
#            "D$|l$", "C$|l$", "U$|l$"],
#           ["R$|w$", "P$|w$", "S$|w$",
#            "R$|t$", "P$|t$", "S$|t$",
#            "R$|l$", "P$|l$", "S$|l$"],          
#           ["D$|r$", "C$|r$", "U$|r$",
#            "D$|p$", "C$|p$", "U$|p$",
#            "D$|s$", "C$|s$", "U$|s$"],
#           ["D$|r_{\\rm A}$", "C$|r_{\\rm A}$", "U$|r_{\\rm A}$",
#            "D$|p_{\\rm A}$", "C$|p_{\\rm A}$", "U$|p_{\\rm A}$",
#            "D$|s_{\\rm A}$", "C$|s_{\\rm A}$", "U$|s_{\\rm A}$"],
#           ["$\mbox{D}_{\\rm A}|r$", "$\mbox{C}_{\\rm A}|r$", "$\mbox{U}_{\\rm A}|r$",
#            "$\mbox{D}_{\\rm A}|p$", "$\mbox{C}_{\\rm A}|p$", "$\mbox{U}_{\\rm A}|p$",
#            "$\mbox{D}_{\\rm A}|s$", "$\mbox{C}_{\\rm A}|s$", "$\mbox{U}_{\\rm A}|s$"],
#           ["$\mbox{D}_{\\rm A}|r_{\\rm A}$", "$\mbox{C}_{\\rm A}|r_{\\rm A}$", "$\mbox{U}_{\\rm A}|r_{\\rm A}$",
#           "$\mbox{D}_{\\rm A}|p_{\\rm A}$", "$\mbox{C}_{\\rm A}|p_{\\rm A}$", "$\mbox{U}_{\\rm A}|p_{\\rm A}$",
#           "$\mbox{D}_{\\rm A}|s_{\\rm A}$", "$\mbox{C}_{\\rm A}|s_{\\rm A}$", "$\mbox{U}_{\\rm A}|s_{\\rm A}$"]]

# xticks = ["D$|w$", "C$|w$", "U$|w$",
#           "D$|t$", "C$|t$", "U$|t$",
#           "D$|l$", "C$|l$", "U$|l$",
#           "R$|w$", "P$|w$", "S$|w$",
#           "R$|t$", "P$|t$", "S$|t$",
#           "R$|l$", "P$|l$", "S$|l$",          
#           "D$|r$", "C$|r$", "U$|r$",
#           "D$|p$", "C$|p$", "U$|p$",
#           "D$|s$", "C$|s$", "U$|s$",
#           "D$|r_{\\rm A}$", "C$|r_{\\rm A}$", "U$|r_{\\rm A}$",
#           "D$|p_{\\rm A}$", "C$|p_{\\rm A}$", "U$|p_{\\rm A}$",
#           "D$|s_{\\rm A}$", "C$|s_{\\rm A}$", "U$|s_{\\rm A}$",
#           "$D_{\\rm A}|r$", "$C_{\\rm A}|r$", "$\mbox{U}_{\\rm A}|r$",
#           "$D_{\\rm A}|p$", "$C_{\\rm A}|p$", "$\mbox{U}_{\\rm A}|p$",
#           "$D_{\\rm A}|s$", "$\mbox{C}_{\\rm A}|s$", "$\mbox{U}_{\\rm A}|s$",
#           "$D_{\\rm A}|r_{\\rm A}$", "$\mbox{C}_{\\rm A}|r_{\\rm A}$", "$\mbox{U}_{\\rm A}|r_{\\rm A}$",
#           "$D_{\\rm A}|p_{\\rm A}$", "$\mbox{C}_{\\rm A}|p_{\\rm A}$", "$\mbox{U}_{\\rm A}|p_{\\rm A}$",
#           "$D_{\\rm A}|s_{\\rm A}$", "$\mbox{C}_{\\rm A}|s_{\\rm A}$", "$\mbox{U}_{\\rm A}|s_{\\rm A}$"]


#         "IR   $\\mbox{D}_{\\rm A}\\mbox{C}_{\\rm A}\\mbox{U}_{\\rm A}|rps$",
#         "IR   DCU$|r_{\\rm A}p_{\\rm A}s_{\\rm A}$",
#         "IR   $\\mbox{D}_{\\rm A}\\mbox{C}_{\\rm A}\\mbox{U}_{\\rm A}|r_{\\rm A}p_{\\rm A}s_{\\rm A}$",
          
#sFrmwks=["DCU$|wtl$", "RPS$|wtl$", "DCU$|rps$", "DCU$|r_{\\rm A}p_{\\rm A}s_{\\rm A}$", "$\mbox{D}_{\\rm A}\mbox{C}_{\\rm A}\mbox{U}_{\\rm A}|rps$", "$\mbox{D}_{\\rm A}\mbox{C}_{\\rm A}\mbox{U}_{\\rm A}|r_{\\rm A}p_{\\rm A}s_{\\rm A}$"]
#sFrmwks=["DCU$|wtl$", "RPS$|wtl$", "DCU$|rps$", "DCU$|rapasa$", "$DaCaUa|rps$", "$DaCaUa|rapasa$"]
# #rnks    = _N.sum(_N.sum(fr_lotsof0s, axis=3), axis=2) + _N.sum(_N.sum(fr_lotsof1s, axis=3), axis=2)#+_N.sum(_N.sum(lm["fr_cmp_fluc_rank1"], axis=3), axis=2)+_N.sum(_N.sum(lm["fr_cmp_fluc_rank2"], axis=3), axis=2)

AQ28scrs = lmScrs["AQ28scrs"]
soc_skils = lmScrs["soc_skils"]
imag     = lmScrs["imag"]
rout     = lmScrs["rout"]
switch     = lmScrs["switch"]
fact_pat     = lmScrs["fact_pat"]
#for i in range(214):
#    _N.random.shuffle(rnks[i])

#_plt.savefig("clusterSDS_TMB2")


#fr5 = _N.zeros(9, dtype=int)
fr54 = _N.zeros(54, dtype=int)
nClusts = 5

vals = ["FF", "BB", "77"]
colors  = []
for nR in range(3):
    for nG in range(3):
        for nB in range(3):
            colors.append("#%(r)s%(g)s%(b)s" % {"r" : vals[nR], "g" : vals[nG], "b" : vals[nB]})
_N.random.shuffle(colors[1:])
markers=["o", "v", "<", "|", "s", "p", "*", "X", "+", "4", "h", "D"]

#fr5[0:9] = _N.arange(ifr*9, (ifr+1)*9)
fr54 = _N.arange(54)
#fr5[9:18] = _N.arange(45, 54)
rnksfltd  = rnks[filtdat][:, fr54]

similarities     = _N.zeros((filtdat.shape[0], filtdat.shape[0]))
for i in range(filtdat.shape[0]):
    for j in range(i+1, filtdat.shape[0]):
        similarities[i, j], pv = _ss.pearsonr(rnksfltd[i], rnksfltd[j])
        similarities[j, i] = similarities[i, j]

# targets=["AQ28scrs", "soc_skils", "imag", "rout", "switch", "fact_pat"]


#for linkage in ["ward", "average", "complete"]:
ac = AggCl(n_clusters=nClusts, linkage="ward").fit(similarities)

#map2d= TSNE(n_components=2)
map2d= umap.UMAP()
#mp2d = map2d.fit_transform(similarities)
trans = map2d.fit(similarities)
mp2d = trans.embedding_

# #  4,4,2,4,4,2,4,4 = 26
# #  5, 5,1,1,4,1,4,1,1,4,1,4 = 34     
#fig = _plt.figure(figsize=(5, 5))

fig = _plt.figure(figsize=(3, 3))
ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor("#333333")
for ic in range(nClusts):
     these = _N.where(ac.labels_ == ic)[0]
     _plt.scatter(mp2d[these, 0], mp2d[these, 1], marker=markers[ic], color=colors[ic], s=15)
     
_plt.xticks([])
_plt.xlabel("TSNE dim 1", fontsize=16)
_plt.yticks([])
_plt.ylabel("TSNE dim 2", fontsize=16)
#_plt.savefig("SDS_CW54_justTSNE")

wgts = _N.arange(1, 55)*10
feat54 = _N.empty((nClusts, 54))
feat54w = _N.empty((nClusts, 54))   # weighted
for ic in range(nClusts):
     ths = _N.where(ac.labels_ == ic)[0]
     for act in range(54):
          sz = (len(_N.where(rnksfltd[ths, act] > 180)[0])/len(ths))*150
          feat54[ic, act] = sz
     feat54w[ic] = wgts*feat54[ic]
#  remap clusters to make it easier to see 2 types of repetoire
remapped = _N.argsort(_N.sum(feat54w, axis=1))
fig = _plt.figure(figsize=(14, 2.8))
ax = _plt.subplot2grid((1, 6), (0, 1), colspan=5)
ax.set_facecolor("#333333")
for _ic in range(nClusts):
     ic = remapped[_ic]
     ths = _N.where(ac.labels_ == ic)[0]
     for act in range(54):
          sz = (len(_N.where(rnksfltd[ths, act] > 180)[0])/len(ths))*150
          _plt.scatter([act], [_ic], s=int(sz), color=colors[_ic])
     
#_plt.xticks(_N.arange(54), xticks, rotation=90, fontsize=17)
#_plt.yticks(_N.arange(nClusts), _N.arange(1, nClusts+1), fontsize=15)

_plt.xlim(-1, 54)
_plt.axvline(x=2.5, ls=":" , color="#FFFFFF")
_plt.axvline(x=5.5, ls=":", color="#FFFFFF")     
_plt.axvline(x=8.5, ls="-" , color="#9999FF")
#-----------------
_plt.axvline(x=11.5, ls=":" , color="#FFFFFF")
_plt.axvline(x=14.5, ls=":", color="#FFFFFF")     
_plt.axvline(x=17.5, ls="-" , color="#9999FF")
#-----------------
_plt.axvline(x=20.5, ls=":" , color="#FFFFFF")
_plt.axvline(x=23.5, ls=":", color="#FFFFFF")     
_plt.axvline(x=26.5, ls="-" , color="#9999FF")
#-----------------
_plt.axvline(x=29.5, ls=":" , color="#FFFFFF")
_plt.axvline(x=32.5, ls=":", color="#FFFFFF")     
_plt.axvline(x=35.5, ls="-" , color="#9999FF")
#-----------------
_plt.axvline(x=38.5, ls=":" , color="#FFFFFF")
_plt.axvline(x=41.5, ls=":", color="#FFFFFF")     
_plt.axvline(x=44.5, ls="-" , color="#9999FF")
#-----------------
_plt.axvline(x=47.5, ls=":" , color="#FFFFFF")
_plt.axvline(x=50.5, ls=":", color="#FFFFFF")     

#_plt.rcParams['backend'] = "ps"
#_plt.rcParams['text.usetex'] = True

# #_plt.hist2d(mp2d[:, 0], mp2d[:, 1], bins=40)

ax = _plt.subplot2grid((1, 6), (0, 0))
ax.set_facecolor("#333333")
#_plt.title(sFrmwks[ifr], fontsize=20)
# _plt.subplot2grid((5*6+spcskp*2+2-1, 1), (ifr*5+spc, 0), rowspan=4)
# _plt.title("%(a)s|%(c)s"  % {"a" : list2str(acts[ifr]), "c" : list2str(conds[ifr])})
#colors=["black", "grey", "blue", "red", "orange", "green", "brown", "yellow", "pink"]
#colors=["black", "blue", "red", "orange", "green", "brown", "yellow", "pink", "grey"]

for _ic in range(nClusts):
     ic = remapped[_ic]
     these = _N.where(ac.labels_ == ic)[0]
     _plt.scatter(mp2d[these, 0], mp2d[these, 1], marker=markers[_ic], color=colors[_ic], s=15)
     
_plt.xticks([])
_plt.yticks([])
fig.subplots_adjust(left=0.03, right=0.97, bottom=0.25, top=0.92)
_plt.savefig("clusterSDS_TMB2_CWB54")



fig = _plt.figure(figsize=(14, 2.8))
ax = _plt.subplot2grid((1, 6), (0, 0))
ax.set_facecolor("#333333")
#_plt.title(sFrmwks[ifr], fontsize=20)
# _plt.subplot2grid((5*6+spcskp*2+2-1, 1), (ifr*5+spc, 0), rowspan=4)
# _plt.title("%(a)s|%(c)s"  % {"a" : list2str(acts[ifr]), "c" : list2str(conds[ifr])})
#colors=["black", "grey", "blue", "red", "orange", "green", "brown", "yellow", "pink"]
#colors=["black", "blue", "red", "orange", "green", "brown", "yellow", "pink", "grey"]

for ic in range(nClusts):
     these = _N.where(ac.labels_ == ic)[0]
     _plt.scatter(mp2d[these, 0], mp2d[these, 1], marker=markers[ic], color=colors[ic], s=15)
     
_plt.xticks([])
_plt.yticks([])

ax = _plt.subplot2grid((1, 6), (0, 1), colspan=5)
ax.set_facecolor("#333333")
for ic in range(nClusts):
     ths = _N.where(ac.labels_ == ic)[0]
     for act in range(54):
          sz = (len(_N.where(rnksfltd[ths, act] > 180)[0])/len(ths))*150
          _plt.scatter([act], [ic], s=int(sz), color=colors[ic])
#_plt.xticks(_N.arange(54), xticks, rotation=90, fontsize=17)
_plt.xticks(_N.arange(54), rotation=90, fontsize=17)
_plt.yticks(_N.arange(nClusts), _N.arange(1, nClusts+1), fontsize=15)

_plt.xlim(-1, 54)
_plt.axvline(x=2.5, ls=":" , color="#FFFFFF")
_plt.axvline(x=5.5, ls=":", color="#FFFFFF")     
_plt.axvline(x=8.5, ls="-" , color="#9999FF")
#-----------------
_plt.axvline(x=11.5, ls=":" , color="#FFFFFF")
_plt.axvline(x=14.5, ls=":", color="#FFFFFF")     
_plt.axvline(x=17.5, ls="-" , color="#9999FF")
#-----------------
_plt.axvline(x=20.5, ls=":" , color="#FFFFFF")
_plt.axvline(x=23.5, ls=":", color="#FFFFFF")     
_plt.axvline(x=26.5, ls="-" , color="#9999FF")
#-----------------
_plt.axvline(x=29.5, ls=":" , color="#FFFFFF")
_plt.axvline(x=32.5, ls=":", color="#FFFFFF")     
_plt.axvline(x=35.5, ls="-" , color="#9999FF")
#-----------------
_plt.axvline(x=38.5, ls=":" , color="#FFFFFF")
_plt.axvline(x=41.5, ls=":", color="#FFFFFF")     
_plt.axvline(x=44.5, ls="-" , color="#9999FF")
#-----------------
_plt.axvline(x=47.5, ls=":" , color="#FFFFFF")
_plt.axvline(x=50.5, ls=":", color="#FFFFFF")     


#_plt.hist2d(mp2d[:, 0], mp2d[:, 1], bins=40)

fig.subplots_adjust(left=0.03, right=0.97, bottom=0.25, top=0.92)
_plt.savefig("clusterSDS_TMB2_CWB54")

# fig = _plt.figure(figsize=(14, 8))
# for ic in range(nClusts):
#      these = _N.where(ac.labels_ == ic)[0]     
#      _plt.subplot2grid((8, 2), (ic, 0))
#      vals = (t_btwn_rounds_filt[0, these])/(t_btwn_rounds_filt[2, these])
#      srtdinds = _N.argsort(vals)
#      mn = _N.mean(vals[srtdinds[1:-1]])
#      _plt.hist(vals, bins=_N.linspace(0., 3.5, 36))
#      _plt.axvline(x=mn, color="black")
# #     _plt.subplot2grid((8, 2), (ic, 1))
# #     _plt.hist(t_btwn_rounds_filt[1, these], bins=_N.linspace(0, 3, 51))
#      # _plt.subplot2grid((8, 2), (ic, 2))
#      # _plt.hist(t_btwn_rounds_filt[2, these], bins=_N.linspace(0, 3, 51))

#      mn0=_N.mean(t_btwn_rounds_filt[0, these]) #lose
#      mn1=_N.mean(t_btwn_rounds_filt[1, these]) #tie
#      mn2=_N.mean(t_btwn_rounds_filt[2, these]) #win
#      sLgtW="***" if mn0 > mn2 else ""
#      print("sz %(sz)d   los %(0).2f    tie %(1).2f    win %(2).2f  %(wrn)s" % {"0" : mn0, "1" : mn1, "2" : mn2, "sz" : len(these), "wrn" : sLgtW})
#      _plt.scatter(mp2d[these, 0], mp2d[these, 1], marker=markers[ic], color=colors[ic], s=17)


# fig = _plt.figure(figsize=(14, 8))
# ist = 0
# for star in ["AQ28scrs", "soc_skils", "imag", "rout", "switch", "fact_pat"]:
#      ist += 1
#      fig.add_subplot(2, 3, ist)
#      _plt.title(star)
#      exec("scrr = %s" % star)
#      fscrr = scrr[filtdat]

#      for icl in range(nClusts):
#           labsF=_N.where(ac.labels_ == icl)[0]
#           _plt.scatter(icl+0.08*_N.random.randn(len(labsF)), fscrr[labsF], color=colors[icl])
#           mn = _N.median(fscrr[labsF])
#           _plt.plot([icl-0.25, icl+0.25], [mn, mn], color="brown", lw=4)



# fig = _plt.figure(figsize=(14, 8))
# ist = 0
# for star in ["AQ28scrs", "soc_skils", "imag", "rout", "switch", "fact_pat"]:
#      ist += 1
#      fig.add_subplot(2, 3, ist)
#      _plt.title(star)
#      exec("scrr = %s" % star)
#      fscrr = scrr[filtdat]
#      # for icl in range(7):
#      #      labsF=_N.where(ac.labels_ == icl)[0]
#      #      _plt.scatter(icl+0.08*_N.random.randn(len(labsF)), fscrr[labsF], color=colors[icl])
#      #      mn = _N.mean(fscrr[labsF])
#      #      _plt.plot([icl-0.25, icl+0.25], [mn, mn], color="brown", lw=4)
#      for icl in range(7):
#           labsF=_N.where(ac.labels_ == icl)[0]
#           not_labsF = _N.where(ac.labels_ != icl)[0]
#           _plt.scatter(icl+0.08*_N.random.randn(len(not_labsF)), fscrr[not_labsF], color="grey")          
#           _plt.scatter(icl+0.08*_N.random.randn(len(labsF)), fscrr[labsF], color=colors[icl])
#           mn = _N.mean(fscrr[labsF])
#           _plt.plot([icl-0.25, icl+0.25], [mn, mn], color="brown", lw=4)
          


# give me
#  ac.labels_       298
#  filtdat          193
#  partIDs          205
#  partIDs_okgames  306
#  filtdat_okgames  298
#  give me scores
# AQ28scrs_filtdat = AQ28scrs[lmF["filtdat"]]
# soc_skils_filtdat = soc_skils[lmF["filtdat"]]
# imag_filtdat = imag[lmF["filtdat"]]
# rout_filtdat = rout[lmF["filtdat"]]
# switch_filtdat = switch[lmF["filtdat"]]
# fact_pat_filtdat = fact_pat[lmF["filtdat"]]

# labels = _N.zeros(len(lmF["partIDs"]), dtype=int)
# hasAQ28_ids_okgames = _N.zeros(len(lmF["partIDs"]), dtype=int)
# hasAQ28_ids_partIDs = []
# for i in range(len(lmF["partIDs"])):   #  For each 
#      #  Find this
#      hasAQ28_ids_okgames[i] = lmF["partIDs_okgames"].index(lmF["partIDs"][i])
     
#      hasAQ28_ids_partIDs.append(
# filtdat_okgames_partIDs = []
# for i in range(len(lmF["filtdat_okgames"])):    For each
#      filtdat_okgames_partIDs.append(lmF["partIDs_okgames"][lmF["filtdat_okgames"][i]])

# filtdat_ac_labels = _N.zeros(len(lmF["filtdat"]), dtype=int)    
# for i in range(len(lmF["filtdat"])):    For each
#      partID = lmF["partIDs"][lmF["filtdat"][i]]
#      filtdat_ac_labels[i] = ac.labels_[filtdat_okgames_partIDs.index(partID)]

# fig = _plt.figure(figsize=(13, 10))     
# ist = 0
# for star in ["AQ28scrs", "soc_skils", "imag", "rout", "switch", "fact_pat"]:
#      ist += 1
#      fig.add_subplot(3, 2, ist)
#      _plt.title(star)
#      exec("targ = %s_filtdat" % star)
#      for ic in range(nClusts):
#           ths = _N.where(filtdat_ac_labels == ic)[0]
#           _plt.scatter(ic + 0.05*_N.random.randn(len(ths)), targ[ths] + 0.08*_N.random.randn(len(ths)))
#           mn = _N.mean(targ[ths])
#           _plt.plot([ic-0.3, ic+0.3], [mn, mn], color="black", lw=3)
     
# _plt.savefig("OUT.png")


# fig = _plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.set_facecolor("#333333")
# _plt.scatter(mp2d[0:271, 0], mp2d[0:271, 1], color="white")
# _plt.scatter(mp2d[271:, 0], mp2d[271:, 1], color="red")

xmin=_N.min(mp2d[:, 0])
xmax=_N.max(mp2d[:, 0])
Ax  = xmax-xmin
xmin-= Ax*0.02
xmax+= Ax*0.02
ymin=_N.min(mp2d[:, 1])
ymax=_N.max(mp2d[:, 1])
Ay  = ymax-ymin
ymin-= Ay*0.02
ymax+= Ay*0.02

for ish in range(1):
     rmp2d = _N.array(mp2d)
     if ish > 0:
          _N.random.shuffle(rmp2d)  #  A = _N.array([[0, 10], [1, 11], [2, 12], [3, 13], [4, 14], [5, 15]])   shuffles along 0th axis
     fig = _plt.figure(figsize=(10.7, 3))
     fig.add_subplot(1, 3, 1)
     _plt.title("TMB2")
     outTMB=_plt.hist2d(rmp2d[0:271, 0], rmp2d[0:271, 1], bins=(_N.linspace(xmin, xmax, 20), _N.linspace(ymin, ymax, 20)), cmap="Greys", density=True)
     _plt.xlabel("TSNE dim 1")
     _plt.ylabel("TSNE dim 2")
     fig.add_subplot(1, 3, 2)
     _plt.title("CW")
     outCW=_plt.hist2d(rmp2d[271:, 0], rmp2d[271:, 1], bins=(_N.linspace(xmin, xmax, 20), _N.linspace(ymin, ymax, 20)), cmap="Greys", density=True)
     _plt.xlabel("TSNE dim 1")
     _plt.ylabel("TSNE dim 2")
     fig.add_subplot(1, 3, 3)
     zmin = _N.min(outTMB[0]-outCW[0])
     zmax = _N.max(outTMB[0]-outCW[0])
     print("-------   %(zmin).2e   %(zmax).2e" % {"zmin" : zmin, "zmax" : zmax})
     bothMax=_N.max([_N.abs(zmin), _N.abs(zmax)])
     #_plt.imshow((outTMB[0]-outCW[0])[::-1, ::-1].T, vmin=(-bothMax), vmax=bothMax, cmap="seismic")
     _plt.imshow((outCW[0]-outTMB[0]).T, vmin=(-bothMax), vmax=bothMax, cmap="seismic", origin="lower")
     _plt.xticks([])
     _plt.yticks([])
     fig.subplots_adjust(wspace=0.35,left=0.15, bottom=0.15, right=0.98)
     _plt.savefig("densitydiff_TSNE_TMB_CW")
