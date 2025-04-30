import numpy as _N
import RPSvAI.utils.readfeats as _rft
import matplotlib.pyplot as _plt
_plt.rcParams["text.usetex"] = True
import scipy.stats as _ss
#from statsmodels.stats.outliers_influence import variance_inflation_factor
import sklearn.preprocessing as _skp
import pickle
import RPSvAI.utils.cv_funcs as cvf
import os
from RPSvAI.utils.dir_util import workdirFN, outdirFN


def str_float_array(arr, sfmt):
    s = "["
    for i in range(arr.shape[0]-1):
        s += (sfmt % arr[i]) + ", "
    s += (sfmt % arr[i]) + "]"       
    return s

def standardize(y):
    ys = y - _N.mean(y)
    ys /= _N.std(ys)
    return ys

def depickle(s):
    import pickle
    with open(s, "rb") as f:
        lm = pickle.load(f)
    return lm

def unskew(dat):
    sk = _N.empty(61)
    im = -1
    if _ss.skew(dat) < 0:
        dat *= -1
    dat -= _N.mean(dat)   #  0-mean
    dat /= _N.std(dat)
    dat -= _N.min(dat)
    amp = _N.max(dat) - _N.min(dat)
    dat += 0.1*amp

    return _N.log(dat)

#win_type = 2   #  window is of fixed number of games
win_type = 2  #  window is of fixed number of games that meet condition 
win     = 3
smth    = 1
label=str(win_type*100 + win*10 + smth)

lblsz=17
tksz=15
lm = depickle(workdirFN("pcpvsv6_%(wt)d%(w)d%(s)d.dmp" % {"wt" : win_type, "w" : win, "s" : smth}))

SHUFFLES = 46
scrs_SH     = _N.empty((6, SHUFFLES+1))

#fig = _plt.figure(figsize=(4.5, 3.4))
fig = _plt.figure(figsize=(4, 3.9))
it  = -1
targets = ["soc_skils", "imag", "rout", "switch", "fact_pat", "AQ28scrs"]
disptar = ["Social Skills", "Imagination", "Routine", "Switching", "Nums \& Patterns", "Composite"]
itar = -1

nFeats=1530
for star in targets:
    it += 1
    for sh in range(SHUFFLES+1):
        if sh == 0:
            scrs = lm["inscr_%(nf)d_%(st)s" % {"st" : star, "nf" : nFeats}]
        else:
            scrs = lm["inscr_%(nf)d_%(st)s_sh%(sh)d" % {"sh" : sh, "st" : star, "nf" : nFeats}]
        scrs_SH[it, sh] = _N.median(scrs)

    _plt.scatter(it + 0.08*_N.random.randn(SHUFFLES), scrs_SH[it, 1:], s=5, color="#BBBBBB")
    _plt.scatter(it, scrs_SH[it, 0], color="black", s=16)    
            
    _plt.plot([it-0.3, it+0.3], [0, 0], ls="--", color="blue")

    signif_feats=_N.where(lm["nonzero_weights_%s" % star] > 90)[0]
    print("***************For target %s*****" % star)
    for isf in signif_feats:
        #tr_featname = _rft.translate_name(featname)
        print("%(lb) 20s    %(nz).2f" % {"lb" : _rft.translate_name(lm["all_feats_label"][isf], latex=False), "nz" : (lm["nonzero_weights_%s" % star][isf]/160)}) # 4 x 40=160
        pcpv = lm["pc_%s" % star][isf]
        print("pcpv  pc=%(pc) 3f  pv=%(Bpv).1e (%(pv).1e)" % {"pc": pcpv[0], "pv" : pcpv[1], "Bpv" : (pcpv[1]*1521)})        

_plt.yticks([-0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06], fontsize=tksz)
_plt.ylim(-0.06, 0.07)
_plt.xticks(_N.arange(6), disptar, rotation=75, fontsize=tksz)
_plt.ylabel(r"\center{Median $R^2$ across folds}", fontsize=lblsz)
fig.subplots_adjust(bottom=0.42, left=0.25, right=0.98, top=0.98)
_plt.savefig(outdirFN("lasso_report", label))


