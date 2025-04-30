import numpy as _N
import matplotlib.pyplot as _plt
_plt.rcParams["text.usetex"] = True
from RPSvAI.utils.dir_util import workdirFN, outdirFN
import warnings
import RPSvAI.utils.readfeats as _rft
import scipy.stats as _ss
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings("ignore")


def depickle(s):
    import pickle
    with open(s, "rb") as f:
        lm = pickle.load(f)
    return lm

win_type = 2
win  = 3
smth  = 1

lm = depickle(workdirFN("pcpvsv6_%(wt)d%(w)d%(s)d.dmp" % {"wt" : win_type, "w" : win, "s" : smth}))

filtdat, all_feats, all_feats_s, all_feats_label, AQ28scrs, soc_skils, imag, rout, switch, fact_pat  = _rft.returnFeatures(win_type, win, smth)
disptar = ["Composite", "Social Skills", "Imagination", "Routine", "Switching", "Nums and Patterns"]
itar = -1

lbsz=19
tksz=17

fig = _plt.figure(figsize=(12, 6))
ig  = -1
for tar in ["AQ28scrs", "soc_skils", "imag", "rout", "switch", "fact_pat"]:
    itar += 1
    exec("target = %s" % tar)
    srtd = _N.argsort(_N.abs(lm["pc_%s" % tar][:, 0]))

    nFeats = all_feats.shape[1]    
    fi = nFeats-1
    pc, pv = _ss.pearsonr(target[filtdat], all_feats[filtdat, srtd[fi]])

    while (pv*nFeats) < (8e-2) and (ig < 7):
        ig += 1
        #fig = _plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(2, 4, ig+1)
        _plt.scatter(target[filtdat] + 0.12*_N.random.randn(len(filtdat)), all_feats[filtdat, srtd[fi]], s=3, color="black")
        pvCorrected = pv*nFeats
        _plt.title("(%(pc).3f, %(pv).3f)" % {"pc" : pc, "ft" : all_feats_label[srtd[fi-1]], "tar" : tar, "pv" : pvCorrected}, fontsize=tksz)


        featname = all_feats_label[srtd[fi]]
        tr_featname = _rft.translate_name(featname)
        _plt.xticks(fontsize=tksz)
        _plt.yticks(fontsize=tksz)
        _plt.xlabel("%s" % disptar[itar], fontsize=lbsz)
        _plt.ylabel(r"%s" % tr_featname, fontsize=lbsz)        
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
        fi -= 1
        pc, pv = _ss.pearsonr(target[filtdat], all_feats[filtdat, srtd[fi]])
fig.subplots_adjust(bottom=0.1, wspace=0.49, hspace=0.48, left=0.09, right=0.99, top=0.9)
#_plt.savefig("%(tar)s_%(ft)s" % {"ft" : featname, "tar" : tar})
_plt.savefig("top_correlations_%(wt)d%(w)d%(s)d" % {"wt" : win_type, "w" : win, "s" : smth})


        
