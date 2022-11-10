#!/usr/bin/python


from sklearn import linear_model
import sklearn.linear_model as _skl
import numpy as _N
import RPSvAI.utils.read_taisen as _rt
import scipy.io as _scio
import scipy.stats as _ss
import matplotlib.pyplot as _plt
import RPSvAI.utils.read_taisen as _rd
import RPSvAI.utils.misc as _Am
from scipy.signal import savgol_filter
from GCoh.eeg_util import unique_in_order_of_appearance, increasing_labels_mapping, rmpd_lab_trnsfrm, find_or_retrieve_GMM_labels, shift_correlated_shuffle, shuffle_discrete_contiguous_regions, mtfftc
from RPSvAI.utils.dir_util import workdirFN, datadirFN
import os
import sys
from sumojam.devscripts.cmdlineargs import process_keyval_args
import pickle
import mne.time_frequency as mtf
import GCoh.eeg_util as _eu
#import RPSvAI.rpsms as rpsms
import GCoh.preprocess_ver as _ppv

import RPSvAI.constants as _cnst
#from RPSvAI.utils.dir_util import getResultFN
import GCoh.datconfig as datconf
import RPSvAI.models.CRutils as _crut
import RPSvAI.models.empirical_ken as _emp
from sklearn.decomposition import PCA
import RPSvAI.AIRPSfeatures as _aift

import GCoh.eeg_util as _eu
import matplotlib.ticker as ticker
from statsmodels import robust

__DSUWTL__ = 0
__RPSWTL__ = 1
__DSURPS__ = 2
__ALL__    = 3

mode       = __ALL__
#mode       = __DSUWTL__
#mode       = __RPSWTL__
#mode       = __DSURPS__
_plt.ioff()
__1st__ = 0
__2nd__ = 1

_ME_WTL = 0
_ME_RPS = 1

_SHFL_KEEP_CONT  = 0
_SHFL_NO_KEEP_CONT  = 1

#  sum_sd
#  entropyL
#  isi_cv, isis_corr


def depickle(s):
     import pickle
     with open(s, "rb") as f:
          lm = pickle.load(f)
     return lm


lm1 = depickle(workdirFN("shuffledCRs_5CFs_TMB2_3_1"))
lm2 = depickle(workdirFN("shuffledCRs_5CFs_TMB2_3_2"))
filtdat = lm1["filtdat"]

ind_of_1_unf = []
for pID in lm2["partIDs"]:
    index = lm1["partIDs"].index(pID)
    ind_of_1_unf.append(index)
#ind_of_1 = _N.intersect1d(ind_of_1_unf, filtdat)
ind_of_1 = _N.array(ind_of_1_unf)

fr_cmp_fluc_rank1_1 = lm1["fr_cmp_fluc_rank1"]
fr_cmp_fluc_rank1_2 = lm2["fr_cmp_fluc_rank1"]


pcs = _N.empty(len(ind_of_1))
for i in range(len(ind_of_1)):
    pc, pv = _ss.pearsonr(fr_cmp_fluc_rank1_1[ind_of_1[i]].flatten(), fr_cmp_fluc_rank1_2[i].flatten())
    print(pc)
    pcs[i] = pc

    

