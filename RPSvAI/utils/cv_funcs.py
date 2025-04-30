from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
import sklearn.linear_model as _skl
import numpy as _N
import scipy.stats as _ss
import warnings
warnings.filterwarnings("ignore")

def pickFeaturesOneCV(X, y, flds=4, REPS=2):
    """
    One call to LassoCV, which will do this REPS number of data folds
    """
    nFeat = X.shape[1]
    datinds = _N.arange(X.shape[0])
    rkf = RepeatedKFold(n_splits=flds, n_repeats=REPS)#, random_state=0)
    train_test = rkf.split(datinds)
    reg = LassoCV(cv=train_test, max_iter=100000).fit(X, y)
    use_features = _N.where(reg.coef_ != 0)[0]
    return use_features, reg.coef_[use_features]

def pickFeaturesTwoCVs(X, y, outer_flds=5, inner_flds=3, outer_REPS=20, innter_REPS=4):
    """
    Outer CV splits all data into 150:50   (for example)
    LassoCV is called on the 150 training set (110:40) to pick features
    We then 
    """
    nFeat = X.shape[1]
    datinds = _N.arange(X.shape[0])
    rkf = RepeatedKFold(n_splits=outer_flds, n_repeats=outer_REPS)#, random_state=0)
    #rkfINNER = RepeatedKFold(n_splits=inner_flds, n_repeats=innter_REPS)#, random_state=0)
    iii = -1
    weights  = _N.zeros((outer_REPS*outer_flds, nFeat))
    #ichosen = _N.zeros(len(cmp_againsts), dtype=_N.int)
    scores   = _N.empty(outer_REPS*outer_flds)
    #oscores   = _N.empty(outer_REPS*outer_flds)    
    tar_prd_CC = _N.empty(outer_REPS*outer_flds)
    
    for train, test in rkf.split(datinds):
        iii += 1
        ####  first, pick alpha using LassoCV
        #train_data_inds = _N.arange(len(train))
        #splits = rkfINNER.split(train_data_inds)
        #reg = LassoCV(cv=splits, max_iter=100000).fit(X[train], y[train])
        reg = LassoCV(cv=inner_flds, max_iter=100000).fit(X[train], y[train])
        #nonzero = _N.where(reg.coef_ != 0)[0]
        #oreg = LinearRegression().fit(X[train][:, nonzero], y[train])        
        scr = reg.score(X[test], y[test])
        #oscr = oreg.score(X[test][:, nonzero], y[test])        
        #predicted = reg.predict(X[test])
        tar_prd_CC[iii], pv = _ss.pearsonr(reg.predict(X[test]), y[test])
        scores[iii] = scr
        #oscores[iii] = oscr
        #print("scr %(1).3f    ols_scr %(2).3f" % {"1" : scr, "2" : scr})
        weights[iii] = reg.coef_

    pcpvs = _N.empty((nFeat, 2))
    for i in range(nFeat):
        pcpvs[i] = _ss.pearsonr(y, X[:, i])

    ths_feats = 1./(_N.std(weights, axis=0) / _N.abs(_N.mean(weights, axis=0)))
    return weights, ths_feats, scores, tar_prd_CC, pcpvs
    
def predictLR(nf, nrep, X, use_features, y, ):
    """
    nf    n_folds
    """
    clf = _skl.LinearRegression()
    scores_folds = []
    scores_thresh = []   #  each thresh, 3 different folds

    N  = X.shape[0]
    fi = -1
    datinds = _N.arange(N)
    Xs_train = _N.empty((N, len(use_features)))
    
    if len(use_features) > 0:
        for i_feat_indx in use_features:
            fi += 1
            Xs_train[:, fi] = X[:, i_feat_indx]

        coefsLR = _N.empty((nrep*nf, len(use_features)))
        #test_sz = nf*(len(filtdat)//nf)-(ns-1)*(len(filtdat)//nf)
        test_sz = N//nf + 1 if N % nf != 0 else N//nf
        print("test_sz   %d" % test_sz)
        obs_v_preds = _N.zeros((nrep*nf, test_sz, 2))

        scoresLR = _N.empty(nrep*nf)
        rkf = RepeatedKFold(n_splits=nf, n_repeats=nrep)#, random_state=0)
        iii = -1

        for train, test in rkf.split(datinds):
            iii += 1
            clf_f = clf.fit(Xs_train[train], y[train])
            scoresLR[iii] = clf_f.score(Xs_train[test], y[test])
            coefsLR[iii] = clf_f.coef_
            obs_v_preds[iii, 0:len(test), 0] = y[test]
            obs_v_preds[iii, 0:len(test), 1] = clf_f.predict(Xs_train[test])


        print("LR     nf %(ns)d    %(mn).4f  %(md).4f" % {"mn" : _N.mean(scoresLR), "md" : _N.median(scoresLR), "ns" : nf})
        #abs_cvs_of_coeffs = _N.abs(_N.std(coefsLR, axis=0) / _N.mean(coefsLR, axis=0))
        #outstrLR = str_float_array(abs_cvs_of_coeffs, "%.2f")

        mnLR = _N.mean(scoresLR)
        mdLR = _N.median(scoresLR)
        return scoresLR, coefsLR, obs_v_preds

    return None, None, None

