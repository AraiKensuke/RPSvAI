import numpy as _N

def get_dbehv(prob_mvs, gk, cond=_N.array([0, 1, 2]), equalize=False):
    ab_d_prob_mvs = _N.abs(_N.diff(prob_mvs, axis=2))  #  time derivative
    if equalize:
        std_r = _N.std(ab_d_prob_mvs, axis=2).reshape(3, 3, 1)
        ab_d_prob_mvs /= std_r
    behv = _N.sum(_N.sum(ab_d_prob_mvs[cond], axis=1), axis=0)  #  1-D timeseries
    _dbehv = _N.diff(behv)       #  use to find maxes of time derivative
    if gk is not None:
        return _N.convolve(_dbehv, gk, mode="same")
    else:
        return _dbehv

def deterministic_rule(all_tds, TO=300):
    """
    """
    #frameworks   = ["DSUWTL", "RPSWTL",
    #                "RPSRPS", "LCBRPS",
    #                "DSUAIRPS", "LCBAIRPS"]

    cv_onrule = _N.empty((all_tds.shape[0], 6, 3, 3))
    cv_offrule = _N.empty((all_tds.shape[0], 6, 3, 3))    
    for shf in range(all_tds.shape[0]):
        tds = all_tds[shf]
        for ifr in range(6):
            for ic in range(3):
                for ia in range(3):
                    if ((ifr == 0) or (ifr == 1)):
                        condColumn = 2        
                        if ic == 2:    #  Lose
                            cond = -1
                        elif ic == 1:
                            cond = 0
                        elif ic == 0:
                            cond = 1
                    if ((ifr == 2) or (ifr == 3)):
                        condColumn = 0        
                        if ic == 2:    #  Lose
                            cond = 3
                        elif ic == 1:
                            cond = 2
                        elif ic == 0:
                            cond = 1
                    if ((ifr == 4) or (ifr == 5)):
                        condColumn = 1
                        if ic == 2:    #  Lose
                            cond = 3
                        elif ic == 1:
                            cond = 2
                        elif ic == 0:
                            cond = 1

                    if (ifr == 0) or (ifr == 2) or (ifr == 4):   #  ACTION DSU
                        rule0 = _N.where((tds[1:-1, condColumn] == cond) &
                                         (((tds[1:-1, 0] == 1) & (tds[2:, 0] == 2)) |
                                          ((tds[1:-1, 0] == 2) & (tds[2:, 0] == 3)) |
                                          ((tds[1:-1, 0] == 3) & (tds[2:, 0] == 1))))[0]
                        rule2 = _N.where((tds[1:-1, condColumn] == cond) &
                                         (((tds[1:-1, 0] == 1) & (tds[2:, 0] == 3)) |
                                          ((tds[1:-1, 0] == 2) & (tds[2:, 0] == 1)) |
                                          ((tds[1:-1, 0] == 3) & (tds[2:, 0] == 2))))[0]
                        rule1 = _N.where((tds[1:-1, condColumn] == cond) &
                                         (tds[1:-1, 0] ==tds[2:, 0]))[0]
                    #if (ifr == 1) or (ifr == 2):   #  ACTION RPS
                    if (ifr == 1):   #  ACTION RPS
                        rule0 = _N.where((tds[1:-1, condColumn] == cond) &
                                         (tds[2:, 0] == 1))[0]
                        rule1 = _N.where((tds[1:-1, condColumn] == cond) &
                                         (tds[2:, 0] == 2))[0]
                        rule2 = _N.where((tds[1:-1, condColumn] == cond) &
                                         (tds[2:, 0] == 3))[0]
                    if (ifr == 3) or (ifr == 5):   #  ACTION LCB
                        rule0 = _N.where((tds[1:-1, condColumn] == cond) &
                                         (((tds[1:-1, 1] == 1) & (tds[2:, 0] == 2)) |
                                          ((tds[1:-1, 1] == 2) & (tds[2:, 0] == 3)) |
                                          ((tds[1:-1, 1] == 3) & (tds[2:, 0] == 1))))[0]
                        rule2 = _N.where((tds[1:-1, condColumn] == cond) &
                                         (((tds[1:-1, 1] == 1) & (tds[2:, 0] == 3)) |
                                          ((tds[1:-1, 1] == 2) & (tds[2:, 0] == 1)) |
                                          ((tds[1:-1, 1] == 3) & (tds[2:, 0] == 2))))[0]
                        rule1 = _N.where((tds[1:-1, condColumn] == cond) &
                                         (tds[1:-1, 1] == tds[2:, 0]))[0]

                    allT = _N.ones(TO)*-1                
                    if ia == 0:   #  transition down
                        allT[rule0] = 1
                        offrule = _N.sort(_N.array(rule1.tolist() + rule2.tolist()))
                    elif ia == 1:
                        allT[rule1] = 1
                        offrule = _N.sort(_N.array(rule0.tolist() + rule2.tolist()))
                    elif ia == 2:
                        allT[rule2] = 1
                        offrule = _N.sort(_N.array(rule0.tolist() + rule1.tolist()))
                    #print(offrule)
                    #if offrule.shape[0] == 0:
                    #    print("shape 0")
                    if offrule.shape[0] > 0:
                        allT[offrule] = 0
                    defts = _N.where(allT > -1)[0]
                    squeezed = allT[defts]
                    offrule = _N.where(squeezed == 0)[0]
                    onrule  = _N.where(squeezed == 1)[0]
                    onrule_isi = _N.diff(onrule)
                    offrule_isi = _N.diff(offrule)
                    cv_onrule[shf, ifr, ic, ia] = _N.std(onrule_isi) / _N.mean(onrule_isi)
                    cv_offrule[shf, ifr, ic, ia] = _N.std(offrule_isi) / _N.mean(offrule_isi)
    return cv_onrule, cv_offrule

def get_dbehv_combined(prob_mvs_list, gk, cond=_N.array([0, 1, 2]), equalize=False, biggest=False, top_comps=9, use_sds=None):
    """
    Rule change is when probability is changing rapidly
    MAXIMA of this   -->  SUM( ABS( dP(act_i | cond_j) / dt ) )
    """

    n_diff_repr = len(prob_mvs_list)
    L = prob_mvs_list[0].shape[2]  # ngames

    all_prob_mvs = _N.empty((9*n_diff_repr, prob_mvs_list[0].shape[2]))

    std_4_repr   = _N.zeros(n_diff_repr)

    if biggest:
        use      = _N.zeros(n_diff_repr*top_comps, dtype=int)
    for nr in range(n_diff_repr):
        all_prob_mvs[nr*9:(nr+1)*9] =  prob_mvs_list[nr].reshape(9, L)
        std_4_repr[nr] = _N.std(prob_mvs_list[nr].reshape(9, L))  #  9 stds
        if biggest:
            if use_sds is None:
                comp_stds = _N.std(prob_mvs_list[nr].reshape(9, L), axis=1)
            else:
                comp_stds = use_sds[nr].reshape(9)
            
            srtdinds = _N.argsort(comp_stds)
            use[nr*top_comps:(nr+1)*top_comps] = srtdinds[9-top_comps:]+nr*top_comps

    ab_d_prob_mvs = _N.abs(_N.diff(all_prob_mvs, axis=1))  #  time derivative

    # if equalize:
    #     std_r = _N.std(ab_d_prob_mvs, axis=1).reshape(9*n_diff_repr, 1)
    #     ab_d_prob_mvs /= std_r

    if biggest:
        #print(use)
        behv = _N.sum(ab_d_prob_mvs[use], axis=0)  #  1-D timeseries
    else:
        behv = _N.sum(ab_d_prob_mvs, axis=0)  #  1-D timeseries        
    """
    if gk is not None:
        fbehv = _N.convolve(behv, gk, mode="same")
        dbehv = _N.diff(fbehv)       #  use to find maxes of time derivative
        return dbehv, fbehv
    else:
        return _dbehv, behv
    
    """
    _dbehv = _N.diff(behv)       #  use to find maxes of time derivative
    if gk is not None:
        return _N.convolve(_dbehv, gk, mode="same"), behv
    else:
        return _dbehv, behv

def get_dbehv_biggest_fluc(prob_mvs_list, chosen_frmwks_list, gk, ranks, ranks0s, len1, min_big_comps=2, big_percentile=0.95, flip_choose_components=False, SHUFFLES=205):
#def get_dbehv_biggest_fluc(prob_mvs_list, gk, cmpZs, min_big_comps=2, big_percentile=0.95, flip_choose_components=False):
    """
    Rule change is when probability is changing rapidly
    MAXIMA of this   -->  SUM( ABS( dP(act_i | cond_j) / dt ) )
    """

    L = prob_mvs_list[0].shape[2]  # ngames

    l_all_prob_mvs = []
    iOKcomps = 0
    for ifr in chosen_frmwks_list:
        for ic in range(3):
            for ia in range(3):
                use = False
                #if (ranks[ifr, ic, ia] / SHUFFLES > big_percentile) or ((ranks0s[ifr, ic, ia] / SHUFFLES > 0.98) and (len1[ifr, ic, ia] > 5)):
                if (ranks[ifr, ic, ia] / SHUFFLES > big_percentile) or (ranks0s[ifr, ic, ia] / SHUFFLES > big_percentile):
                #if (ranks[ifr, ic, ia] / SHUFFLES > big_percentile) and (ranks0s[ifr, ic, ia] / SHUFFLES > 0.5):
                #if (cmpZs[ifr, ic, ia] > 2):
                    if not flip_choose_components:
                        if gk is None:
                            l_all_prob_mvs.append(prob_mvs_list[ifr][ic, ia])
                        else:
                            l_all_prob_mvs.append(_N.convolve(prob_mvs_list[ifr][ic, ia], gk, mode="same"))
                else:
                    if flip_choose_components:
                        if gk is None:

                            l_all_prob_mvs.append(prob_mvs_list[ifr][ic, ia])
                        else:
                            l_all_prob_mvs.append(_N.convolve(prob_mvs_list[ifr][ic, ia], gk, mode="same"))


    all_prob_mvs = _N.array(l_all_prob_mvs)

    n_big_comps = all_prob_mvs.shape[0]

    # std_4_repr   = _N.zeros(n_diff_repr)

    if all_prob_mvs.shape[0] > min_big_comps:  #  all_prob_mvs
        ab_d_prob_mvs = _N.abs(_N.diff(all_prob_mvs, axis=1))  #  time derivative
        
        dbehv = _N.sum(ab_d_prob_mvs, axis=0)  #  1-D timeseries        
        return dbehv, all_prob_mvs.shape[0]
    else:
        return None, 0
    # """
    # if gk is not None:
    #     fbehv = _N.convolve(behv, gk, mode="same")
    #     dbehv = _N.diff(fbehv)       #  use to find maxes of time derivative
    #     return dbehv, fbehv
    # else:
    #     return _dbehv, behv
    
    # """
    # _dbehv = _N.diff(behv)       #  use to find maxes of time derivative
    # if gk is not None:
    #     return _N.convolve(_dbehv, gk, mode="same"), behv
    # else:
    #     return _dbehv, behv

def get_dbehv_biggest_flucDBG(prob_mvs_list, chosen_frmwks_list, gk, ranks, ranks0s, len1, min_big_comps=2, big_percentile=0.95, flip_choose_components=False, SHUFFLES=205):
#def get_dbehv_biggest_fluc(prob_mvs_list, gk, cmpZs, min_big_comps=2, big_percentile=0.95, flip_choose_components=False):
    """
    Rule change is when probability is changing rapidly
    MAXIMA of this   -->  SUM( ABS( dP(act_i | cond_j) / dt ) )
    """

    L = prob_mvs_list[0].shape[2]  # ngames

    l_all_prob_mvs = []
    iOKcomps = 0

    RULES = []
    for ifr in chosen_frmwks_list:
        for ic in range(3):
            for ia in range(3):
                use = False
                #if (ranks[ifr, ic, ia] / SHUFFLES > big_percentile) or ((ranks0s[ifr, ic, ia] / SHUFFLES > 0.98) and (len1[ifr, ic, ia] > 5)):
                if (ranks[ifr, ic, ia] / SHUFFLES > big_percentile) or (ranks0s[ifr, ic, ia] / SHUFFLES > big_percentile):
                #if (ranks[ifr, ic, ia] / SHUFFLES > big_percentile) and (ranks0s[ifr, ic, ia] / SHUFFLES > 0.5):
                #if (cmpZs[ifr, ic, ia] > 2):
                    if not flip_choose_components:
                        if gk is None:
                            l_all_prob_mvs.append(prob_mvs_list[ifr][ic, ia])
                        else:
                            l_all_prob_mvs.append(_N.convolve(prob_mvs_list[ifr][ic, ia], gk, mode="same"))
                else:
                    if flip_choose_components:
                        if gk is None:

                            l_all_prob_mvs.append(prob_mvs_list[ifr][ic, ia])
                        else:
                            l_all_prob_mvs.append(_N.convolve(prob_mvs_list[ifr][ic, ia], gk, mode="same"))


    all_prob_mvs = _N.array(l_all_prob_mvs)

    n_big_comps = all_prob_mvs.shape[0]

    # std_4_repr   = _N.zeros(n_diff_repr)

    if all_prob_mvs.shape[0] > min_big_comps:  #  all_prob_mvs
        ab_d_prob_mvs = _N.abs(_N.diff(all_prob_mvs, axis=1))  #  time derivative
        
        dbehv = _N.sum(ab_d_prob_mvs, axis=0)  #  1-D timeseries        
        return dbehv, all_prob_mvs, all_prob_mvs.shape[0]
    else:
        return None, None, 0
    # """
    # if gk is not None:
    #     fbehv = _N.convolve(behv, gk, mode="same")
    #     dbehv = _N.diff(fbehv)       #  use to find maxes of time derivative
    #     return dbehv, fbehv
    # else:
    #     return _dbehv, behv
    
    # """
    # _dbehv = _N.diff(behv)       #  use to find maxes of time derivative
    # if gk is not None:
    #     return _N.convolve(_dbehv, gk, mode="same"), behv
    # else:
    #     return _dbehv, behv

def get_dbehv_biggest_fluc_seprules(prob_mvs_list, chosen_frmwks_list, gk, ranks, ranks0s, min_big_comps=2, big_percentile=0.95, flip_choose_components=False, SHUFFLES=205):
#def get_dbehv_biggest_fluc(prob_mvs_list, gk, cmpZs, min_big_comps=2, big_percentile=0.95, flip_choose_components=False):
    """
    Rule change is when probability is changing rapidly
    MAXIMA of this   -->  SUM( ABS( dP(act_i | cond_j) / dt ) )
    """

    L = prob_mvs_list[0].shape[2]  # ngames

    l_all_prob_mvs = []
    l_lohis        = []
    iOKcomps = 0

    RULES = []
    for ifr in chosen_frmwks_list:
        for ic in range(3):
            for ia in range(3):
                use = False
                #if (ranks[ifr, ic, ia] / SHUFFLES > big_percentile) or ((ranks0s[ifr, ic, ia] / SHUFFLES > 0.98) and (len1[ifr, ic, ia] > 5)):
                OFFs = (ranks[ifr, ic, ia] / SHUFFLES > big_percentile)
                ONs  = (ranks0s[ifr, ic, ia] / SHUFFLES > big_percentile)
                if OFFs or ONs:
                #if (ranks[ifr, ic, ia] / SHUFFLES > big_percentile) and (ranks0s[ifr, ic, ia] / SHUFFLES > 0.5):
                #if (cmpZs[ifr, ic, ia] > 2):
                    if OFFs and not ONs:
                        l_lohis.append(0)
                    elif not OFFs and ONs:
                        l_lohis.append(1)
                    else:
                        l_lohis.append(2)
                    
                    if not flip_choose_components:
                        if gk is None:
                            l_all_prob_mvs.append(prob_mvs_list[ifr][ic, ia])
                        else:
                            l_all_prob_mvs.append(_N.convolve(prob_mvs_list[ifr][ic, ia], gk, mode="same"))
                else:
                    if flip_choose_components:
                        if gk is None:

                            l_all_prob_mvs.append(prob_mvs_list[ifr][ic, ia])
                        else:
                            l_all_prob_mvs.append(_N.convolve(prob_mvs_list[ifr][ic, ia], gk, mode="same"))


    all_prob_mvs = _N.array(l_all_prob_mvs)
    lohis        = _N.array(l_lohis)

    n_big_comps = all_prob_mvs.shape[0]

    # std_4_repr   = _N.zeros(n_diff_repr)

    if all_prob_mvs.shape[0] > min_big_comps:  #  all_prob_mvs
        ab_d_prob_mvs = _N.abs(_N.diff(all_prob_mvs, axis=1))  #  time derivative
        
        #dbehv = _N.sum(ab_d_prob_mvs, axis=0)  #  1-D timeseries        
        return ab_d_prob_mvs, all_prob_mvs, all_prob_mvs.shape[0], lohis
    else:
        return None, None, 0, None
    # """
    # if gk is not None:
    #     fbehv = _N.convolve(behv, gk, mode="same")
    #     dbehv = _N.diff(fbehv)       #  use to find maxes of time derivative
    #     return dbehv, fbehv
    # else:
    #     return _dbehv, behv
    
    # """
    # _dbehv = _N.diff(behv)       #  use to find maxes of time derivative
    # if gk is not None:
    #     return _N.convolve(_dbehv, gk, mode="same"), behv
    # else:
    #     return _dbehv, behv
    
def get_dbehv_combined_choose_(prob_mvs, gk, equalize=False):
    n_diff_repr = len(prob_mvs)   #  list of prob_mvs for each representation
    L = prob_mvs[0].shape[2]
    
    l_all_prob_mvs = []#_N.empty((9*n_diff_repr, prob_mvs_list[0].shape[2]))

    SHUFFLES = prob_mvs[0].shape[0]-1
    chng_pms = 0
    for nr in range(n_diff_repr):
        #l_bigchgs = []
        s = _N.std(prob_mvs[nr], axis=2)   #  std of move prob over all games
        std0 = _N.std(s[1:], axis=0)       #  std of SHUFFLED 
        m0   = _N.mean(s[1:], axis=0)
        z0   = (s[0] - m0) / std0
        bigchgs  = _N.where(z0 > 1.)[0]
        # for i in range(9):        
        #     ths = _N.where(s[0, i] > s[1:, i])[0]
        #     if len(ths) > int(SHUFFLES*0.95):
        #         l_bigchgs.append(i)
        # bigchgs = _N.array(l_bigchgs)
        chng_pms += len(bigchgs)

        for ib in bigchgs:
            l_all_prob_mvs.append(_N.array(prob_mvs[nr][0, ib]))
    if chng_pms == 0:
        return 0, None
    all_prob_mvs = _N.array(l_all_prob_mvs)
    ab_d_prob_mvs = _N.abs(_N.diff(all_prob_mvs, axis=1))  #  time derivative
    # if equalize:
    #     std_r = _N.std(ab_d_prob_mvs, axis=1).reshape(9*n_diff_repr, 1)
    #     ab_d_prob_mvs /= std_r
    behv = _N.sum(ab_d_prob_mvs, axis=0)  #  1-D timeseries
    _dbehv = _N.diff(behv)       #  use to find maxes of time derivative
    if gk is not None:
        return chng_pms, _N.convolve(_dbehv, gk, mode="same")
    else:
        return chng_pms, _dbehv
    
def entropy3(sig, N):
    cube = _N.zeros((N, N, N))   #  W T L conditions or
    iN   = 1./N

    #print(sig.shape[0])
    for i in range(sig.shape[0]):
        ix = int(sig[i, 0]/iN)
        iy = int(sig[i, 1]/iN)
        iz = int(sig[i, 2]/iN)
        ix = ix if ix < N else N-1
        iy = iy if iy < N else N-1
        iz = iz if iz < N else N-1
        cube[ix, iy, iz] += 1

    #print(cube)
    entropy  = 0
    for i in range(N):
        for j in range(N):
            for k in range(N):
                p_ijk = cube[i, j, k] / len(sig)
                if p_ijk > 0:
                    entropy += -p_ijk * _N.log(p_ijk)
    return entropy

def entropy2(sig, N):
    #  calculate the entropy
    square = _N.zeros((N, N))
    iN   = 1./N
    for i in range(len(sig)):
        ix = int(sig[i, 0]/iN)
        iy = int(sig[i, 1]/iN)
        ix = ix if ix < N else N-1
        iy = iy if iy < N else N-1
        square[ix, iy] += 1

    entropy  = 0
    for i in range(N):
        for j in range(N):
                p_ij = square[i, j] / len(sig)
                if p_ij > 0:
                    entropy += -p_ij * _N.log(p_ij)
    return entropy
