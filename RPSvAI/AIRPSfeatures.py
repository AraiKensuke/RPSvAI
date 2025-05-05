import numpy as _N
import scipy.stats as _ss
import RPSvAI.utils.misc as misc

#  entropy of a given action over the 3 conditions  (how different is probability of an action over the 3 different conditions?)
#  entropy of a actions for a given condition  (how different is probability of the actions for a given condition?)

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

def entropy1(__sig, N, repeat=None, nz=0, maxval=1.):
    """
    _sig   T x 3
    """
    line = _N.zeros(N)   #  W T L conditions or
    maxval = _N.max(__sig)
    minval = _N.min(__sig)
    if maxval == minval:
        return 0
    _sig   = _N.array(__sig)
    __sig -= minval
    maxval -= minval
    iN   = maxval/N

    #print(sig.shape[0])

    if repeat is not None:
        newlen = _sig.shape[0]*repeat
        sig = _N.empty(newlen)
        sig[:, 0] = _N.repeat(_sig[:, 0], repeat) + nz*_N.random.randn(newlen)
    else:
        sig = _sig
    
    for i in range(sig.shape[0]):
        ix = int(sig[i]/iN)
        ix = ix if ix < N else N-1
        line[ix] += 1

    entropy  = 0
    for i in range(N):
        p_i = line[i] / len(sig)
        if p_i > 0:
            entropy += -p_i * _N.log(p_i)
    return entropy

def entropy3(sig, N, repeat=None, nz=0, maxval=1., returnCube=False):
    """
    _sig   T x 3
    """
    cube = _N.zeros((N, N, N))   #  W T L conditions or
    #iN   = maxval/N
    _sig = _N.array(sig)
    maxval = _N.max(_sig)
    minval = _N.min(_sig)
    _sig -= minval
    iN   = maxval/N

    #print(sig.shape[0])

    if repeat is not None:
        newlen = _sig.shape[0]*repeat
        sig = _N.empty((newlen, 3))
        sig[:, 0] = _N.repeat(_sig[:, 0], repeat) + nz*_N.random.randn(newlen)
        sig[:, 1] = _N.repeat(_sig[:, 1], repeat) + nz*_N.random.randn(newlen)
        sig[:, 2] = _N.repeat(_sig[:, 2], repeat) + nz*_N.random.randn(newlen)
    else:
        sig = _sig
    
    for i in range(sig.shape[0]):
        ix = int(sig[i, 0]/iN)
        iy = int(sig[i, 1]/iN)
        iz = int(sig[i, 2]/iN)
        if ix > N:
            print("%(ix)d   %(N)d" % {"ix" : ix, "N" : N})
        if iy > N:
            print("%(iy)d   %(N)d" % {"iy" : iy, "N" : N})
        if iz > N:
            print("%(iz)d   %(N)d" % {"iz" : iz, "N" : N})
            
        ix = ix if ix < N else N-1
        iy = iy if iy < N else N-1
        iz = iz if iz < N else N-1
        cube[ix, iy, iz] += 1

    entropy  = 0
    for i in range(N):
        for j in range(N):
            for k in range(N):
                p_ijk = cube[i, j, k] / len(sig)
                if p_ijk > 0:
                    entropy += -p_ijk * _N.log(p_ijk)
    if returnCube == False:
        return entropy
    else:
        return entropy, cube

def entropyCRprobs(prob, fix="action", normalize=True, PCS=6, PCS1=10):
    """
    fix:  action or condition
    entropyD, entropyS, entropyU = entropyCRprobs(prob_mvs, fix="condition", normalize=True)
    """

    ##  prob[(condition), (action), t]
    if fix=="action":  #
        entAct0 = entropy3(prob[:, 0].T, PCS)
        entAct1 = entropy3(prob[:, 1].T, PCS)
        entAct2 = entropy3(prob[:, 2].T, PCS)
        if normalize:
            #  for a fixed action, 
            act_cond1= entropy1(prob[0, 0], PCS1)
            act_cond2= entropy1(prob[1, 0], PCS1)
            act_cond3= entropy1(prob[2, 0], PCS1)            
            entAct0 /= (act_cond1 + act_cond2 + act_cond3)
            act_cond1= entropy1(prob[0, 1], PCS1)
            act_cond2= entropy1(prob[1, 1], PCS1)
            act_cond3= entropy1(prob[2, 1], PCS1)            
            entAct1 /= (act_cond1 + act_cond2 + act_cond3)
            act_cond1= entropy1(prob[0, 2], PCS1)
            act_cond2= entropy1(prob[1, 2], PCS1)
            act_cond3= entropy1(prob[2, 2], PCS1)            
            entAct2 /= (act_cond1 + act_cond2 + act_cond3)
        return entAct0, entAct1, entAct2
            
    if fix=="condition":  #  
        entCond0 = entropy3(prob[0].T, PCS)
        entCond1 = entropy3(prob[1].T, PCS)
        entCond2 = entropy3(prob[2].T, PCS)
        if normalize:
            #  for a fixed action, 
            cond_act1= entropy1(prob[0, 0], PCS1)
            cond_act2= entropy1(prob[1, 0], PCS1)
            cond_act3= entropy1(prob[2, 0], PCS1)            
            entCond0 /= (cond_act1 + cond_act2 + cond_act3)
            cond_act1= entropy1(prob[0, 1], PCS1)
            cond_act2= entropy1(prob[1, 1], PCS1)
            cond_act3= entropy1(prob[2, 1], PCS1)            
            entCond1 /= (cond_act1 + cond_act2 + cond_act3)
            cond_act1= entropy1(prob[0, 2], PCS1)
            cond_act2= entropy1(prob[1, 2], PCS1)
            cond_act3= entropy1(prob[2, 2], PCS1)            
            entCond2 /= (cond_act1 + cond_act2 + cond_act3)
        return entCond0, entCond1, entCond2            
            
    
def corr_btwn_probCRcomps(prob_mvs):
    pc0001, pv01_0 = _ss.pearsonr(prob_mvs[0, 0], prob_mvs[0, 1])    
    pc0002, pv01_1 = _ss.pearsonr(prob_mvs[0, 0], prob_mvs[0, 2])
    pc0010, pv01_1 = _ss.pearsonr(prob_mvs[0, 0], prob_mvs[1, 0])
    pc0011, pv01_1 = _ss.pearsonr(prob_mvs[0, 0], prob_mvs[1, 1])
    pc0012, pv01_1 = _ss.pearsonr(prob_mvs[0, 0], prob_mvs[1, 2])        
    pc0020, pv01_1 = _ss.pearsonr(prob_mvs[0, 0], prob_mvs[2, 0])
    pc0021, pv01_1 = _ss.pearsonr(prob_mvs[0, 0], prob_mvs[2, 1])
    pc0022, pv01_1 = _ss.pearsonr(prob_mvs[0, 0], prob_mvs[2, 2])
    ################
    pc0102, pv01_0 = _ss.pearsonr(prob_mvs[0, 1], prob_mvs[0, 2])    
    pc0110, pv01_1 = _ss.pearsonr(prob_mvs[0, 1], prob_mvs[1, 0])
    pc0111, pv01_1 = _ss.pearsonr(prob_mvs[0, 1], prob_mvs[1, 1])
    pc0112, pv01_1 = _ss.pearsonr(prob_mvs[0, 1], prob_mvs[1, 2])        
    pc0120, pv01_1 = _ss.pearsonr(prob_mvs[0, 1], prob_mvs[2, 0])
    pc0121, pv01_1 = _ss.pearsonr(prob_mvs[0, 1], prob_mvs[2, 1])
    pc0122, pv01_1 = _ss.pearsonr(prob_mvs[0, 1], prob_mvs[2, 2])        
    ################
    pc0210, pv01_0 = _ss.pearsonr(prob_mvs[0, 2], prob_mvs[1, 0])    
    pc0211, pv01_1 = _ss.pearsonr(prob_mvs[0, 2], prob_mvs[1, 1])
    pc0212, pv01_1 = _ss.pearsonr(prob_mvs[0, 2], prob_mvs[1, 2])        
    pc0220, pv01_1 = _ss.pearsonr(prob_mvs[0, 2], prob_mvs[2, 0])
    pc0221, pv01_1 = _ss.pearsonr(prob_mvs[0, 2], prob_mvs[2, 1])
    pc0222, pv01_1 = _ss.pearsonr(prob_mvs[0, 2], prob_mvs[2, 2])        
    ################
    pc1011, pv01_1 = _ss.pearsonr(prob_mvs[1, 0], prob_mvs[1, 1])
    pc1012, pv01_1 = _ss.pearsonr(prob_mvs[1, 0], prob_mvs[1, 2])        
    pc1020, pv01_1 = _ss.pearsonr(prob_mvs[1, 0], prob_mvs[2, 0])
    pc1021, pv01_1 = _ss.pearsonr(prob_mvs[1, 0], prob_mvs[2, 1])
    pc1022, pv01_1 = _ss.pearsonr(prob_mvs[1, 0], prob_mvs[2, 2])        
    ################
    pc1112, pv01_1 = _ss.pearsonr(prob_mvs[1, 1], prob_mvs[1, 2])        
    pc1120, pv01_1 = _ss.pearsonr(prob_mvs[1, 1], prob_mvs[2, 0])
    pc1121, pv01_1 = _ss.pearsonr(prob_mvs[1, 1], prob_mvs[2, 1])
    pc1122, pv01_1 = _ss.pearsonr(prob_mvs[1, 1], prob_mvs[2, 2])        
    ################
    pc1220, pv01_1 = _ss.pearsonr(prob_mvs[1, 2], prob_mvs[2, 0])
    pc1221, pv01_1 = _ss.pearsonr(prob_mvs[1, 2], prob_mvs[2, 1])
    pc1222, pv01_1 = _ss.pearsonr(prob_mvs[1, 2], prob_mvs[2, 2])        
    ################
    pc2021, pv01_1 = _ss.pearsonr(prob_mvs[2, 0], prob_mvs[2, 1])
    pc2022, pv01_1 = _ss.pearsonr(prob_mvs[2, 0], prob_mvs[2, 2])
    ################    
    pc2122, pv01_1 = _ss.pearsonr(prob_mvs[2, 1], prob_mvs[2, 2])

    return _N.array([pc0001, pc0002, pc0010, pc0011, pc0012, pc0020, pc0021, pc0022,
                     pc0102, pc0110, pc0111, pc0112, pc0120, pc0121, pc0122,
                     pc0210, pc0211, pc0212, pc0220, pc0221, pc0222,
                     pc1011, pc1012, pc1020, pc1021, pc1022,
                     pc1112, pc1120, pc1121, pc1122,
                     pc1220, pc1221, pc1222,
                     pc2021, pc2022,
                     pc2122])

def cntrmvs_DSUWTL(prob_mvs, TO):
    """
    count number of counter and non-counter moves (as defined by mental model)
    """
    cntr = 0
    n_cntr = 0
    maxp_chg_times_wtl = []

    for wtl in [0, 2]:
        #  A counter move.
        #  If my most probably move now is UP.  The next time same condition
        #  happens, AI will
        #  WIN WIN WIN WIN WIN WIN case
        #  .... was at WIN-STAY
        #  .... 
        #  HP  R R                   S (S)   AI thinks HP will stay, and DN.
        #  AI  S                     P  R    HP should S->P  DN to counter
        #  so if my highest probability is now Win-Stay,
        #  if I then transition to Win-Dn, that is a counter move.
        #  .... was at WIN-DN        
        #  HP  R R2                   S (S)   AI thinks HP will stay, and DN.
        #  AI  S                     P  R    HP should S->P  DN to counter
        #  so if my highest probability is now Win-Stay,
        #  if I then transition to Win-Dn, that is a counter move.
        
        #
        #  TIE TIE TIE TIE TIE TIE case
        #  .... was at TIE-STAY
        #  HP  R R                   S (S)   AI thinks HP will stay, and UP.
        #  AI  R                     S (R)   HP should S->P DN to counter
        #  TIE  (UP)
        #  HP  R P                   S (R)   AI thinks HP will stay
        #  AI  R                     S (P)    and downgrades.  HP should S->S (stay)
        #  TIE  (DN)
        #  HP  R S                   S (P)   AI thinks HP will stay
        #  AI  R                     S (S)    and downgrades.  HP should UP

        ###  DSURPS thinking
        #
        #  STAY  (HP only cares about own last move)
        #  HP  R R                   R (S)    the counter is D
        #  AI  X                     P (P)
        #  if (hghst_probs[ig] == 0) and (hghst_probs[ig+1] == 1):  (in case of DSURPS

        hghst_prob_actions = prob_mvs[wtl].argmax(axis=0)
        #print("BEFORE.....................   repeats")

        
        # repeats = _N.array(misc.repeated_array_entry(hghst_prob_actions))
        # #   1 1 1 2 2 1 1 1 1 1 1 1
        # #print(repeats)        #  [3, 2, 10, 2, 12, 9]
        # blips = _N.where(repeats < 2)[0] # blips [1, 3]
        # for ib in blips[0:-1]:
        #     #  sum(repeats[0:1]) = 3
        #     #  repeats[0:2]            
        #     fr = _N.sum(repeats[0:ib]) 
        #     to = _N.sum(repeats[0:ib+1])
        #     #print("%(f)d  %(t)d" % {"f" : fr, "t" : to})
        #     #print(hghst_prob_actions[fr:to])
        #     #print(hghst_prob_actions[fr-1])
        #     hghst_prob_actions[fr:to] = hghst_prob_actions[fr-1]

        # #repeats = _N.array(misc.repeated_array_entry(hghst_prob_actions))
        # #print(repeats)
        
        maxp_chg_times = _N.where(_N.diff(hghst_prob_actions) != 0)[0]
        maxp_chg_times_wtl.extend(maxp_chg_times)
        
        # maxp_chg_time_intvs = _N.diff(maxp_chg_times)
        # one_bf_changes = maxp_chg_times[_N.where(maxp_chg_time_intvs < 2)[0]]
        # for ibf in one_bf_changes:
        #     hghst_probs[ibf+1:ibf+4] = hghst_probs[ibf]
        
        for ig in range(5, TO-5-5):
            ############################################
            #if (hghst_prob_actions[ig] == 1) and (hghst_prob_actions[ig+1] == 0):
            if (hghst_prob_actions[ig] == 1) and (hghst_prob_actions[ig+1] == 0):
                #  went from STAY to DN  (counter)
                cntr += 1
            elif (hghst_prob_actions[ig] == 1) and (hghst_prob_actions[ig+1] == 2):
                #  went from STAY to UP  (not counter)
                n_cntr += 1
            ############################################                    
            if (hghst_prob_actions[ig] == 2) and (hghst_prob_actions[ig+1] == 1):
                #  went from UP to STAY
                cntr += 1
            elif (hghst_prob_actions[ig] == 2) and (hghst_prob_actions[ig+1] == 0):
                #  went from UP to DN
                n_cntr += 1
            ############################################                    
            if (hghst_prob_actions[ig] == 0) and (hghst_prob_actions[ig+1] == 2):
                #  went from DN to UP
                cntr += 1
            elif (hghst_prob_actions[ig] == 0) and (hghst_prob_actions[ig+1] == 1):
                #  went from DN to STAY
                n_cntr += 1
                    
    l = _N.unique(_N.array(maxp_chg_times_wtl)).tolist()
    i = len(l)-2
    while i > 0:
        if l[i+1] - l[i] == 1:
            l.pop(i+1)

        i -= 1
    maxs = _N.array(l)+2
    return cntr, n_cntr

def cntrmvs_RPSWTL(prob_mvs, TO):
    """
    count number of counter and non-counter moves (as defined by mental model)
    """
    cntr = 0
    n_cntr = 0
    maxp_chg_times_wtl = []

    for wtl in [0, 2]:
        #  A counter move.
        #  If my most probably move now is UP.  The next time same condition
        #  happens, AI will
        #  WIN    highsest action is 0.  changes to 1  (non-counter is 2)
        #  R      R   S     non-counter  P
        #  S      P  (P)
        #      higest action is 1, counter is 2.  non-counte is 0
        #  S      S  (P)
        #  P      R  (R)
        #
        #  P      R
        #  R      P
        #
        hghst_prob_actions = prob_mvs[wtl].argmax(axis=0)
        maxp_chg_times = _N.where(_N.diff(hghst_prob_actions) != 0)[0]
        maxp_chg_times_wtl.extend(maxp_chg_times)
        
        # maxp_chg_time_intvs = _N.diff(maxp_chg_times)
        # one_bf_changes = maxp_chg_times[_N.where(maxp_chg_time_intvs < 2)[0]]
        # for ibf in one_bf_changes:
        #     hghst_probs[ibf+1:ibf+4] = hghst_probs[ibf]
        

        for ig in range(5, TO-5-5):
            ############################################
            if (hghst_prob_actions[ig] == 0) and (hghst_prob_actions[ig+1] == 1):
                #  went from R to DN  (counter)
                cntr += 1
            elif (hghst_prob_actions[ig] == 0) and (hghst_prob_actions[ig+1] == 2):
                #  went from STAY to UP  (not counter)
                n_cntr += 1
            ############################################                    
            if (hghst_prob_actions[ig] == 1) and (hghst_prob_actions[ig+1] == 2):
                #  went from UP to STAY
                cntr += 1
            elif (hghst_prob_actions[ig] == 1) and (hghst_prob_actions[ig+1] == 0):
                #  went from UP to DN
                n_cntr += 1
            ############################################                    
            if (hghst_prob_actions[ig] == 2) and (hghst_prob_actions[ig+1] == 0):
                #  went from DN to UP
                cntr += 1
            elif (hghst_prob_actions[ig] == 2) and (hghst_prob_actions[ig+1] == 1):
                #  went from DN to STAY
                n_cntr += 1
                    
    l = _N.unique(_N.array(maxp_chg_times_wtl)).tolist()
    i = len(l)-2
    while i > 0:
        if l[i+1] - l[i] == 1:
            l.pop(i+1)

        i -= 1
    maxs = _N.array(l)+2
    return cntr, n_cntr

def cntrmvs_DSURPS(prob_mvs, TO):
    """
    count number of counter and non-counter moves (as defined by mental model)
    """
    cntr = 0
    n_cntr = 0
    maxp_chg_times_rps = []

    for rps in [0, 1, 2]:
        #  A counter move.
        #  If my most probably move now is UP.  The next time same condition
        #  happens, AI will
        #  (HP only cares about own last move)        
        ###  DSURPS thinking
        #  CONDITION: R
        ###
        #  DN
        #  HP  R S                   S (P)    the counter is U  (not counter is going to S)
        #  AI  X                     R (R)
        #  STAY  (HP only cares about own last move)
        #  HP  R R                   R (S)    the counter is D  (not counter is U)
        #  AI  X                     P (P)
        ###
        #  UP
        #  HP  R P                   P (R)    the counter is S  (not counter is D)
        #  AI  X                     S (S)

        ###  DSURPS thinking
        #  CONDITION: S
        ###
        #  DN
        #  HP  R S                   S (P)    the counter is U
        #  AI  X                     R (R)
        #  STAY
        #  HP  S S                   S (P)    the counter is D
        #  AI  X                     R (R)
        ###
        #  UP
        #  HP  R P                   P (R)    the counter is S
        #  AI  X                     S (S)
        
        
        #  if (hghst_probs[ig] == 0) and (hghst_probs[ig+1] == 1):  (in case of DSURPS

        hghst_prob_actions = prob_mvs[rps].argmax(axis=0)  
        maxp_chg_times = _N.where(_N.diff(hghst_prob_actions) != 0)[0]
        maxp_chg_times_rps.extend(maxp_chg_times)
        
        # maxp_chg_time_intvs = _N.diff(maxp_chg_times)
        # one_bf_changes = maxp_chg_times[_N.where(maxp_chg_time_intvs < 2)[0]]
        # for ibf in one_bf_changes:
        #     hghst_probs[ibf+1:ibf+4] = hghst_probs[ibf]
        
        for ig in range(5, TO-5-5):
            if True:
                ############################################
                if (hghst_prob_actions[ig] == 0) and (hghst_prob_actions[ig+1] == 2):
                    #  went from DN to UP
                    cntr += 1
                elif (hghst_prob_actions[ig] == 0) and (hghst_prob_actions[ig+1] == 1):
                    #  went from DN to ST
                    n_cntr += 1
                ############################################                    
                if (hghst_prob_actions[ig] == 1) and (hghst_prob_actions[ig+1] == 0):
                    #  went from UP to STAY
                    cntr += 1
                elif (hghst_prob_actions[ig] == 1) and (hghst_prob_actions[ig+1] == 2):
                    #  went from UP to DN
                    n_cntr += 1
                ############################################                    
                if (hghst_prob_actions[ig] == 2) and (hghst_prob_actions[ig+1] == 1):
                    #  went from DN to UP
                    cntr += 1
                elif (hghst_prob_actions[ig] == 2) and (hghst_prob_actions[ig+1] == 0):
                    #  went from DN to STAY
                    n_cntr += 1
                    
    l = _N.unique(_N.array(maxp_chg_times_rps)).tolist()
    i = len(l)-2
    while i > 0:
        if l[i+1] - l[i] == 1:
            l.pop(i+1)

        i -= 1
    maxs = _N.array(l)+2
    return cntr, n_cntr

def cntrmvs_DSUAIRPS(prob_mvs, TO):
    """
    count number of counter and non-counter moves (as defined by mental model)
    """
    cntr = 0
    n_cntr = 0
    maxp_chg_times_rps = []

    for airps in [0, 1, 2]:
        #  A counter move.
        #  If my most probably move now is UP.  The next time same condition
        #  happens, AI will
        #  (HP only cares about own last move)        
        ###  DSURPS thinking
        #  CONDITION: R
        ###
        #  DN
        #  HP  R S                   S (P)    the counter is U  (not counter is going to S)
        #  AI  X                     R (R)
        #  STAY  (HP only cares about own last move)
        #  HP  R R                   R (S)    the counter is D  (not counter is U)
        #  AI  X                     P (P)
        ###
        #  UP
        #  HP  R P                   P (R)    the counter is S  (not counter is D)
        #  AI  X                     S (S)

        ###  DSURPS thinking
        #  CONDITION: S
        ###
        #  DN
        #  HP  R S                   S (P)    the counter is U
        #  AI  X                     R (R)
        #  STAY
        #  HP  S S                   S (P)    the counter is D
        #  AI  X                     R (R)
        ###
        #  UP
        #  HP  R P                   P (R)    the counter is S
        #  AI  X                     S (S)
        
        
        #  if (hghst_probs[ig] == 0) and (hghst_probs[ig+1] == 1):  (in case of DSURPS

        hghst_prob_actions = prob_mvs[airps].argmax(axis=0)  
        maxp_chg_times = _N.where(_N.diff(hghst_prob_actions) != 0)[0]
        maxp_chg_times_rps.extend(maxp_chg_times)
        
        # maxp_chg_time_intvs = _N.diff(maxp_chg_times)
        # one_bf_changes = maxp_chg_times[_N.where(maxp_chg_time_intvs < 2)[0]]
        # for ibf in one_bf_changes:
        #     hghst_probs[ibf+1:ibf+4] = hghst_probs[ibf]
        
        for ig in range(5, TO-5-5):
            if True:
                ############################################
                if (hghst_prob_actions[ig] == 0) and (hghst_prob_actions[ig+1] == 2):
                    #  went from DN to UP
                    cntr += 1
                elif (hghst_prob_actions[ig] == 0) and (hghst_prob_actions[ig+1] == 1):
                    #  went from DN to ST
                    n_cntr += 1
                ############################################                    
                if (hghst_prob_actions[ig] == 1) and (hghst_prob_actions[ig+1] == 0):
                    #  went from UP to STAY
                    cntr += 1
                elif (hghst_prob_actions[ig] == 1) and (hghst_prob_actions[ig+1] == 2):
                    #  went from UP to DN
                    n_cntr += 1
                ############################################                    
                if (hghst_prob_actions[ig] == 2) and (hghst_prob_actions[ig+1] == 1):
                    #  went from DN to UP
                    cntr += 1
                elif (hghst_prob_actions[ig] == 2) and (hghst_prob_actions[ig+1] == 0):
                    #  went from DN to STAY
                    n_cntr += 1
                    
    l = _N.unique(_N.array(maxp_chg_times_rps)).tolist()
    i = len(l)-2
    while i > 0:
        if l[i+1] - l[i] == 1:
            l.pop(i+1)

        i -= 1
    maxs = _N.array(l)+2
    return cntr, n_cntr


#  p(UP | W)
#  p(UP | WW)

def wtl_after_lcb(_hnd_dat, TO, pid,
                  win_aft_L, tie_aft_L, los_aft_L,
                  win_aft_C, tie_aft_C, los_aft_C,
                  win_aft_B, tie_aft_B, los_aft_B,
                  win_aft_D, tie_aft_D, los_aft_D,
                  win_aft_S, tie_aft_S, los_aft_S,
                  win_aft_U, tie_aft_U, los_aft_U):

    Ls = _N.where((((_hnd_dat[0:TO-2, 1] == 1) & (_hnd_dat[1:TO-1, 0] == 2)) |
                   ((_hnd_dat[0:TO-2, 1] == 2) & (_hnd_dat[1:TO-1, 0] == 3)) |
                   ((_hnd_dat[0:TO-2, 1] == 3) & (_hnd_dat[1:TO-1, 0] == 1))))[0]
    Cs = _N.where((_hnd_dat[0:TO-2, 1] == _hnd_dat[1:TO-1, 0]))[0]
    Bs = _N.where((((_hnd_dat[0:TO-2, 1] == 1) & (_hnd_dat[1:TO-1, 0] == 3)) |
                   ((_hnd_dat[0:TO-2, 1] == 2) & (_hnd_dat[1:TO-1, 0] == 1)) |
                   ((_hnd_dat[0:TO-2, 1] == 3) & (_hnd_dat[1:TO-1, 0] == 2))))[0]

    Ds = _N.where((((_hnd_dat[0:TO-2, 0] == 1) & (_hnd_dat[1:TO-1, 0] == 2)) |
                   ((_hnd_dat[0:TO-2, 0] == 2) & (_hnd_dat[1:TO-1, 0] == 3)) |
                   ((_hnd_dat[0:TO-2, 0] == 3) & (_hnd_dat[1:TO-1, 0] == 1))))[0]
    Ss = _N.where((_hnd_dat[0:TO-2, 0] == _hnd_dat[1:TO-1, 0]))[0]
    Us = _N.where((((_hnd_dat[0:TO-2, 0] == 1) & (_hnd_dat[1:TO-1, 0] == 3)) |
                   ((_hnd_dat[0:TO-2, 0] == 2) & (_hnd_dat[1:TO-1, 0] == 1)) |
                   ((_hnd_dat[0:TO-2, 0] == 3) & (_hnd_dat[1:TO-1, 0] == 2))))[0]
    
    win_aft_Ls = _N.where(_hnd_dat[Ls + 1, 2]  == 1)[0]
    tie_aft_Ls = _N.where(_hnd_dat[Ls + 1, 2]  == 0)[0]
    los_aft_Ls = _N.where(_hnd_dat[Ls + 1, 2]  == -1)[0]    
    win_aft_Cs = _N.where(_hnd_dat[Cs + 1, 2]  == 1)[0]
    tie_aft_Cs = _N.where(_hnd_dat[Cs + 1, 2]  == 0)[0]
    los_aft_Cs = _N.where(_hnd_dat[Cs + 1, 2]  == -1)[0]    
    win_aft_Bs = _N.where(_hnd_dat[Bs + 1, 2]  == 1)[0]
    tie_aft_Bs = _N.where(_hnd_dat[Bs + 1, 2]  == 0)[0]
    los_aft_Bs = _N.where(_hnd_dat[Bs + 1, 2]  == -1)[0]

    win_aft_L[pid-1] = len(win_aft_Ls) / len(Ls)
    tie_aft_L[pid-1] = len(tie_aft_Ls) / len(Ls)
    los_aft_L[pid-1] = len(los_aft_Ls) / len(Ls)    
    win_aft_C[pid-1] = len(win_aft_Cs) / len(Cs)
    tie_aft_C[pid-1] = len(tie_aft_Cs) / len(Cs)
    los_aft_C[pid-1] = len(los_aft_Cs) / len(Cs)    
    win_aft_B[pid-1] = len(win_aft_Bs) / len(Bs)
    tie_aft_B[pid-1] = len(tie_aft_Bs) / len(Bs)
    los_aft_B[pid-1] = len(los_aft_Bs) / len(Bs)

    win_aft_Ds = _N.where(_hnd_dat[Ls + 1, 2]  == 1)[0]
    tie_aft_Ds = _N.where(_hnd_dat[Ls + 1, 2]  == 0)[0]
    los_aft_Ds = _N.where(_hnd_dat[Ls + 1, 2]  == -1)[0]    
    win_aft_Ss = _N.where(_hnd_dat[Cs + 1, 2]  == 1)[0]
    tie_aft_Ss = _N.where(_hnd_dat[Cs + 1, 2]  == 0)[0]
    los_aft_Ss = _N.where(_hnd_dat[Cs + 1, 2]  == -1)[0]    
    win_aft_Us = _N.where(_hnd_dat[Bs + 1, 2]  == 1)[0]
    tie_aft_Us = _N.where(_hnd_dat[Bs + 1, 2]  == 0)[0]
    los_aft_Us = _N.where(_hnd_dat[Bs + 1, 2]  == -1)[0]

    win_aft_D[pid-1] = len(win_aft_Ds) / len(Ds)
    tie_aft_D[pid-1] = len(tie_aft_Ds) / len(Ds)
    los_aft_D[pid-1] = len(los_aft_Ds) / len(Ds)    
    win_aft_S[pid-1] = len(win_aft_Ss) / len(Ss)
    tie_aft_S[pid-1] = len(tie_aft_Ss) / len(Ss)
    los_aft_S[pid-1] = len(los_aft_Ss) / len(Ss)    
    win_aft_U[pid-1] = len(win_aft_Us) / len(Us)
    tie_aft_U[pid-1] = len(tie_aft_Us) / len(Us)
    los_aft_U[pid-1] = len(los_aft_Us) / len(Us)    
    

def wtl_after_wtl(_hnd_dat, TO, pid,
                  win_aft_win, tie_aft_win, los_aft_win,
                  win_aft_tie, tie_aft_tie, los_aft_tie,
                  win_aft_los, tie_aft_los, los_aft_los,
                  L_aft_air, C_aft_air, B_aft_air,
                  L_aft_ais, C_aft_ais, B_aft_ais,
                  L_aft_aip, C_aft_aip, B_aft_aip,
                  win_aft_AIR, tie_aft_AIR, los_aft_AIR,
                  win_aft_AIS, tie_aft_AIS, los_aft_AIS,
                  win_aft_AIP, tie_aft_AIP, los_aft_AIP):
                  # L_aft_win, C_aft_win, B_aft_win,
                  # L_aft_tie, C_aft_tie, B_aft_tie,
                  # L_aft_los, C_aft_los, B_aft_los,
    
    
    ####
    wins = _N.where(_hnd_dat[0:TO-2, 2] == 1)[0]
    loses = _N.where(_hnd_dat[0:TO-2, 2] == -1)[0]
    ties = _N.where(_hnd_dat[0:TO-2, 2] == 0)[0]
    
    ww   = _N.where(_hnd_dat[wins+1, 2] == 1)[0]
    wt   = _N.where(_hnd_dat[wins+1, 2] == 0)[0]
    wl   = _N.where(_hnd_dat[wins+1, 2] == -1)[0]
    wr   = _N.where(_hnd_dat[wins+1, 0] == 1)[0]
    wp   = _N.where(_hnd_dat[wins+1, 0] == 2)[0]
    ws   = _N.where(_hnd_dat[wins+1, 0] == 3)[0]

    air = _N.where(_hnd_dat[0:TO-2, 1] == 1)[0]
    ais = _N.where(_hnd_dat[0:TO-2, 1] == 2)[0]
    aip = _N.where(_hnd_dat[0:TO-2, 1] == 3)[0]        
    
    win_aft_win[pid-1] = len(ww) / len(wins)
    tie_aft_win[pid-1] = len(wt) / len(wins)
    los_aft_win[pid-1] = len(wl) / len(wins)    
    #tie_aft_win[pid-1] = len(wt) / len(ties)
    #los_aft_win[pid-1] = len(wl) / len(loses)
    # R_aft_win[pid-1] = len(wr) / len(wins)
    # P_aft_win[pid-1] = len(wp) / len(wins)
    # S_aft_win[pid-1] = len(ws) / len(wins)
    wL = _N.where((_hnd_dat[0:TO-2, 2] == 1) & # choose BEAT last AI
                  (((_hnd_dat[0:TO-2, 1] == 1) & (_hnd_dat[1:TO-1, 0] == 2)) |
                   ((_hnd_dat[0:TO-2, 1] == 2) & (_hnd_dat[1:TO-1, 0] == 3)) |
                   ((_hnd_dat[0:TO-2, 1] == 3) & (_hnd_dat[1:TO-1, 0] == 1))))[0]
    wC = _N.where((_hnd_dat[0:TO-2, 2] == 1) & # choose COPY last AI
                  (_hnd_dat[0:TO-2, 1] == _hnd_dat[1:TO-1, 0]))[0]
    wB = _N.where((_hnd_dat[0:TO-2, 2] == 1) & # choose BEAT last AI
                  (((_hnd_dat[0:TO-2, 1] == 1) & (_hnd_dat[1:TO-1, 0] == 3)) |
                   ((_hnd_dat[0:TO-2, 1] == 2) & (_hnd_dat[1:TO-1, 0] == 1)) |
                   ((_hnd_dat[0:TO-2, 1] == 3) & (_hnd_dat[1:TO-1, 0] == 2))))[0]
    tL = _N.where((_hnd_dat[0:TO-2, 2] == 0) & # choose BEAT last AI
                  (((_hnd_dat[0:TO-2, 1] == 1) & (_hnd_dat[1:TO-1, 0] == 2)) |
                   ((_hnd_dat[0:TO-2, 1] == 2) & (_hnd_dat[1:TO-1, 0] == 3)) |
                   ((_hnd_dat[0:TO-2, 1] == 3) & (_hnd_dat[1:TO-1, 0] == 1))))[0]
    tC = _N.where((_hnd_dat[0:TO-2, 2] == 0) & # choose COPY last AI
                  (_hnd_dat[0:TO-2, 1] == _hnd_dat[1:TO-1, 0]))[0]
    tB = _N.where((_hnd_dat[0:TO-2, 2] == 0) & # choose BEAT last AI
                  (((_hnd_dat[0:TO-2, 1] == 1) & (_hnd_dat[1:TO-1, 0] == 3)) |
                   ((_hnd_dat[0:TO-2, 1] == 2) & (_hnd_dat[1:TO-1, 0] == 1)) |
                   ((_hnd_dat[0:TO-2, 1] == 3) & (_hnd_dat[1:TO-1, 0] == 2))))[0]
    lL = _N.where((_hnd_dat[0:TO-2, 2] == -1) & # choose BEAT last AI
                  (((_hnd_dat[0:TO-2, 1] == 1) & (_hnd_dat[1:TO-1, 0] == 2)) |
                   ((_hnd_dat[0:TO-2, 1] == 2) & (_hnd_dat[1:TO-1, 0] == 3)) |
                   ((_hnd_dat[0:TO-2, 1] == 3) & (_hnd_dat[1:TO-1, 0] == 1))))[0]
    lC = _N.where((_hnd_dat[0:TO-2, 2] == -1) & # choose COPY last AI
                  (_hnd_dat[0:TO-2, 1] == _hnd_dat[1:TO-1, 0]))[0]
    lB = _N.where((_hnd_dat[0:TO-2, 2] == -1) & # choose BEAT last AI
                  (((_hnd_dat[0:TO-2, 1] == 1) & (_hnd_dat[1:TO-1, 0] == 3)) |
                   ((_hnd_dat[0:TO-2, 1] == 2) & (_hnd_dat[1:TO-1, 0] == 1)) |
                   ((_hnd_dat[0:TO-2, 1] == 3) & (_hnd_dat[1:TO-1, 0] == 2))))[0]


    airL = _N.where((_hnd_dat[0:TO-2, 1] == 1) & # choose BEAT last AI
                    (((_hnd_dat[0:TO-2, 1] == 1) & (_hnd_dat[1:TO-1, 0] == 2)) |
                     ((_hnd_dat[0:TO-2, 1] == 2) & (_hnd_dat[1:TO-1, 0] == 3)) |
                     ((_hnd_dat[0:TO-2, 1] == 3) & (_hnd_dat[1:TO-1, 0] == 1))))[0]
    airC = _N.where((_hnd_dat[0:TO-2, 1] == 1) & # choose COPY last AI
                    (_hnd_dat[0:TO-2, 1] == _hnd_dat[1:TO-1, 0]))[0]
    airB = _N.where((_hnd_dat[0:TO-2, 1] == 1) & # choose BEAT last AI
                    (((_hnd_dat[0:TO-2, 1] == 1) & (_hnd_dat[1:TO-1, 0] == 3)) |
                     ((_hnd_dat[0:TO-2, 1] == 2) & (_hnd_dat[1:TO-1, 0] == 1)) |
                     ((_hnd_dat[0:TO-2, 1] == 3) & (_hnd_dat[1:TO-1, 0] == 2))))[0]
    aisL = _N.where((_hnd_dat[0:TO-2, 1] == 2) & # choose BEAT last AI
                    (((_hnd_dat[0:TO-2, 1] == 1) & (_hnd_dat[1:TO-1, 0] == 2)) |
                     ((_hnd_dat[0:TO-2, 1] == 2) & (_hnd_dat[1:TO-1, 0] == 3)) |
                     ((_hnd_dat[0:TO-2, 1] == 3) & (_hnd_dat[1:TO-1, 0] == 1))))[0]
    aisC = _N.where((_hnd_dat[0:TO-2, 1] == 2) & # choose COPY last AI
                    (_hnd_dat[0:TO-2, 1] == _hnd_dat[1:TO-1, 0]))[0]
    aisB = _N.where((_hnd_dat[0:TO-2, 1] == 2) & # choose BEAT last AI
                    (((_hnd_dat[0:TO-2, 1] == 1) & (_hnd_dat[1:TO-1, 0] == 3)) |
                     ((_hnd_dat[0:TO-2, 1] == 2) & (_hnd_dat[1:TO-1, 0] == 1)) |
                     ((_hnd_dat[0:TO-2, 1] == 3) & (_hnd_dat[1:TO-1, 0] == 2))))[0]
    aipL = _N.where((_hnd_dat[0:TO-2, 1] == 3) & # choose BEAT last AI
                    (((_hnd_dat[0:TO-2, 1] == 1) & (_hnd_dat[1:TO-1, 0] == 2)) |
                     ((_hnd_dat[0:TO-2, 1] == 2) & (_hnd_dat[1:TO-1, 0] == 3)) |
                     ((_hnd_dat[0:TO-2, 1] == 3) & (_hnd_dat[1:TO-1, 0] == 1))))[0]
    aipC = _N.where((_hnd_dat[0:TO-2, 1] == 3) & # choose COPY last AI
                    (_hnd_dat[0:TO-2, 1] == _hnd_dat[1:TO-1, 0]))[0]
    aipB = _N.where((_hnd_dat[0:TO-2, 1] == 3) & # choose BEAT last AI
                    (((_hnd_dat[0:TO-2, 1] == 1) & (_hnd_dat[1:TO-1, 0] == 3)) |
                     ((_hnd_dat[0:TO-2, 1] == 2) & (_hnd_dat[1:TO-1, 0] == 1)) |
                     ((_hnd_dat[0:TO-2, 1] == 3) & (_hnd_dat[1:TO-1, 0] == 2))))[0]
    L_aft_air[pid-1] = len(airL) / len(air)
    C_aft_air[pid-1] = len(airC) / len(air)
    B_aft_air[pid-1] = len(airB) / len(air)
    L_aft_ais[pid-1] = len(aisL) / len(ais)
    C_aft_ais[pid-1] = len(aisC) / len(ais)
    B_aft_ais[pid-1] = len(aisB) / len(ais)
    L_aft_aip[pid-1] = len(aipL) / len(aip)
    C_aft_aip[pid-1] = len(aipC) / len(aip)
    B_aft_aip[pid-1] = len(aipB) / len(aip)
                    
                  
    ####    
    lw   = _N.where(_hnd_dat[loses+1, 2] == 1)[0]
    lt   = _N.where(_hnd_dat[loses+1, 2] == 0)[0]
    ll   = _N.where(_hnd_dat[loses+1, 2] == -1)[0]
    lr   = _N.where(_hnd_dat[loses+1, 0] == 1)[0]
    lp   = _N.where(_hnd_dat[loses+1, 0] == 2)[0]
    ls   = _N.where(_hnd_dat[loses+1, 0] == 3)[0]        

    win_aft_los[pid-1] = len(lw) / len(loses)
    tie_aft_los[pid-1] = len(lt) / len(loses)
    los_aft_los[pid-1] = len(ll) / len(loses)    
    #win_aft_los[pid-1] = len(lw) / len(wins)    
    #tie_aft_los[pid-1] = len(lt) / len(ties)

    # R_aft_los[pid-1] = len(lr) / len(loses)
    # P_aft_los[pid-1] = len(lp) / len(loses)
    # S_aft_los[pid-1] = len(ls) / len(loses)    

    ####
    tw   = _N.where(_hnd_dat[ties+1, 2] == 1)[0]
    tt   = _N.where(_hnd_dat[ties+1, 2] == 0)[0]
    tl   = _N.where(_hnd_dat[ties+1, 2] == -1)[0]
    tr   = _N.where(_hnd_dat[ties+1, 0] == 1)[0]
    tp   = _N.where(_hnd_dat[ties+1, 0] == 2)[0]
    ts   = _N.where(_hnd_dat[ties+1, 0] == 3)[0]
    #nTies[pid-1] = len(ties)    

    win_aft_tie[pid-1] = len(tw) / len(ties)
    los_aft_tie[pid-1] = len(tl) / len(ties)
    tie_aft_tie[pid-1] = len(tt) / len(ties)    
    #win_aft_tie[pid-1] = len(tw) / len(wins)
    #los_aft_tie[pid-1] = len(tl) / len(loses)
    # R_aft_tie[pid-1] = len(tr) / len(ties)
    # P_aft_tie[pid-1] = len(tp) / len(ties)
    # S_aft_tie[pid-1] = len(ts) / len(ties)

    ######################
    AIRs = _N.where(_hnd_dat[0:TO-2, 1] == 1)[0]
    air_w   = _N.where(_hnd_dat[AIRs+1, 2] == 1)[0]
    air_t   = _N.where(_hnd_dat[AIRs+1, 2] == 0)[0]
    air_l   = _N.where(_hnd_dat[AIRs+1, 2] == -1)[0]
    win_aft_AIR[pid-1] = len(air_w) / len(AIRs)
    tie_aft_AIR[pid-1] = len(air_t) / len(AIRs)
    los_aft_AIR[pid-1] = len(air_l) / len(AIRs)
    
    AISs = _N.where(_hnd_dat[0:TO-2, 1] == 2)[0]
    ais_w   = _N.where(_hnd_dat[AISs+1, 2] == 1)[0]
    ais_t   = _N.where(_hnd_dat[AISs+1, 2] == 0)[0]
    ais_l   = _N.where(_hnd_dat[AISs+1, 2] == -1)[0]
    win_aft_AIS[pid-1] = len(ais_w) / len(AISs)
    tie_aft_AIS[pid-1] = len(ais_t) / len(AISs)
    los_aft_AIS[pid-1] = len(ais_l) / len(AISs)
    
    AIPs = _N.where(_hnd_dat[0:TO-2, 1] == 3)[0]
    aip_w   = _N.where(_hnd_dat[AIPs+1, 2] == 1)[0]
    aip_t   = _N.where(_hnd_dat[AIPs+1, 2] == 0)[0]
    aip_l   = _N.where(_hnd_dat[AIPs+1, 2] == -1)[0]
    win_aft_AIP[pid-1] = len(aip_w) / len(AIPs)
    tie_aft_AIP[pid-1] = len(aip_t) / len(AIPs)
    los_aft_AIP[pid-1] = len(aip_l) / len(AIPs)


    # L_aft_win[pid-1] = len(wL) / len(wins)
    # C_aft_win[pid-1] = len(wC) / len(wins)
    # B_aft_win[pid-1] = len(wB) / len(wins)
    # L_aft_tie[pid-1] = len(tL) / len(ties)
    # C_aft_tie[pid-1] = len(tC) / len(ties)
    # B_aft_tie[pid-1] = len(tB) / len(ties)
    # L_aft_los[pid-1] = len(lL) / len(loses)
    # C_aft_los[pid-1] = len(lC) / len(loses)
    # B_aft_los[pid-1] = len(lB) / len(loses)
                  

    
    
    
def resptime_aft_wtl(_hnd_dat, TO, pid, inp_meth, time_aft_win, time_aft_tie, time_aft_los):
    tm_t0 = 1
    tm_t1 = TO-1
    #tm_t0 = 1
    #tm_t1 = TO//2

    #  _hnd_dat[0, 3]    is time at which first move was made
    #  one back
        
    resp_tms             = _N.zeros(_hnd_dat.shape[0], dtype=_N.int32)
    resp_tms[1:]         = _N.diff(_hnd_dat[:, 3])

    n_used_inp_meths = len(_N.unique(inp_meth))

    mouseOffset = 0
    n_mouse = 0
    n_keys  = 0
    mouse_resp_t = 0
    key_resp_t   = 0    

    if n_used_inp_meths == 2:
        mouse_inp = _N.where(inp_meth == 0)[0]
        key_inp   = _N.where(inp_meth == 1)[0]
        if len(mouse_inp) > 1:
            mouse_resp_t = _N.mean(resp_tms[mouse_inp])
        else:
            mouse_resp_t = resp_tms[mouse_inp[0]]
        if len(key_inp) > 1:
            key_resp_t = _N.mean(resp_tms[key_inp])
        else:
            key_resp_t = resp_tms[key_inp[0]]
            
        key_resp_t   = _N.mean(resp_tms[key_inp])
        #print("%(nm)d)  %(m).1f    %(nk)d)  %(k).1f" % {"m" : mouse_resp_t, "k" : key_resp_t, "nm" : len(mouse_inp), "nk" : len(key_inp)})
        mouseOffset = mouse_resp_t - key_resp_t
        n_mouse = len(mouse_inp)
        n_keys = len(key_inp)

    resp_time_all        = resp_tms[tm_t0:tm_t1-1] - (1-inp_meth[tm_t0:tm_t1-1])*mouseOffset        
    time_all             = _N.mean(resp_time_all)
    
    winsS = _N.where(_hnd_dat[tm_t0:tm_t1-1, 2] == 1)[0]+tm_t0
    mn = _N.mean(resp_tms[winsS+1] - (1-inp_meth[winsS+1])*mouseOffset) / time_all
    sd = _N.std(resp_tms[winsS+1] - (1-inp_meth[winsS+1])*mouseOffset) / time_all    
    time_aft_win[pid-1]  = sd

    losesS = _N.where(_hnd_dat[tm_t0:tm_t1-1, 2] == -1)[0]+tm_t0
    mn = _N.mean(resp_tms[losesS+1] - (1-inp_meth[losesS+1])*mouseOffset) / time_all
    sd = _N.std(resp_tms[losesS+1] - (1-inp_meth[losesS+1])*mouseOffset) / time_all    
    
    time_aft_los[pid-1]  = sd

    tiesS = _N.where(_hnd_dat[tm_t0:tm_t1-1, 2] == 0)[0]+tm_t0
    mn = _N.mean(resp_tms[tiesS+1] - (1-inp_meth[tiesS+1])*mouseOffset) / time_all
    sd = _N.std(resp_tms[tiesS+1] - (1-inp_meth[tiesS+1])*mouseOffset) / time_all    
    
    time_aft_tie[pid-1]  = sd
    return n_mouse, n_keys, mouse_resp_t, key_resp_t, resp_time_all



def resptime_b4aft_wtl(_hnd_dat, TO, pid, inp_meth, time_b4aft_win_mn, time_b4aft_win_sd, time_b4aft_tie_mn, time_b4aft_tie_sd, time_b4aft_los_mn, time_b4aft_los_sd):
    #  I want to compare tie 
    tm_t0 = 2
    tm_t1 = TO-1
    #tm_t0 = 1
    #tm_t1 = TO//2

    #  _hnd_dat[0, 3]    is time at which first move was made
    #  one back
        
    resp_tms             = _N.zeros(_hnd_dat.shape[0], dtype=_N.int32)
    resp_tms[1:]         = _N.diff(_hnd_dat[:, 3])

    n_used_inp_meths = len(_N.unique(inp_meth))

    mouseOffset = 0
    n_mouse = 0
    n_keys  = 0
    mouse_resp_t = 0
    key_resp_t   = 0

    if n_used_inp_meths == 2:
        mouse_inp = _N.where(inp_meth == 0)[0]
        key_inp   = _N.where(inp_meth == 1)[0]
        if len(mouse_inp) > 1:
            mouse_resp_t = _N.mean(resp_tms[mouse_inp])
        else:
            mouse_resp_t = resp_tms[mouse_inp[0]]
        if len(key_inp) > 1:
            key_resp_t = _N.mean(resp_tms[key_inp])
        else:
            key_resp_t = resp_tms[key_inp[0]]
            
        key_resp_t   = _N.mean(resp_tms[key_inp])
        #print("%(nm)d)  %(m).1f    %(nk)d)  %(k).1f" % {"m" : mouse_resp_t, "k" : key_resp_t, "nm" : len(mouse_inp), "nk" : len(key_inp)})
        mouseOffset = mouse_resp_t - key_resp_t
        n_mouse = len(mouse_inp)
        n_keys = len(key_inp)
    
    allS = _N.where(_hnd_dat[tm_t0:tm_t1-1, 2] != 2)[0]+tm_t0    # all moves
    tAftBefA  = _N.mean(resp_tms[allS+1] / resp_tms[allS])
    tAftBefS  = _N.std(resp_tms[allS+1] / resp_tms[allS])    
    
    winsS = _N.where(_hnd_dat[tm_t0:tm_t1-1, 2] == 1)[0]+tm_t0
    tAft  = resp_tms[winsS+1] 
    tBef  = resp_tms[winsS] 

    time_b4aft_win_mn[pid-1] = _N.mean(tAft / tBef) / tAftBefA
    if _N.isnan(time_b4aft_win_mn[pid-1]):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print((tAft / tBef))
        print(_N.mean(tAft / tBef))
    time_b4aft_win_sd[pid-1] = _N.std(tAft / tBef) / tAftBefS   
    

    losesS = _N.where(_hnd_dat[tm_t0:tm_t1-1, 2] == -1)[0]+tm_t0
    tAft  = resp_tms[losesS+1]
    tBef  = resp_tms[losesS]
    time_b4aft_los_mn[pid-1] = _N.mean(tAft / tBef) / tAftBefA
    if _N.isnan(time_b4aft_los_mn[pid-1]):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print((tAft / tBef))
        print(_N.mean(tAft / tBef))
    
    time_b4aft_los_sd[pid-1] = _N.std(tAft / tBef) / tAftBefS   

    tiesS = _N.where(_hnd_dat[tm_t0:tm_t1-1, 2] == 0)[0]+tm_t0
    tAft  = resp_tms[tiesS+1]
    tBef  = resp_tms[tiesS]
    time_b4aft_tie_mn[pid-1] = _N.mean(tAft / tBef) / tAftBefA
    if _N.isnan(time_b4aft_tie_mn[pid-1]):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print((tAft / tBef))
        print(_N.mean(tAft / tBef))
    
    time_b4aft_tie_sd[pid-1] = _N.std(tAft / tBef) / tAftBefS 

def action_result(_hnd_dat, TO, pid, u_or_d_res, dn_res, up_res, stay_res, u_or_d_tie, stay_tie):
    ###
    #  RSP   1 (R)   3  (P)  lose
    u_or_d = _N.where(_hnd_dat[0:TO-2, 0] != _hnd_dat[1:TO-1, 0])[0]
    #  R->S   S->P
    dn_only = _N.where(((_hnd_dat[0:TO-2, 0] == 1) & (_hnd_dat[1:TO-1, 0] == 2)) |
                       ((_hnd_dat[0:TO-2, 0] == 2) & (_hnd_dat[1:TO-1, 0] == 3)) |
                       ((_hnd_dat[0:TO-2, 0] == 3) & (_hnd_dat[1:TO-1, 0] == 1)))[0]
    up_only = _N.where(((_hnd_dat[0:TO-2, 0] == 1) & (_hnd_dat[1:TO-1, 0] == 3)) |
                       ((_hnd_dat[0:TO-2, 0] == 2) & (_hnd_dat[1:TO-1, 0] == 1)) |
                       ((_hnd_dat[0:TO-2, 0] == 3) & (_hnd_dat[1:TO-1, 0] == 2)))[0]
    st_only = _N.where(_hnd_dat[0:TO-2, 0] == _hnd_dat[1:TO-1, 0])[0]
    up_isi = _N.diff(up_only)
    dn_isi = _N.diff(dn_only)
    st_isi = _N.diff(st_only)
    Rs     = _N.where(_hnd_dat[:, 0] == 1)[0]
    Ss     = _N.where(_hnd_dat[:, 0] == 2)[0]
    Ps     = _N.where(_hnd_dat[:, 0] == 3)[0]
    R_isi  = _N.diff(Rs)
    S_isi  = _N.diff(Ss)
    P_isi  = _N.diff(Ps)

    # up_cvs[pid-1] = _N.std(up_isi) / _N.mean(up_isi)
    # dn_cvs[pid-1] = _N.std(dn_isi) / _N.mean(dn_isi)
    # st_cvs[pid-1] = _N.std(st_isi) / _N.mean(st_isi)
    # R_cvs[pid-1] = _N.std(R_isi) / _N.mean(R_isi)
    # S_cvs[pid-1] = _N.std(S_isi) / _N.mean(S_isi)
    # P_cvs[pid-1] = _N.std(P_isi) / _N.mean(P_isi)        

    #
    u_or_d_res[pid-1] = _N.sum(_hnd_dat[u_or_d+1, 2])
    ties_after_u_or_d = _N.where(_hnd_dat[u_or_d+1, 2] == 0)[0]
    u_or_d_tie[pid-1] = len(ties_after_u_or_d)
    up_res[pid-1] = _N.sum(_hnd_dat[up_only+1, 2])
    dn_res[pid-1] = _N.sum(_hnd_dat[dn_only+1, 2])    
    stay   = _N.where(_hnd_dat[0:TO-2, 0] == _hnd_dat[1:TO-1, 0])[0]    
    stay_res[pid-1] = _N.sum(_hnd_dat[stay+1, 2])
    ties_after_stay = _N.where(_hnd_dat[stay+1, 2] == 0)[0]
    stay_tie[pid-1] = len(ties_after_stay)
    

def rulechange(_hnd_dat, signal_5_95, pfrm_change36, pfrm_change69, pfrm_change912, imax_imin_pfrm36, imax_imin_pfrm69, imax_imin_pfrm912, all_avgs, SHUFFLES, t0, t1, maxs, cut, pid):
    inds =_N.arange(_hnd_dat.shape[0])
    
    for sh in range(SHUFFLES+1):
        if sh > 0:
            _N.random.shuffle(inds)
        hnd_dat = _hnd_dat[inds]

        avgs = _N.empty((len(maxs)-2*cut, t1-t0))
        #print("len(maxs)  %d" % len(maxs))
        #print(maxs)

        for im in range(cut, len(maxs)-cut):
            #print(hnd_dat[maxs[im]+t0:maxs[im]+t1, 2].shape)
            #print("%(1)d %(2)d" % {"1" : maxs[im]+t0, "2" : maxs[im]+t1})
            st = 0
            en = t1-t0
            if maxs[im] + t0 < 0:   #  just don't use this one
                print("DON'T USE THIS ONE")
                avgs[im-1, :] = 0
            else:
                try:
                    avgs[im-1, :] = hnd_dat[maxs[im]+t0:maxs[im]+t1, 2]
                except ValueError:
                    print("*****  %(1)d  %(2)d" % {"1" : maxs[im]+t0, "2" : maxs[im]+t1})
                    print(avgs[im-1, :].shape)
                    print(hnd_dat[maxs[im]+t0:maxs[im]+t1, 2])
                    

        all_avgs[pid-1, sh] = _N.mean(avgs, axis=0)
        #fig.add_subplot(5, 5, pid)
        #_plt.plot(_N.mean(avgs, axis=0))

    srtd   = _N.sort(all_avgs[pid-1, 1:], axis=0)
    signal_5_95[pid-1, 1] = srtd[int(0.05*SHUFFLES)]
    signal_5_95[pid-1, 2] = srtd[int(0.95*SHUFFLES)]
    signal_5_95[pid-1, 0] = all_avgs[pid-1, 0]
    signal_5_95[pid-1, 3] = (signal_5_95[pid-1, 0] - signal_5_95[pid-1, 1]) / (signal_5_95[pid-1, 2] - signal_5_95[pid-1, 1])

    #pfrm_change36[pid-1] = _N.max(signal_5_95[pid-1, 0, 3:6]) - _N.min(signal_5_95[pid-1, 0, 3:6])

    sInds = _N.argsort(signal_5_95[pid-1, 0, 3:6])
    #sInds = _N.argsort(signal_5_95[pid-1, 0, 1:5])
    if sInds[2] - sInds[0] > 0:
        m36 = 1
    else:
        m36 = -1
    sInds = _N.argsort(signal_5_95[pid-1, 0, 6:9])
    #sInds = _N.argsort(signal_5_95[pid-1, 0, 5:10])
    if sInds[2] - sInds[0] > 0:
        m69 = 1
    else:
        m69 = -1
    sInds = _N.argsort(signal_5_95[pid-1, 0, 9:12])
    #sInds = _N.argsort(signal_5_95[pid-1, 0, 10:15])
    if sInds[2] - sInds[0] > 0:
        m912 = 1
    else:
        m912 = -1

    imax36 = _N.argmax(signal_5_95[pid-1, 0, 3:6])+3
    imin36 = _N.argmin(signal_5_95[pid-1, 0, 3:6])+3
    imax69 = _N.argmax(signal_5_95[pid-1, 0, 6:9])+6
    imin69 = _N.argmin(signal_5_95[pid-1, 0, 6:9])+6    
    imax912= _N.argmax(signal_5_95[pid-1, 0, 9:12])+9
    imin912= _N.argmin(signal_5_95[pid-1, 0, 9:12])+9    

    # imax36 = _N.argmax(signal_5_95[pid-1, 0, 1:5])+1
    # imin36 = _N.argmin(signal_5_95[pid-1, 0, 1:5])+1
    # imax69 = _N.argmax(signal_5_95[pid-1, 0, 5:10])+5
    # imin69 = _N.argmin(signal_5_95[pid-1, 0, 5:10])+5    
    # imax912= _N.argmax(signal_5_95[pid-1, 0, 10:13])+10
    # imin912= _N.argmin(signal_5_95[pid-1, 0, 10:13])+10   
    
    imax_imin_pfrm36[pid-1, 0] = imin36
    imax_imin_pfrm36[pid-1, 1] = imax36
    imax_imin_pfrm69[pid-1, 0] = imin69
    imax_imin_pfrm69[pid-1, 1] = imax69
    imax_imin_pfrm912[pid-1, 0]= imin912
    imax_imin_pfrm912[pid-1, 1]= imax912
    
    pfrm_change36[pid-1] = signal_5_95[pid-1, 0, imax36] - signal_5_95[pid-1, 0, imin36]
    pfrm_change69[pid-1] = signal_5_95[pid-1, 0, imax69] - signal_5_95[pid-1, 0, imin69]
    #pfrm_change69[pid-1] = _N.max(signal_5_95[pid-1, 0, 6:9]) - _N.min(signal_5_95[pid-1, 0, 6:9])
    pfrm_change69[pid-1] = _N.mean(signal_5_95[pid-1, 0, 7:11]) - _N.mean(signal_5_95[pid-1, 0, 3:7])

    pfrm_change912[pid-1]= signal_5_95[pid-1, 0, imax912] - signal_5_95[pid-1, 0, imin912]

    #fig.add_subplot(6, 6, pid)

    # _plt.title(netwins[pid-1] )
    # _plt.plot(ts, signal_5_95[pid-1, 0], marker=".", ms=10)
    # _plt.plot(ts, signal_5_95[pid-1, 1])
    # _plt.plot(ts, signal_5_95[pid-1, 2])
    # _plt.axvline(x=-0.5, ls="--")

    be = _N.where(signal_5_95[pid-1, 0] < signal_5_95[pid-1, 1])[0]
    # if len(be) > 0:
    #     belows.extend(be)
    # ab = _N.where(signal_5_95[pid-1, 0] > signal_5_95[pid-1, 2])[0]        
    # if len(ab) > 0:
    #     aboves.extend(ab)


def perceptron_features(all_AI_weights, all_AI_preds, partIDs):
    strt = 30
    aAw = all_AI_weights[:, strt:]   #  len(partIDs) x (T+1) x 3 x 3 x 2
    ################

    #  axis=1 is over time
    #  Instead of std, how about 
    mn_aAw1                 = _N.mean(aAw, axis=1)
    sd_aAw1                 = _N.std(aAw, axis=1)
    aift1 = _N.mean(_N.diff(mn_aAw1, axis=3), axis=1) # poor reproduce
    aift2 = _N.std(_N.diff( mn_aAw1, axis=3), axis=1) # poor reproduce  
    aift3 = _N.mean(_N.diff(mn_aAw1, axis=3), axis=2) # poor reproduce
    aift4 = _N.std(_N.diff( mn_aAw1, axis=3), axis=2) # poor reproduce
    aift5 = _N.mean(_N.diff(sd_aAw1, axis=3), axis=1) # poor reproduce
    aift6 = _N.std( _N.diff(sd_aAw1, axis=3), axis=1) # poor reproduce
    aift7 = _N.mean(_N.diff(sd_aAw1, axis=3), axis=2) # poor reproduce
    aift8 = _N.std(_N.diff(sd_aAw1, axis=3), axis=2) # poor reproduce

    mn_aAw2                 = _N.mean(aAw, axis=2)
    sd_aAw2                 = _N.std(aAw, axis=2)
    aift9 = _N.mean(_N.diff( mn_aAw2, axis=3), axis=1) # poor reproduce    
    aift10 = _N.std(_N.diff( mn_aAw2, axis=3), axis=1) # poor reproduce    
    aift11 = _N.mean(_N.diff(mn_aAw2, axis=3), axis=2) # poor reproduce  
    aift12 = _N.std(_N.diff(mn_aAw2, axis=3), axis=2) # poor reproduce    
    aift13 = _N.mean(_N.diff( sd_aAw2, axis=3), axis=1) # poor reproduce
    aift14 = _N.std(_N.diff( sd_aAw2, axis=3), axis=1) # poor reproduce
    aift15 = _N.mean(_N.diff(sd_aAw2, axis=3), axis=2) # poor reproduce
    aift16 = _N.std(_N.diff(sd_aAw2, axis=3), axis=2) # poor reproduce

    mn_aAw3                 = _N.mean(aAw, axis=3)
    sd_aAw3                 = _N.std(aAw, axis=3)
    aift17 = _N.mean(_N.diff( mn_aAw3, axis=3), axis=1) # poor reproduce    
    aift18 = _N.std(_N.diff( mn_aAw3, axis=3), axis=1) # poor reproduce    
    aift19 = _N.mean(_N.diff(mn_aAw3, axis=3), axis=2) # poor reproduce  
    aift20 = _N.std(_N.diff(mn_aAw3, axis=3), axis=2) # poor reproduce    
    aift21 = _N.mean(_N.diff( sd_aAw3, axis=3), axis=1) # poor reproduce
    aift22 = _N.std(_N.diff( sd_aAw3, axis=3), axis=1) # poor reproduce
    aift23 = _N.mean(_N.diff(sd_aAw3, axis=3), axis=2) # poor reproduce
    aift24 = _N.std(_N.diff(sd_aAw3, axis=3), axis=2) # poor reproduce

    #########  based on diffAIw
    #  the last axis has 2 values (N=2)
    diffAIw = _N.diff(aAw, axis=4).squeeze()
    stg1  = _N.std(diffAIw, axis=3)   #  len(partIDs) x (T+1) x 3
    stg2  = _N.mean(diffAIw, axis=3)   #  len(partIDs) x (T+1) x 3
    stg3  = _N.std(diffAIw, axis=2)   #  len(partIDs) x (T+1) x 3
    stg4  = _N.mean(diffAIw, axis=2)   #  len(partIDs) x (T+1) x 3

    sdFt1 = _N.std(stg1, axis=1)      #  len(partIDs) x 3  difference in R,P,S
    sdFt2 = _N.std(stg2, axis=1)      #  len(partIDs) x 3  difference in R,P,S
    sdFt3 = _N.std(stg3, axis=1)      #  len(partIDs) x 3  difference in R,P,S
    sdFt4 = _N.std(stg4, axis=1)      #  len(partIDs) x 3  difference in R,P,S
    mnFt1  = _N.mean(stg1, axis=1)    # the diff   #  len(partIDs) x (T+1) x 3
    mnFt2  = _N.mean(stg2, axis=1)    # the diff   #  len(partIDs) x (T+1) x 3
    mnFt3  = _N.mean(stg3, axis=1)    # the diff   #  len(partIDs) x (T+1) x 3
    mnFt4  = _N.mean(stg4, axis=1)    # the diff   #  len(partIDs) x (T+1) x 3

    s_rps = _N.sum(_N.std(diffAIw, axis=1), axis=2) #  

    #  IT TURNS OUT mnFt4 is same as aift1
    #  the last axis has 2 values (N=2)    
    #########  based on sumAIw
    # # ################
    sumAIw = _N.sum(aAw, axis=4).squeeze()
    stg5  = _N.mean(sumAIw, axis=3)   #  len(partIDs) x (T+1) x 3
    stg6  = _N.std(sumAIw, axis=3)   #  len(partIDs) x (T+1) x 3    
    stg7  = _N.mean(sumAIw, axis=2)   #  len(partIDs) x (T+1) x 3
    stg8  = _N.std(sumAIw, axis=2)   #  len(partIDs) x (T+1) x 3

    sdFt5 = _N.std(stg5, axis=1)      #  len(partIDs) x 3  difference in R,P,S
    sdFt6 = _N.std(stg6, axis=1)      #  len(partIDs) x 3  difference in R,P,S
    sdFt7 = _N.std(stg7, axis=1)      #  len(partIDs) x 3  difference in R,P,S
    sdFt8 = _N.std(stg8, axis=1)      #  len(partIDs) x 3  difference in R,P,S 
    mnFt5  = _N.mean(stg5, axis=1)      #  len(partIDs) x 3  difference in R,P,S
    mnFt6  = _N.mean(stg6, axis=1)      #  len(partIDs) x 3  difference in R,P,S
    mnFt7  = _N.mean(stg7, axis=1)      #  len(partIDs) x 3  difference in R,P,S
    mnFt8  = _N.mean(stg8, axis=1)      #  len(partIDs) x 3  difference in R,P,S
    

    # AIfts1allcomps = _N.mean(_N.sum(sumAIw, axis=3), axis=1)
    # AIfts1 = AIfts1allcomps[:, 0]
    # AIfts2 = AIfts1allcomps[:, 1]
    # AIfts3 = AIfts1allcomps[:, 2]
    AIent1  = _N.empty(len(partIDs))    
    AIent2  = _N.empty(len(partIDs))
    AIent3  = _N.empty(len(partIDs))
    AIent4  = _N.empty(len(partIDs))    
    AIent5  = _N.empty(len(partIDs))    
    AIent6  = _N.empty(len(partIDs))
    AIent7  = _N.empty(len(partIDs))
    AIent8  = _N.empty(len(partIDs))    

    # ###  Does it look like we are well-defined by CR?

    for pid in range(len(partIDs)):
        minV = _N.min(stg1[pid])
        maxV = _N.max(stg1[pid])
        stg1[pid] -= minV
        maxV -= minV
        print(stg1[pid])
        AIent1[pid] = entropy3(stg1[pid], 5, maxval=(maxV-minV))

        minV = _N.min(stg2[pid])
        maxV = _N.max(stg2[pid])
        stg2[pid] -= minV
        maxV -= minV
        AIent2[pid] = entropy3(stg2[pid], 5, maxval=(maxV-minV))

        minV = _N.min(stg3[pid])
        maxV = _N.max(stg3[pid])
        stg3[pid] -= minV
        maxV -= minV
        AIent3[pid] = entropy3(stg3[pid], 5, maxval=(maxV-minV))

        minV = _N.min(stg4[pid])
        maxV = _N.max(stg4[pid])
        stg4[pid] -= minV
        maxV -= minV
        AIent4[pid] = entropy3(stg4[pid], 5, maxval=(maxV-minV))

        minV = _N.min(stg5[pid])
        maxV = _N.max(stg5[pid])
        stg5[pid] -= minV
        maxV -= minV
        AIent5[pid] = entropy3(stg5[pid], 5, maxval=(maxV-minV))

        minV = _N.min(stg6[pid])
        maxV = _N.max(stg6[pid])
        stg6[pid] -= minV
        maxV -= minV
        AIent6[pid] = entropy3(stg6[pid], 5, maxval=(maxV-minV))

        minV = _N.min(stg7[pid])
        maxV = _N.max(stg7[pid])
        stg7[pid] -= minV
        maxV -= minV
        AIent7[pid] = entropy3(stg7[pid], 5, maxval=(maxV-minV))

        minV = _N.min(stg8[pid])
        maxV = _N.max(stg8[pid])
        stg8[pid] -= minV
        maxV -= minV
        AIent8[pid] = entropy3(stg8[pid], 5, maxval=(maxV-minV))
        
    srtds = _N.sort(all_AI_preds[:, strt:], axis=2)

    amps  = (srtds[:, :, 2] - srtds[:, :, 0])   #  range 

    diff_top2 = (srtds[:, :, 2] - srtds[:, :, 1]) / _N.mean(amps, axis=1).reshape(len(partIDs), 1)
    diff_top3 = (srtds[:, :, 1] - srtds[:, :, 0]) / _N.mean(amps, axis=1).reshape(len(partIDs), 1)
    #diff_top2 = (srtds[:, :, 2] - srtds[:, :, 1]) / (srtds[:, :, 2] - srtds[:, :, 0])
    mn_diff_top2 = _N.std(diff_top2, axis=1) / _N.mean(diff_top2, axis=1) 
    sd_diff_top2 = 1./_N.std(diff_top2, axis=1)

    #return FEAT1, FEAT2, FEAT3, FEAT4, FEAT5, FEAT6, AIent1, AIent2, AIent3, AIent4, AIent5, AIent6, AIent7, AIent8, mn_diff_top2, sd_diff_top2, mnFt1, mnFt2, mnFt3, mnFt4, mnFt5, mnFt6, mnFt7, mnFt8, sdFt1, sdFt2, sdFt3, sdFt4, sdFt5, sdFt6, sdFt7, sdFt8, aift1[:, :, 0], aift2[:, :, 0], aift3[:, :, 0], aift4[:, :, 0], aift5[:, :, 0], aift6[:, :, 0], aift7[:, :, 0], aift8[:, :, 0], aift9[:, :, 0], aift10[:, :, 0], aift11[:, :, 0], aift12[:, :, 0]
    return s_rps[:, 0], s_rps[:, 1], s_rps[:, 2], AIent1, AIent2, AIent3, AIent4, AIent5, AIent6, AIent7, AIent8, mn_diff_top2, sd_diff_top2, mnFt1, mnFt2, mnFt3, mnFt4, mnFt5, mnFt6, mnFt7, mnFt8, sdFt1, sdFt2, sdFt3, sdFt4, sdFt5, sdFt6, sdFt7, sdFt8, aift1[:, :, 0], aift2[:, :, 0], aift3[:, :, 0], aift4[:, :, 0], aift5[:, :, 0], aift6[:, :, 0], aift7[:, :, 0], aift8[:, :, 0], aift9[:, :, 0], aift10[:, :, 0], aift11[:, :, 0], aift12[:, :, 0], aift13[:, :, 0], aift14[:, :, 0], aift15[:, :, 0], aift16[:, :, 0], aift17[:, :, 0], aift18[:, :, 0], aift19[:, :, 0], aift20[:, :, 0], aift21[:, :, 0], aift22[:, :, 0], aift23[:, :, 0], aift24[:, :, 0]

def get_maxes(behv, thrs, win=3, r1=0.25, r2=0.5, pcs=30, nI=4, thrI=2):
    maxima = _N.where((behv[0:-3] < behv[1:-2]) & (behv[1:-2] > behv[2:-1]))[0]
    minima = _N.where((behv[0:-3] > behv[1:-2]) & (behv[1:-2] < behv[2:-1]))[0]
    nMins = len(minima)
    nMaxs = len(maxima)        
    
    start_thr = _N.sort(behv[minima + 1])[int(r1*nMins)]  #  we don't want maxes to be below any mins
    thr_max   = r2*(_N.max(behv[maxima]) - _N.min(behv[maxima])) + _N.min(behv[maxima]) 
    
    dthr      = (thr_max - start_thr) / pcs
    
    bDone     = False
    i = -1
    while (not bDone) and (i < pcs):
        i += 1
        max_thr = start_thr + dthr*i
        maxs = maxima[_N.where(behv[maxima+1] > max_thr)[0]] + win//2+1
        intvs = _N.diff(maxs)
        #print("!!!!!!!")
        #print(intvs)
        
        if len(_N.where(intvs <= thrI)[0]) < nI:   #  not too many of these
            bDone = True
            #thrs[pid-1] = i
    if not bDone:   #  didn't find it.
        max_thr = start_thr + dthr*28
        maxs = maxima[_N.where(behv[maxima+1] > max_thr)[0]] + win//2+1
        #thrs[pid-1] = 28
        #####!!!!!  len(maxs) < cut means nothing to trigger average
    return maxs
