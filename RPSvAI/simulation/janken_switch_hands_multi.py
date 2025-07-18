import numpy as _N

#  frameworks
#  DKU_wtl
#  RPS_wtl
##  RPS_rps
#  DKU_rps
#  LCB_rps
#  DKU_airps
#  LCB_airps

def next_hand(frmwk, Tmat, prev_hnd_wtl, prev_hnd, prev_ai_hnd):
    #print("framwk  %d" % frmwk)
    crats= _N.zeros(4)
    if frmwk == 0:  # DKU_wtl
        rats = Tmat[prev_hnd_wtl-1]
        crats[1:4] = _N.cumsum(rats)

        rnd   = _N.random.rand()
        nxt_mv = _N.where((rnd >= crats[0:-1]) & (rnd < crats[1:]))[0]

        #  nxt_mv == 0   stay 
        #  nxt_mv == 1   go_to_weaker
        #  nxt_mv == 2   go_to_stronger

        if nxt_mv == 0:       #  go to weaker hnd
            if prev_hnd == 1:   #  goo(1)    ->  go to choki(2)
                return 2
            elif prev_hnd == 2: #  choki(2)  ->  paa(3)
                return 3
            elif prev_hnd == 3: #  paa    ->  goo
                return 1
            else:
                print("shouldn't be here")
        elif nxt_mv == 1:
            return prev_hnd
        elif nxt_mv == 2:       #  go to stronger hnd
            if prev_hnd == 1:   #  goo(1)    ->  paa(3)
                return 3
            elif prev_hnd == 2: #  choki(2)  ->  goo(1)
                return 1
            elif prev_hnd == 3: #  paa(3)    ->  choki(2)
                return 2
            else:
                print("shouldn't be here")
    elif frmwk == 1:   #  RPS_wtl
        rats = Tmat[prev_hnd_wtl-1]
        crats[1:4] = _N.cumsum(rats)

        rnd   = _N.random.rand()
        return _N.where((rnd >= crats[0:-1]) & (rnd < crats[1:]))[0][0] + 1

        # RSP
    # elif frmwk == 2:   #  RPS_rps
    #     rats = Tmat[prev_hnd-1]
    #     crats[1:4] = _N.cumsum(rats)

    #     rnd   = _N.random.rand()
    #     return _N.where((rnd >= crats[0:-1]) & (rnd < crats[1:]))[0][0] + 1
    # elif frmwk == 2:   #  DSU_rps
    #     rats = Tmat[prev_hnd-1]
    #     crats[1:4] = _N.cumsum(rats)

    #     rnd   = _N.random.rand()
    #     nxt_mv = _N.where((rnd >= crats[0:-1]) & (rnd < crats[1:]))[0]

    #     #  nxt_mv == 0   stay 
    #     #  nxt_mv == 1   go_to_weaker
    #     #  nxt_mv == 2   go_to_stronger

    #     print("!!!!!!!!!!!!!!!!!!!!!!!!!")
    #     if nxt_mv == 0:       #  go to weaker hnd
    #         if prev_hnd == 1:   #  goo(1)    ->  go to choki(2)
    #             return 2
    #         elif prev_hnd == 2: #  choki(2)  ->  paa(3)
    #             return 3
    #         elif prev_hnd == 3: #  paa    ->  goo
    #             return 1
    #         else:
    #             print("shouldn't be here")
    #     elif nxt_mv == 1:
    #         return prev_hnd
    #     elif nxt_mv == 2:       #  go to stronger hnd
    #         if prev_hnd == 1:   #  goo(1)    ->  paa(3)
    #             return 3
    #         elif prev_hnd == 2: #  choki(2)  ->  goo(1)
    #             return 1
    #         elif prev_hnd == 3: #  paa(3)    ->  choki(2)
    #             return 2
    #         else:
    #             print("shouldn't be here")
    #     # RSP
    elif frmwk == 2:   #  DCU|rps
        rats = Tmat[prev_hnd-1]
        crats[1:4] = _N.cumsum(rats)

        rnd   = _N.random.rand()

        action = _N.where((rnd >= crats[0:-1]) & (rnd < crats[1:]))[0][0] + 1
        if action == 1:   #  Downgrade own hand
            #  return hand that b
            if prev_hnd == 1:   #  AI goo(1)    ->  choki(2)
                return 2
            elif prev_hnd == 2: #  AI choki(2)  ->  paa(3)
                return 3
            elif prev_hnd == 3: #  AI paa(3)    ->  goo(2)
                return 1
        elif action == 2:   #  Copy AIhand
            #  return hand that b
            return prev_hnd
        elif action == 3:   #  BEAT AI hand
            #  return hand that b
            if prev_hnd == 1:   #  AI goo(1)    ->  paa(3)
                return 3
            elif prev_hnd == 2: #  AI choki(2)  ->  goo(1)
                return 1
            elif prev_hnd == 3: #  AI paa(3)    ->  choki(2)
                return 2
        #return _N.where((rnd >= crats[0:-1]) & (rnd < crats[1:]))[0][0] + 1
        # RSP
    elif frmwk == 3:   #  DaCaUa_rps
        rats = Tmat[prev_hnd-1]
        crats[1:4] = _N.cumsum(rats)

        rnd   = _N.random.rand()

        action = _N.where((rnd >= crats[0:-1]) & (rnd < crats[1:]))[0][0] + 1
        if action == 1:   #  LOSE to AI hand
            #  return hand that b
            if prev_ai_hnd == 1:   #  AI goo(1)    ->  choki(2)
                return 2
                #return 3
            elif prev_ai_hnd == 2: #  AI choki(2)  ->  paa(3)
                return 3
                #return 1
            elif prev_ai_hnd == 3: #  AI paa(3)    ->  goo(2)
                return 1
                #return 2
        elif action == 2:   #  Copy AIhand
            #  return hand that b
            return prev_ai_hnd
        elif action == 3:   #  BEAT AI hand
            #  return hand that b
            if prev_ai_hnd == 1:   #  AI goo(1)    ->  paa(3)
                return 3
                #return 2
            elif prev_ai_hnd == 2: #  AI choki(2)  ->  goo(1)
                return 1
                #return 3
            elif prev_ai_hnd == 3: #  AI paa(3)    ->  choki(2)
                return 2
                #return 1
        #return _N.where((rnd >= crats[0:-1]) & (rnd < crats[1:]))[0][0] + 1
        # RSP
    elif frmwk == 4:   #  DCU|rapasa
        rats = Tmat[prev_ai_hnd-1]
        crats[1:4] = _N.cumsum(rats)
        rnd   = _N.random.rand()
        nxt_mv = _N.where((rnd >= crats[0:-1]) & (rnd < crats[1:]))[0]

        #  nxt_mv == 0   stay 
        #  nxt_mv == 1   go_to_weaker
        #  nxt_mv == 2   go_to_stronger

        if nxt_mv == 0:         #  go to weaker hand
            if prev_hnd == 1:   #  goo(1)    ->  go to choki(2)
                return 2
                #return 3
            elif prev_hnd == 2: #  choki(2)  ->  paa(3)
                return 3
                #return 1
            elif prev_hnd == 3: #  paa    ->  goo
                return 1
                #return 2
            else:
                print("shouldn't be here")
        elif nxt_mv == 1:       #  stay
            return prev_hnd
        elif nxt_mv == 2:       #  go to stronger hnd
            if prev_hnd == 1:   #  goo(1)    ->  paa(3)
                return 3
                #return 2
            elif prev_hnd == 2: #  choki(2)  ->  goo(1)
                return 1
                #return 3
            elif prev_hnd == 3: #  paa(3)    ->  choki(2)
                return 2
                #return 1
            else:
                print("shouldn't be here")
    elif frmwk == 5:   #  DaCaUa|rapasa
        rats = Tmat[prev_ai_hnd-1]
        print(rats)
        print("---------------------")
        #print(rats)
        crats[1:4] = _N.cumsum(rats)

        rnd   = _N.random.rand()
        #print("prev AI %(ph)d    HP hand %(nh)d" % {"ph" : prev_ai_hnd, "nh" : (_N.where((rnd >= crats[0:-1]) & (rnd < crats[1:]))[0][0] + 1)})
        action = _N.where((rnd >= crats[0:-1]) & (rnd < crats[1:]))[0][0] + 1
        if action == 1:   #  LOSE to AI hand
            #  return hand that b
            if prev_ai_hnd == 1:   #  AI goo(1)    ->  choki(2)
                rethnd=2
            elif prev_ai_hnd == 2: #  AI choki(2)  ->  paa(3)
                rethnd=3
            elif prev_ai_hnd == 3: #  AI paa(3)    ->  goo(2)
                rethnd=1
        elif action == 2:   #  Copy AIhand
            #  return hand that b
            rethnd=prev_ai_hnd
        elif action == 3:   #  BEAT AI hand
            #  return hand that b
            if prev_ai_hnd == 1:   #  AI goo(1)    ->  paa(3)
                rethnd=3
            elif prev_ai_hnd == 2: #  AI choki(2)  ->  goo(1)
                rethnd=1
            elif prev_ai_hnd == 3: #  AI paa(3)    ->  choki(2)
                rethnd=2
        #print("prevai=%(pia)d   act=%(act)d    ret=%(rh)d" % {"pia" : prev_ai_hnd, "act" : action, "rh" : rethnd})
        return rethnd
        

        # RSP

        
