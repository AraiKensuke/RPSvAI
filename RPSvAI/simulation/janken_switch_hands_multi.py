import numpy as _N

#  frameworks
#  DSU_WTL
#  RPS_WTL
#  RPS_RPS
#  DSU_AIRPS
#  RPS_AIRPS

def next_hand(frmwk, Tmat, prev_hnd_wtl, prev_hnd, prev_ai_hnd):
    print("framwk  %d" % frmwk)
    crats= _N.zeros(4)
    if frmwk == 0:  # DSUWTL
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
    elif frmwk == 1:   #  RPSWTL
        rats = Tmat[prev_hnd_wtl-1]
        crats[1:4] = _N.cumsum(rats)

        rnd   = _N.random.rand()
        return _N.where((rnd >= crats[0:-1]) & (rnd < crats[1:]))[0][0] + 1

        # RSP
    elif frmwk == 2:   #  RPSRPS
        rats = Tmat[prev_hnd-1]
        crats[1:4] = _N.cumsum(rats)

        rnd   = _N.random.rand()
        return _N.where((rnd >= crats[0:-1]) & (rnd < crats[1:]))[0][0] + 1

        # RSP
    elif frmwk == 3:   #  DSU_AIRPS
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
            elif prev_hnd == 2: #  choki(2)  ->  paa(3)
                return 3
            elif prev_hnd == 3: #  paa    ->  goo
                return 1
            else:
                print("shouldn't be here")
        elif nxt_mv == 1:       #  stay
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
    elif frmwk == 4:   #  RPS_AIRPS
        rats = Tmat[prev_ai_hnd-1]
        print("---------------------")
        print(prev_ai_hnd)
        print(rats)
        crats[1:4] = _N.cumsum(rats)

        rnd   = _N.random.rand()
        print("prev AI %(ph)d    HP hand %(nh)d" % {"ph" : prev_ai_hnd, "nh" : (_N.where((rnd >= crats[0:-1]) & (rnd < crats[1:]))[0][0] + 1)})
        return _N.where((rnd >= crats[0:-1]) & (rnd < crats[1:]))[0][0] + 1

        # RSP

        
