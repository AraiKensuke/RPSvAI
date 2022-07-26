import numpy as _N

class perceptron:
    N = None                         # how much history

    v = None                  # inputs into predictive units
    x = None                  # past moves by player 
    w = None                  # weights

    def __init__(self, NN):
        self.N = NN          
        self.v = _N.zeros(3)
        self.x = _N.zeros(3*self.N+1)
        self.x[3*self.N] = -1    # threshold      x[3*N] never changes
        #self.x[3*self.N] = 0    # threshold      x[3*N] never changes
        self.w = _N.zeros((3, 3*self.N+1))

    def binary_rep(self, m):   
        """
        binary representation of player's last move  m = 1, 2, 3 (g,c,p)
        goo, choki, paa       prev = [1, -1, -1], [-1, 1, -1] or [-1, -1, 1]
        """
        binary_rep  = _N.ones(3)*-1
        binary_rep[m-1] = +1       #  previous player move
        return binary_rep
        
    #  after user inputs 1 (goo), 2 (choki), 3 (paa), the machine 
    #  reveals its own hand, and winner is decided.  input <= 0, and game ends

    #  v is input to each of the 3 units.  unit w/ largest input wins
    #  x are the N past moves.
    #  x[0:3]   x[3:6]  ... x[12:15]   x[15].  what is x[3*N]? for
    # v[3],x[3*N+1],w[9*N+3]

    def predict(self, prev_pair, update=True, uw=1., update_rule=1):
        """
        prev_m is previous player move.  
        """
        prev_m  = int(prev_pair[1])
        #print("...................In predict  player:   %(1)d" % {"1" : prev_m})
        N  = self.N
        x  = self.x
        #  x[0] is R0
        #  x[1] is S0
        #  x[2] is P0
        #  x[3] is R1
        #  x[4] is S1
        #  x[5] is P1        
        
        w  = self.w
        v  = self.v
        #print("old x")
        #print(x)
        
        #  
        
        #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #print(w)
        #  m is previous player move.  (last game)
        #  x is past data, w is weights, v is prediction to be compared w/ user
        #  we return 1, 2 or 3

        #  i,j,k,kmax,vmax,prec[3];
        if prev_m <= 0:    #  END GAMES
            return -1 #  return index of maximum output unit

        HUM_hand_bin  = self.binary_rep(prev_m)  # [1,-1,-1], [-1,1,-1] or [-1,-1,1]
        #  current value of v was already compared to prev_m.  Here we do 
        #  it again to decide whether weights w should be changed.

        if update_rule == 1:
            for i in range(3):   #  for each predictive unit
                #  if +/- of predict units don't match users's move, then 
                #  perform error correction learning  (change weight)
                if HUM_hand_bin[i] * v[i] <= 0:             
                    #w[i] += uw*HUM_hand_bin[i]*x
                    #w[i] += uw*HUM_hand_bin[i]*x
                    ####  THIS IS WHAT WE MEAN
                    #for k in range(3*N):
                    #    w[i, k] += HUM_hand_bin[i] * x[k]
                    #for k in range(3*N):
                    #
                    # w[i, 0] += HUM_hand_bin[i] * x[0]
                    # w[i, 1] += HUM_hand_bin[i] * x[2]
                    # w[i, 2] += HUM_hand_bin[i] * x[4]
                    # w[i, 3] += HUM_hand_bin[i] * x[1]
                    # w[i, 4] += HUM_hand_bin[i] * x[3]
                    # w[i, 5] += HUM_hand_bin[i] * x[5]                    
                    w[i, 0] += HUM_hand_bin[i] * x[0]
                    w[i, 1] += HUM_hand_bin[i] * x[3]
                    w[i, 2] += HUM_hand_bin[i] * x[1]
                    w[i, 3] += HUM_hand_bin[i] * x[4]
                    w[i, 4] += HUM_hand_bin[i] * x[2]
                    w[i, 5] += HUM_hand_bin[i] * x[5]                    
                    
        elif update_rule == 2:    #  probably more familiar update rule
            for k in range(3):   #  for each predictive unit
                w[i] += uw*(HUM_hand_bin[i] - v[i])*x

        ###  (g0 c0 p0) is newest player hand, binary representation
        ###  shift over x=[g1 c1 p1   g2 c2 p2   g3 c3 p3]
        ###  to         x=[g0 c0 p0   g1 c1 p1   g2 c2 p2]  (g3 c3 p3 disgarded)
        for i in range(3*N-3):
            x[3*N-1-i] = x[3*N-4-i]
        x[0:3] = HUM_hand_bin[0:3]
        #print("new x")
        #print(x)

        #  calculate perceptron output for each perceptron
        for k in range(3):
            v[k] = _N.dot(w[k], x)

        #  index - 1 of unit which got largest inpu

        vmax=v[0]
        kmax= 0
        for k in range(1, 3):
            if v[k] >= vmax:
                vmax = v[k] 
                kmax = k
        #print("****************************")
        #print(w)

        #  this prediction will be compared with player's hand 
        #  received after this function returns, 
        #  this function returns 1, 2 or 3
        return kmax + 1  #  return index of unit that got largest input
