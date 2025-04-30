import numpy as _N

def gauKer(w):
    """
    1-D gaussian kernel.  Use with numpy.convolve
    """
    wf = _N.empty(8*w+1)

    for i in range(-4*w, 4*w+1):
        wf[i+4*w] = _N.exp(-0.5*(i*i)/(w*w))
    wf /= _N.sum(wf)
    return wf

def depickle(s):
     import pickle
     with open(s, "rb") as f:
          lm = pickle.load(f)
     return lm

def repeated_array_entry(arr):
    """
    return me 
    """
    longest_repeats = []
    i = 0
    L = arr.shape[0]
    while i < L-1:
        if arr[i] == arr[i+1]:  #  Found a repeat
            j = i
            keep_going = True
            while (j < L-1) and keep_going:
                if arr[j] != arr[j+1]:
                    longest_repeats.append(j - i+1)
                    keep_going = False
                j += 1
            if keep_going:  #  hit end of loop while a repeat
                longest_repeats.append(j - i+1)                
            i = j-1  #  j+1 is not equal
        else:     #  Not a repeat
            longest_repeats.append(1)
        i += 1
    if arr[L-2] != arr[L-1]:   #  last 2 are not repeats
        longest_repeats.append(1)

    return longest_repeats
 
