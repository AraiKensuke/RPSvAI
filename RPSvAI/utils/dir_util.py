#resultDir = "/Users/arai/nctc/Workspace/AIiRPS_Results"
import os

def workdirFN(fn):
    try:
        return "%(rd)s/%(fn)s" % {"rd" : os.environ["RPSWORKDIR"], "fn" : fn}
    except KeyError:
        print("!!!! Please set environment variable RPSWORKDIR!")
        exit()
        
