#resultDir = "/Users/arai/nctc/Workspace/AIiRPS_Results"
import os

def datadirFN(fn):
    try:
        return "%(rd)s/%(fn)s" % {"rd" : os.environ["TAISENDATA"], "fn" : fn}
    except KeyError:
        print("!!!! Please set environment variable TAISENDATA!")
        exit()

def workdirFN(fn):
    try:
        return "%(rd)s/%(fn)s" % {"rd" : os.environ["RPSWORKDIR"], "fn" : fn}
    except KeyError:
        print("!!!! Please set environment variable RPSWORKDIR!")
        exit()

def outdirFN(fn, label):
    try:
        outdir = "%(od)s_%(lb)s" % {"od" : os.environ["RPSOUTPUTDIR"], "lb" : label}
        if not os.access(outdir, os.F_OK):
            os.mkdir(outdir)
        return "%(od)s/%(fn)s" % {"od" : outdir, "fn" : fn}
    except KeyError:
        print("!!!! Please set environment variable RPSOUTPUTDIR!")
        exit()
        
