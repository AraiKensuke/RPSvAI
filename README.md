# RPSvAI
RPS against AI

Code and data to analyze RPS game play against an AI agent.

Code to extract features from game play.  Simpler features are things like net wins.  More complicated features
use the idea of Covert Adjustments of Behavior (CAB).

Setting up.

You need python 3, numpy, matplotlib, scipy.  This program is run from the command line.

You need to set 2 environment variables:
```
    TAISENDATA - where data from the RPSvAI collected from web go.
    RPSWORKDIR - directory where intermediate and final results of analysis go.
```

Please add this line to .pythonrc.py
```
def rexf(fn):   
    globals().update(runpy.run_path(fn, run_name="__main__"))
```
    
First thing

Build list of usable datafiles, and store in $RPSWORKDIR/TMB2fns_[1].txt

```
(shell)$ python TMB_easy_dirnames TMB2 
Calculate conditional action probabilities (like p(UP | WIN)), and the 
(shell)$ python
> rexf("calcCR_test.py")
```
