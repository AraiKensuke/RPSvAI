# RPSvAI
RPS against AI

Code and data to analyze repeated RPS game play against an AI agent.

Our hypothesis is that personality traits (especially those known to correlate to presence / severity of psychiatric dysfunction), manifest 
themselves in RPS game play. If we know what to look for, RPS may be a good diagnostic tool for the psychiatrist, and so our job is to find out
what we need to look for.

This repository contains the game play data and responses to the Autism Quotient Short questionnaire from several hundred anonymous players on the 
internet.  The code extracts features from game play, and runs a cross-validated linear regression and feature selection to test whether AQS 
scores can be explained with features from game play.  Features can be simple things like the net nnumber of wins, or more complicated features that
use the idea of Covert Adjustments of Behavior (CAB).

## Python environment

You need python 3, numpy, matplotlib, scipy and sklearn.  This program is run from the command line.

You need to set 3 environment variables:
```
    TAISENDATA - where data from the RPSvAI collected from web go.
    RPSWORKDIR - directory where intermediate results of analysis, like calculated conditional probabilities, go.
    RPSOUTPUTDIR - directory where final results like figures are output.
```

Please add this line to .pythonrc.py, which allows us to run python scripts from command line
```
def rexf(fn):   
    """
    rexf(pythonscript.py)  - execute a python script while in CLI mode
    """
    globals().update(runpy.run_path(fn, run_name="__main__"))
```
    
##  Using the code

We ran 2 types of experiments.  TMB1 is the experiment where players played against 4 types of fixed rules for 30 games each.  TMB2 was the experiment
where players played 300 games against an AI that learned the player's biases on-line, so the AI kept up if players changed gameplay rules.  We first
describe TMB2 experiment.

Given the nature of data collection - anonymous participants recruited online, we don't have a way to control or demand a level of quality
control found in laboratory experiments.  Players might be rushing through the games, they might not complete all portions of the data
collection, many players might be playing at the same time.  We first build a list of data collected, and store in $RPSWORKDIR/TMB2fns_[1].txt
```
(shell)$ python TMB_easy_dirnames TMB2 
```
We next calculate the conditonal action probabilitie, like p(UP | WIN)), and store the results in $RPSWORKDIR because this will be used again
many times in subsequent analysis.  We also calculate these probabilities on shuffled data, where the game order is randomized.
```
(shell)$ python
> rexf("calcCR_test.py")
```
We next derive various features from the data, as well as calculate AQS scores.  We also filter out data we don't think represents properly
played game, ie inter-game-intervals too short to suggest thoughful play, noticeable outlier in AQS score from population etc.
```
> rexf("RPS_biomark.py")
```
We next perform a cross-validated and feature selected linear regression for AQS score vs. calculated features.
```
> rexf("lassoAQ28_strict.py")
```

The output images are written to $RPSOUTPUTDIR.  
