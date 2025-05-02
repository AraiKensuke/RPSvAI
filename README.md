
Code for paper [Deviation from Nash mixed equilibrium in repeated rock-paper-scissors, K. Arai, S. Jacob., A. Widge, and A. Yousefi, Scientific Reports 2025.](https://www.nature.com/articles/s41598-025-95444-6)

1)  Install [Anaconda or Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main).

2)  Download zip file from repository.
   Choose or create directory to unpack: `INSTALLDIR`.  Unpacking zip file in this directory creates RPSvAI-main folder in `INSTALLDIR`.

3)  `conda create -f $INSTALLDIR/RPSvAI-main/environment.yml` to create environment to run code.

4)  Environment variables to set.\
`TAISENDATA=$INSTALLDIR/DATA`\
`RPSWORKDIR=$INSTALLDIR/WorkDir`\
`RPSOUTPUTDIR=$INSTALLDIR/OutDir`

5)  Folder `$INSTALLDIR/RPSvAI-main/DATA` contains anonymized human vs AI RPS data from TestMyBrain.org-hosted web-based experiment.  In each directory organized by collection date, there are 1 or more directories called \
    `$INSTALLDIR/RPSvAI-main/DATA/TMB2/YYYYMMDD/YYYYMMDD-HHMM-SS/x`, \
    where x=1, 2, ... are the game number (each game 300 rounds) for this participant.  Under this directory, there is a file called block1_AI.dat, a flat text file containing game data.

#  Analysis
The same code is used to analyze the 1st and 2nd run (< 20 subjects) data of human subjects vs AI RPS, and data generated in simulation vs AI RPS.  Some subjects completed both the game and the AQ28 questionnaire, and some subjects also appeared to not be engaged during the game (ie making responses too quickly suggests mashing keyboard).  There is a 2-step filtering mechanism to remove these subjects for analysis.
##  Human subject data
Data collected from TestMyBrain.org-hosted experiment reides in 
#  Creating simulated data.
We included simulation code for validating our analysis.  
simulation/simulate_human_JS_5CFs

#  Analyzing human RPS data
5)  Analyze Human RPS data
### calculate the conditional response probabilities
calcCR.py
### calculate Featuers from data
RPS_features.py
### calculate Featuers from data
RPS_cmp_frameworks.py
### calculate rule change times
RPS_rulechange
### 
patternCFs

Retest-reliability  (preliminary, 15 participants).   Re-run 5), but with settings edited
###  calculate 
TMB2_reliability
TMB2_reliability_shuffle

#  Using features of behavior, predict
lassoAQ28_v6  (w/ AIcert)
lassoAQ28_report

----

reliabilit
visit=1
visits=[1, 2]
run again with
visit=2
visits=[1, 2]
