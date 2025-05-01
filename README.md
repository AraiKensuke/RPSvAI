
Code for paper [Deviation from Nash mixed equilibrium in repeated rock-paper-scissors, K. Arai, S. Jacob., A. Widge, and A. Yousefi, Scientific Reports 2025.](https://www.nature.com/articles/s41598-025-95444-6)

1)  Install [Anaconda or Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main).

2)  Download zip file from repository.
   Choose or create directory to unpack: `INSTALLDIR`.  Unpacking zip file in this directory creates RPSvAI-main folder in `INSTALLDIR`.

3)  `conda create -f $INSTALLDIR/RPSvAI-main/environment.yml` to create environment to run code.

4)  Folder `$INSTALLDIR/RPSvAI-main/DATA` contains anonymized human vs AI RPS data.  In each directory organized by collection date, there are 1 or more directories called \
    `$INSTALLDIR/RPSvAI-main/DATA/TMB2/YYYYMMDD/YYYYMMDD-HHMM-SS/x`, \
    where x=1, 2, ... are the game number (each game 300 rounds) for this participant.  Under this directory, there is a file called block1_AI.dat, a flat text file containing game data.

5)  Environment variables to set.\
`TAISENDATA=$INSTALLDIR/DATA`\
`RPSWORKDIR=$INSTALLDIR/WorkDir`\
`RPSOUTPUTDIR=$INSTALLDIR/OutDir`

3)  Create conda environment

4)  Create simulated data
simulation/simulate_human_JS_5CFs

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
