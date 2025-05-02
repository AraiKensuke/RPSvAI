
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

#  Data
The same code is used to analyze the 1st and 2nd run (< 20 subjects) data of human subjects vs AI RPS, and data generated in simulation vs AI RPS.  Some subjects completed both the game and the AQ28 questionnaire, and some subjects also appeared to not be engaged during the game (ie making responses too quickly suggests mashing keyboard).  There is a 2-step filtering mechanism to remove these subjects for analysis.
xx
In addition to human subject data, we included simulation code for validating our analysis code.  
`$INSTALLDIR/RPSvAI-main/RPSvAI/simulation/simulate_human_JS_5CFs.py` allows users to define the repertoire of rules, the switching time intervals etc, and generate games played against the same AI human subjects faced.  The data is given a name `expt=` line in the code.  Setting this and running the simulation will create folder `$INSTALLDIR/RPSvAI-main/DATA/SIMHUM`, where x is an integer < 100 chosen by the user and speficied in `$INSTALLDIR/RPSvAI-main/RPSvAI/simulation/simulate_human_JS_5CFs.py`.  Data file stored in this folder with generated YYYYMMDD and HHMM-SS to make files that are in the same format as data collected from humans.

#  Analysis
### calculate the conditional response probabilities under each of the 54 possible rules.
`$INSTALLDIR/RPSvAI-main/RPSvAI/calcCR.py`
### calculate Featuers from data   
`$INSTALLDIR/RPSvAI-main/RPSvAI/RPS_features.py`
### search for rule usage
`$INSTALLDIR/RPSvAI-main/RPSvAI/RPS_cmp_frameworks.py`
### calculate rule change times
`$INSTALLDIR/RPSvAI-main/RPSvAI/RPS_rulechange.py`
### calculate fraction of population using each of the 54 possible rules
`$INSTALLDIR/RPSvAI-main/RPSvAI/patternCFs.py`
### cluster subjects by types of rules in their repertoire
`$INSTALLDIR/RPSvAI-main/RPSvAI/clusterSDS.py`

#  Analysis (human subjects only)
#  Preliminary analysis of test-retest-reliability  (15 participants).   Re-run 5), but with settings edited
`$INSTALLDIR/RPSvAI-main/RPSvAI/TMB2_reliability.py`
`$INSTALLDIR/RPSvAI-main/RPSvAI/TMB2_reliability_shuffle.py`

#  Correlation and predictability of AQ28 subscores using RPS behavioral features.
`$INSTALLDIR/RPSvAI-main/RPSvAI/lassoAQ28_v6.py`
`$INSTALLDIR/RPSvAI-main/RPSvAI/lassoAQ28_report.py`

visit=1
visits=[1, 2]
run again with
visit=2
visits=[1, 2]
