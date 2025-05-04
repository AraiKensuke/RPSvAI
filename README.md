
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
### calculate the time-dependent conditional response probabilities under each of the 6x9 = 54 possible rules.
`$INSTALLDIR/RPSvAI-main/RPSvAI/calcCR.py`
calculates CR probability calculated in sliding windows.  In paper, the parameters used were `win_type=2, wins=3, gk=1`
Outputs pickle files
`$RPSWORKDIR/YYYYMMDD_HHMM-SS/231/variousCR_1.dmp`.  Choosing different parameters will result in output to a differrent directory, ie changing `gk` to 2 will output to directory `232`.  Temporal order of rounds are shuffled and CR timeseries calculated to assess whether a given rule is in the rule repertoire of a particular subject.
### calculate features from data   
`$INSTALLDIR/RPSvAI-main/RPSvAI/RPS_features.py`    
Outputs pickle file
`$RPSWORKDIR/TMB2_AQ28_vs_RPSfeatures_1_of_1_231.dmp`.  Features are dependent on sliding window parameters above, and so output filename reflects this.  This file contains features for all subjects.  1_of_1 refers to the output being for the 1st time playing 300-round game.  When analyzing preliminary test-retest reliability of the features, we run this command again for the 2nd game, in which case output file is `$RPSWORKDIR/TMB2_AQ28_vs_RPSfeatures_1_of_[1,2]_231.dmp`. 
### search for rule usage  
`$INSTALLDIR/RPSvAI-main/RPSvAI/RPS_cmp_frameworks.py`
Using shuffled CRs, look for signatures of rule usage, like excess rounds where conditional probability is near 1 or 0 compared to the shuffled distribution.  Outputs pickle file `shuffledCRs_TMB2_231.dmp`. 
### calculate the rule-change triggered net win rate
`$INSTALLDIR/RPSvAI-main/RPSvAI/RPS_rulechange.py`
Calculate Fig.5.  Setting `expt` to `TMB2` will use human vs AI data, `SIMHUMxx` to use simulated human vs AI data.
### calculate fraction of population using each of the 54 possible rules
`$INSTALLDIR/RPSvAI-main/RPSvAI/patternCFs.py`.  Recreates Fig. 4B.
### project features of rulechange timeseries of all subjects to 2D
`$INSTALLDIR/RPSvAI-main/RPSvAI/clusterSDS.py`.  Recreates Fig. 4C.

#  Analysis (human subjects only)
#  Preliminary analysis of test-retest-reliability  (15 participants).   Re-run 5), but with settings edited
`$INSTALLDIR/RPSvAI-main/RPSvAI/TMB2_reliability.py`\
`$INSTALLDIR/RPSvAI-main/RPSvAI/TMB2_reliability_shuffle.py`

For analysis of data where player played at least 1 300-round game, use parameters.

`visit=1`\
`visits=[1]`

run analysis of data of 1st game where player played at least 2 300-round games, use parameters

`visit=1`\
`visits=[1, 2]`

run analysis of data of 2nd game where player played at least 2 300-round games, use parameters

`visit=2`\
`visits=[1, 2]`

#  Correlation and predictability of AQ28 subscores using RPS behavioral features.
`$INSTALLDIR/RPSvAI-main/RPSvAI/lassoAQ28_v6.py`\
`$INSTALLDIR/RPSvAI-main/RPSvAI/lassoAQ28_report.py`

#  Create simulated game data
`$INSTALLDIR/RPSvAI-main/RPSvAI/simulation/janken_simulate_human_JS_5CFs.py`
Specify rules used in rule-changing simulation against AI.  Use this data for validation of 2D mapping results, and rule-change detection.
