
Code for paper Deviation from [Nash mixed equilibrium in repeated rock-paper-scissors, K. Arai, S. Jacob., A. Widge, and A. Yousefi, Scientific Reports 2025].(https://www.nature.com/articles/s41598-025-95444-6)

1)  Download zip file from repository.
Creates RPSvAI-main

2)  Environment variables to set.
TAISENDATA
RPSWORKDIR
RPSOUTPUTDIR

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
