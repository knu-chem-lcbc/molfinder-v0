#!/bin/bash
niter=300
nbank=1000
rseed=3748273
denominator=5
w1_coef=0.8 # coefficient of similiarity/QED terms.
dist_power=0.95
target_smi="CCN1c2ccccc2Cc3c(O)ncnc13"

python3 molfinder.py  -N ${nbank} -r ${rseed} -n ${niter} -v ${denominator} -c ${w1_coef} -dc ${dist_power} -t ${target_smi} 1> log 2> /dev/null
