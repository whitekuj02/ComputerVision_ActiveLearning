#!/bin/bash

export PATH=~/anaconda3/bin:~/anaconda3/condabin:$PATH

. ~/anaconda3/etc/profile.d/conda.sh

conda activate

conda activate aicon

today=`date +%y%m%d_%H%M%S`

echo $today

nohup python3 base_code.py >log/log.$today&