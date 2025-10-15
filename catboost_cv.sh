#!/usr/bin/env bash
date
export CUDA_VISIBLE_DEVICES=0,1
source /Storage/animesh/Miniconda3/miniconda3/bin/activate
conda activate ml4py312
export MPLCONFIGDIR=/Storage/animesh/miniconda3/envs/desi/.matplotlib
# Activate venv explicitly


python -u /Storage/animesh/PECVEL/Codes/catboost_cv.py  --gpu --iterations 50000 --depth 10 --learning_rate 0.05 --mode train --model_path /Storage/animesh/CATBOOST_model_v1.cbm

echo "Job has been completed named V1 $date" > job_completed.txt


