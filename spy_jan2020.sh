#!/bin/bash
#$ -cwd
#$ -m abe
#$ -M xh2436@nyu.edu

echo "Job start at $(date)"
python3 spy_jan2020.py
echo "Job finished at $(date)"
