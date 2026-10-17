#!/bin/bash
#SBATCH -p ZEN4 --qos=nu-zen4 -c 32 --mem=64G --time=04:00:00 -J eval
#SBATCH -o logs/slurm_eval_%j.out
# TODO: run eval/evaluate.py on a CPU node
