#!/bin/bash
#SBATCH -p H100 --qos=nu-h100 --gres=gpu:1 -c 16 --mem=64G --time=2-00:00:00 -J vllm
#SBATCH -o logs/slurm_vllm_%j.out
# TODO: start vLLM with Qwen2.5-VL-7B-Instruct from models/ in offline mode
