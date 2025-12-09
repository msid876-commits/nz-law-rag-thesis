#!/bin/bash
#SBATCH --job-name=law_rag
#SBATCH --time=00:30:00
#SBATCH --output=law_rag_output.log
#SBATCH --error=law_rag_error.log
#SBATCH --gres=gpu:1

cd /data/msid876/nz-law-rag
export HF_HOME=/data/msid876/.cache/huggingface
export TRANSFORMERS_CACHE=/data/msid876/.cache/huggingface
source venv/bin/activate
python test8.py
