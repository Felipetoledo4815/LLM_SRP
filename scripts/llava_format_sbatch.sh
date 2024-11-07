#!/bin/bash

#SBATCH --job-name=LlaVAformat
#SBATCH --partition=standard
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --account=lesslab

split=$1
mode=$2

module load miniforge

conda deactivate
conda activate .llm_srp/

python scripts/dataset_to_llava_format_mode"$mode".py --split $split