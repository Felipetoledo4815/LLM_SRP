#!/bin/bash

#SBATCH --job-name=LlaVAformat
#SBATCH --partition=standard
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --account=lesslab

split=$1

python scripts/dataset_to_llava_format.py --split $split