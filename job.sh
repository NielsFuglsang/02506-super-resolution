#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J superres
#BSUB -n 1
#BSUB -W 1:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o logs/superres_%J.out
#BSUB -e logs/superres_%J.err

module load cuda/11.1
source .venv/bin/activate

which python3

echo "Running script..."
python3 main.py