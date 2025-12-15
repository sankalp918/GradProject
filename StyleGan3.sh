#!/bin/bash
#SBATCH --account=def-abed-ab
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --time=120:00:00
#SBATCH --job-name=stylegan3
 
module load StdEnv/2020 gcc/9.3.0 python/3.10 cuda/11.1.1
source $HOME/StyleGan3-vene/bin/activate
 
cd /home/spate193/stylegan3
 
python train.py \
    --outdir=/home/spate193/scratch/training-runs \
    --cfg=stylegan3-t \
    --data=/home/spate193/scratch/stylegan_data/normal_dataset.zip \
    --gpus=4 \
    --batch=128 \
    --gamma=8.2 \
    --mirror=1 \
    --snap=25 \
    --kimg=20000 \
    --metrics=none \
    --aug=ada \
    --target=0.6
