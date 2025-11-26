#!/bin/bash
#SBATCH --account=def-abed-ab
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --time=2:00:00
#SBATCH --job-name=CompVision
#SBATCH --output=%x-%j.out


# ---- Compute Canada module stack ----
module load StdEnv/2020 gcc/9.3.0 python/3.10 cuda/11.4 cudnn/8.2 scipy-stack/2022a opencv/4.5.5

source ~/VIRTUAL/bin/activate

# Optional deterministic CPU fallbacks; uncomment when debugging numerics
# export TF_ENABLE_ONEDNN_OPTS=0

# ---- Launch training ----
python fruit_sorting_noise_robust_tf.py \
    --data_root /home/spate193/.cache/kagglehub/datasets/moltean/fruits/versions/61/fruits-360_100x100/fruits-360 \
    --image_size 100 \
    --batch_size 64 \
    --noise_mode symmetric \
    --noise_rate 0.2 \
    --warmup_epochs 6 \
    --refine_epochs 10 \
    --output_dir /home/spate193/scratch/fruit360_sym20_run