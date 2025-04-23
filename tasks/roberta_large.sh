#!/bin/bash

#SBATCH --job-name=roberta_large
#SBATCH --output=./res/roberta_large_res.txt
#SBATCH --error=./res/roberta_large_error.txt
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e1350606@u.nus.edu

source ../miniconda3/etc/profile.d/conda.sh
conda activate qwen

# GE
python train.py --dataset go_emotion --batch_size 16 --seed 1234
python train.py --dataset go_emotion --batch_size 16 --seed 2345
python train.py --dataset go_emotion --batch_size 16 --seed 3456
python train.py --dataset go_emotion --batch_size 16 --seed 4567
python train.py --dataset go_emotion --batch_size 16 --seed 5678

# ED
python train.py --dataset ED --batch_size 16 --seed 1234
python train.py --dataset ED --batch_size 16 --seed 2345
python train.py --dataset ED --batch_size 16 --seed 3456
python train.py --dataset ED --batch_size 16 --seed 4567
python train.py --dataset ED --batch_size 16 --seed 5678