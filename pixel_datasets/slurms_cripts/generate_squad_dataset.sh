#!/bin/bash
#SBATCH --job-name=generate_squad
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00
#SBATCH --mem=16G
#SBATCH --array=[0-0]
#SBATCH --exclude=hendrixgpu20fl
#SBATCH --nodelist=hendrixgpu11fl

/home/knf792/anaconda3/envs/Genalog/bin/python /home/knf792/PycharmProjects/pixel-2/generate_squad_dataset.py ${SLURM_ARRAY_TASK_ID}
