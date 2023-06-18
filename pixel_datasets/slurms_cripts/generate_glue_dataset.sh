#!/bin/bash
#SBATCH --job-name=generate_glue
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --array=[0-20]
#SBATCH --exclude=hendrixgpu20fl,hendrixgpu18fl,hendrixgpu17fl,hendrixgpu16fl,hendrixgpu19fl

/home/knf792/anaconda3/envs/Genalog/bin/python /home/knf792/PycharmProjects/pixel-2/generate_and_OCR_and_upload_glue_dataset.py 6 ${SLURM_ARRAY_TASK_ID} 20
