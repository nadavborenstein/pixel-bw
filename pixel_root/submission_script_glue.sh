#!/bin/sh
#PBS -A DD-22-70
#PBS -q qnvidia
#PBS -l select=1,walltime=48:00:00
#PBS -N pixel

# Above we define that we select 1 full nodes (16 GPUs) on the qnivida queue for 48h (this is the maximum run time)
# Note: it may make sense to use a large number of nodes to train w/ large batch size for fewer steps
# We use compute hours from the project DD-22-70

cd $PBS_O_WORKDIR

# load cuda
module load CUDA/11.3.1
module load cuDNN/8.2.1.32-CUDA-11.3.1

# load conda env
module load Anaconda3/2022.05
eval "$(conda shell.bash hook)"
conda activate pixel-env

# log into wandb
wandb login

# add git-lfs binaries to PATH to upload models to huggingface hub via git-lfs
export PATH=$PATH:/home/it4i-nadavb/apps/lfs/bin
export PYTHONPATH="/home/it4i-nadavb/pixel"

# WANDB setup
export WANDB_API_KEY="6e62fdec81f0b81d70d3756f1391f44988935dab"
export WANDB_ENTITY="nadav_b"
export WANDB_PROJECT="pixel-finetuning"
export WANDB_DIR="/scratch/project/dd-22-70/wandb/"

# Training environment setup
if [[ -z "${PBS_NODEFILE}" ]]; then
    RANKS=$HOSTNAME
    NNODES=1
else
    MASTER_RANK=$(head -n 1 $PBS_NODEFILE)
    RANKS=$(tr '\n' ' ' < $PBS_NODEFILE)
    NNODES=$(< $PBS_NODEFILE wc -l)
fi

# Construct string that is prepended to the training command
# This ensures all steps are executed on all nodes
PRELOAD="module load Anaconda3/2022.05 ; "
PRELOAD+='eval "$(conda shell.bash hook)" ; '
PRELOAD+="module load CUDA/11.3.1 ; "
PRELOAD+="module load cuDNN/8.2.1.32-CUDA-11.3.1 ; "
PRELOAD+="conda activate pixel-env ; "
PRELOAD+="wandb login ; "
PRELOAD+="export PATH=$PATH:/home/it4i-nadavb/apps/lfs/bin ; "
PRELOAD+="export PYTHONPATH=/home/it4i-nadavb/pixel ; "
PRELOAD+="export WANDB_API_KEY=6e62fdec81f0b81d70d3756f1391f44988935dab ; "
PRELOAD+="export WANDB_ENTITY=nadav_b ; "
PRELOAD+="export WANDB_PROJECT=pixel-finetuning ; "
PRELOAD+="export WANDB_DIR=/scratch/project/dd-22-70/wandb ; "
PRELOAD+="export OUTPUT_DIR=/scratch/project/dd-22-70/experiments/pixel-glue ; "
PRELOAD+="export RUN_NAME=pixel-glue-finetuning; "
PRELOAD+="export OMP_NUM_THREADS=8 ; "

# Set variables locally
export OUTPUT_DIR=/scratch/project/dd-22-70/experiments/pixel-glue/
export RUN_NAME=pixel-glue-finetuning
mkdir -p ${OUTPUT_DIR}/${RUN_NAME}

# Distributed launch configuration
LAUNCHER="python -m torch.distributed.run "
LAUNCHER+="--nnodes=$NNODES --nproc_per_node=auto --max_restarts 0 "
if [[ "$NNODES" -eq 1 ]]; then
    LAUNCHER+="--standalone "
else
    LAUNCHER+="--rdzv_backend=c10d --rdzv_endpoint=$MASTER_RANK "
fi

# Training command
CMD="pixel/glue_finetuning.py \
    --model_name_or_path=Nadav/PretrainedPHD \
    --task_name=sst2 \
    --output_dir=${OUTPUT_DIR}/${RUN_NAME} \
    --overwrite_output_dir=true \
    --job_dir=${OUTPUT_DIR}/${RUN_NAME} \
    --max_steps=3000 \
    --do_train=true \
    --do_eval=true \
    --base_learning_rate=3e-5 \
    --weight_decay=0.00 \
    --warmup_steps=100 \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=16 \
    --gradient_accumulation_steps=1 \
    --logging_strategy=steps \
    --logging_steps=50 \
    --evaluation_strategy=steps \
    --eval_steps=100 \
    --save_strategy=steps \
    --save_steps=30000 \
    --seed=123 \
    --use_auth_token=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 \
    --fp16=true \
    --half_precision_backend=amp \
    --report_to=wandb \
    --push_to_hub=false \
    --hub_model_id=copenlu/pixel \
    --hub_strategy=checkpoint \
    --hub_token=6e62fdec81f0b81d70d3756f1391f44988935dab \
    --hub_private_repo=true \
    --ddp_find_unused_parameters=false \
    --dataloader_num_workers=16 \
    --dropout_prob=0.1 \
    --load_best_model_at_end=true"


FULL_CMD=" $PRELOAD $LAUNCHER $CMD $@ "
echo "Training Command: $FULL_CMD"

# Launch the pytorch processes on each worker (use ssh for remote nodes)
RANK=0
for NODE in $RANKS; do
    if [[ "$NODE" == "$HOSTNAME" ]]; then
        echo "Launching rank $RANK on local node $NODE"
        eval $FULL_CMD &
    else
        echo "Launching rank $RANK on remote node $NODE"
        ssh $NODE "cd $PWD; $FULL_CMD" &
    fi
    RANK=$((RANK+1))
done

wait
