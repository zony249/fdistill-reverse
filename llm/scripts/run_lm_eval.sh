#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=a100:4
#SBATCH --mem=64000M
#SBATCH --time=1-00:00
#SBATCH --account=
#SBATCH --output=slurm-logs/%j-llama-3-70B-distributed.out


nvidia-smi
nvidia-smi topo -m

export HEAD_NODE=$(hostname) # store head node's address

# export HF_DATASETS_OFFLINE=1
# export HF_HOME=~/large-file-storage
export CUDA_VISIBLE_DEVICES=""
export NUM_PROCESSES=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
export MODEL=""
export TASK=hellaswag # change to coqa if necessary

accelerate launch \
    --num_processes=$NUM_PROCESSES \
    -m lm_eval \
        --model hf \
        --model_args pretrained=$MODEL,parallelize=False,dtype=bfloat16,trust_remote_code=true \
        --tasks=hellaswag \
        --num_fewshot=0 \
        --gen_kwargs num_beams=1 \
        --batch_size 30 \
        --output_path runs/lm_eval \
        # --log_samples \