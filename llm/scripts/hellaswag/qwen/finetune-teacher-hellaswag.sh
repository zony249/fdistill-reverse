#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=l40s:4
#SBATCH --mem=56000M
#SBATCH --time=3-00:00
#SBATCH --account=
#SBATCH --output=slurm-logs/%j-Qwen3-8b-hellaswag-finetune-faster.out


nvidia-smi 
nvidia-smi topo -m 



export CUDA_VISIBLE_DEVICES=""
export HF_HOME=$SCRATCH
export MODEL=Qwen/Qwen3-8B

accelerate launch \
    --config-file=accl-config/fsdp-qwen3-conf.yaml \
    finetune.py \
        --task=hellaswag \
        --base_model=$MODEL \
        --lora_adapter="random_init" \
        --epochs=10 \
        --batch_size=6 \
        --lr=2e-5 \
        --eval_every_steps=500 \
        --gradient_accumulation_steps=1 \
        --output_dir=runs/Qwen3-8B-finetune-hellaswag--$(date +%Y-%m-%d--%T) \
        --force_load_local_dataset \
        --local_dataset_dir=hellaswag_local \
