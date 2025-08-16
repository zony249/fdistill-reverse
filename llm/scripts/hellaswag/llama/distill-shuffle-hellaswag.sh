#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=l40s:4
#SBATCH --mem=96000M
#SBATCH --time=3-00:00
#SBATCH --account=aip-lilimou
#SBATCH --output=slurm-logs/%j-Llama-3.1-8b-hellaswag-shuffle-distill.out



nvidia-smi 
nvidia-smi topo -m 



# export CUDA_VISIBLE_DEVICES=""
export NUM_PROCESSES=4
export HF_HOME=$SCRATCH
export TEACHER=meta-llama/Llama-3.1-8B-finetune-hellaswag

    # --config-file=accl-config/fsdp-qwen3-parallel-conf.yaml \
accelerate launch \
    --num_processes=$NUM_PROCESSES \
    distillation.py \
        --task=hellaswag \
        --base_model=weight_copy \
        --teacher_model=$TEACHER \
        --lora_adapter=random_init \
        --epochs=160 \
        --batch_size=1 \
        --lr=3e-4 \
        --warmup_steps=1000 \
        --gradient_accumulation_steps=2 \
        --eval_every_steps=1000 \
        --ce_alpha=1.0 --kl_alpha=1.0 --hidden_alpha=3.0 \
        --matching_location=shuffle \
        --output_dir=runs/Llama-3.1-8B-hellaswag-shuffle-distill--$(date +%Y-%m-%d--%T) \
        --force_load_local_dataset \
        --local_dataset_dir=hellaswag_local \
