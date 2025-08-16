#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=l40s:4
#SBATCH --mem=96000M
#SBATCH --time=1-00:00
#SBATCH --account=
#SBATCH --output=slurm-logs/%j-Qwen3-8b-coqa-reverse-distill.out



nvidia-smi 
nvidia-smi topo -m 



# export CUDA_VISIBLE_DEVICES=
# export NCCL_P2P_DISABLE=1
export HF_HOME=$SCRATCH

    # --config-file=accl-config/fsdp-qwen3-parallel-conf.yaml \
accelerate launch \
    --num_processes=4 \
    distillation.py \
        --task=coqa \
        --base_model=weight_copy \
        --teacher_model=Qwen/Qwen3-8B-finetune-coqa-2 \
        --lora_adapter=random_init \
        --epochs=15 \
        --batch_size=1 \
        --lr=1e-4 \
        --warmup_steps=1000 \
        --gradient_accumulation_steps=2 \
        --eval_every_steps=500 \
        --ce_alpha=1.0 --kl_alpha=1.0 --hidden_alpha=3.0 \
        --matching_location=reverse \
        --output_dir=runs/q3-8b-coqa-reverse-distill--$(date +%Y-%m-%d--%T) \
        --force_load_local_dataset \
        --local_dataset_dir=coqa_local \
