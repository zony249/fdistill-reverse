#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=l40s:4
#SBATCH --mem=64000M
#SBATCH --time=2-00:00
#SBATCH --account=aip-lilimou
#SBATCH --output=slurm-logs/%j-gpt-oss-20b-coqa-finetune.out


nvidia-smi 
nvidia-smi topo -m 



# export CUDA_VISIBLE_DEVICES=""
export LD_LIBRARY_PATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/CUDA/cuda12.6/cudnn/9.10.0.56/lib64:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export HF_HOME=$SCRATCH
export MODEL=openai/gpt-oss-20b

accelerate launch \
    --config-file=accl-config/deepspeed-3-gpt-conf.yaml \
    finetune.py \
        --task=coqa \
        --base_model=$MODEL \
        --lora_adapter="random_init" \
        --epochs=2 \
        --batch_size=1 \
        --lr=1e-5 \
        --eval_every_steps=100 \
        --gradient_accumulation_steps=2 \
        --output_dir=runs/gpt-oss-20b-finetune-coqa--$(date +%Y-%m-%d--%T) \
        --force_load_local_dataset \
        --local_dataset_dir=coqa_local \
