#!/bin/bash 

#SBATCH --cpus-per-task=4 # number of cores
#SBATCH --mem=32000 # 100M for the whole job 
#SBATCH --time=3-00:00 # walltime in d-hh:mm or hh:mm:ss format
#SBATCH --account=
#SBATCH --gres=gpu:1 # GPUs per node
#SBATCH --output=slurm-logs/slurm-%j-dart-teacher.out

nvidia-smi


export MODEL_NAME=facebook/bart-large
export EXP_NAME=$(date +%m-%d-%y--%T)--dart-teacher
export MODEL_OUTPUT_PATH=runs/$EXP_NAME

mkdir runs

python3 finetune.py \
    --learning_rate=1e-5 \
    --gpus 1 \
    --do_train \
    --do_predict \
    --n_val -1 \
    --num_train_epochs 16 \
    --warmup_steps=100\
    --train_batch_size=16 --eval_batch_size=4 --gradient_accumulation_steps=1 \
    --max_source_length 128 --max_target_length=256 --val_max_target_length=256 --test_max_target_length=256\
    --val_check_interval 0.5 --eval_beams 5\
    --data_dir dart_new \
    --model_name_or_path $MODEL_NAME \
    --output_dir $MODEL_OUTPUT_PATH \
    --overwrite_output_dir\
    "$@"
