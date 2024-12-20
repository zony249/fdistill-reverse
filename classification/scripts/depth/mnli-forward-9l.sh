#!/bin/bash

#SBATCH --cpus-per-task=4 # number of cores
#SBATCH --mem=32000 # 100M for the whole job 
#SBATCH --time=0-12:00 # walltime in d-hh:mm or hh:mm:ss format
#SBATCH --account=
#SBATCH --gres=gpu:1 # GPUs per node
#SBATCH --output=slurm-logs/slurm-%j-mnli-9l-forward.out

nvidia-smi


export TASK_NAME=mnli
export EXP_NAME=$(date +%m-%d-%y--%T)--$TASK_NAME-forward-9l
export OUTPUT=$SCRATCH/fdistill-reverse/runs/glue/$EXP_NAME

python -m debugpy --listen 0.0.0.0:5678 distillation.py \
  --model_name_or_path models/$TASK_NAME-teacher \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_predict \
  --num_student_layers=9 \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --eval_steps=1000 \
  --eval_batch_size=16 \
  --eval_accumulation_steps=16 \
  --learning_rate 1e-5 \
  --num_train_epochs 10 \
  --alpha_mle=1. --alpha_kl=1. --alpha_hidden=3. \
  --output_dir $OUTPUT \