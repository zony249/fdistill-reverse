#!/bin/bash

#SBATCH --cpus-per-task=4 # number of cores
#SBATCH --mem=32000 # 100M for the whole job 
#SBATCH --time=2-00:00 # walltime in d-hh:mm or hh:mm:ss format
#SBATCH --account=def-lilimou 
#SBATCH --gres=gpu:1 # GPUs per node
#SBATCH --output=slurm-logs/slurm-%j-finetune-teacher-cola.out

nvidia-smi


export TASK_NAME=qqp
export EXP_NAME=$(date +%m-%d-%y--%T)--$TASK_NAME-student-reverse
export OUTPUT=runs/$EXP_NAME

python -m debugpy --listen 0.0.0.0:5678 eval2.py \
  --model_name_or_path models/$TASK_NAME-student-reverse/best_tfmr/ \
  --task_name $TASK_NAME \
  --do_eval \
  --max_seq_length 128 \
  --output_dir $OUTPUT \
