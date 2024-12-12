#!/bin/bash

#SBATCH --cpus-per-task=4 # number of cores
#SBATCH --mem=32000 # 100M for the whole job 
#SBATCH --time=2-00:00 # walltime in d-hh:mm or hh:mm:ss format
#SBATCH --account=
#SBATCH --gres=gpu:1 # GPUs per node
#SBATCH --output=slurm-logs/slurm-%j-finetune-teacher-qnli.out

nvidia-smi

sleep $((RANDOM % 100))
export TASK_NAME=qnli
export EXP_NAME=$(date +%m-%d-%y--%T)--$TASK_NAME-teacher
export OUTPUT=$SCRATCH/fdistill-reverse/runs/glue/$EXP_NAME

mkdir $SCRATCH/fdistill-reverse/runs/glue

python finetune.py \
  --model_name_or_path bert-base-uncased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --output_dir $OUTPUT \
  --seed $((RANDOM % 10000)) \