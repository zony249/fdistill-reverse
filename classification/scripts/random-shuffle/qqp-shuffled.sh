#!/bin/bash

#SBATCH --cpus-per-task=4 # number of cores
#SBATCH --mem=32000 # 100M for the whole job 
#SBATCH --time=0-12:00 # walltime in d-hh:mm or hh:mm:ss format
#SBATCH --account=
#SBATCH --gpus-per-node=a100:1 # GPUs per node
#SBATCH --output=slurm-logs/slurm-%j-qqp-shuffled-3l.out

nvidia-smi


export TASK_NAME=qqp
export EXP_NAME=$(date +%m-%d-%y--%T)--$TASK_NAME-shuffled-3l
export OUTPUT=$SCRATCH/fdistill-reverse/runs/glue/$EXP_NAME

mkdir $SCRATCH/fdistill-reverse/runs/glue
sleep $((RANDOM % 100))

python distillation.py \
  --model_name_or_path models/$TASK_NAME-teacher \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_predict \
  --num_student_layers=3 \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --eval_steps=1000 \
  --eval_batch_size=16 \
  --eval_accumulation_steps=16 \
  --learning_rate 1e-5 \
  --num_train_epochs 10 \
  --alpha_mle=1. --alpha_kl=1. --alpha_hidden=3. \
  --random_matching \
  --seed $((RANDOM % 10000)) \
  --output_dir $OUTPUT