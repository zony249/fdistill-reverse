#!/bin/bash

#SBATCH --cpus-per-task=4 # number of cores
#SBATCH --mem=32000 # 100M for the whole job 
#SBATCH --time=2-00:00 # walltime in d-hh:mm or hh:mm:ss format
#SBATCH --account=
#SBATCH --gres=gpu:1 # GPUs per node
#SBATCH --output=slurm-logs/slurm-%j-finetune-teacher-mnli.out

nvidia-smi


export TASK_NAME=mnli
export SUFFIX=untrained-random
export TEACHER_MODEL=models/$TASK_NAME-teacher/best_tfmr/
export STUDENT_MODEL=models/$TASK_NAME-$SUFFIX/best_tfmr
export EXP_NAME=$(date +%m-%d-%y--%T)--$TASK_NAME-dist-$SUFFIX
export OUTPUT=runs/$EXP_NAME

python distance_analysis.py \
  --model_name_or_path $TEACHER_MODEL \
  --student_model $STUDENT_MODEL \
  --task_name $TASK_NAME \
  --do_eval \
  --max_seq_length 128 \
  --output_dir $OUTPUT \
  # --from_student_layer 3 \
  # --center_hidden_states \
