#!/bin/bash





export TASK_NAME=mrpc
export EXP_NAME=$(date +%m-%d-%y--%T)--mrpc-teacher
export OUTPUT=runs/$EXP_NAME

python finetune.py \
  --model_name_or_path bert-base-uncased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_predict \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --eval_steps=25 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --output_dir $OUTPUT