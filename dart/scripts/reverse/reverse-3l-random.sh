#!/bin/bash 

#SBATCH --cpus-per-task=4 # number of cores
#SBATCH --mem=32000 # 100M for the whole job 
#SBATCH --time=3-00:00 # walltime in d-hh:mm or hh:mm:ss format
#SBATCH --account=
#SBATCH --gres=gpu:1 # GPUs per node
#SBATCH --output=slurm-logs/slurm-%j-dart-reverse-3l-random.out

nvidia-smi

export TEACHER=models/dart-teacher-regularized/best_tfmr
export EXP_NAME=$(date +%m-%d-%y--%T)--dart-reverse-3l
export OUTPUT=$SCRATCH/fdistill-reverse/runs/dart/$EXP_NAME


python3 distillation.py \
  --teacher $TEACHER \
  --data_dir dart \
  --tokenizer_name $TEACHER \
  --student_decoder_layers 3 --student_encoder_layers 3 \
  --learning_rate=1e-4 \
  --freeze_embeds \
  --do_train \
  --gpus 1\
  --val_check_interval 0.3 --n_val -1 --eval_beams 5 --length_penalty=1. \
  --max_source_length=128 --max_target_length=128 --val_max_target_length=128 --test_max_target_length=128 \
  --model_name_or_path IGNORED \
  --alpha_hid=3. --alpha_mlm=1. --alpha_ce=1. \
  --train_batch_size=8 --eval_batch_size=8 --gradient_accumulation_steps=1 \
  --num_train_epochs=18 \
  --warmup_steps 100\
  --output_dir $OUTPUT\
  --overwrite_output_dir \
  --random_init_student \
  --reverse_encoder --reverse_decoder \
  --seed $((RANDOM % 10000)) \
  "$@"
