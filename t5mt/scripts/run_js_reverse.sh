#!/bin/sh

#SBATCH --cpus-per-task=4 # number of cores
#SBATCH --mem=32000 # 100M for the whole job 
#SBATCH --time=7-00:00 # walltime in d-hh:mm or hh:mm:ss format
#SBATCH --account=def-lilimou 
#SBATCH --gres=gpu:1 # GPUs per node
#SBATCH --output=slurm-logs/slurm-%j-wmt-student-js-forward.out

export KL_METHOD="js"
export MODEL_SAVE_PATH=runs/wmt-student-js-reverse-$(date +%m-%d-%y--%T)
export INIT_MODEL=models/wmt-student-predistill-reverse
export DATA_DIR=wmt_en-ro_100k_teacher_modified
export TEACHER_MODEL=models/wmt-teacher-t5-base/best_tfmr


python3 kd.py \
  --teacher $TEACHER_MODEL\
  --adafactor \
  --data_dir $DATA_DIR \
  --adafactor \
  --tokenizer_name $INIT_MODEL\
  --learning_rate=5e-4 \
  --do_train \
  --gpus 1\
  --sample_beams 5 --do_sample --top_k 40 --beta 0.25\
  --task translation\
  --temperature 1.\
  --freeze_embeds \
  --val_check_interval 0.5 --n_val -1 --eval_beams 5 --length_penalty=1. \
  --model_name_or_path IGNORED\
  --student $INIT_MODEL \
  --alpha_hid=0. --alpha_ce=1. --alpha_mlm=0.\
  --train_batch_size=8 --eval_batch_size=5 --gradient_accumulation_steps=1 \
  --warmup_steps 100 \
  --output_dir $MODEL_SAVE_PATH \
  --overwrite_output_dir\
  --kd_method $KL_METHOD\
  --num_train_epochs 12\
  "$@"

mkdir $MODEL_SAVE_PATH