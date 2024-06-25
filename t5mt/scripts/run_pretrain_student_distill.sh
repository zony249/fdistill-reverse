#!/bin/sh

#SBATCH --cpus-per-task=4 # number of cores
#SBATCH --mem=32000 # 100M for the whole job 
#SBATCH --time=3-00:00 # walltime in d-hh:mm or hh:mm:ss format
#SBATCH --account=def-lilimou 
#SBATCH --gres=gpu:1 # GPUs per node
#SBATCH --output=slurm-logs/slurm-%j-wmt-student-predistill-forward-deeper.out

nvidia-smi

export TEACHER=models/wmt-teacher-t5-base/best_tfmr
export OUTPUT_NAME=student-predistill-forward-$(date +%m-%d-%y--%T)
export TMP_OUTPUT_PATH=$SLURM_TMPDIR/$OUTPUT_NAME
export MODEL_OUTPUT_PATH=$SCRATCH/fdistill-reverse/runs

python distillation.py \
  --teacher $TEACHER \
  --num_train_epochs 28\
  --adafactor \
  --data_dir wmt_en-ro_100k \
  --tokenizer_name $TEACHER \
  --student_decoder_layers 4 --student_encoder_layers 4 \
  --learning_rate=1e-3 \
  --freeze_embeds \
  --temperature 2. \
  --do_train \
  --task translation\
  --gpus 1\
  --val_check_interval 0.3 --n_val -1 --eval_beams 4 --length_penalty=1. \
  --model_name_or_path IGNORED --normalize_hidden\
  --alpha_hid=3.\
  --train_batch_size=8 --eval_batch_size=8 --gradient_accumulation_steps=1 \
  --warmup_steps 500\
  --output_dir $TMP_OUTPUT_PATH \
  --overwrite_output_dir\
  "$@"



mv $TMP_OUTPUT_PATH $MODEL_OUTPUT_PATH
