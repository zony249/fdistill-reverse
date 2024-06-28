#!/bin/sh

#SBATCH --cpus-per-task=4 # number of cores
#SBATCH --mem=32000 # 100M for the whole job 
#SBATCH --time=2-23:00 # walltime in d-hh:mm or hh:mm:ss format
#SBATCH --account=def-lilimou 
#SBATCH --gres=gpu:1 # GPUs per node
#SBATCH --output=slurm-logs/slurm-%j-ef-dr-same.out

nvidia-smi

export TEACHER=facebook/bart-large-xsum
export OUTPUT_NAME=$(date +%m-%d-%y--%T)-ef-dr-same
export MODEL_OUTPUT_PATH=runs/$OUTPUT_NAME

mkdir runs

python distillation.py \
  --teacher $TEACHER \
  --num_train_epochs 40\
  --data_dir xsum \
  --tokenizer_name $TEACHER \
  --student_decoder_layers 3 --student_encoder_layers 3 \
  --learning_rate=5e-4 \
  --freeze_embeds \
  --temperature 2. \
  --do_train \
  --gpus 1\
  --val_check_interval 0.3 --n_val -1 --eval_beams 2 --length_penalty=0.5 \
  --max_target_length=60 --val_max_target_length=60 --test_max_target_length=100 \
  --model_name_or_path IGNORED \
  --alpha_hid=3. --alpha_ce=0. --alpha_mlm=0.2\
  --train_batch_size=16 --eval_batch_size=16 --gradient_accumulation_steps=1 \
  --sortish_sampler \
  --warmup_steps 500\
  --output_dir $MODEL_OUTPUT_PATH\
  --overwrite_output_dir\
  --reverse_decoder --copy_same_order\
  "$@"



