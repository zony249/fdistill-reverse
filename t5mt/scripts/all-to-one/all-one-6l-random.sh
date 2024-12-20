#!/bin/sh

#SBATCH --cpus-per-task=4 # number of cores
#SBATCH --mem=32000 # 100M for the whole job 
#SBATCH --time=2-23:00 # walltime in d-hh:mm or hh:mm:ss format
#SBATCH --account=
#SBATCH --gpus-per-node=a100:1 # GPUs per node
#SBATCH --output=slurm-logs/slurm-%j-all-one-6l-random.out

nvidia-smi

export TEACHER=models/wmt-teacher-t5-base/best_tfmr
export OUTPUT_NAME=$(date +%m-%d-%y--%T)--all-one-6l
export MODEL_OUTPUT_PATH=$SCRATCH/fdistill-reverse/runs/t5mt/$OUTPUT_NAME
export SEED=$((RANDOM % 10000))

mkdir $SCRATCH/fdistill-reverse/runs/t5mt/

python distillation.py \
  --teacher $TEACHER \
  --num_train_epochs 40\
  --adafactor \
  --data_dir wmt_en-ro_100k \
  --tokenizer_name $TEACHER \
  --student_decoder_layers 6 --student_encoder_layers 6 \
  --learning_rate=1e-4 \
  --freeze_embeds \
  --temperature 2. \
  --do_train \
  --task translation\
  --gpus 1\
  --val_check_interval 0.3 --n_val -1 --eval_beams 4 --length_penalty=1. \
  --model_name_or_path IGNORED --normalize_hidden\
  --alpha_hid=3. --alpha_ce=1. --alpha_mlm=1.\
  --train_batch_size=8 --eval_batch_size=8 --gradient_accumulation_steps=1 \
  --warmup_steps 500\
  --output_dir $MODEL_OUTPUT_PATH\
  --overwrite_output_dir\
  --match_all_layers --to 6 \
  --random_init_student \
  --seed $SEED \
  "$@"



