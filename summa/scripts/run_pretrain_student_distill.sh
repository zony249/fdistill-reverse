#!/bin/sh


#SBATCH --cpus-per-task=4 # number of cores
#SBATCH --mem=32000 # 100M for the whole job 
#SBATCH --time=3-00:00 # walltime in d-hh:mm or hh:mm:ss format
#SBATCH --account=def-lilimou 
#SBATCH --gres=gpu:1 # GPUs per node
#SBATCH --output=slurm-logs/slurm-%j-xsum-student-predistill-forward.out

nvidia-smi

export EXP_NAME=$(date +%d-%m-%y--%T)--xsum-student-predistill-forward
export MODEL_SAVE_DIR=runs/
export TMP_OUTPUT_PATH=$SLURM_TMPDIR/$EXP_NAME

python distillation.py \
  --teacher facebook/bart-large-xsum \
  --data_dir xsum \
  --tokenizer_name facebook/bart-large-xsum \
  --student_decoder_layers 3 --student_encoder_layers 3 \
  --freeze_embeds \
  --learning_rate=5e-4 \
  --temperature 2.\
  --do_train \
  --gpus 1\
  --val_check_interval 0.3 --n_val -1 --eval_beams 2 --length_penalty=0.5 \
  --max_target_length=60 --val_max_target_length=60 --test_max_target_length=100 \
  --model_name_or_path IGNORED \
  --alpha_hid=3. \
  --train_batch_size=16 --eval_batch_size=8 --gradient_accumulation_steps=1 \
  --sortish_sampler \
  --num_train_epochs=40 \
  --warmup_steps 500 \
  --output_dir $MODEL_SAVE_DIR \
  --overwrite_output_dir \
  --reverse \
