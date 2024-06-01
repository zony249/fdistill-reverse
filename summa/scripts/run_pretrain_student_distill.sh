#!/bin/sh


#SBATCH --cpus-per-task=4 # number of cores
#SBATCH --mem=32000 # 100M for the whole job 
#SBATCH --time=0-01:00 # walltime in d-hh:mm or hh:mm:ss format
#SBATCH --account=def-lilimou 
#SBATCH --gres=gpu:1 # GPUs per node
#SBATCH --output=slurm-logs/slurm-%j-wmt-student-js-eval.out

nvidia-smi

export MODEL_SAVE_DIR=runs/$(date +%d-%m-%y--%T)--xsum-student-predistill-forward


python distillation.py \
  --teacher facebook/bart-large-xsum \
  --data_dir xsum \
  --tokenizer_name facebook/bart-large-xsum \
  --student_decoder_layers 3 --student_encoder_layers 3 \
  --freeze_embeds \
  --learning_rate=3e-4 \
  --temperature 2.\
  --do_train \
  --gpus 1\
  --val_check_interval 0.3 --n_val -1 --eval_beams 2 --length_penalty=0.5 \
  --max_target_length=60 --val_max_target_length=60 --test_max_target_length=100 \
  --model_name_or_path IGNORED \
  --alpha_hid=3. \
  --train_batch_size=8 --eval_batch_size=8 --gradient_accumulation_steps=2 \
  --sortish_sampler \
  --num_train_epochs=28\
  --warmup_steps 500 \
  --output_dir '' \
  --overwrite_output_dir\
  "$@"
