export KL_METHOD='js'
export SAVE_MODEL=runs/$(date +%m-%d-%y--%T)--xsum-student-js-forward
export INIT_MODEL=models/xsum-student-predistill
export TEACHER_MODEL=facebook/bart-large-xsum
export DATA_DIR=xsum

python kd.py \
  --teacher $TEACHER_MODEL\
  --data_dir $DATA_DIR\
  --tokenizer_name $TEACHER_MODEL \
  --learning_rate=1e-4 \
  --do_train \
  --gpus 1\
  --sample_beams 5 --do_sample --top_k 30 --beta 0.85\
  --temperature 1.\
  --freeze_embeds \
  --val_check_interval 0.5 --n_val -1 --eval_beams 5 --length_penalty=0.5 \
  --model_name_or_path IGNORED\
  --student $INIT_MODEL\
  --alpha_hid=0. --alpha_ce=1. --alpha_mlm=0.\
  --max_target_length=60 --val_max_target_length=60 --test_max_target_length=100 \
  --train_batch_size=8 --eval_batch_size=5 --gradient_accumulation_steps=2 \
  --warmup_steps 100 \
  --output_dir $SAVE_MODEL \
  --overwrite_output_dir\
  --kd_method $KL_METHOD\
  --num_train_epochs 12\
  "$@"
