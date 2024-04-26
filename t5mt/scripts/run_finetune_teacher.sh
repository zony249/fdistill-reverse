#!/bin/sh

export TEACHER="facebook/mbart-large-cc25"
export OUTPUT="runs/wmt-teacher-$(date +%m-%d-%y--%T)"

mkdir runs
mkdir ${OUTPUT}

python3 finetune.py \
    --learning_rate=5e-4\
    --do_train \
    --val_check_interval=0.5 \
    --adafactor \
    --num_train_epochs 9 \
    --data_dir wmt_en_ro \
    --max_source_length 300 --max_target_length 300 --val_max_target_length 300 --test_max_target_length 300 \
    --train_batch_size=8 --eval_batch_size=4 --eval_beams 2\
    --n_val -1\
    --seed 42\
    --task translation \
    --warmup_steps 500 \
    --gpus 1\
    --output_dir ${OUTPUT} \
    --model_name_or_path ${TEACHER} \
    --tokenizer_name ${TEACHER} \
    --overwrite_output_dir \
    "$@"
