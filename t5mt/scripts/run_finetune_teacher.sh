#!/bin/sh


#SBATCH --cpus-per-task=4 # number of cores
#SBATCH --mem=32000 # 100M for the whole job 
#SBATCH --time=2-00:00 # walltime in d-hh:mm or hh:mm:ss format
#SBATCH --account=
#SBATCH --gres=gpu:1 # GPUs per node
#SBATCH --output=slurm-logs/slurm-%j-t5-base-wmt-100k-new-model.out

nvidia-smi

MINWAIT=1
MAXWAIT=100
sleep $((MINWAIT+RANDOM % (MAXWAIT-MINWAIT)))

export ROOT=$SCRATCH/fdistill-reverse
export TEACHER="t5-base"
export OUTPUT=$ROOT/runs/wmt-teacher-$(date +%m-%d-%y--%T)
export SEED=$((RANDOM))

echo "using seed ${SEED}"

mkdir $ROOT/runs
mkdir ${OUTPUT}

python3 finetune.py \
    --learning_rate=5e-4\
    --do_train \
    --val_check_interval=0.5 \
    --adafactor \
    --num_train_epochs 9 \
    --data_dir wmt_en-ro_100k \
    --max_source_length 300 --max_target_length 300 --val_max_target_length 300 --test_max_target_length 300 \
    --train_batch_size=8 --eval_batch_size=4 --eval_beams 2\
    --n_val -1\
    --seed $((RANDOM))\
    --task translation \
    --warmup_steps 500 \
    --gpus 1\
    --output_dir ${OUTPUT} \
    --model_name_or_path ${TEACHER} \
    --tokenizer_name ${TEACHER} \
    --overwrite_output_dir \
    "$@"
