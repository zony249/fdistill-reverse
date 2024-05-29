#!/bin/sh

#SBATCH --cpus-per-task=4 # number of cores
#SBATCH --mem=32000 # 100M for the whole job 
#SBATCH --time=7-00:00 # walltime in d-hh:mm or hh:mm:ss format
#SBATCH --account=def-lilimou 
#SBATCH --gres=gpu:1 # GPUs per node
#SBATCH --output=slurm-logs/slurm-%j-wmt-teacher-generate-pseudolabel.out

export TO_DIR=wmt_en-ro_100k_teacher_modified

export ORG_DIR=wmt_en-ro_100k
cp $ORG_DIR/ $TO_DIR/ -r

python3 run_eval.py \
  --model_name models/wmt-teacher-t5-base/best_tfmr\
  --input_path $ORG_DIR/train.source\
  --save_path $TO_DIR/train.target \
  --reference_path $ORG_DIR/train.target \
  --score_path $SLURM_TMPDIR/_tmp.json \
  --max_input_length 300\
  --device cuda --num_beams 5\
  "$@"

#cp $SLURM_TMPDIR/teacher.out $TO_DIR/train.target
