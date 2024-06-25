#!/bin/sh


#SBATCH --cpus-per-task=4 # number of cores
#SBATCH --mem=32000 # 100M for the whole job 
#SBATCH --time=0-01:00 # walltime in d-hh:mm or hh:mm:ss format
#SBATCH --account=def-lilimou 
#SBATCH --gres=gpu:1 # GPUs per node
#SBATCH --output=slurm-logs/slurm-%j-wmt-student-js-eval.out

nvidia-smi

export MODEL_NAME=models/06-19-24--22:57:21--xsum-student-mle/best_tfmr
echo $MODEL_NAME
export BEAM=6
export SAVE_PATH=runs/pred-$(date +%m-%d-%y--%T)-teacher-eval


python3 run_eval.py \
  --model_name $MODEL_NAME \
  --input_path xsum/test.source \
  --save_path $SAVE_PATH/res.out \
  --reference_path xsum/test.target \
  --score_path $SAVE_PATH/metrics.json \
  --num_beams $BEAM\
  --length_penalty 0.5\
  --device cuda
  "$@"
