#!/bin/sh


#SBATCH --cpus-per-task=4 # number of cores
#SBATCH --mem=32000 # 100M for the whole job 
#SBATCH --time=0-01:00 # walltime in d-hh:mm or hh:mm:ss format
#SBATCH --account=def-lilimou 
#SBATCH --gres=gpu:1 # GPUs per node
#SBATCH --output=slurm-logs/slurm-%j-wmt-student-js-eval.out

export MODEL_NAME=$SCRATCH/fdistill-reverse/runs/student-js-05-27-24--11:33:39/best_tfmr
echo $MODEL_NAME
export BEAM=5
export OUTPUT=$SCRATCH/fdistill-reverse/evals/pred-$(date +%m-%d-%y)

mkdir $SCRATCH/fdistill-reverse/evals

python3 run_eval.py \
  --model_name $MODEL_NAME \
  --input_path wmt_en-ro_100k/test.source \
  --save_path $OUTPUT/res.out \
  --reference_path wmt_en-ro_100k/test.target \
  --score_path $OUTPUT/metrics.json \
  --device cuda\
  "$@"