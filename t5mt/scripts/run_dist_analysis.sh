
#!/bin/sh


#SBATCH --cpus-per-task=4 # number of cores
#SBATCH --mem=32000 # 100M for the whole job 
#SBATCH --time=0-01:00 # walltime in d-hh:mm or hh:mm:ss format
#SBATCH --account=
#SBATCH --gres=gpu:1 # GPUs per node
#SBATCH --output=slurm-logs/slurm-%j-wmt-student-predistill-forward-eval.out

export MODEL_NAME=models/wmt-teacher-t5-base/best_tfmr
export STUDENT=models/er-dr-6-layer/best_tfmr
echo $MODEL_NAME
export BEAM=1
export OUTPUT=runs/pred-$(date +%m-%d-%y--%T)


python3 distance_analysis.py \
  --model_name $MODEL_NAME \
  --student_model $STUDENT \
  --input_path wmt_en-ro_100k/test.source \
  --save_path $OUTPUT/res.out \
  --reference_path wmt_en-ro_100k/test.target \
  --score_path $OUTPUT/metrics.json \
  --device cuda\
  --num_beams=$BEAM \
  --pca_mode=decoder_only\
  --bs 16 \
  --reverse_decoder \
  --reverse_encoder \
  # --display_centroids \
  # --reverse_decoder \
  # --unnormalized \
  "$@"