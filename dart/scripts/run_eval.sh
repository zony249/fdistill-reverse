
export MODEL_NAME=models/dart-teacher-regularized/best_tfmr
export EXP_NAME=pred-$(date +%m-%d-%y--%T)
export BEAM=6

mkdir runs/$EXP_NAME

python3 run_eval.py \
  --model_name $MODEL_NAME\
  --input_path dart_eval_data/test.src \
  --ref_dir dart_eval_data \
  --save_path runs/$EXP_NAME/res.out \
  --reference_path dart_eval_data/test.ref0 \
  --score_path runs/$EXP_NAME/metrics.out \
  --device cuda --num_beams $BEAM
  "$@"