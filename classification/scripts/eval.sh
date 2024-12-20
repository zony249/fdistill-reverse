#!/bin/bash

#SBATCH --cpus-per-task=4 # number of cores
#SBATCH --mem=32000 # 100M for the whole job 
#SBATCH --time=2-00:00 # walltime in d-hh:mm or hh:mm:ss format
#SBATCH --account=
#SBATCH --gres=gpu:1 # GPUs per node
#SBATCH --output=slurm-logs/slurm-%j-finetune-teacher-cola.out

nvidia-smi


export TASK_NAME=sst2
export EXP_NAME=$(date +%m-%d-%y--%T)--$TASK_NAME-student-reverse
export OUTPUT=runs/$EXP_NAME

python eval2.py \
  --model_name_or_path runs/$TASK_NAME-student-reverse/best_tfmr \
  --task_name $TASK_NAME \
  --do_eval \
  --max_seq_length 128 \
  --output_dir $OUTPUT \

aaa="xxx"
if [[ $TASK_NAME == "mnli" ]]; then
   export TASK=MNLI-m
elif [[ $TASK_NAME == "qqp" ]]; then
   export TASK=QQP
elif [[ $TASK_NAME == "qnli" ]]; then
   export TASK=QNLI
elif [[ $TASK_NAME == "sst2" ]]; then
   export TASK=SST-2
elif [[ $TASK_NAME == "mrpc" ]]; then
   export TASK=MRPC
elif [[ $TASK_NAME == "rte" ]]; then
   export TASK=RTE
elif [[ $TASK_NAME == "stsb" ]]; then
   export TASK=STS-B
elif [[ $TASK_NAME == "cola" ]]; then
   export TASK=CoLA
else 
    echo "FAILURE"
fi


python convert_glue_preds.py --input_file $OUTPUT/test_results_$TASK_NAME.txt --task $TASK
if [[ $TASK_NAME == "mnli" ]]; then 
   export TASK=MNLI-mm
   python convert_glue_preds.py --input_file $OUTPUT/test_results_$TASK_NAME-mm.txt --task $TASK
fi