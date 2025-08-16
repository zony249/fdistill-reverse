#!/bin/bash 


sbatch scripts/coqa/llama/distill-forward-coqa.sh
sbatch scripts/coqa/llama/distill-all-one-coqa.sh
sbatch scripts/coqa/llama/distill-none-coqa.sh
sbatch scripts/coqa/llama/distill-reverse-coqa.sh
sbatch scripts/coqa/llama/distill-shuffle-coqa.sh



sbatch scripts/hellaswag/llama/distill-forward-hellaswag.sh
sbatch scripts/hellaswag/llama/distill-all-one-hellaswag.sh
sbatch scripts/hellaswag/llama/distill-none-hellaswag.sh
sbatch scripts/hellaswag/llama/distill-reverse-hellaswag.sh
sbatch scripts/hellaswag/llama/distill-shuffle-hellaswag.sh