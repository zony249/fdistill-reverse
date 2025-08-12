import os 
from copy import deepcopy
import shutil 
from argparse import Namespace, ArgumentParser
from peft import get_peft_model, LoraConfig, TaskType 


import torch
from datasets import load_dataset
from sft_trainer import SFTTrainer, DistillTrainer
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer
)
from sft_trainer import SFTConfig
from data_utils import get_dataset_and_task_processor
from transformers.models.qwen3 import Qwen3ForCausalLM

from accelerate import Accelerator, load_checkpoint_and_dispatch, init_empty_weights

from model_utils import load_model, create_student_from_teacher


if __name__ == "__main__": 

    # Finetune args
    parser = ArgumentParser("finetune.py")
    parser.add_argument("--output_dir", type=str, default="runs")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--task", type=str, required=True, choices=["hellaswag", "wikitext", "coqa"])
    parser.add_argument("--lora_adapter", type=str, default=None, help="\"random_init\", name of adapter, or None")
    parser.add_argument("--eval_every_steps", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--force_load_local_dataset", action="store_true")
    parser.add_argument("--local_dataset_dir", type=str, default=None)

    # Distillation args
    parser.add_argument("--teacher_model", type=str, required=True)
    parser.add_argument("--ce_alpha", type=float, default=1.0)
    parser.add_argument("--kl_alpha", type=float, default=1.0)
    parser.add_argument("--hidden_alpha", type=float, default=3.0)
    parser.add_argument("--matching_location", type=str, default="striped", choices=["last", "striped", "forward", "reverse", "all_one", "shuffle"])
    parser.add_argument("--kl_mode", type=str, default="reverse", choices=["forward", "reverse"])

    args = parser.parse_args()

    # os.makedirs(args.output_dir)

    # load student and teacher models 
    if args.base_model == "weight_copy" or args.base_model == "random_init": 
        model, tok = create_student_from_teacher(args.teacher_model, mode=args.base_model)
    else: 
        model, tok = load_model(args.base_model, torch_dtype=torch.bfloat16,)

    # with init_empty_weights():
    teacher_model, teacher_tok = load_model(args.teacher_model, torch_dtype=torch.bfloat16)

    # teacher_model = load_checkpoint_and_dispatch(
    #     teacher_model, checkpoint=args.teacher_model, device_map="auto", no_split_module_classes=['Qwen3DecoderLayer']
    # )


    datasets, compute_metrics = get_dataset_and_task_processor(args.task, 
                                                               tok=tok, 
                                                               val_test_only=False, 
                                                               load_from_disk=args.force_load_local_dataset, 
                                                               local_dataset_dir=args.local_dataset_dir)

    trainset = datasets["train"]
    eval_set = datasets["validation"]



    # Initialize or load adapters
    if args.lora_adapter == "random_init": 
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=128, 
            lora_alpha=32, 
            lora_dropout=0.1, 
            target_modules = "all-linear"
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        # load lora adapter for model 
    elif args.lora_adapter is not None: 
        raise NotImplementedError("TODO: Implement loading trained adapters")



    accelerator = Accelerator() 
    teacher_model = accelerator.prepare_model(teacher_model, device_placement=True, evaluation_mode=True)

    trainer_cfg = SFTConfig(
        output_dir=args.output_dir, 
        num_train_epochs=args.epochs, 
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size, 
        learning_rate=args.lr, 
        eval_strategy="steps", 
        warmup_steps=args.warmup_steps, 
        eval_steps=args.eval_every_steps, 
        save_steps=args.eval_every_steps, 
        gradient_accumulation_steps=args.gradient_accumulation_steps, 
        save_strategy="best", 
        metric_for_best_model="loss", 
        batch_eval_metrics=True)


    trainer = DistillTrainer(model=model, 
                         processing_class=tok,
                         teacher=teacher_model,
                         args=trainer_cfg, 
                         train_dataset=trainset,
                         eval_dataset=eval_set,  
                         ce_alpha=args.ce_alpha, 
                         kl_alpha=args.kl_alpha, 
                         hidden_alpha=args.hidden_alpha, 
                         matching_location=args.matching_location, 
                         kl_mode=args.kl_mode, 
                         compute_metrics=compute_metrics)
    trainer.train()