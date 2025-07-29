import os 
import sys 
from copy import deepcopy 
from argparse import Namespace, ArgumentParser 
import yaml

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoConfig,
    AutoModel, 
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    AwqConfig 
)

if __name__ == "__main__": 

    parser = ArgumentParser(description="args") 
    parser.add_argument("model_name_or_path", type=str, help="huggingface id") 
    parser.add_argument("--type", type=str, default="causal", choices=[None, "causal", "seq2seq"], help="type of model")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--save_to", type=str, default=None, help="Directory to save the model to.")

    args = parser.parse_args()


    if args.type == "causal": 
        AutoClass = AutoModelForCausalLM 
    elif args.type == "seq2seq": 
        AutoClass = AutoModelForSeq2SeqLM
    else: 
        AutoClass = AutoModel
        
    

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    model = AutoClass.from_pretrained(args.model_name_or_path, torch_dtype=dtype)
                                    #   quantization_config=quant_config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    config = AutoConfig.from_pretrained(args.model_name_or_path)


    save_dest = os.path.join(args.save_to, args.model_name_or_path) if args.save_to is not None else args.model_name_or_path

    model.save_pretrained(save_dest)
    tokenizer.save_pretrained(save_dest)
