import os 
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer 
from peft import PeftModelForCausalLM
from model_utils import create_student_from_teacher
from argparse import ArgumentParser, Namespace 


parser = ArgumentParser("merge adapters with base model") 
parser.add_argument("--base_model", required=True, type=str) 
parser.add_argument("--weight_copy", action="store_true")
parser.add_argument("--adapter", required=True, type=str)
parser.add_argument("--save_as", default=None, type=str)

args = parser.parse_args()

base = args.base_model
adapter = args.adapter
save_as = "/".join(adapter.split("/")[:-1]) if args.save_as is None else args.save_as

print(f"Saving to {save_as}")

if args.weight_copy:
    base_model, tok = create_student_from_teacher(base, "weight_copy")
else: 
    base_model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
# base_model.parallelize(4)
tokenizer = AutoTokenizer.from_pretrained(base)

peft_model = PeftModelForCausalLM.from_pretrained(base_model, adapter)
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained(save_as)
tokenizer.save_pretrained(save_as)

num_params = 0
for p in merged_model.parameters():
    num_params += p.numel() 
print("====== NUMBER OF PARAMETERS ======")
print(f"{num_params:,}")