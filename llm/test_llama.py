import os
from transformers import AutoTokenizer, AutoConfig
# from models import Qwen3ForCausalLM, Qwen3ModelParallel, Qwen3ForCausalLMParallel
from transformers import AutoModelForCausalLM
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

config = AutoConfig.from_pretrained("runs/transfers/Llama-3.1-8B-coqa-all_one-distill")
model = AutoModelForCausalLM.from_pretrained("runs/transfers/Llama-3.1-8B-coqa-all_one-distill", device_map="auto")

# model.parallelize(4)
tok = AutoTokenizer.from_pretrained("runs/transfers/Llama-3.1-8B-coqa-all_one-distill") 




prompt = "Tell me about LLMs please and thank"
inputs = tok([prompt], return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, output_hidden_states=True, output_attentions=True)
print("".join(tok.batch_decode(outputs[0])))
# print(outputs.hidden_states)