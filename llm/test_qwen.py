import os
from transformers import AutoTokenizer, AutoConfig
from models import Qwen3ForCausalLM, Qwen3ModelParallel, Qwen3ForCausalLMParallel
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", device_map="auto")

# model.parallelize(4)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B") 




prompt = "Tell me about LLMs please and thank"
inputs = tok([prompt], return_tensors="pt").to(model.device)

outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
# print("".join(tok.batch_decode(outputs[0])))
print(outputs.hidden_states)