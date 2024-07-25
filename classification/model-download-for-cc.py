from transformers import MBartForConditionalGeneration
from transformers import T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from copy import deepcopy
import torch 
from torch import nn

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
tok = AutoTokenizer.from_pretrained("bert-base-uncased")

config_mnli = AutoConfig.from_pretrained(
    "bert-base-uncased", 
    num_labels=3,
    finetuning_task="mnli",
)
config_stsb = AutoConfig.from_pretrained(
    "bert-base-uncased", 
    num_labels=1,
    finetuning_task="stsb",
)


model_mnli= AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    config=config_mnli,
)
model_stsb= AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    config=config_stsb,
)

model.save_pretrained("bert-base-uncased")
tok.save_pretrained("bert-base-uncased")

model_mnli.save_pretrained("bert-base-uncased-mnli")
tok.save_pretrained("bert-base-uncased-mnli")

model_stsb.save_pretrained("bert-base-uncased-stsb")
tok.save_pretrained("bert-base-uncased-stsb")
# model = T5ForConditionalGeneration.from_pretrained("t5-base")
# tok = AutoTokenizer.from_pretrained("t5-base")

# model.save_pretrained("t5-base")
# tok.save_pretrained("t5-base")


print("Done. ")